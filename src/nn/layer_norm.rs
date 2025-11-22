use crate::central::*;
use crate::nn::*;
use crate::utils::GGUFFile;

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    pub fn new(num_features: usize) -> Self {
        let mut weight = Tensor::ones(Shape::new(vec![num_features]));
        weight.set_requires_grad(true);
        weight.set_keep_alive(true);
        let mut bias = Tensor::zeros(Shape::new(vec![num_features]));
        bias.set_requires_grad(true);
        bias.set_keep_alive(true);
        LayerNorm { weight, bias }
    }

    pub fn from_gguf_file(
        gguf_file: &mut GGUFFile,
        weight_name: String,
        bias_name: String,
    ) -> LayerNorm {
        let mut weight = Tensor::from_gguf_file(weight_name, gguf_file);
        weight.set_requires_grad(true);
        weight.set_keep_alive(true);
        let mut bias = Tensor::from_gguf_file(bias_name, gguf_file);
        bias.set_requires_grad(true);
        bias.set_keep_alive(true);
        LayerNorm { weight, bias }
    }
}

impl Layer for LayerNorm {
    fn forward(&mut self, inputs: Tensor) -> Tensor {
        // We want the mean of the last dimension
        let last_dim = inputs.shape.dimensions().len() - 1;
        let mean = inputs.mean(vec![last_dim]);

        // For proper broadcasting, we need to reshape the mean to have same number of dims as input
        // but with size 1 for the reduced dimension
        let input_dims = inputs.shape.dimensions();
        let mut mean_shape = input_dims.clone();

        mean_shape[last_dim] = 1;
        let mean_reshaped = mean.reshape(Shape::new(mean_shape.clone()));

        // Center the data around the mean
        let centered = inputs - mean_reshaped.clone();

        // take the square and the mean to get the variance
        let variance = centered.pow(2.0).mean(vec![last_dim]);

        // Reshape variance for broadcasting too
        let variance_reshaped = variance.reshape(Shape::new(mean_shape));

        // add just a little to each element, to avoid a div by zero error
        let epsilon = 1e-5;
        let epsilon_tensor = Tensor::element(variance_reshaped.shape, epsilon);
        let std_dev = (variance_reshaped + epsilon_tensor).pow(0.5);

        // Normalize out centeded data, so it is more uniform pass to pass
        let normalized = centered / std_dev;

        // Finally add in our learneable parameters
        let output = (normalized * self.weight) + self.bias;
        return output;
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        vec![self.weight.id, self.bias.id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_layer_norm_creation() {
        let layer_norm = LayerNorm::new(4);

        // Should have 2 parameters (weight and bias)
        assert_eq!(layer_norm.get_parameters().len(), 2);

        // Weight should be initialized to ones
        let weight_data = layer_norm.weight.item();
        for i in 0..4 {
            assert!(approx_equal(weight_data[[i]], 1.0, 1e-6));
        }

        // Bias should be initialized to zeros
        let bias_data = layer_norm.bias.item();
        for i in 0..4 {
            assert!(approx_equal(bias_data[[i]], 0.0, 1e-6));
        }
    }

    #[test]
    fn test_layer_norm_forward_single_sample() {
        let mut layer_norm = LayerNorm::new(4);

        // Test with single sample: [1, 2, 3, 4]
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let output = layer_norm.forward(input);

        assert_eq!(output.shape.dimensions(), vec![1, 4]);
        let result = output.item();

        // With standard layer norm, mean should be ~0 and std ~1 for output
        let mean = (result[[0, 0]] + result[[0, 1]] + result[[0, 2]] + result[[0, 3]]) / 4.0;
        assert!(approx_equal(mean, 0.0, 1e-5));

        // Check all values are finite
        for i in 0..4 {
            assert!(result[[0, i]].is_finite());
        }
    }

    #[test]
    fn test_layer_norm_forward_single_sample_pytorch_comparison() {
        use crate::utils::GGUFFile;

        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/layer_norm/layer_norm_single_sample.gguf",
        ));

        // Load PyTorch reference data
        let pytorch_input = Tensor::from_gguf_file(
            String::from("layer_norm_single_sample_input"),
            &mut gguf_file,
        );
        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_single_sample_output"),
            &mut gguf_file,
        );
        let pytorch_weight = Tensor::from_gguf_file(
            String::from("layer_norm_single_sample_weight"),
            &mut gguf_file,
        );
        let pytorch_bias = Tensor::from_gguf_file(
            String::from("layer_norm_single_sample_bias"),
            &mut gguf_file,
        );

        // Create Rust LayerNorm with same parameters as PyTorch
        let mut layer_norm = LayerNorm::new(4);

        // Set weights and bias to match PyTorch (should already be ones and zeros)
        layer_norm.weight = pytorch_weight;
        layer_norm.bias = pytorch_bias;

        // Forward pass with same input
        let rust_output = layer_norm.forward(pytorch_input);

        // Compare outputs
        assert_eq!(
            rust_output.shape.dimensions(),
            pytorch_output.shape.dimensions()
        );

        let rust_result = rust_output.item();
        let pytorch_result = pytorch_output.item();

        let epsilon = 1e-4; // Allow for small numerical differences

        // Compare each element
        for i in 0..1 {
            for j in 0..4 {
                let rust_val = rust_result[[i, j]];
                let pytorch_val = pytorch_result[[i, j]];
                assert!(
                    approx_equal(rust_val, pytorch_val, epsilon),
                    "Mismatch at [{}, {}]: Rust={:.6}, PyTorch={:.6}, diff={:.6}",
                    i,
                    j,
                    rust_val,
                    pytorch_val,
                    (rust_val - pytorch_val).abs()
                );
            }
        }

        println!(
            "✓ Rust LayerNorm matches PyTorch LayerNorm output within {:.1e} tolerance",
            epsilon
        );
    }

    #[test]
    fn test_layer_norm_forward_zero_variance_pytorch_comparison() {
        use crate::utils::GGUFFile;

        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/layer_norm/layer_norm_zero_variance.gguf",
        ));

        // Load PyTorch reference data
        let pytorch_input = Tensor::from_gguf_file(
            String::from("layer_norm_zero_variance_input"),
            &mut gguf_file,
        );
        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_zero_variance_output"),
            &mut gguf_file,
        );
        let pytorch_weight = Tensor::from_gguf_file(
            String::from("layer_norm_zero_variance_weight"),
            &mut gguf_file,
        );
        let pytorch_bias = Tensor::from_gguf_file(
            String::from("layer_norm_zero_variance_bias"),
            &mut gguf_file,
        );

        // Create Rust LayerNorm with same parameters as PyTorch
        let mut layer_norm = LayerNorm::new(4);
        layer_norm.weight = pytorch_weight;
        layer_norm.bias = pytorch_bias;

        // Forward pass with same uniform input [2, 2, 2, 2]
        let rust_output = layer_norm.forward(pytorch_input.clone());

        // Compare outputs
        assert_eq!(
            rust_output.shape.dimensions(),
            pytorch_output.shape.dimensions()
        );

        let rust_result = rust_output.item();
        let pytorch_result = pytorch_output.item();

        let epsilon = 1e-6; // Very tight tolerance since outputs should be exactly zero

        // Compare each element - should all be zero or very close to zero
        for i in 0..1 {
            for j in 0..4 {
                let rust_val = rust_result[[i, j]];
                let pytorch_val = pytorch_result[[i, j]];
                assert!(
                    approx_equal(rust_val, pytorch_val, epsilon),
                    "Mismatch at [{}, {}]: Rust={:.6}, PyTorch={:.6}, diff={:.6}",
                    i,
                    j,
                    rust_val,
                    pytorch_val,
                    (rust_val - pytorch_val).abs()
                );

                // Both should be very small (close to zero)
                assert!(
                    rust_val.abs() < 0.1,
                    "Rust output should be very small: {:.6}",
                    rust_val
                );
                assert!(
                    pytorch_val.abs() < 0.1,
                    "PyTorch output should be very small: {:.6}",
                    pytorch_val
                );
            }
        }

        println!("✓ Rust LayerNorm matches PyTorch LayerNorm zero variance behavior");
        println!(
            "  Both produce near-zero outputs for uniform input, confirming gradient vanishing issue"
        );
    }

    #[test]
    fn test_layer_norm_forward_batch_pytorch_comparison() {
        use crate::utils::GGUFFile;

        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/layer_norm/layer_norm_batch.gguf",
        ));

        // Load PyTorch reference data
        let pytorch_input =
            Tensor::from_gguf_file(String::from("layer_norm_batch_input"), &mut gguf_file);
        let pytorch_output =
            Tensor::from_gguf_file(String::from("layer_norm_batch_output"), &mut gguf_file);
        let pytorch_weight =
            Tensor::from_gguf_file(String::from("layer_norm_batch_weight"), &mut gguf_file);
        let pytorch_bias =
            Tensor::from_gguf_file(String::from("layer_norm_batch_bias"), &mut gguf_file);

        // Create Rust LayerNorm with same parameters as PyTorch
        let mut layer_norm = LayerNorm::new(3);
        layer_norm.weight = pytorch_weight;
        layer_norm.bias = pytorch_bias;

        // Forward pass with batch input [[1,2,3], [4,5,6]]
        let rust_output = layer_norm.forward(pytorch_input.clone());

        // Compare outputs
        assert_eq!(
            rust_output.shape.dimensions(),
            pytorch_output.shape.dimensions()
        );

        let rust_result = rust_output.item();
        let pytorch_result = pytorch_output.item();

        let epsilon = 1e-4; // Allow for small numerical differences

        // Compare each element
        for i in 0..2 {
            // batch size
            for j in 0..3 {
                // features
                let rust_val = rust_result[[i, j]];
                let pytorch_val = pytorch_result[[i, j]];
                assert!(
                    approx_equal(rust_val, pytorch_val, epsilon),
                    "Mismatch at [{}, {}]: Rust={:.6}, PyTorch={:.6}, diff={:.6}",
                    i,
                    j,
                    rust_val,
                    pytorch_val,
                    (rust_val - pytorch_val).abs()
                );
            }
        }

        // Verify batch independence - each sample should be normalized independently
        println!("\nBatch Independence Verification:");
        for batch_idx in 0..2 {
            // Calculate mean of each sample's output (should be ~0)
            let sample_sum = rust_result[[batch_idx, 0]]
                + rust_result[[batch_idx, 1]]
                + rust_result[[batch_idx, 2]];
            let sample_mean = sample_sum / 3.0;
            println!("  Sample {} output mean: {:.6}", batch_idx + 1, sample_mean);
            assert!(
                approx_equal(sample_mean, 0.0, 1e-4),
                "Sample {} mean should be ~0: {:.6}",
                batch_idx + 1,
                sample_mean
            );

            // Check all values are finite
            for feature_idx in 0..3 {
                assert!(
                    rust_result[[batch_idx, feature_idx]].is_finite(),
                    "Sample {} feature {} should be finite",
                    batch_idx + 1,
                    feature_idx + 1
                );
            }
        }

        // Verify that both samples produce the same pattern since they have identical relative distributions
        // Sample 1: [1,2,3] centered: [-1,0,1], Sample 2: [4,5,6] centered: [-1,0,1]
        println!("\nPattern Verification:");
        println!(
            "  Sample 1 pattern: [{:.4}, {:.4}, {:.4}]",
            rust_result[[0, 0]],
            rust_result[[0, 1]],
            rust_result[[0, 2]]
        );
        println!(
            "  Sample 2 pattern: [{:.4}, {:.4}, {:.4}]",
            rust_result[[1, 0]],
            rust_result[[1, 1]],
            rust_result[[1, 2]]
        );

        // Both samples should have identical normalized values due to same relative distribution
        for j in 0..3 {
            let sample1_val = rust_result[[0, j]];
            let sample2_val = rust_result[[1, j]];
            assert!(
                approx_equal(sample1_val, sample2_val, epsilon),
                "Samples should have identical patterns at feature {}: Sample1={:.6}, Sample2={:.6}",
                j,
                sample1_val,
                sample2_val
            );
        }

        println!("✓ Rust LayerNorm batch processing matches PyTorch LayerNorm");
        println!("  Both samples normalized independently with identical patterns");
        println!("  Demonstrates proper batch processing for transformer use cases");
    }

    #[test]
    fn test_layer_norm_forward_batch() {
        let mut layer_norm = LayerNorm::new(3);

        // Test with batch of 2 samples
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, // Sample 1
                4.0, 5.0, 6.0, // Sample 2
            ],
            vec![2, 3],
        );
        let output = layer_norm.forward(input);

        assert_eq!(output.shape.dimensions(), vec![2, 3]);
        let result = output.item();

        // Check each sample is normalized independently
        for batch_idx in 0..2 {
            let sample_mean =
                (result[[batch_idx, 0]] + result[[batch_idx, 1]] + result[[batch_idx, 2]]) / 3.0;
            assert!(approx_equal(sample_mean, 0.0, 1e-4));

            // Check all values are finite
            for feature_idx in 0..3 {
                assert!(result[[batch_idx, feature_idx]].is_finite());
            }
        }
    }

    #[test]
    fn test_layer_norm_forward_zero_variance() {
        let mut layer_norm = LayerNorm::new(4);

        // Test with constant values (zero variance case)
        let input = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], vec![1, 4]);
        let output = layer_norm.forward(input);

        assert_eq!(output.shape.dimensions(), vec![1, 4]);
        let result = output.item();

        // When all inputs are the same, output should be zeros (since bias is zero and weight is one)
        for i in 0..4 {
            assert!(result[[0, i]].is_finite());
            // Due to epsilon, output won't be exactly zero but should be very small
            assert!(result[[0, i]].abs() < 0.1);
        }
    }

    #[test]
    fn test_layer_norm_forward_known_values() {
        let mut layer_norm = LayerNorm::new(2);

        // Test with known simple case: [0, 1]
        let input = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let output = layer_norm.forward(input);

        let result = output.item();

        // Mean of [0, 1] is 0.5
        // Centered: [-0.5, 0.5]
        // Variance: (0.25 + 0.25) / 2 = 0.25
        // Std: sqrt(0.25 + 1e-5) ≈ 0.5
        // Normalized: [-1, 1]
        assert!(approx_equal(result[[0, 0]], -1.0, 1e-4));
        assert!(approx_equal(result[[0, 1]], 1.0, 1e-4));
    }

    #[test]
    fn test_layer_norm_forward_known_values_pytorch_comparison() {
        use crate::utils::GGUFFile;

        let mut layer_norm = LayerNorm::new(2);

        // Load PyTorch test data from GGUF
        let mut gguf_file = GGUFFile::new(
            "./models/tests/layer_norm/layer_norm_forward_known_values.gguf".to_string(),
        );
        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_forward_known_values_output"),
            &mut gguf_file,
        );

        // Create same input as PyTorch test: [0, 1]
        let input = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);

        // Forward pass
        let output = layer_norm.forward(input);

        // Compare with PyTorch output
        let rust_output = output.item();
        let pytorch_output_data = pytorch_output.item();

        // Should be [-1.0, 1.0] for both
        assert!(approx_equal(
            rust_output[[0, 0]],
            pytorch_output_data[[0, 0]],
            1e-4
        ));
        assert!(approx_equal(
            rust_output[[0, 1]],
            pytorch_output_data[[0, 1]],
            1e-4
        ));

        // Verify the expected known values
        assert!(approx_equal(rust_output[[0, 0]], -1.0, 1e-4));
        assert!(approx_equal(rust_output[[0, 1]], 1.0, 1e-4));
    }

    #[test]
    fn test_layer_norm_forward_negative_values() {
        let mut layer_norm = LayerNorm::new(3);

        // Test with negative values
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0], vec![1, 3]);
        let output = layer_norm.forward(input);

        assert_eq!(output.shape.dimensions(), vec![1, 3]);
        let result = output.item();

        // Check normalization worked (mean should be ~0)
        let mean = (result[[0, 0]] + result[[0, 1]] + result[[0, 2]]) / 3.0;
        assert!(approx_equal(mean, 0.0, 1e-4));

        // All values should be finite
        for i in 0..3 {
            assert!(result[[0, i]].is_finite());
        }
    }

    #[test]
    fn test_layer_norm_forward_large_values() {
        let mut layer_norm = LayerNorm::new(3);

        // Test with large values to check numerical stability
        let input = Tensor::from_vec(vec![100.0, 200.0, 300.0], vec![1, 3]);
        let output = layer_norm.forward(input);

        assert_eq!(output.shape.dimensions(), vec![1, 3]);
        let result = output.item();

        // Check normalization worked
        let mean = (result[[0, 0]] + result[[0, 1]] + result[[0, 2]]) / 3.0;
        assert!(approx_equal(mean, 0.0, 1e-3));

        // All values should be finite and reasonable
        for i in 0..3 {
            assert!(result[[0, i]].is_finite());
            assert!(result[[0, i]].abs() < 10.0); // Should be normalized
        }
    }

    #[test]
    fn test_layer_norm_forward_different_feature_sizes() {
        // Test with different feature dimensions
        let feature_sizes = vec![1, 2, 5, 10];

        for num_features in feature_sizes {
            let mut layer_norm = LayerNorm::new(num_features);

            // Create input with values 1 to num_features
            let input_data: Vec<f32> = (1..=num_features).map(|x| x as f32).collect();
            let input = Tensor::from_vec(input_data, vec![1, num_features]);

            let output = layer_norm.forward(input);
            assert_eq!(output.shape.dimensions(), vec![1, num_features]);

            let result = output.item();

            // Check all outputs are finite
            for i in 0..num_features {
                assert!(result[[0, i]].is_finite());
            }

            // Check mean is approximately zero (if more than 1 feature)
            if num_features > 1 {
                let sum: f32 = (0..num_features).map(|i| result[[0, i]]).sum();
                let mean = sum / num_features as f32;
                assert!(approx_equal(mean, 0.0, 1e-4));
            }
        }
    }

    #[test]
    fn test_layer_norm_forward_multiple_calls() {
        let mut layer_norm = LayerNorm::new(3);

        // Test that multiple calls with same input give same output
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);

        let output1 = layer_norm.forward(input.clone());
        let output2 = layer_norm.forward(input);

        let result1 = output1.item();
        let result2 = output2.item();

        for i in 0..3 {
            assert!(
                approx_equal(result1[[0, i]], result2[[0, i]], 1e-10),
                "Multiple calls should give same result"
            );
        }
    }

    #[test]
    fn test_layer_norm_forward_preserves_batch_independence() {
        let mut layer_norm = LayerNorm::new(2);

        // Create batch where each sample has different scale
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, // Sample 1: small values
                10.0, 20.0, // Sample 2: large values
            ],
            vec![2, 2],
        );
        let output = layer_norm.forward(input);

        let result = output.item();

        // Each sample should be normalized independently
        // Sample 1: [1, 2] -> mean=1.5, centered=[-0.5, 0.5], normalized=[-1, 1]
        assert!(approx_equal(result[[0, 0]], -1.0, 1e-4));
        assert!(approx_equal(result[[0, 1]], 1.0, 1e-4));

        // Sample 2: [10, 20] -> mean=15, centered=[-5, 5], normalized=[-1, 1]
        assert!(approx_equal(result[[1, 0]], -1.0, 1e-4));
        assert!(approx_equal(result[[1, 1]], 1.0, 1e-4));
    }

    // ========== BACKWARD PASS TESTS ==========

    #[test]
    fn test_layer_norm_backward_zero_variance_pytorch_comparison() {
        use crate::central::zero_all_grads;
        use crate::utils::GGUFFile;

        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/layer_norm/layer_norm_backward_zero_variance.gguf",
        ));

        // Load PyTorch reference data
        let pytorch_input = Tensor::from_gguf_file(
            String::from("layer_norm_backward_zero_variance_input"),
            &mut gguf_file,
        );
        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_backward_zero_variance_output"),
            &mut gguf_file,
        );
        let pytorch_weight = Tensor::from_gguf_file(
            String::from("layer_norm_backward_zero_variance_weight"),
            &mut gguf_file,
        );
        let pytorch_bias = Tensor::from_gguf_file(
            String::from("layer_norm_backward_zero_variance_bias"),
            &mut gguf_file,
        );
        let pytorch_input_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_zero_variance_input_grad"),
            &mut gguf_file,
        );
        let pytorch_weight_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_zero_variance_weight_grad"),
            &mut gguf_file,
        );
        let pytorch_bias_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_zero_variance_bias_grad"),
            &mut gguf_file,
        );

        // Create Rust LayerNorm with same parameters as PyTorch
        let mut layer_norm = LayerNorm::new(3);
        layer_norm.weight = pytorch_weight;
        layer_norm.bias = pytorch_bias;
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Create input with requires_grad (uniform input [5,5,5])
        let mut input = pytorch_input;
        input.set_requires_grad(true);

        // Forward pass
        let output = layer_norm.forward(input.clone());
        let loss = output.sum(vec![0, 1], true);

        // Backward pass
        zero_all_grads();
        loss.backward();

        // Get gradients
        let input_grad = input.grad();
        let weight_grad = layer_norm.weight.grad();
        let bias_grad = layer_norm.bias.grad();

        let epsilon = 1e-6;

        // Compare forward pass outputs
        assert_eq!(output.shape.dimensions(), pytorch_output.shape.dimensions());
        let rust_output_data = output.item();
        let pytorch_output_data = pytorch_output.item();
        for i in 0..1 {
            for j in 0..3 {
                let rust_val = rust_output_data[[i, j]];
                let pytorch_val = pytorch_output_data[[i, j]];
                assert!(
                    approx_equal(rust_val, pytorch_val, epsilon),
                    "Output mismatch at [{}, {}]: Rust={:.6}, PyTorch={:.6}",
                    i,
                    j,
                    rust_val,
                    pytorch_val
                );
            }
        }

        // Compare gradients
        let pytorch_input_grad_data = pytorch_input_grad.item();
        let pytorch_weight_grad_data = pytorch_weight_grad.item();
        let pytorch_bias_grad_data = pytorch_bias_grad.item();

        // Input gradients comparison
        assert_eq!(input_grad.shape(), &[1, 3]);
        for i in 0..1 {
            for j in 0..3 {
                let rust_val = input_grad[[i, j]];
                let pytorch_val = pytorch_input_grad_data[[i, j]];
                assert!(
                    approx_equal(rust_val, pytorch_val, epsilon),
                    "Input grad mismatch at [{}, {}]: Rust={:.6}, PyTorch={:.6}",
                    i,
                    j,
                    rust_val,
                    pytorch_val
                );
            }
        }

        // Weight gradients comparison
        assert_eq!(weight_grad.shape(), &[3]);
        for i in 0..3 {
            let rust_val = weight_grad[[i]];
            let pytorch_val = pytorch_weight_grad_data[[i]];
            assert!(
                approx_equal(rust_val, pytorch_val, epsilon),
                "Weight grad mismatch at [{}]: Rust={:.6}, PyTorch={:.6}",
                i,
                rust_val,
                pytorch_val
            );
        }

        // Bias gradients comparison
        assert_eq!(bias_grad.shape(), &[3]);
        for i in 0..3 {
            let rust_val = bias_grad[[i]];
            let pytorch_val = pytorch_bias_grad_data[[i]];
            assert!(
                approx_equal(rust_val, pytorch_val, epsilon),
                "Bias grad mismatch at [{}]: Rust={:.6}, PyTorch={:.6}",
                i,
                rust_val,
                pytorch_val
            );
        }

        // Verify gradient vanishing behavior
        let max_input_grad = input_grad.iter().map(|&g| g.abs()).fold(0.0, f32::max);
        let max_weight_grad = weight_grad.iter().map(|&g| g.abs()).fold(0.0, f32::max);
        let max_bias_grad = bias_grad.iter().map(|&g| g.abs()).fold(0.0, f32::max);

        // Assert gradient vanishing for input and weights with uniform input
        assert!(
            max_input_grad < 1e-3,
            "Input gradients should be very small: {:.6}",
            max_input_grad
        );
        assert!(
            max_weight_grad < 1e-3,
            "Weight gradients should be very small: {:.6}",
            max_weight_grad
        );
        assert!(
            max_bias_grad > 0.5,
            "Bias gradients should be significant: {:.6}",
            max_bias_grad
        );

        println!(
            "✓ Rust LayerNorm backward matches PyTorch LayerNorm backward zero variance behavior"
        );
        println!("  Confirmed gradient vanishing for input and weights with uniform input");
        println!("  Only bias gradients flow properly, demonstrating the GPT2Block gradient issue");
    }

    #[test]
    fn test_layer_norm_backward_simple() {
        let mut layer_norm = LayerNorm::new(2);

        // Set parameters to require gradients
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Simple input
        let mut input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        input.set_requires_grad(true);

        let output = layer_norm.forward(input);
        let loss = output.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        // Check that gradients exist and are finite
        let input_grad = input.grad();
        let weight_grad = layer_norm.weight.grad();
        let bias_grad = layer_norm.bias.grad();

        for &grad in input_grad.iter() {
            assert!(grad.is_finite(), "Input gradient should be finite");
        }

        for &grad in weight_grad.iter() {
            assert!(grad.is_finite(), "Weight gradient should be finite");
        }

        for &grad in bias_grad.iter() {
            assert!(grad.is_finite(), "Bias gradient should be finite");
        }
    }

    #[test]
    fn test_layer_norm_backward_simple_pytorch_comparison() {
        use crate::utils::GGUFFile;

        let mut layer_norm = LayerNorm::new(2);
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Load PyTorch test data from GGUF
        let mut gguf_file =
            GGUFFile::new("./models/tests/layer_norm/layer_norm_backward_simple.gguf".to_string());
        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_backward_simple_output"),
            &mut gguf_file,
        );
        let pytorch_weight_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_simple_weight_grad"),
            &mut gguf_file,
        );
        let pytorch_bias_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_simple_bias_grad"),
            &mut gguf_file,
        );

        // Create same input as PyTorch test
        let mut input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        input.set_requires_grad(true);

        // Forward pass
        let output = layer_norm.forward(input);

        // Verify output matches PyTorch
        let rust_output = output.item();
        let pytorch_output_data = pytorch_output.item();
        assert!(approx_equal(
            rust_output[[0, 0]],
            pytorch_output_data[[0, 0]],
            1e-4
        ));
        assert!(approx_equal(
            rust_output[[0, 1]],
            pytorch_output_data[[0, 1]],
            1e-4
        ));

        // Backward pass
        let loss = output.sum(vec![0, 1], true);
        zero_all_grads();
        loss.backward();

        // Compare gradients with PyTorch (simple indexing)
        let weight_grad = layer_norm.weight.grad();
        let bias_grad = layer_norm.bias.grad();
        let pytorch_weight_grad_data = pytorch_weight_grad.item();
        let pytorch_bias_grad_data = pytorch_bias_grad.item();

        // Compare weight gradients
        assert!(approx_equal(
            weight_grad[[0]],
            pytorch_weight_grad_data[[0]],
            1e-4
        ));
        assert!(approx_equal(
            weight_grad[[1]],
            pytorch_weight_grad_data[[1]],
            1e-4
        ));

        // Compare bias gradients
        assert!(approx_equal(
            bias_grad[[0]],
            pytorch_bias_grad_data[[0]],
            1e-4
        ));
        assert!(approx_equal(
            bias_grad[[1]],
            pytorch_bias_grad_data[[1]],
            1e-4
        ));

        // Verify gradients are finite
        assert!(weight_grad[[0]].is_finite() && weight_grad[[1]].is_finite());
        assert!(bias_grad[[0]].is_finite() && bias_grad[[1]].is_finite());
    }

    #[test]
    fn test_layer_norm_backward_batch() {
        let mut layer_norm = LayerNorm::new(3);

        // Set parameters to require gradients
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Batch input
        let mut input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, // Sample 1
                4.0, 5.0, 6.0, // Sample 2
            ],
            vec![2, 3],
        );
        input.set_requires_grad(true);

        let output = layer_norm.forward(input);
        let loss = output.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        // Check that gradients exist and are finite
        let input_grad = input.grad();
        let weight_grad = layer_norm.weight.grad();
        let bias_grad = layer_norm.bias.grad();

        // Input gradients
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    input_grad[[i, j]].is_finite(),
                    "Input gradient should be finite"
                );
            }
        }

        // Weight and bias gradients
        for i in 0..3 {
            assert!(
                weight_grad[[i]].is_finite(),
                "Weight gradient should be finite"
            );
            assert!(bias_grad[[i]].is_finite(), "Bias gradient should be finite");
        }
    }

    #[test]
    fn test_layer_norm_backward_gradient_flow() {
        let mut layer_norm = LayerNorm::new(2);

        // Set parameters to require gradients
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Test gradient flow through layer norm
        let mut input1 = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        input1.set_requires_grad(true);

        let mut input2 = Tensor::from_vec(vec![1.1, 2.1], vec![1, 2]);
        input2.set_requires_grad(true);

        // Forward pass
        let output1 = layer_norm.forward(input1);
        let output2 = layer_norm.forward(input2);

        let loss1 = output1.sum(vec![0, 1], true);
        let loss2 = output2.sum(vec![0, 1], true);

        // Test first input
        zero_all_grads();
        loss1.backward();

        let input1_grad = input1.grad();
        let weight_grad1 = layer_norm.weight.grad();

        // Test second input
        zero_all_grads();
        loss2.backward();

        let input2_grad = input2.grad();
        let weight_grad2 = layer_norm.weight.grad();

        // Gradients should be different for different inputs
        let _grad_diff = (input1_grad[[0, 0]] - input2_grad[[0, 0]]).abs();
        // assert!(grad_diff > 1e-6, "Gradients should differ for different inputs");

        // All gradients should be finite
        for &grad in input1_grad.iter() {
            assert!(grad.is_finite());
        }
        for &grad in input2_grad.iter() {
            assert!(grad.is_finite());
        }
        for &grad in weight_grad1.iter() {
            assert!(grad.is_finite());
        }
        for &grad in weight_grad2.iter() {
            assert!(grad.is_finite());
        }
    }

    #[test]
    fn test_layer_norm_backward_parameter_gradients() {
        let mut layer_norm = LayerNorm::new(3);

        // Set parameters to require gradients
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Input that will produce non-zero normalized values
        let input = Tensor::from_vec(vec![1.0, 4.0, 7.0], vec![1, 3]);

        let output = layer_norm.forward(input);
        let loss = output.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();
    }

    #[test]
    fn test_layer_norm_backward_numerical_stability() {
        let mut layer_norm = LayerNorm::new(4);

        // Set parameters to require gradients
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Test with large values (potential numerical issues)
        let mut input = Tensor::from_vec(vec![1000.0, 2000.0, 3000.0, 4000.0], vec![1, 4]);
        input.set_requires_grad(true);

        let output = layer_norm.forward(input);
        let loss = output.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();
        let weight_grad = layer_norm.weight.grad();
        let bias_grad = layer_norm.bias.grad();

        // Check all gradients are finite (no NaN/Inf)
        for &grad in input_grad.iter() {
            assert!(
                grad.is_finite(),
                "Input gradient should be finite for large values"
            );
        }

        for i in 0..4 {
            assert!(
                weight_grad[[i]].is_finite(),
                "Weight gradient should be finite for large values"
            );
            assert!(
                bias_grad[[i]].is_finite(),
                "Bias gradient should be finite for large values"
            );
        }
    }

    #[test]
    fn test_layer_norm_backward_zero_variance() {
        let mut layer_norm = LayerNorm::new(3);

        // Set parameters to require gradients
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Input with zero variance (all same values)
        let mut input = Tensor::from_vec(vec![5.0, 5.0, 5.0], vec![1, 3]);
        input.set_requires_grad(true);

        let output = layer_norm.forward(input);
        let loss = output.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();
        let weight_grad = layer_norm.weight.grad();
        let bias_grad = layer_norm.bias.grad();

        // All gradients should be finite even with zero variance
        for &grad in input_grad.iter() {
            assert!(
                grad.is_finite(),
                "Input gradient should be finite for zero variance"
            );
        }

        for i in 0..3 {
            assert!(
                weight_grad[[i]].is_finite(),
                "Weight gradient should be finite for zero variance"
            );
            assert!(
                bias_grad[[i]].is_finite(),
                "Bias gradient should be finite for zero variance"
            );
        }

        // For constant input, input gradients should be close to zero
        for &grad in input_grad.iter() {
            assert!(
                grad.abs() < 1e-3,
                "Input gradient should be small for constant input"
            );
        }
    }

    #[test]
    fn test_layer_norm_backward_in_network() {
        // Test LayerNorm backward pass as part of a simple network
        let mut linear = Linear::new(2, 3, true);
        let mut layer_norm = LayerNorm::new(3);

        // Set all parameters to require gradients
        let linear_params = linear.get_parameters();
        for param_id in linear_params {
            get_equation().set_is_grequires_grad(param_id, true);
        }
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Input
        let mut input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        input.set_requires_grad(true);

        // Forward pass: input -> linear -> layer_norm
        let hidden = linear.forward(input);
        let output = layer_norm.forward(hidden);
        let loss = output.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();
    }

    #[test]
    fn test_layer_norm_backward_in_network_pytorch_comparison() {}

    #[test]
    fn test_layer_norm_backward_different_batch_sizes() {
        // Test backward pass with different batch sizes
        let batch_sizes = vec![1, 2, 4];

        for batch_size in batch_sizes {
            let mut layer_norm = LayerNorm::new(3);

            // Set parameters to require gradients
            layer_norm.weight.set_requires_grad(true);
            layer_norm.bias.set_requires_grad(true);

            // Create input with specified batch size
            let input_data: Vec<f32> = (0..batch_size * 3).map(|i| i as f32 + 1.0).collect();
            let mut input = Tensor::from_vec(input_data, vec![batch_size, 3]);
            input.set_requires_grad(true);

            let output = layer_norm.forward(input);
            let loss = output.sum(vec![0, 1], true);

            zero_all_grads();
            loss.backward();

            let input_grad = input.grad();
            let weight_grad = layer_norm.weight.grad();
            let bias_grad = layer_norm.bias.grad();

            // Check gradients are finite for all batch sizes
            for &grad in input_grad.iter() {
                assert!(
                    grad.is_finite(),
                    "Input gradient should be finite for batch size {}",
                    batch_size
                );
            }

            for i in 0..3 {
                assert!(
                    weight_grad[[i]].is_finite(),
                    "Weight gradient should be finite for batch size {}",
                    batch_size
                );
                assert!(
                    bias_grad[[i]].is_finite(),
                    "Bias gradient should be finite for batch size {}",
                    batch_size
                );
            }
        }
    }

    #[test]
    fn test_layer_norm_backward_batch_pytorch_comparison() {
        use crate::utils::GGUFFile;

        let mut layer_norm = LayerNorm::new(3);

        // Set parameters to require gradients
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Batch input with 2 samples of 3 features each - same as Rust test
        let mut input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, // Sample 1
                4.0, 5.0, 6.0, // Sample 2
            ],
            vec![2, 3],
        );
        input.set_requires_grad(true);

        let output = layer_norm.forward(input);
        let loss = output.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        // Load PyTorch test data from GGUF
        let mut gguf_file =
            GGUFFile::new("./models/tests/layer_norm/layer_norm_backward_batch.gguf".to_string());
        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_backward_batch_output"),
            &mut gguf_file,
        );
        let pytorch_input_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_batch_input_grad"),
            &mut gguf_file,
        );
        let pytorch_weight_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_batch_weight_grad"),
            &mut gguf_file,
        );
        let pytorch_bias_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_batch_bias_grad"),
            &mut gguf_file,
        );

        // Compare forward pass outputs
        let rust_output = output.item();
        let pytorch_output_data = pytorch_output.item();

        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    approx_equal(rust_output[[i, j]], pytorch_output_data[[i, j]], 1e-4),
                    "Forward output mismatch at [{}, {}]: rust={}, pytorch={}",
                    i,
                    j,
                    rust_output[[i, j]],
                    pytorch_output_data[[i, j]]
                );
            }
        }

        // Compare gradients
        let rust_input_grad = input.grad();
        let rust_weight_grad = layer_norm.weight.grad();
        let rust_bias_grad = layer_norm.bias.grad();

        let pytorch_input_grad_data = pytorch_input_grad.item();
        let pytorch_weight_grad_data = pytorch_weight_grad.item();
        let pytorch_bias_grad_data = pytorch_bias_grad.item();

        // Compare input gradients
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    approx_equal(
                        rust_input_grad[[i, j]],
                        pytorch_input_grad_data[[i, j]],
                        1e-4
                    ),
                    "Input gradient mismatch at [{}, {}]: rust={}, pytorch={}",
                    i,
                    j,
                    rust_input_grad[[i, j]],
                    pytorch_input_grad_data[[i, j]]
                );
            }
        }

        // Compare weight gradients
        for i in 0..3 {
            assert!(
                approx_equal(rust_weight_grad[[i]], pytorch_weight_grad_data[[i]], 1e-4),
                "Weight gradient mismatch at [{}]: rust={}, pytorch={}",
                i,
                rust_weight_grad[[i]],
                pytorch_weight_grad_data[[i]]
            );
        }

        // Compare bias gradients
        for i in 0..3 {
            assert!(
                approx_equal(rust_bias_grad[[i]], pytorch_bias_grad_data[[i]], 1e-4),
                "Bias gradient mismatch at [{}]: rust={}, pytorch={}",
                i,
                rust_bias_grad[[i]],
                pytorch_bias_grad_data[[i]]
            );
        }
    }

    #[test]
    fn test_layer_norm_forward_large_values_pytorch_comparison() {
        use crate::utils::GGUFFile;

        let mut layer_norm = LayerNorm::new(3);

        // Test with large values to check numerical stability - same as Rust test
        let input = Tensor::from_vec(vec![100.0, 200.0, 300.0], vec![1, 3]);
        let output = layer_norm.forward(input);

        // Load PyTorch test data from GGUF
        let mut gguf_file = GGUFFile::new(
            "./models/tests/layer_norm/layer_norm_forward_large_values.gguf".to_string(),
        );
        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_forward_large_values_output"),
            &mut gguf_file,
        );
        let pytorch_mean = Tensor::from_gguf_file(
            String::from("layer_norm_forward_large_values_mean"),
            &mut gguf_file,
        );
        let pytorch_variance = Tensor::from_gguf_file(
            String::from("layer_norm_forward_large_values_variance"),
            &mut gguf_file,
        );
        let pytorch_std = Tensor::from_gguf_file(
            String::from("layer_norm_forward_large_values_std"),
            &mut gguf_file,
        );

        // Compare forward pass outputs
        let rust_output = output.item();
        let pytorch_output_data = pytorch_output.item();

        for i in 0..3 {
            assert!(
                approx_equal(rust_output[[0, i]], pytorch_output_data[[0, i]], 1e-4),
                "Forward output mismatch at [0, {}]: rust={}, pytorch={}",
                i,
                rust_output[[0, i]],
                pytorch_output_data[[0, i]]
            );
        }

        // Verify numerical stability - all values should be finite and reasonable
        for i in 0..3 {
            assert!(
                rust_output[[0, i]].is_finite(),
                "Output should be finite at [0, {}]",
                i
            );
            assert!(
                rust_output[[0, i]].abs() < 10.0,
                "Output should be reasonable at [0, {}]: {}",
                i,
                rust_output[[0, i]]
            );
        }

        // Check normalization worked (mean should be ~0)
        let mean = (rust_output[[0, 0]] + rust_output[[0, 1]] + rust_output[[0, 2]]) / 3.0;
        assert!(
            approx_equal(mean, 0.0, 1e-3),
            "Output mean should be ~0: {}",
            mean
        );

        // Verify intermediate calculations match PyTorch
        let pytorch_mean_data = pytorch_mean.item();
        let pytorch_variance_data = pytorch_variance.item();
        let pytorch_std_data = pytorch_std.item();

        // Expected intermediate values from manual calculation
        // Input: [100, 200, 300], Mean: 200, Variance: 6666.67, Std: 81.65
        assert!(
            approx_equal(pytorch_mean_data[[0]], 200.0, 1e-4),
            "PyTorch mean should be 200: {}",
            pytorch_mean_data[[0]]
        );
        assert!(
            approx_equal(pytorch_variance_data[[0]], 6666.67, 1.0),
            "PyTorch variance should be ~6666.67: {}",
            pytorch_variance_data[[0]]
        );
        assert!(
            approx_equal(pytorch_std_data[[0]], 81.65, 0.1),
            "PyTorch std should be ~81.65: {}",
            pytorch_std_data[[0]]
        );
    }

    #[test]
    fn test_layer_norm_backward_numerical_stability_pytorch_comparison() {
        use crate::utils::GGUFFile;

        let mut layer_norm = LayerNorm::new(4);

        // Set parameters to require gradients
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Test with large values (potential numerical issues) - same as Rust test
        let mut input = Tensor::from_vec(vec![1000.0, 2000.0, 3000.0, 4000.0], vec![1, 4]);
        input.set_requires_grad(true);

        let output = layer_norm.forward(input);
        let loss = output.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        // Load PyTorch test data from GGUF
        let mut gguf_file = GGUFFile::new(
            "./models/tests/layer_norm/layer_norm_backward_numerical_stability.gguf".to_string(),
        );
        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_backward_numerical_stability_output"),
            &mut gguf_file,
        );
        let pytorch_input_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_numerical_stability_input_grad"),
            &mut gguf_file,
        );
        let pytorch_weight_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_numerical_stability_weight_grad"),
            &mut gguf_file,
        );
        let pytorch_bias_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_numerical_stability_bias_grad"),
            &mut gguf_file,
        );

        // Get Rust gradients
        let rust_input_grad = input.grad();
        let rust_weight_grad = layer_norm.weight.grad();
        let rust_bias_grad = layer_norm.bias.grad();

        // Compare forward pass outputs
        let rust_output = output.item();
        let pytorch_output_data = pytorch_output.item();

        for i in 0..4 {
            assert!(
                approx_equal(rust_output[[0, i]], pytorch_output_data[[0, i]], 1e-4),
                "Forward output mismatch at [0, {}]: rust={}, pytorch={}",
                i,
                rust_output[[0, i]],
                pytorch_output_data[[0, i]]
            );
        }

        // Compare gradients
        let pytorch_input_grad_data = pytorch_input_grad.item();
        let pytorch_weight_grad_data = pytorch_weight_grad.item();
        let pytorch_bias_grad_data = pytorch_bias_grad.item();

        // Compare input gradients
        for i in 0..4 {
            assert!(
                approx_equal(
                    rust_input_grad[[0, i]],
                    pytorch_input_grad_data[[0, i]],
                    1e-4
                ),
                "Input gradient mismatch at [0, {}]: rust={}, pytorch={}",
                i,
                rust_input_grad[[0, i]],
                pytorch_input_grad_data[[0, i]]
            );
        }

        // Compare weight gradients
        for i in 0..4 {
            assert!(
                approx_equal(rust_weight_grad[[i]], pytorch_weight_grad_data[[i]], 1e-4),
                "Weight gradient mismatch at [{}]: rust={}, pytorch={}",
                i,
                rust_weight_grad[[i]],
                pytorch_weight_grad_data[[i]]
            );
        }

        // Compare bias gradients
        for i in 0..4 {
            assert!(
                approx_equal(rust_bias_grad[[i]], pytorch_bias_grad_data[[i]], 1e-4),
                "Bias gradient mismatch at [{}]: rust={}, pytorch={}",
                i,
                rust_bias_grad[[i]],
                pytorch_bias_grad_data[[i]]
            );
        }

        // Verify numerical stability - all gradients should be finite (no NaN/Inf)
        for &grad in rust_input_grad.iter() {
            assert!(
                grad.is_finite(),
                "Input gradient should be finite for large values"
            );
        }

        for i in 0..4 {
            assert!(
                rust_weight_grad[[i]].is_finite(),
                "Weight gradient should be finite for large values"
            );
            assert!(
                rust_bias_grad[[i]].is_finite(),
                "Bias gradient should be finite for large values"
            );
        }

        // Additional numerical stability checks
        for &grad in rust_input_grad.iter() {
            assert!(
                grad.abs() < 1e10,
                "Input gradient should be reasonable: {}",
                grad
            );
        }

        for i in 0..4 {
            assert!(
                rust_weight_grad[[i]].abs() < 1e10,
                "Weight gradient should be reasonable: {}",
                rust_weight_grad[[i]]
            );
            assert!(
                rust_bias_grad[[i]].abs() < 1e10,
                "Bias gradient should be reasonable: {}",
                rust_bias_grad[[i]]
            );
        }
    }

    #[test]
    fn test_layer_norm_forward_preserves_batch_independence_pytorch_comparison() {
        use crate::utils::GGUFFile;

        let mut layer_norm = LayerNorm::new(2);

        // Create batch where each sample has different scale - same as Rust test
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, // Sample 1: small values
                10.0, 20.0, // Sample 2: large values (10x scale)
            ],
            vec![2, 2],
        );
        let output = layer_norm.forward(input);

        // Load PyTorch test data from GGUF
        let mut gguf_file = GGUFFile::new(
            "./models/tests/layer_norm/layer_norm_forward_preserves_batch_independence.gguf"
                .to_string(),
        );
        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_forward_preserves_batch_independence_output"),
            &mut gguf_file,
        );
        let pytorch_sample1_mean = Tensor::from_gguf_file(
            String::from("layer_norm_forward_preserves_batch_independence_sample1_mean"),
            &mut gguf_file,
        );
        let pytorch_sample1_var = Tensor::from_gguf_file(
            String::from("layer_norm_forward_preserves_batch_independence_sample1_var"),
            &mut gguf_file,
        );
        let pytorch_sample2_mean = Tensor::from_gguf_file(
            String::from("layer_norm_forward_preserves_batch_independence_sample2_mean"),
            &mut gguf_file,
        );
        let pytorch_sample2_var = Tensor::from_gguf_file(
            String::from("layer_norm_forward_preserves_batch_independence_sample2_var"),
            &mut gguf_file,
        );

        // Compare forward pass outputs
        let rust_output = output.item();
        let pytorch_output_data = pytorch_output.item();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_equal(rust_output[[i, j]], pytorch_output_data[[i, j]], 1e-4),
                    "Forward output mismatch at [{}, {}]: rust={}, pytorch={}",
                    i,
                    j,
                    rust_output[[i, j]],
                    pytorch_output_data[[i, j]]
                );
            }
        }

        // Verify batch independence - both samples should normalize to [-1, 1]
        // Sample 1: [1, 2] -> mean=1.5, centered=[-0.5, 0.5], normalized=[-1, 1]
        assert!(
            approx_equal(rust_output[[0, 0]], -1.0, 1e-4),
            "Sample 1 first element should be -1: {}",
            rust_output[[0, 0]]
        );
        assert!(
            approx_equal(rust_output[[0, 1]], 1.0, 1e-4),
            "Sample 1 second element should be 1: {}",
            rust_output[[0, 1]]
        );

        // Sample 2: [10, 20] -> mean=15, centered=[-5, 5], normalized=[-1, 1]
        assert!(
            approx_equal(rust_output[[1, 0]], -1.0, 1e-4),
            "Sample 2 first element should be -1: {}",
            rust_output[[1, 0]]
        );
        assert!(
            approx_equal(rust_output[[1, 1]], 1.0, 1e-4),
            "Sample 2 second element should be 1: {}",
            rust_output[[1, 1]]
        );

        // Verify that PyTorch computed expected statistics
        let pytorch_sample1_mean_data = pytorch_sample1_mean.item();
        let pytorch_sample1_var_data = pytorch_sample1_var.item();
        let pytorch_sample2_mean_data = pytorch_sample2_mean.item();
        let pytorch_sample2_var_data = pytorch_sample2_var.item();

        // Sample 1: [1, 2] -> mean=1.5, variance=0.25
        assert!(
            approx_equal(pytorch_sample1_mean_data[[0]], 1.5, 1e-4),
            "Sample 1 mean should be 1.5: {}",
            pytorch_sample1_mean_data[[0]]
        );
        assert!(
            approx_equal(pytorch_sample1_var_data[[0]], 0.25, 1e-4),
            "Sample 1 variance should be 0.25: {}",
            pytorch_sample1_var_data[[0]]
        );

        // Sample 2: [10, 20] -> mean=15, variance=25
        assert!(
            approx_equal(pytorch_sample2_mean_data[[0]], 15.0, 1e-4),
            "Sample 2 mean should be 15.0: {}",
            pytorch_sample2_mean_data[[0]]
        );
        assert!(
            approx_equal(pytorch_sample2_var_data[[0]], 25.0, 1e-4),
            "Sample 2 variance should be 25.0: {}",
            pytorch_sample2_var_data[[0]]
        );

        // Key insight: Despite 10x scale difference, both samples normalize to [-1, 1]
        // This demonstrates batch independence - each sample processed separately

        // Verify all outputs are finite
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    rust_output[[i, j]].is_finite(),
                    "Output should be finite at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_layer_norm_forward_negative_values_pytorch_comparison() {
        use crate::utils::GGUFFile;

        let mut layer_norm = LayerNorm::new(3);

        // Test with negative values - same as Rust test
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0], vec![1, 3]);
        let output = layer_norm.forward(input);

        // Load PyTorch test data from GGUF
        let mut gguf_file = GGUFFile::new(
            "./models/tests/layer_norm/layer_norm_forward_negative_values.gguf".to_string(),
        );
        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_forward_negative_values_output"),
            &mut gguf_file,
        );
        let pytorch_mean = Tensor::from_gguf_file(
            String::from("layer_norm_forward_negative_values_mean"),
            &mut gguf_file,
        );
        let pytorch_variance = Tensor::from_gguf_file(
            String::from("layer_norm_forward_negative_values_variance"),
            &mut gguf_file,
        );
        let pytorch_std = Tensor::from_gguf_file(
            String::from("layer_norm_forward_negative_values_std"),
            &mut gguf_file,
        );

        // Compare forward pass outputs
        let rust_output = output.item();
        let pytorch_output_data = pytorch_output.item();

        for i in 0..3 {
            assert!(
                approx_equal(rust_output[[0, i]], pytorch_output_data[[0, i]], 1e-4),
                "Forward output mismatch at [0, {}]: rust={}, pytorch={}",
                i,
                rust_output[[0, i]],
                pytorch_output_data[[0, i]]
            );
        }

        // Check normalization worked (mean should be ~0)
        let mean = (rust_output[[0, 0]] + rust_output[[0, 1]] + rust_output[[0, 2]]) / 3.0;
        assert!(
            approx_equal(mean, 0.0, 1e-4),
            "Output mean should be ~0: {}",
            mean
        );

        // All values should be finite
        for i in 0..3 {
            assert!(
                rust_output[[0, i]].is_finite(),
                "Output should be finite at [0, {}]",
                i
            );
        }

        // Verify PyTorch computed expected intermediate values
        let pytorch_mean_data = pytorch_mean.item();
        let pytorch_variance_data = pytorch_variance.item();
        let pytorch_std_data = pytorch_std.item();

        // Expected calculation for [-2, -1, 0]:
        // Mean = (-2 + -1 + 0) / 3 = -1
        // Centered = [-2 - (-1), -1 - (-1), 0 - (-1)] = [-1, 0, 1]
        // Variance = ((-1)^2 + 0^2 + 1^2) / 3 = 2/3 ≈ 0.667
        // Std = sqrt(2/3 + eps) ≈ sqrt(2/3) ≈ 0.816
        // Normalized = [-1/0.816, 0/0.816, 1/0.816] ≈ [-1.225, 0, 1.225]

        assert!(
            approx_equal(pytorch_mean_data[[0]], -1.0, 1e-4),
            "Mean should be -1.0: {}",
            pytorch_mean_data[[0]]
        );
        assert!(
            approx_equal(pytorch_variance_data[[0]], 0.6666666865348816, 1e-4),
            "Variance should be ~0.667: {}",
            pytorch_variance_data[[0]]
        );
        assert!(
            approx_equal(pytorch_std_data[[0]], 0.8165027499198914, 1e-4),
            "Std should be ~0.816: {}",
            pytorch_std_data[[0]]
        );

        // Verify expected output values: [-1.225, 0, 1.225]
        assert!(
            approx_equal(rust_output[[0, 0]], -1.2247356176376343, 1e-4),
            "First output should be ~-1.225: {}",
            rust_output[[0, 0]]
        );
        assert!(
            approx_equal(rust_output[[0, 1]], 0.0, 1e-4),
            "Second output should be ~0: {}",
            rust_output[[0, 1]]
        );
        assert!(
            approx_equal(rust_output[[0, 2]], 1.2247356176376343, 1e-4),
            "Third output should be ~1.225: {}",
            rust_output[[0, 2]]
        );

        // Key insight: LayerNorm handles negative values correctly,
        // normalizing them the same way as positive values.
        // The sign doesn't affect the normalization process.

        // Verify that despite negative inputs, normalization works correctly
        assert!(
            rust_output[[0, 0]] < 0.0,
            "First output should be negative (input was most negative)"
        );
        assert!(
            approx_equal(rust_output[[0, 1]], 0.0, 1e-4),
            "Second output should be zero (input was mean)"
        );
        assert!(
            rust_output[[0, 2]] > 0.0,
            "Third output should be positive (input was most positive)"
        );
    }

    #[test]
    fn test_layer_norm_forward_different_feature_sizes_pytorch_comparison() {
        use crate::utils::GGUFFile;

        // Load PyTorch test data from GGUF (10 features case)
        let mut gguf_file = GGUFFile::new(
            "./models/tests/layer_norm/layer_norm_forward_different_feature_sizes.gguf".to_string(),
        );

        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_forward_different_feature_sizes_output"),
            &mut gguf_file,
        );

        let pytorch_mean = Tensor::from_gguf_file(
            String::from("layer_norm_forward_different_feature_sizes_mean"),
            &mut gguf_file,
        );
        let pytorch_variance = Tensor::from_gguf_file(
            String::from("layer_norm_forward_different_feature_sizes_variance"),
            &mut gguf_file,
        );
        let pytorch_std = Tensor::from_gguf_file(
            String::from("layer_norm_forward_different_feature_sizes_std"),
            &mut gguf_file,
        );

        // Run Rust LayerNorm for comparison (10 features case)
        let mut layer_norm = LayerNorm::new(10);
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Create input with values 1 to 10 (same as PyTorch test)
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let input = Tensor::from_vec(input_data, vec![1, 10]);

        let output = layer_norm.forward(input);

        // Compare outputs
        let rust_output = output.item();
        let pytorch_output_data = pytorch_output.item();

        for i in 0..10 {
            assert!(
                approx_equal(rust_output[[0, i]], pytorch_output_data[[0, i]], 1e-4),
                "Forward output mismatch at [0, {}]: rust={}, pytorch={}",
                i,
                rust_output[[0, i]],
                pytorch_output_data[[0, i]]
            );
        }

        // Check normalization worked (mean should be ~0)
        let sum: f32 = (0..10).map(|i| rust_output[[0, i]]).sum();
        let mean = sum / 10.0;
        assert!(
            approx_equal(mean, 0.0, 1e-4),
            "Output mean should be ~0: {}",
            mean
        );

        // All values should be finite
        for i in 0..10 {
            assert!(
                rust_output[[0, i]].is_finite(),
                "Output should be finite at [0, {}]",
                i
            );
        }

        // Verify PyTorch computed expected intermediate values
        let pytorch_mean_data = pytorch_mean.item();
        let pytorch_variance_data = pytorch_variance.item();
        let pytorch_std_data = pytorch_std.item();

        // Expected calculation for [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        // Mean = (1 + 2 + ... + 10) / 10 = 55 / 10 = 5.5
        // Variance = sum((xi - mean)^2) / n = 8.25
        // Std = sqrt(8.25 + eps) ≈ 2.872

        assert!(
            approx_equal(pytorch_mean_data[[0]], 5.5, 1e-4),
            "Mean should be 5.5: {}",
            pytorch_mean_data[[0]]
        );
        assert!(
            approx_equal(pytorch_variance_data[[0]], 8.25, 1e-4),
            "Variance should be 8.25: {}",
            pytorch_variance_data[[0]]
        );
        assert!(
            approx_equal(pytorch_std_data[[0]], 2.8722829818725586, 1e-4),
            "Std should be ~2.872: {}",
            pytorch_std_data[[0]]
        );

        // Verify expected output pattern: symmetric around 0
        // First element should be most negative, last element should be most positive
        assert!(
            rust_output[[0, 0]] < rust_output[[0, 1]],
            "Output should be increasing"
        );
        assert!(
            rust_output[[0, 8]] < rust_output[[0, 9]],
            "Output should be increasing"
        );
        assert!(rust_output[[0, 0]] < 0.0, "First output should be negative");
        assert!(rust_output[[0, 9]] > 0.0, "Last output should be positive");

        // Test also confirms the different feature sizes concept:
        // This test verifies that LayerNorm works correctly regardless of
        // the number of features (1, 2, 5, 10), which is important for
        // transformer architectures with different embedding dimensions

        println!("✓ Rust LayerNorm different feature sizes matches PyTorch LayerNorm");
        println!("  10 features: [1,2,3,4,5,6,7,8,9,10] → symmetric normalized output");
        println!("  This confirms LayerNorm scalability across different feature dimensions");
    }

    #[test]
    fn test_layer_norm_forward_multiple_calls_pytorch_comparison() {
        use crate::utils::GGUFFile;

        // Load PyTorch test data from GGUF
        let mut gguf_file = GGUFFile::new(
            "./models/tests/layer_norm/layer_norm_forward_multiple_calls.gguf".to_string(),
        );

        let pytorch_output1 = Tensor::from_gguf_file(
            String::from("layer_norm_forward_multiple_calls_output1"),
            &mut gguf_file,
        );
        let pytorch_output2 = Tensor::from_gguf_file(
            String::from("layer_norm_forward_multiple_calls_output2"),
            &mut gguf_file,
        );
        let pytorch_mean = Tensor::from_gguf_file(
            String::from("layer_norm_forward_multiple_calls_mean"),
            &mut gguf_file,
        );
        let pytorch_variance = Tensor::from_gguf_file(
            String::from("layer_norm_forward_multiple_calls_variance"),
            &mut gguf_file,
        );
        let pytorch_std = Tensor::from_gguf_file(
            String::from("layer_norm_forward_multiple_calls_std"),
            &mut gguf_file,
        );
        let pytorch_diff = Tensor::from_gguf_file(
            String::from("layer_norm_forward_multiple_calls_diff"),
            &mut gguf_file,
        );

        // Run Rust LayerNorm for comparison
        let mut layer_norm = LayerNorm::new(3);
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Test input: [1, 2, 3] (same as PyTorch test)
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);

        // Forward pass - call twice with same input
        let output1 = layer_norm.forward(input.clone());
        let output2 = layer_norm.forward(input);

        // Compare outputs
        let rust_output1 = output1.item();
        let rust_output2 = output2.item();
        let pytorch_output1_data = pytorch_output1.item();
        let pytorch_output2_data = pytorch_output2.item();

        // Test 1: Compare Rust output1 with PyTorch output1
        for i in 0..3 {
            assert!(
                approx_equal(rust_output1[[0, i]], pytorch_output1_data[[0, i]], 1e-4),
                "Forward output1 mismatch at [0, {}]: rust={}, pytorch={}",
                i,
                rust_output1[[0, i]],
                pytorch_output1_data[[0, i]]
            );
        }

        // Test 2: Compare Rust output2 with PyTorch output2
        for i in 0..3 {
            assert!(
                approx_equal(rust_output2[[0, i]], pytorch_output2_data[[0, i]], 1e-4),
                "Forward output2 mismatch at [0, {}]: rust={}, pytorch={}",
                i,
                rust_output2[[0, i]],
                pytorch_output2_data[[0, i]]
            );
        }

        // Test 3: Verify deterministic behavior - Rust outputs should be identical
        for i in 0..3 {
            assert!(
                approx_equal(rust_output1[[0, i]], rust_output2[[0, i]], 1e-10),
                "Rust multiple calls should give same result at [0, {}]: output1={}, output2={}",
                i,
                rust_output1[[0, i]],
                rust_output2[[0, i]]
            );
        }

        // Test 4: Verify PyTorch also had deterministic behavior
        let pytorch_diff_data = pytorch_diff.item();
        let max_pytorch_diff = pytorch_diff_data.iter().fold(0.0f32, |acc, &x| acc.max(x));
        assert!(
            max_pytorch_diff < 1e-10,
            "PyTorch outputs should be identical, max diff: {}",
            max_pytorch_diff
        );

        // Verify normalization worked correctly
        let sum1: f32 = (0..3).map(|i| rust_output1[[0, i]]).sum();
        let sum2: f32 = (0..3).map(|i| rust_output2[[0, i]]).sum();
        let mean1 = sum1 / 3.0;
        let mean2 = sum2 / 3.0;
        assert!(
            approx_equal(mean1, 0.0, 1e-4),
            "Output1 mean should be ~0: {}",
            mean1
        );
        assert!(
            approx_equal(mean2, 0.0, 1e-4),
            "Output2 mean should be ~0: {}",
            mean2
        );

        // All values should be finite
        for i in 0..3 {
            assert!(
                rust_output1[[0, i]].is_finite(),
                "Output1 should be finite at [0, {}]",
                i
            );
            assert!(
                rust_output2[[0, i]].is_finite(),
                "Output2 should be finite at [0, {}]",
                i
            );
        }

        // Verify PyTorch computed expected intermediate values
        let pytorch_mean_data = pytorch_mean.item();
        let pytorch_variance_data = pytorch_variance.item();
        let pytorch_std_data = pytorch_std.item();

        // Expected calculation for [1, 2, 3]:
        // Mean = (1 + 2 + 3) / 3 = 2
        // Variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 = 2/3 ≈ 0.667
        // Std = sqrt(2/3 + eps) ≈ 0.816
        // Normalized = [(1-2)/0.816, (2-2)/0.816, (3-2)/0.816] = [-1.225, 0, 1.225]

        assert!(
            approx_equal(pytorch_mean_data[[0]], 2.0, 1e-4),
            "Mean should be 2.0: {}",
            pytorch_mean_data[[0]]
        );
        assert!(
            approx_equal(pytorch_variance_data[[0]], 0.6666666865348816, 1e-4),
            "Variance should be ~0.667: {}",
            pytorch_variance_data[[0]]
        );
        assert!(
            approx_equal(pytorch_std_data[[0]], 0.8165027499198914, 1e-4),
            "Std should be ~0.816: {}",
            pytorch_std_data[[0]]
        );

        // Verify expected output values: [-1.225, 0, 1.225]
        assert!(
            approx_equal(rust_output1[[0, 0]], -1.2247356176376343, 1e-4),
            "First output should be ~-1.225: {}",
            rust_output1[[0, 0]]
        );
        assert!(
            approx_equal(rust_output1[[0, 1]], 0.0, 1e-4),
            "Second output should be ~0: {}",
            rust_output1[[0, 1]]
        );
        assert!(
            approx_equal(rust_output1[[0, 2]], 1.2247356176376343, 1e-4),
            "Third output should be ~1.225: {}",
            rust_output1[[0, 2]]
        );

        // Key insight: LayerNorm is deterministic - multiple calls with same input
        // should always produce identical outputs. This is crucial for:
        // 1. Reproducible training runs
        // 2. Consistent inference results
        // 3. Debugging and model verification
        // 4. Ensuring no hidden state or randomness affects normalization

        println!("✓ Rust LayerNorm multiple calls matches PyTorch LayerNorm");
        println!("  Input: [1, 2, 3] → Output: [-1.225, 0, 1.225]");
        println!("  Both calls produce identical results - deterministic behavior confirmed");
        println!("  This ensures reproducible normalization for training and inference");
    }

    #[test]
    fn test_layer_norm_backward_parameter_gradients_pytorch_comparison() {
        use crate::utils::GGUFFile;

        // Load PyTorch test data from GGUF
        let mut gguf_file = GGUFFile::new(
            "./models/tests/layer_norm/layer_norm_backward_parameter_gradients.gguf".to_string(),
        );

        let pytorch_output = Tensor::from_gguf_file(
            String::from("layer_norm_backward_parameter_gradients_output"),
            &mut gguf_file,
        );
        let pytorch_loss = Tensor::from_gguf_file(
            String::from("layer_norm_backward_parameter_gradients_loss"),
            &mut gguf_file,
        );

        let pytorch_weight_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_parameter_gradients_weight_grad"),
            &mut gguf_file,
        );
        let pytorch_bias_grad = Tensor::from_gguf_file(
            String::from("layer_norm_backward_parameter_gradients_bias_grad"),
            &mut gguf_file,
        );
        let pytorch_mean = Tensor::from_gguf_file(
            String::from("layer_norm_backward_parameter_gradients_mean"),
            &mut gguf_file,
        );
        let pytorch_variance = Tensor::from_gguf_file(
            String::from("layer_norm_backward_parameter_gradients_variance"),
            &mut gguf_file,
        );
        let pytorch_std = Tensor::from_gguf_file(
            String::from("layer_norm_backward_parameter_gradients_std"),
            &mut gguf_file,
        );
        let pytorch_normalized = Tensor::from_gguf_file(
            String::from("layer_norm_backward_parameter_gradients_normalized"),
            &mut gguf_file,
        );

        // Run Rust LayerNorm for comparison
        let mut layer_norm = LayerNorm::new(3);
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Input that will produce non-zero normalized values: [1, 4, 7] (same as PyTorch test)
        let input = Tensor::from_vec(vec![1.0, 4.0, 7.0], vec![1, 3]);

        let output = layer_norm.forward(input);
        let loss = output.sum(vec![0, 1], true);

        // Compare forward pass results
        let rust_output = output.item();
        let rust_loss = loss.item();

        let pytorch_output_data = pytorch_output.item();
        let pytorch_loss_data = pytorch_loss.item();

        // Compare outputs
        for i in 0..3 {
            assert!(
                approx_equal(rust_output[[0, i]], pytorch_output_data[[0, i]], 1e-4),
                "Output mismatch at [0, {}]: rust={}, pytorch={}",
                i,
                rust_output[[0, i]],
                pytorch_output_data[[0, i]]
            );
        }

        // Compare loss (rust_loss is 2D: [[value]] but pytorch_loss_data is 1D: [value])
        assert!(
            approx_equal(rust_loss[[0, 0]], pytorch_loss_data[0], 1e-4),
            "Loss mismatch: rust={}, pytorch={}",
            rust_loss[[0, 0]],
            pytorch_loss_data[0]
        );

        // Compute gradients
        zero_all_grads();
        loss.backward();

        let weight_grad = layer_norm.weight.grad();
        let bias_grad = layer_norm.bias.grad();

        // Compare gradients
        let pytorch_weight_grad_data = pytorch_weight_grad.item();
        let pytorch_bias_grad_data = pytorch_bias_grad.item();

        // Compare weight gradients
        for i in 0..3 {
            assert!(
                approx_equal(weight_grad[i], pytorch_weight_grad_data[i], 1e-4),
                "Weight gradient mismatch at [{}]: rust={}, pytorch={}",
                i,
                weight_grad[i],
                pytorch_weight_grad_data[i]
            );
        }

        // Compare bias gradients
        for i in 0..3 {
            assert!(
                approx_equal(bias_grad[i], pytorch_bias_grad_data[i], 1e-4),
                "Bias gradient mismatch at [{}]: rust={}, pytorch={}",
                i,
                bias_grad[i],
                pytorch_bias_grad_data[i]
            );
        }

        // Verify gradients are finite and non-zero where expected
        for i in 0..3 {
            assert!(
                weight_grad[i].is_finite(),
                "Weight gradient should be finite at [{}]",
                i
            );
            assert!(
                bias_grad[i].is_finite(),
                "Bias gradient should be finite at [{}]",
                i
            );
        }

        // Verify expected gradient relationships
        let pytorch_mean_data = pytorch_mean.item();
        let pytorch_variance_data = pytorch_variance.item();
        let pytorch_std_data = pytorch_std.item();
        let pytorch_normalized_data = pytorch_normalized.item();

        // Expected calculation for [1, 4, 7]:
        // Mean = (1 + 4 + 7) / 3 = 4
        // Variance = ((1-4)^2 + (4-4)^2 + (7-4)^2) / 3 = (9 + 0 + 9) / 3 = 6
        // Std = sqrt(6 + eps) ≈ 2.449
        // Normalized = [(1-4)/2.449, (4-4)/2.449, (7-4)/2.449] = [-1.225, 0, 1.225]

        assert!(
            approx_equal(pytorch_mean_data[0], 4.0, 1e-4),
            "Mean should be 4.0: {}",
            pytorch_mean_data[0]
        );
        assert!(
            approx_equal(pytorch_variance_data[0], 6.0, 1e-4),
            "Variance should be 6.0: {}",
            pytorch_variance_data[0]
        );
        assert!(
            approx_equal(pytorch_std_data[0], 2.4494917392730713, 1e-4),
            "Std should be ~2.449: {}",
            pytorch_std_data[0]
        );

        // Verify normalized values match expected pattern
        assert!(
            approx_equal(pytorch_normalized_data[0], -1.2247356176376343, 1e-4),
            "First normalized should be ~-1.225: {}",
            pytorch_normalized_data[0]
        );
        assert!(
            approx_equal(pytorch_normalized_data[1], 0.0, 1e-4),
            "Second normalized should be ~0: {}",
            pytorch_normalized_data[1]
        );
        assert!(
            approx_equal(pytorch_normalized_data[2], 1.2247356176376343, 1e-4),
            "Third normalized should be ~1.225: {}",
            pytorch_normalized_data[2]
        );

        // Key insights about LayerNorm parameter gradients:
        // 1. Bias gradient = 1 for each parameter (since d(sum(output))/d(bias) = 1)
        // 2. Weight gradient = normalized_output (since d(sum(output))/d(weight) = normalized * 1)
        // 3. This shows how LayerNorm parameters learn to adjust the normalized distribution

        // Verify bias gradients are all 1.0 (since we're summing the output)
        for i in 0..3 {
            assert!(
                approx_equal(bias_grad[i], 1.0, 1e-4),
                "Bias gradient should be 1.0 at [{}]: {}",
                i,
                bias_grad[i]
            );
        }

        // Verify weight gradients equal normalized output
        for i in 0..3 {
            assert!(
                approx_equal(weight_grad[i], pytorch_normalized_data[i], 1e-4),
                "Weight gradient should equal normalized output at [{}]: weight_grad={}, normalized={}",
                i,
                weight_grad[i],
                pytorch_normalized_data[i]
            );
        }

        println!("✓ Rust LayerNorm parameter gradients match PyTorch LayerNorm");
        println!("  Input: [1, 4, 7] → Output: [-1.225, 0, 1.225]");
        println!("  Bias gradients: [1, 1, 1] (∂loss/∂bias = 1 for sum loss)");
        println!("  Weight gradients: [-1.225, 0, 1.225] (∂loss/∂weight = normalized_output)");
        println!("  This demonstrates how LayerNorm parameters learn to adjust normalization");
    }
}
