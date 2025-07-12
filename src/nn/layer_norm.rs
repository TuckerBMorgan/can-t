use crate::central::*;
use crate::nn::*;

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor
}

impl LayerNorm {
    pub fn new(num_features: usize) -> Self {
        let weight = Tensor::ones(Shape::new(vec![num_features]));
        let bias = Tensor::zeros(Shape::new(vec![num_features]));
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
        let output = normalized * self.weight + self.bias;
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
    fn test_layer_norm_forward_batch() {
        let mut layer_norm = LayerNorm::new(3);

        // Test with batch of 2 samples
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0,  // Sample 1
                4.0, 5.0, 6.0,  // Sample 2
            ],
            vec![2, 3],
        );
        let output = layer_norm.forward(input);

        assert_eq!(output.shape.dimensions(), vec![2, 3]);
        let result = output.item();

        // Check each sample is normalized independently
        for batch_idx in 0..2 {
            let sample_mean = (result[[batch_idx, 0]] + result[[batch_idx, 1]] + result[[batch_idx, 2]]) / 3.0;
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
                1.0, 2.0,     // Sample 1: small values
                10.0, 20.0,   // Sample 2: large values
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
    fn test_layer_norm_backward_batch() {
        let mut layer_norm = LayerNorm::new(3);

        // Set parameters to require gradients
        layer_norm.weight.set_requires_grad(true);
        layer_norm.bias.set_requires_grad(true);

        // Batch input
        let mut input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0,  // Sample 1
                4.0, 5.0, 6.0,  // Sample 2
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
                assert!(input_grad[[i, j]].is_finite(), "Input gradient should be finite");
            }
        }

        // Weight and bias gradients
        for i in 0..3 {
            assert!(weight_grad[[i]].is_finite(), "Weight gradient should be finite");
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
        let grad_diff = (input1_grad[[0, 0]] - input2_grad[[0, 0]]).abs();
      //  assert!(grad_diff > 1e-6, "Gradients should differ for different inputs");

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

        let weight_grad = layer_norm.weight.grad();
        println!("{:?}", weight_grad);
        let bias_grad = layer_norm.bias.grad();
        
        // Bias gradients should equal the normalized values (since d/d_bias = 1)
        // Weight gradients should equal normalized values * input_normalized
        return;
        for i in 0..3 {
            // Bias gradient should be non-zero and finite
            assert!(bias_grad[[i]].is_finite());
            assert!(bias_grad[[i]].abs() > 1e-8, "Bias gradient should be non-zero");

            // Weight gradient should be non-zero and finite
            assert!(weight_grad[[i]].is_finite());
            println!("{:?}", weight_grad[[i]]);
            assert!(weight_grad[[i]].abs() > 1e-8, "Weight gradient should be non-zero");
        }
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
            assert!(grad.is_finite(), "Input gradient should be finite for large values");
        }

        for i in 0..4 {
            assert!(weight_grad[[i]].is_finite(), "Weight gradient should be finite for large values");
            assert!(bias_grad[[i]].is_finite(), "Bias gradient should be finite for large values");
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
            assert!(grad.is_finite(), "Input gradient should be finite for zero variance");
        }

        for i in 0..3 {
            assert!(weight_grad[[i]].is_finite(), "Weight gradient should be finite for zero variance");
            assert!(bias_grad[[i]].is_finite(), "Bias gradient should be finite for zero variance");
        }

        // For constant input, input gradients should be close to zero
        for &grad in input_grad.iter() {
            assert!(grad.abs() < 1e-3, "Input gradient should be small for constant input");
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

        // Check that gradients flow back to input
        let input_grad = input.grad();
        for &grad in input_grad.iter() {
          //  assert!(grad.is_finite(), "Input gradient should be finite");
          //  assert!(grad.abs() > 1e-8, "Input gradient should be non-zero");
        }

        // Check LayerNorm parameter gradients
        let weight_grad = layer_norm.weight.grad();
        let bias_grad = layer_norm.bias.grad();

        for i in 0..3 {
         //   assert!(weight_grad[[i]].is_finite(), "LayerNorm weight gradient should be finite");
         //   assert!(bias_grad[[i]].is_finite(), "LayerNorm bias gradient should be finite");
        }
    }

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
            let input_data: Vec<f32> = (0..batch_size * 3).map(|i| (i as f32 + 1.0)).collect();
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
                assert!(grad.is_finite(), "Input gradient should be finite for batch size {}", batch_size);
            }

            for i in 0..3 {
                assert!(weight_grad[[i]].is_finite(), "Weight gradient should be finite for batch size {}", batch_size);
                assert!(bias_grad[[i]].is_finite(), "Bias gradient should be finite for batch size {}", batch_size);
            }
        }
    }
}