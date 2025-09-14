use crate::central::*;
use crate::nn::*;
use crate::utils::GGUFFile;

pub struct Linear {
    pub weights: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, has_bias: bool) -> Linear {
        let mut weights = Tensor::randn(Shape::new(vec![in_features, out_features]));
        weights.set_requires_grad(true);
        let mut bias = None;
        if has_bias {
            bias = Some(Tensor::zeros(Shape::new(vec![out_features])));
            bias.as_mut().unwrap().set_requires_grad(true);
        }
        Linear { weights, bias }
    }

    pub fn from_tensors(weights: Tensor, bias: Option<Tensor>) -> Linear {
        Linear {
            weights,
            bias
        }
    }   
    /// Creates a Linear layer from a GGUF file by loading weight and bias tensors
    /// # Arguments
    /// * `weight_tensor_name` - The name of the weight tensor in the GGUF file
    /// * `bias_tensor_name` - Optional name of the bias tensor in the GGUF file
    /// * `gguf_file` - The GGUF file data structure to load from
    pub fn from_gguf_file(
        weight_tensor_name: String,
        bias_tensor_name: Option<String>,
        gguf_file: &mut GGUFFile,
    ) -> Linear {
        let mut weights = Tensor::from_gguf_file(weight_tensor_name, gguf_file);
        // GGUF saves linear layers in pytorch format, which is inverse from cant
        let mut weights = weights.transpose(0, 1);
        weights.set_requires_grad(true);

        let mut bias = None;
        if let Some(bias_name) = bias_tensor_name {
            let mut bias_tensor = Tensor::from_gguf_file(bias_name, gguf_file);
            bias_tensor.set_requires_grad(true);
            bias = Some(bias_tensor);
        }

        Linear { weights, bias }
    }
}

impl Layer for Linear {
    fn forward(&mut self, inputs: Tensor) -> Tensor {
        let output = inputs << self.weights;//.transpose(0, 1);
        match &self.bias {
            Some(bias) => output + *bias,
            None => output,
        }
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        let mut ids = vec![self.weights.id];
        match &self.bias {
            Some(bias) => ids.push(bias.id),
            _ => {}
        }

        return ids;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_linear_creation_with_bias() {
        let layer = Linear::new(3, 2, true);

        // Check weight shape is [3, 2] (in_features, out_features)
        assert_eq!(layer.weights.shape.dimensions(), vec![3, 2]);

        // Check bias exists and has correct shape [2]
        assert!(layer.bias.is_some());
        let bias = layer.bias.as_ref().unwrap();
        assert_eq!(bias.shape.dimensions(), vec![2]);

        // Check bias is initialized to zeros
        let bias_data = bias.item();
        for i in 0..2 {
            assert!(approx_equal(bias_data[[i]], 0.0, 1e-6));
        }
    }

    #[test]
    fn test_linear_creation_without_bias() {
        let layer = Linear::new(4, 3, false);

        // Check weight shape is [4, 3]
        assert_eq!(layer.weights.shape.dimensions(), vec![4, 3]);

        // Check bias is None
        assert!(layer.bias.is_none());
    }

    #[test]
    fn test_linear_forward_with_bias() {
        let mut layer = Linear::new(2, 3, true);

        // Set known weights for predictable output
        // Weight matrix [2, 3]:
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        layer.weights = Tensor::from_vec(weight_data, vec![2, 3]);

        // Set known bias [1, -1, 2]
        let bias_data = vec![1.0, -1.0, 2.0];
        layer.bias = Some(Tensor::from_vec(bias_data, vec![3]));

        // Input: single sample [1, 2] -> shape [1, 2]
        let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);

        let output = layer.forward(input);

        // Expected calculation:
        // input @ weights + bias
        // [1, 2] @ [[1, 2, 3], [4, 5, 6]] + [1, -1, 2]
        // [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] + [1, -1, 2]
        // [9, 12, 15] + [1, -1, 2] = [10, 11, 17]

        assert_eq!(output.shape.dimensions(), vec![1, 3]);
        let result = output.item();
        assert!(approx_equal(result[[0, 0]], 10.0, 1e-5));
        assert!(approx_equal(result[[0, 1]], 11.0, 1e-5));
        assert!(approx_equal(result[[0, 2]], 17.0, 1e-5));
    }

    #[test]
    fn test_linear_forward_without_bias() {
        let mut layer = Linear::new(2, 3, false);

        // Set known weights
        let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        layer.weights = Tensor::from_vec(weight_data, vec![2, 3]);

        // Input: single sample [1, 2]
        let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);

        let output = layer.forward(input);

        // Expected: [1, 2] @ [[1, 2, 3], [4, 5, 6]] = [9, 12, 15]
        assert_eq!(output.shape.dimensions(), vec![1, 3]);
        let result = output.item();
        assert!(approx_equal(result[[0, 0]], 9.0, 1e-5));
        assert!(approx_equal(result[[0, 1]], 12.0, 1e-5));
        assert!(approx_equal(result[[0, 2]], 15.0, 1e-5));
    }

    #[test]
    fn test_linear_forward_batch() {
        let mut layer = Linear::new(2, 3, true);

        // Set known weights and bias
        let weight_data = vec![1.0, 0.0, -1.0, 0.0, 1.0, 1.0];
        layer.weights = Tensor::from_vec(weight_data, vec![2, 3]);
        layer.bias = Some(Tensor::from_vec(vec![0.5, -0.5, 1.0], vec![3]));

        // Batch input: 2 samples, each with 2 features -> [2, 2]
        let input = Tensor::from_vec(vec![1.0, 2.0, -1.0, 3.0], vec![2, 2]);

        let output = layer.forward(input);

        // Expected shape: [2, 3] (batch_size, out_features)
        assert_eq!(output.shape.dimensions(), vec![2, 3]);

        let result = output.item();

        // Sample 1: [1, 2] @ [[1, 0, -1], [0, 1, 1]] + [0.5, -0.5, 1.0]
        //          [1*1 + 2*0, 1*0 + 2*1, 1*(-1) + 2*1] + [0.5, -0.5, 1.0]
        //          [1, 2, 1] + [0.5, -0.5, 1.0] = [1.5, 1.5, 2.0]
        assert!(approx_equal(result[[0, 0]], 1.5, 1e-5));
        assert!(approx_equal(result[[0, 1]], 1.5, 1e-5));
        assert!(approx_equal(result[[0, 2]], 2.0, 1e-5));

        // Sample 2: [-1, 3] @ [[1, 0, -1], [0, 1, 1]] + [0.5, -0.5, 1.0]
        //          [-1*1 + 3*0, -1*0 + 3*1, -1*(-1) + 3*1] + [0.5, -0.5, 1.0]
        //          [-1, 3, 4] + [0.5, -0.5, 1.0] = [-0.5, 2.5, 5.0]
        assert!(approx_equal(result[[1, 0]], -0.5, 1e-5));
        assert!(approx_equal(result[[1, 1]], 2.5, 1e-5));
        assert!(approx_equal(result[[1, 2]], 5.0, 1e-5));
    }

    #[test]
    fn test_linear_get_parameters_with_bias() {
        let layer = Linear::new(3, 2, true);

        let params = layer.get_parameters();

        // Should have 2 parameters: weights and bias
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], layer.weights.id);
        assert_eq!(params[1], layer.bias.as_ref().unwrap().id);
    }

    #[test]
    fn test_linear_get_parameters_without_bias() {
        let layer = Linear::new(3, 2, false);

        let params = layer.get_parameters();

        // Should have 1 parameter: only weights
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], layer.weights.id);
    }

    #[test]
    fn test_linear_weight_initialization_range() {
        let layer = Linear::new(10, 5, true); // Smaller layer for more manageable testing

        let weight_data = layer.weights.item();
        let weight_values: Vec<f32> = weight_data.iter().copied().collect();

        // Check that weights are not all zeros (should be random)
        let all_zeros = weight_values.iter().all(|&x| x == 0.0);
        assert!(!all_zeros, "Weights should not all be zero");

        // Check that weights are in reasonable range (not too large)
        let max_abs = weight_values.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(
            max_abs < 5.0,
            "Weights should not be too large: max_abs = {}",
            max_abs
        );

        // Check that we have at least some positive and negative values
        let has_positive = weight_values.iter().any(|&x| x > 0.0);
        let has_negative = weight_values.iter().any(|&x| x < 0.0);
        assert!(has_positive, "Should have some positive weights");
        assert!(has_negative, "Should have some negative weights");

        // Check that variance is reasonable (not all values identical)
        let mean = weight_values.iter().sum::<f32>() / weight_values.len() as f32;
        let variance = weight_values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / weight_values.len() as f32;
        assert!(
            variance > 1e-6,
            "Weights should have some variance, got {}",
            variance
        );
    }

    #[test]
    fn test_linear_different_dimensions() {
        // Test various input/output dimension combinations
        let test_cases = vec![
            (1, 1),    // Minimal case
            (5, 1),    // Many-to-one
            (1, 10),   // One-to-many
            (128, 64), // Typical neural network sizes
        ];

        for (in_dim, out_dim) in test_cases {
            let mut layer = Linear::new(in_dim, out_dim, true);

            // Test single sample
            let input = Tensor::ones(Shape::new(vec![1, in_dim]));
            let output = layer.forward(input);

            assert_eq!(output.shape.dimensions(), vec![1, out_dim]);

            // Test batch
            let batch_size = 3;
            let batch_input = Tensor::ones(Shape::new(vec![batch_size, in_dim]));
            let batch_output = layer.forward(batch_input);

            assert_eq!(batch_output.shape.dimensions(), vec![batch_size, out_dim]);
        }
    }

    #[test]
    fn test_linear_chain() {
        // Test chaining multiple linear layers
        let mut layer1 = Linear::new(3, 4, true);
        let mut layer2 = Linear::new(4, 2, true);

        let input = Tensor::from_vec(vec![1.0, -1.0, 0.5], vec![1, 3]);

        let hidden = layer1.forward(input);
        assert_eq!(hidden.shape.dimensions(), vec![1, 4]);

        let output = layer2.forward(hidden);
        assert_eq!(output.shape.dimensions(), vec![1, 2]);

        // Ensure output is finite and reasonable
        let result = output.item();
        assert!(result[[0, 0]].is_finite());
        assert!(result[[0, 1]].is_finite());
    }

    // ========== BACKWARD PASS TESTS ==========

    #[test]
    fn test_linear_backward_basic_with_bias() {
        use crate::central::zero_all_grads;
        
        let mut layer = Linear::new(2, 3, true);
        
        // Set known weights and bias for predictable gradients
        layer.weights = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        layer.weights.set_requires_grad(true);
        
        layer.bias = Some(Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![3]));
        layer.bias.as_mut().unwrap().set_requires_grad(true);
        
        // Input with requires_grad
        let mut input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        input.set_requires_grad(true);
        
        let output = layer.forward(input.clone());
        let loss = output.sum(vec![0, 1], true);
        
        zero_all_grads();
        loss.backward();
        
        // Check input gradients
        let input_grad = input.grad();
        assert_eq!(input_grad.shape(), &[1, 2]);
        
        // Input gradient should be sum of weight columns (since loss = sum(output))
        // grad_input = weights.sum(axis=1) = [1+2+3, 4+5+6] = [6, 15]
        assert!(approx_equal(input_grad[[0, 0]], 6.0, 1e-5));
        assert!(approx_equal(input_grad[[0, 1]], 15.0, 1e-5));
        
        // Check weight gradients
        let weight_grad = layer.weights.grad();
        assert_eq!(weight_grad.shape(), &[2, 3]);
        
        // Weight gradients: outer product of input and output gradient
        // Since output gradient is [1, 1, 1] and input is [1, 2]:
        // grad_weights = input.T @ output_grad = [[1], [2]] @ [[1, 1, 1]] = [[1, 1, 1], [2, 2, 2]]
        assert!(approx_equal(weight_grad[[0, 0]], 1.0, 1e-5));
        assert!(approx_equal(weight_grad[[0, 1]], 1.0, 1e-5));
        assert!(approx_equal(weight_grad[[0, 2]], 1.0, 1e-5));
        assert!(approx_equal(weight_grad[[1, 0]], 2.0, 1e-5));
        assert!(approx_equal(weight_grad[[1, 1]], 2.0, 1e-5));
        assert!(approx_equal(weight_grad[[1, 2]], 2.0, 1e-5));
        
        // Check bias gradients (should equal output gradients)
        let bias_grad = layer.bias.as_ref().unwrap().grad();
        assert_eq!(bias_grad.shape(), &[3]);
        
        // Bias gradient should be [1, 1, 1] since loss = sum(output)
        assert!(approx_equal(bias_grad[[0]], 1.0, 1e-5));
        assert!(approx_equal(bias_grad[[1]], 1.0, 1e-5));
        assert!(approx_equal(bias_grad[[2]], 1.0, 1e-5));
    }

    #[test]
    fn test_linear_backward_without_bias() {
        use crate::central::zero_all_grads;
        
        let mut layer = Linear::new(3, 2, false);
        
        // Set known weights
        layer.weights = Tensor::from_vec(vec![1.0, -1.0, 2.0, 0.0, -1.0, 3.0], vec![3, 2]);
        layer.weights.set_requires_grad(true);
        
        let mut input = Tensor::from_vec(vec![1.0, 2.0, -1.0], vec![1, 3]);
        input.set_requires_grad(true);
        
        let output = layer.forward(input.clone());
        let loss = output.sum(vec![0, 1], true);
        
        zero_all_grads();
        loss.backward();
        
        // Check input gradients
        let input_grad = input.grad();
        // grad_input = weights.sum(axis=1) = [1+(-1), 2+0, (-1)+3] = [0, 2, 2]
        assert!(approx_equal(input_grad[[0, 0]], 0.0, 1e-5));
        assert!(approx_equal(input_grad[[0, 1]], 2.0, 1e-5));
        assert!(approx_equal(input_grad[[0, 2]], 2.0, 1e-5));
        
        // Check weight gradients
        let weight_grad = layer.weights.grad();
        // grad_weights = input.T @ output_grad = [[1], [2], [-1]] @ [[1, 1]] = [[1, 1], [2, 2], [-1, -1]]
        assert!(approx_equal(weight_grad[[0, 0]], 1.0, 1e-5));
        assert!(approx_equal(weight_grad[[0, 1]], 1.0, 1e-5));
        assert!(approx_equal(weight_grad[[1, 0]], 2.0, 1e-5));
        assert!(approx_equal(weight_grad[[1, 1]], 2.0, 1e-5));
        assert!(approx_equal(weight_grad[[2, 0]], -1.0, 1e-5));
        assert!(approx_equal(weight_grad[[2, 1]], -1.0, 1e-5));
    }

    #[test]
    fn test_linear_backward_batch() {
        use crate::central::zero_all_grads;
        
        let mut layer = Linear::new(2, 2, true);
        
        // Set simple weights for easy verification
        layer.weights = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]); // Identity matrix
        layer.weights.set_requires_grad(true);
        
        layer.bias = Some(Tensor::zeros(Shape::new(vec![2])));
        layer.bias.as_mut().unwrap().set_requires_grad(true);
        
        // Batch input: 2 samples
        let mut input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        input.set_requires_grad(true);
        
        let output = layer.forward(input.clone());
        let loss = output.sum(vec![0, 1], true);
        
        zero_all_grads();
        loss.backward();
        
        // Check input gradients (sum across samples)
        let input_grad = input.grad();
        // Each input element should have gradient = sum of corresponding weight row
        assert!(approx_equal(input_grad[[0, 0]], 1.0, 1e-5)); // First row, first col
        assert!(approx_equal(input_grad[[0, 1]], 1.0, 1e-5)); // First row, second col  
        assert!(approx_equal(input_grad[[1, 0]], 1.0, 1e-5)); // Second row, first col
        assert!(approx_equal(input_grad[[1, 1]], 1.0, 1e-5)); // Second row, second col
        
        // Check bias gradients (sum across batch)
        let bias_grad = layer.bias.as_ref().unwrap().grad();
        // Each bias element should have gradient = batch_size (since each sample contributes 1)
        assert!(approx_equal(bias_grad[[0]], 2.0, 1e-5)); // sum across batch for first output
        assert!(approx_equal(bias_grad[[1]], 2.0, 1e-5)); // sum across batch for second output
    }

    #[test]
    fn test_linear_backward_chained_layers() {
        use crate::central::zero_all_grads;
        
        // Create a simple 2-layer network: 2 -> 3 -> 1
        let mut layer1 = Linear::new(2, 3, true);
        let mut layer2 = Linear::new(3, 1, true);
        
        // Set simple weights for predictable gradients
        layer1.weights = Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0], vec![2, 3]);
        layer1.weights.set_requires_grad(true);
        layer1.bias = Some(Tensor::zeros(Shape::new(vec![3])));
        layer1.bias.as_mut().unwrap().set_requires_grad(true);
        
        layer2.weights = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3, 1]);
        layer2.weights.set_requires_grad(true);
        layer2.bias = Some(Tensor::zeros(Shape::new(vec![1])));
        layer2.bias.as_mut().unwrap().set_requires_grad(true);
        
        let mut input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        input.set_requires_grad(true);
        
        // Forward pass through both layers
        let hidden = layer1.forward(input.clone());
        let output = layer2.forward(hidden);
        let loss = output.sum(vec![0, 1], true);
        
        zero_all_grads();
        loss.backward();
        
        // Check that gradients flow through both layers
        let input_grad = input.grad();
        assert!(input_grad.iter().all(|&g| g.is_finite()));
        assert!(input_grad.iter().any(|&g| g.abs() > 1e-6)); // Should have non-trivial gradients
        
        // Check layer1 gradients
        let layer1_weight_grad = layer1.weights.grad();
        assert!(layer1_weight_grad.iter().all(|&g| g.is_finite()));
        assert!(layer1_weight_grad.iter().any(|&g| g.abs() > 1e-6));
        
        let layer1_bias_grad = layer1.bias.as_ref().unwrap().grad();
        assert!(layer1_bias_grad.iter().all(|&g| g.is_finite()));
        assert!(layer1_bias_grad.iter().any(|&g| g.abs() > 1e-6));
        
        // Check layer2 gradients
        let layer2_weight_grad = layer2.weights.grad();
        assert!(layer2_weight_grad.iter().all(|&g| g.is_finite()));
        assert!(layer2_weight_grad.iter().any(|&g| g.abs() > 1e-6));
        
        let layer2_bias_grad = layer2.bias.as_ref().unwrap().grad();
        assert!(layer2_bias_grad.iter().all(|&g| g.is_finite()));
        assert!(layer2_bias_grad.iter().any(|&g| g.abs() > 1e-6));
    }

    #[test]
    fn test_linear_backward_with_relu_activation() {
        use crate::central::zero_all_grads;
        
        let mut layer1 = Linear::new(2, 3, true);
        let mut layer2 = Linear::new(3, 1, true);
        
        // Set weights that will produce both positive and negative values for ReLU
        layer1.weights = Tensor::from_vec(vec![1.0, -1.0, 2.0, -2.0, 1.0, -1.0], vec![2, 3]);
        layer1.weights.set_requires_grad(true);
        layer1.bias = Some(Tensor::from_vec(vec![0.5, -0.5, 1.0], vec![3]));
        layer1.bias.as_mut().unwrap().set_requires_grad(true);
        
        layer2.weights = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3, 1]);
        layer2.weights.set_requires_grad(true);
        layer2.bias = Some(Tensor::zeros(Shape::new(vec![1])));
        layer2.bias.as_mut().unwrap().set_requires_grad(true);
        
        let mut input = Tensor::from_vec(vec![1.0, 1.0], vec![1, 2]);
        input.set_requires_grad(true);
        
        // Forward: input -> linear1 -> relu -> linear2
        let mut hidden_raw = layer1.forward(input.clone());
        let hidden_relu = hidden_raw.relu(); // Apply ReLU activation
        let output = layer2.forward(hidden_relu);
        let loss = output.sum(vec![0, 1], true);
        
        zero_all_grads();
        loss.backward();
        
        // Check that gradients exist and are reasonable
        let input_grad = input.grad();
        assert!(input_grad.iter().all(|&g| g.is_finite()));
        
        // ReLU should block some gradients where activations were negative
        let layer1_weight_grad = layer1.weights.grad();
        assert!(layer1_weight_grad.iter().all(|&g| g.is_finite()));
        
        // Some gradients might be zero due to ReLU blocking
        let has_zero_grads = layer1_weight_grad.iter().any(|&g| g == 0.0);
        let has_nonzero_grads = layer1_weight_grad.iter().any(|&g| g.abs() > 1e-6);
        
        // We should have at least some non-zero gradients
        assert!(has_nonzero_grads, "Should have some non-zero gradients despite ReLU");
    }

    #[test]
    fn test_linear_backward_with_tanh_activation() {
        use crate::central::zero_all_grads;
        
        let mut layer1 = Linear::new(2, 2, true);
        let mut layer2 = Linear::new(2, 1, false);
        
        // Set moderate weights to avoid tanh saturation
        layer1.weights = Tensor::from_vec(vec![0.5, -0.5, 0.3, 0.7], vec![2, 2]);
        layer1.weights.set_requires_grad(true);
        layer1.bias = Some(Tensor::from_vec(vec![0.1, -0.1], vec![2]));
        layer1.bias.as_mut().unwrap().set_requires_grad(true);
        
        layer2.weights = Tensor::from_vec(vec![1.0, -1.0], vec![2, 1]);
        layer2.weights.set_requires_grad(true);
        
        let mut input = Tensor::from_vec(vec![1.0, -1.0], vec![1, 2]);
        input.set_requires_grad(true);
        
        // Forward: input -> linear1 -> tanh -> linear2
        let hidden_raw = layer1.forward(input.clone());
        let hidden_tanh = hidden_raw.tanh(); // Apply tanh activation
        let output = layer2.forward(hidden_tanh);
        let loss = output.pow(2.0).sum(vec![0, 1], true); // Squared loss for stronger gradients
        
        zero_all_grads();
        loss.backward();
        
        // Check that gradients flow through tanh
        let input_grad = input.grad();
        assert!(input_grad.iter().all(|&g| g.is_finite()));
        assert!(input_grad.iter().any(|&g| g.abs() > 1e-6));
        
        // Tanh should modulate gradients but not block them completely
        let layer1_weight_grad = layer1.weights.grad();
        assert!(layer1_weight_grad.iter().all(|&g| g.is_finite()));
        assert!(layer1_weight_grad.iter().any(|&g| g.abs() > 1e-6));
        
        // Check that tanh affects gradient magnitude (should be scaled by tanh derivative)
        let layer1_bias_grad = layer1.bias.as_ref().unwrap().grad();
        assert!(layer1_bias_grad.iter().all(|&g| g.is_finite()));
        assert!(layer1_bias_grad.iter().any(|&g| g.abs() > 1e-6));
    }

    #[test]
    fn test_linear_backward_deep_network() {
        use crate::central::zero_all_grads;
        
        // Create a deeper network: 3 -> 4 -> 3 -> 2 -> 1
        let mut layer1 = Linear::new(3, 4, true);
        let mut layer2 = Linear::new(4, 3, true);
        let mut layer3 = Linear::new(3, 2, true);
        let mut layer4 = Linear::new(2, 1, true);
        
        // Initialize with small random-like weights
        layer1.weights = Tensor::from_vec((0..12).map(|i| (i as f32) * 0.1 - 0.6).collect(), vec![3, 4]);
        layer1.weights.set_requires_grad(true);
        layer1.bias = Some(Tensor::from_vec(vec![0.1, -0.1, 0.05, -0.05], vec![4]));
        layer1.bias.as_mut().unwrap().set_requires_grad(true);
        
        layer2.weights = Tensor::from_vec((0..12).map(|i| (i as f32) * 0.08 - 0.5).collect(), vec![4, 3]);
        layer2.weights.set_requires_grad(true);
        layer2.bias = Some(Tensor::from_vec(vec![0.02, -0.02, 0.03], vec![3]));
        layer2.bias.as_mut().unwrap().set_requires_grad(true);
        
        layer3.weights = Tensor::from_vec(vec![0.3, -0.3, 0.2, -0.2, 0.1, -0.1], vec![3, 2]);
        layer3.weights.set_requires_grad(true);
        layer3.bias = Some(Tensor::from_vec(vec![0.01, -0.01], vec![2]));
        layer3.bias.as_mut().unwrap().set_requires_grad(true);
        
        layer4.weights = Tensor::from_vec(vec![0.5, -0.5], vec![2, 1]);
        layer4.weights.set_requires_grad(true);
        layer4.bias = Some(Tensor::zeros(Shape::new(vec![1])));
        layer4.bias.as_mut().unwrap().set_requires_grad(true);
        
        let mut input = Tensor::from_vec(vec![1.0, 0.5, -0.5], vec![1, 3]);
        input.set_requires_grad(true);
        
        // Forward pass through deep network with activations
        let h1 = layer1.forward(input.clone()).tanh();
        let h2 = layer2.forward(h1).relu();
        let h3 = layer3.forward(h2).tanh();
        let output = layer4.forward(h3);
        let loss = output.pow(2.0).sum(vec![0, 1], true);
        
        zero_all_grads();
        loss.backward();
        
        // Check that gradients flow through all layers
        let input_grad = input.grad();
        assert!(input_grad.iter().all(|&g| g.is_finite()));
        
        // Verify each layer has meaningful gradients
        let layers_and_names = vec![
            (&layer1, "layer1"),
            (&layer2, "layer2"), 
            (&layer3, "layer3"),
            (&layer4, "layer4"),
        ];
        
        for (layer, name) in layers_and_names {
            let weight_grad = layer.weights.grad();
            assert!(
                weight_grad.iter().all(|&g| g.is_finite()),
                "{} weights should have finite gradients", name
            );
            
            let bias_grad = layer.bias.as_ref().unwrap().grad();
            assert!(
                bias_grad.iter().all(|&g| g.is_finite()),
                "{} bias should have finite gradients", name
            );
            
            // Should have at least some non-trivial gradients
            let has_significant_grad = weight_grad.iter().any(|&g| g.abs() > 1e-8) ||
                                     bias_grad.iter().any(|&g| g.abs() > 1e-8);
            assert!(
                has_significant_grad,
                "{} should have some significant gradients", name
            );
        }
    }

    #[test]
    fn test_linear_backward_different_loss_functions() {
        use crate::central::zero_all_grads;
        
        let mut layer = Linear::new(2, 2, true);
        
        // Set fixed weights for comparison
        layer.weights = Tensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], vec![2, 2]);
        layer.weights.set_requires_grad(true);
        layer.bias = Some(Tensor::from_vec(vec![0.1, -0.1], vec![2]));
        layer.bias.as_mut().unwrap().set_requires_grad(true);
        
        let mut input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        input.set_requires_grad(true);
        
        // Test 1: Sum loss
        let output1 = layer.forward(input.clone());
        let loss1 = output1.sum(vec![0, 1], true);
        
        zero_all_grads();
        loss1.backward();
        
        let weight_grad1 = layer.weights.grad().iter().copied().collect::<Vec<f32>>();
        let bias_grad1 = layer.bias.as_ref().unwrap().grad().iter().copied().collect::<Vec<f32>>();
        
        // Test 2: Mean loss (should produce different gradients)
        let output2 = layer.forward(input.clone());
        let loss2 = output2.mean(vec![0, 1]);
        
        zero_all_grads();
        loss2.backward();
        
        let weight_grad2 = layer.weights.grad().iter().copied().collect::<Vec<f32>>();
        let bias_grad2 = layer.bias.as_ref().unwrap().grad().iter().copied().collect::<Vec<f32>>();
        
        // Gradients should be different due to different loss functions
        let weight_grads_differ = weight_grad1.iter().zip(weight_grad2.iter())
            .any(|(g1, g2)| (g1 - g2).abs() > 1e-6);
        let bias_grads_differ = bias_grad1.iter().zip(bias_grad2.iter())
            .any(|(g1, g2)| (g1 - g2).abs() > 1e-6);
        
        assert!(weight_grads_differ || bias_grads_differ, 
                "Different loss functions should produce different gradients");
    }

    #[test]
    fn test_linear_backward_gradient_accumulation() {
        use crate::central::zero_all_grads;
        
        let mut layer = Linear::new(2, 1, true);
        
        layer.weights = Tensor::from_vec(vec![1.0, -1.0], vec![2, 1]);
        layer.weights.set_requires_grad(true);
        layer.bias = Some(Tensor::from_vec(vec![0.5], vec![1]));
        layer.bias.as_mut().unwrap().set_requires_grad(true);
        
        let mut input1 = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        input1.set_requires_grad(true);
        let mut input2 = Tensor::from_vec(vec![2.0, 1.0], vec![1, 2]);
        input2.set_requires_grad(true);
        
        zero_all_grads();
        
        // First forward/backward pass
        let output1 = layer.forward(input1.clone());
        let loss1 = output1.sum(vec![0, 1], true);
        loss1.backward();
        
        let weight_grad_after_first = layer.weights.grad().iter().copied().collect::<Vec<f32>>();
        let bias_grad_after_first = layer.bias.as_ref().unwrap().grad().iter().copied().collect::<Vec<f32>>();
        
        // Second forward/backward pass (should accumulate gradients)
        let output2 = layer.forward(input2);
        let loss2 = output2.sum(vec![0, 1], true);
        loss2.backward();
        
        let weight_grad_after_second = layer.weights.grad().iter().copied().collect::<Vec<f32>>();
        let bias_grad_after_second = layer.bias.as_ref().unwrap().grad().iter().copied().collect::<Vec<f32>>();
        
        // Gradients should have accumulated (increased in magnitude)
        for i in 0..weight_grad_after_first.len() {
            assert!(
                weight_grad_after_second[i].abs() >= weight_grad_after_first[i].abs(),
                "Weight gradients should accumulate: {} vs {}", 
                weight_grad_after_second[i], weight_grad_after_first[i]
            );
        }
        
        for i in 0..bias_grad_after_first.len() {
            assert!(
                bias_grad_after_second[i].abs() >= bias_grad_after_first[i].abs(),
                "Bias gradients should accumulate: {} vs {}", 
                bias_grad_after_second[i], bias_grad_after_first[i]
            );
        }
    }
}
