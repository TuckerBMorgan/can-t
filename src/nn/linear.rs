use crate::central::*;
use crate::nn::*;


pub struct Linear {
    weights: Tensor,
    bias: Option<Tensor>
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, has_bias: bool) -> Linear {
        let weights = Tensor::randn(Shape::new(vec![in_features, out_features]));
        let mut bias = None;
        if has_bias {
            bias = Some(Tensor::zeros(Shape::new(vec![out_features])));
        }
        Linear {
            weights,
            bias
        }
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
            _ => {

            }
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
        let layer = Linear::new(10, 5, true);  // Smaller layer for more manageable testing
        
        let weight_data = layer.weights.item();
        let weight_values: Vec<f32> = weight_data.iter().copied().collect();
        
        // Check that weights are not all zeros (should be random)
        let all_zeros = weight_values.iter().all(|&x| x == 0.0);
        assert!(!all_zeros, "Weights should not all be zero");
        
        // Check that weights are in reasonable range (not too large)
        let max_abs = weight_values.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_abs < 5.0, "Weights should not be too large: max_abs = {}", max_abs);
        
        // Check that we have at least some positive and negative values
        let has_positive = weight_values.iter().any(|&x| x > 0.0);
        let has_negative = weight_values.iter().any(|&x| x < 0.0);
        assert!(has_positive, "Should have some positive weights");
        assert!(has_negative, "Should have some negative weights");
        
        // Check that variance is reasonable (not all values identical)
        let mean = weight_values.iter().sum::<f32>() / weight_values.len() as f32;
        let variance = weight_values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / weight_values.len() as f32;
        assert!(variance > 1e-6, "Weights should have some variance, got {}", variance);
    }

    #[test]
    fn test_linear_different_dimensions() {
        // Test various input/output dimension combinations
        let test_cases = vec![
            (1, 1),   // Minimal case
            (5, 1),   // Many-to-one
            (1, 10),  // One-to-many
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
}