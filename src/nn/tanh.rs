use crate::central::*;
use crate::nn::*;


pub struct TanhLayer {

}

impl TanhLayer {
    pub fn new() -> TanhLayer {
        TanhLayer {

        }
    }
}

impl Layer for TanhLayer {
    fn forward(&mut self, input: Tensor) -> Tensor {
        return input.tanh();
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        return vec![];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_tanh_layer_creation() {
        let layer = TanhLayer::new();
        
        // TanhLayer should have no parameters
        assert_eq!(layer.get_parameters().len(), 0);
    }

    #[test]
    fn test_tanh_layer_forward_single_value() {
        let mut layer = TanhLayer::new();
        
        // Test with a single value
        let input = Tensor::from_vec(vec![0.0], vec![1]);
        let output = layer.forward(input);
        
        assert_eq!(output.shape.dimensions(), vec![1]);
        let result = output.item();
        assert!(approx_equal(result[[0]], 0.0, 1e-6)); // tanh(0) = 0
    }

    #[test]
    fn test_tanh_layer_forward_known_values() {
        let mut layer = TanhLayer::new();
        
        // Test with known tanh values
        let test_cases = vec![
            (0.0, 0.0),           // tanh(0) = 0
            (1.0, 0.761594156),   // tanh(1) ≈ 0.7616
            (-1.0, -0.761594156), // tanh(-1) ≈ -0.7616
            (2.0, 0.964027580),   // tanh(2) ≈ 0.9640
            (-2.0, -0.964027580), // tanh(-2) ≈ -0.9640
        ];
        
        for (input_val, expected) in test_cases {
            let input = Tensor::from_vec(vec![input_val], vec![1]);
            let output = layer.forward(input);
            
            let result = output.item();
            assert!(approx_equal(result[[0]], expected, 1e-5), 
                   "tanh({}) = {}, expected {}", input_val, result[[0]], expected);
        }
    }

    #[test]
    fn test_tanh_layer_forward_vector() {
        let mut layer = TanhLayer::new();
        
        // Test with a vector of values
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let output = layer.forward(input);
        
        assert_eq!(output.shape.dimensions(), vec![5]);
        let result = output.item();
        
        // Check each value
        let expected = vec![-0.964027580, -0.761594156, 0.0, 0.761594156, 0.964027580];
        for i in 0..5 {
            assert!(approx_equal(result[[i]], expected[i], 1e-5));
        }
    }

    #[test]
    fn test_tanh_layer_forward_matrix() {
        let mut layer = TanhLayer::new();
        
        // Test with a 2x3 matrix
        let input = Tensor::from_vec(
            vec![-1.0, 0.0, 1.0, -0.5, 0.5, 2.0], 
            vec![2, 3]
        );
        let output = layer.forward(input);
        
        assert_eq!(output.shape.dimensions(), vec![2, 3]);
        let result = output.item();
        
        // Check specific values
        assert!(approx_equal(result[[0, 0]], (-1.0f32).tanh(), 1e-5)); // tanh(-1.0)
        assert!(approx_equal(result[[0, 1]], 0.0, 1e-5));              // tanh(0.0)
        assert!(approx_equal(result[[0, 2]], (1.0f32).tanh(), 1e-5));  // tanh(1.0)
        assert!(approx_equal(result[[1, 0]], (-0.5f32).tanh(), 1e-5)); // tanh(-0.5)
        assert!(approx_equal(result[[1, 1]], (0.5f32).tanh(), 1e-5));  // tanh(0.5)
        assert!(approx_equal(result[[1, 2]], (2.0f32).tanh(), 1e-5));  // tanh(2.0)
    }

    #[test]
    fn test_tanh_layer_forward_batch() {
        let mut layer = TanhLayer::new();
        
        // Test with batch input [batch_size=2, features=3]
        let input = Tensor::from_vec(
            vec![
                1.0, -1.0, 0.0,  // Sample 1
                0.5, -0.5, 2.0,  // Sample 2
            ],
            vec![2, 3]
        );
        let output = layer.forward(input);
        
        assert_eq!(output.shape.dimensions(), vec![2, 3]);
        let result = output.item();
        
        // Check batch 1
        assert!(approx_equal(result[[0, 0]], (1.0f32).tanh(), 1e-5));
        assert!(approx_equal(result[[0, 1]], (-1.0f32).tanh(), 1e-5));
        assert!(approx_equal(result[[0, 2]], 0.0, 1e-5));
        
        // Check batch 2
        assert!(approx_equal(result[[1, 0]], (0.5f32).tanh(), 1e-5));
        assert!(approx_equal(result[[1, 1]], (-0.5f32).tanh(), 1e-5));
        assert!(approx_equal(result[[1, 2]], (2.0f32).tanh(), 1e-5));
    }

    #[test]
    fn test_tanh_layer_range_properties() {
        let mut layer = TanhLayer::new();
        
        // Test that tanh is bounded between -1 and 1
        let large_values = vec![-100.0, -10.0, -5.0, 5.0, 10.0, 100.0];
        let input = Tensor::from_vec(large_values, vec![6]);
        let output = layer.forward(input);
        
        let result = output.item();
        for i in 0..6 {
            assert!(result[[i]] >= -1.0, "tanh output should be >= -1, got {}", result[[i]]);
            assert!(result[[i]] <= 1.0, "tanh output should be <= 1, got {}", result[[i]]);
        }
        
        // Check that very large positive values approach 1
        assert!(result[[5]] > 0.999); // tanh(100) should be very close to 1
        
        // Check that very large negative values approach -1
        assert!(result[[0]] < -0.999); // tanh(-100) should be very close to -1
    }

    #[test]
    fn test_tanh_layer_antisymmetry() {
        let mut layer = TanhLayer::new();
        
        // Test that tanh(-x) = -tanh(x)
        let values = vec![0.1, 0.5, 1.0, 1.5, 2.0];
        
        for &val in &values {
            let positive_input = Tensor::from_vec(vec![val], vec![1]);
            let negative_input = Tensor::from_vec(vec![-val], vec![1]);
            
            let positive_output = layer.forward(positive_input);
            let negative_output = layer.forward(negative_input);
            
            let pos_result = positive_output.item()[[0]];
            let neg_result = negative_output.item()[[0]];
            
            assert!(approx_equal(pos_result, -neg_result, 1e-6), 
                   "tanh({}) = {}, tanh({}) = {}, should be antisymmetric", 
                   val, pos_result, -val, neg_result);
        }
    }

    #[test]
    fn test_tanh_layer_zero_gradient_at_saturation() {
        let mut layer = TanhLayer::new();
        
        // Test with values that should saturate tanh (large absolute values)
        let mut input = Tensor::from_vec(vec![10.0], vec![1]);
        input.set_requires_grad(true);
        
        let output = layer.forward(input);
        let loss = output.sum(vec![0], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        
        // Gradient should be very small for saturated values
        assert!(grad[[0]].abs() < 0.01, 
               "Gradient at saturation should be small, got {}", grad[[0]]);
    }

    #[test]
    fn test_tanh_layer_gradient_at_zero() {
        let mut layer = TanhLayer::new();
        
        // Test gradient at zero (should be 1.0)
        let mut input = Tensor::from_vec(vec![0.0], vec![1]);
        input.set_requires_grad(true);
        
        let output = layer.forward(input);
        let loss = output.sum(vec![0], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        
        // tanh'(0) = 1 - tanh²(0) = 1 - 0² = 1
        assert!(approx_equal(grad[[0]], 1.0, 1e-5), 
               "Gradient of tanh at 0 should be 1.0, got {}", grad[[0]]);
    }

    #[test]
    fn test_tanh_layer_in_network() {
        // Test TanhLayer as part of a simple network with random weights
        let mut linear = Linear::new(2, 3, true);
        let mut tanh = TanhLayer::new();
        let mut output_linear = Linear::new(3, 1, true);
        
        // Forward pass through network with random weights
        let input = Tensor::from_vec(vec![1.0, -1.0], vec![1, 2]);
        
        let hidden = linear.forward(input);
        let activated = tanh.forward(hidden);
        let output = output_linear.forward(activated);
        
        // Output should be finite and in reasonable range
        let result = output.item();
        assert!(result[[0, 0]].is_finite());
        
        // Since tanh bounds outputs to [-1, 1], and we have reasonable network sizes,
        // the final output should be bounded
        assert!(result[[0, 0]].abs() < 100.0); // Should be reasonable magnitude
    }

    #[test] 
    fn test_tanh_layer_multiple_calls() {
        let mut layer = TanhLayer::new();
        
        // Test that multiple calls with same input give same output
        let input = Tensor::from_vec(vec![0.5, -0.5, 1.0], vec![3]);
        
        let output1 = layer.forward(input.clone());
        let output2 = layer.forward(input);
        
        let result1 = output1.item();
        let result2 = output2.item();
        
        for i in 0..3 {
            assert!(approx_equal(result1[[i]], result2[[i]], 1e-10), 
                   "Multiple calls should give same result");
        }
    }
}