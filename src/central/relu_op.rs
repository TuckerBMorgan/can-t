use crate::central::*;

impl Tensor {
    pub fn relu(&mut self) -> Tensor {
        let data = {
            let equation = get_equation();

            let data = equation.get_data_flat_buffer(self.id);
            let mapped : Vec<f32> = data.iter().map(|x|x.max(0.0)).collect();
            mapped
        };

        return Tensor::create_tensor_data_and_shape_and_operation(self.shape, data, Operation::RELU(self.id));
    }
}

pub fn backward_for_relu(backprop_packet: BackproagationPacket) {
    if let Operation::RELU(source)  = backprop_packet.operation{
        let data = backprop_packet.equation.get_data_flat_buffer(source);
        let mapped : Vec<f32> = data.iter().map(|x|if *x > 0.0{
            return 1.0
        }
        else {
            return 0.0;
        } ).collect();

        let grad = backprop_packet.equation.get_grad_flat_buffer(backprop_packet.incoming_grad);
        let result = backprop_packet.equation.mul_vector(&mapped, grad);
        backprop_packet.equation.add_tensor_grad(source, result);
    }
    else {
        panic!("Wrong operation called for backward_for_relu")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_relu_forward_single_value() {
        // Test with a single positive value
        let mut input = Tensor::from_vec(vec![2.5], vec![1]);
        let output = input.relu();
        
        assert_eq!(output.shape.dimensions(), vec![1]);
        let result = output.item();
        assert!(approx_equal(result[[0]], 2.5, 1e-6));
    }

    #[test]
    fn test_relu_forward_single_negative() {
        // Test with a single negative value
        let mut input = Tensor::from_vec(vec![-2.5], vec![1]);
        let output = input.relu();
        
        assert_eq!(output.shape.dimensions(), vec![1]);
        let result = output.item();
        assert!(approx_equal(result[[0]], 0.0, 1e-6));
    }

    #[test]
    fn test_relu_forward_zero() {
        // Test with zero
        let mut input = Tensor::from_vec(vec![0.0], vec![1]);
        let output = input.relu();
        
        assert_eq!(output.shape.dimensions(), vec![1]);
        let result = output.item();
        assert!(approx_equal(result[[0]], 0.0, 1e-6));
    }

    #[test]
    fn test_relu_forward_mixed_values() {
        // Test with mix of positive, negative, and zero values
        let test_values = vec![-3.0, -1.5, 0.0, 1.5, 3.0];
        let expected_values = vec![0.0, 0.0, 0.0, 1.5, 3.0];
        
        let mut input = Tensor::from_vec(test_values, vec![5]);
        let output = input.relu();
        
        assert_eq!(output.shape.dimensions(), vec![5]);
        let result = output.item();
        
        for i in 0..5 {
            assert!(approx_equal(result[[i]], expected_values[i], 1e-6),
                   "ReLU({}) = {}, expected {}", 
                   if i < 5 { vec![-3.0, -1.5, 0.0, 1.5, 3.0][i] } else { 0.0 }, 
                   result[[i]], expected_values[i]);
        }
    }

    #[test]
    fn test_relu_forward_vector() {
        // Test with a vector of values
        let mut input = Tensor::from_vec(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0], 
            vec![5]
        );
        let output = input.relu();
        
        assert_eq!(output.shape.dimensions(), vec![5]);
        let result = output.item();
        
        // Check each value
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        for i in 0..5 {
            assert!(approx_equal(result[[i]], expected[i], 1e-6));
        }
    }

    #[test]
    fn test_relu_forward_matrix() {
        // Test with a 2x3 matrix
        let mut input = Tensor::from_vec(
            vec![-1.0, 0.0, 1.0, -0.5, 0.5, 2.0], 
            vec![2, 3]
        );
        let output = input.relu();
        
        assert_eq!(output.shape.dimensions(), vec![2, 3]);
        let result = output.item();
        
        // Check specific values
        assert!(approx_equal(result[[0, 0]], 0.0, 1e-6));  // ReLU(-1.0) = 0.0
        assert!(approx_equal(result[[0, 1]], 0.0, 1e-6));  // ReLU(0.0) = 0.0
        assert!(approx_equal(result[[0, 2]], 1.0, 1e-6));  // ReLU(1.0) = 1.0
        assert!(approx_equal(result[[1, 0]], 0.0, 1e-6));  // ReLU(-0.5) = 0.0
        assert!(approx_equal(result[[1, 1]], 0.5, 1e-6));  // ReLU(0.5) = 0.5
        assert!(approx_equal(result[[1, 2]], 2.0, 1e-6));  // ReLU(2.0) = 2.0
    }

    #[test]
    fn test_relu_forward_batch() {
        // Test with batch input [batch_size=2, features=3]
        let mut input = Tensor::from_vec(
            vec![
                1.0, -1.0, 0.0,  // Sample 1
                0.5, -0.5, 2.0,  // Sample 2
            ],
            vec![2, 3]
        );
        let output = input.relu();
        
        assert_eq!(output.shape.dimensions(), vec![2, 3]);
        let result = output.item();
        
        // Check batch 1
        assert!(approx_equal(result[[0, 0]], 1.0, 1e-6));
        assert!(approx_equal(result[[0, 1]], 0.0, 1e-6));
        assert!(approx_equal(result[[0, 2]], 0.0, 1e-6));
        
        // Check batch 2
        assert!(approx_equal(result[[1, 0]], 0.5, 1e-6));
        assert!(approx_equal(result[[1, 1]], 0.0, 1e-6));
        assert!(approx_equal(result[[1, 2]], 2.0, 1e-6));
    }

    #[test]
    fn test_relu_forward_large_values() {
        // Test that ReLU handles large values correctly
        let large_values = vec![-1000.0, -100.0, -10.0, 10.0, 100.0, 1000.0];
        let expected = vec![0.0, 0.0, 0.0, 10.0, 100.0, 1000.0];
        
        let mut input = Tensor::from_vec(large_values.clone(), vec![6]);
        let output = input.relu();
        
        let result = output.item();
        for i in 0..6 {
            assert!(approx_equal(result[[i]], expected[i], 1e-6),
                   "ReLU({}) = {}, expected {}", large_values[i], result[[i]], expected[i]);
        }
    }

    #[test]
    fn test_relu_forward_properties() {
        // Test key ReLU properties
        let mut input = Tensor::from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0], vec![5]);
        let output = input.relu();
        
        let result = output.item();
        
        // Property 1: All outputs should be non-negative
        for i in 0..5 {
            assert!(result[[i]] >= 0.0, "ReLU output should be non-negative, got {}", result[[i]]);
        }
        
        // Property 2: For positive inputs, output equals input
        assert!(approx_equal(result[[3]], 1.0, 1e-6)); // ReLU(1.0) = 1.0
        assert!(approx_equal(result[[4]], 5.0, 1e-6)); // ReLU(5.0) = 5.0
        
        // Property 3: For negative inputs, output is zero
        assert!(approx_equal(result[[0]], 0.0, 1e-6)); // ReLU(-5.0) = 0.0
        assert!(approx_equal(result[[1]], 0.0, 1e-6)); // ReLU(-1.0) = 0.0
        
        // Property 4: ReLU(0) = 0
        assert!(approx_equal(result[[2]], 0.0, 1e-6)); // ReLU(0.0) = 0.0
    }

    #[test]
    fn test_relu_forward_shape_preservation() {
        // Test that ReLU preserves input shape
        let shapes = vec![
            vec![1],
            vec![5],
            vec![2, 3],
            vec![2, 3, 4],
            vec![1, 1, 1, 1],
        ];
        
        for shape in shapes {
            let size: usize = shape.iter().product();
            let data: Vec<f32> = (0..size).map(|i| i as f32 - (size/2) as f32).collect();
            
            let mut input = Tensor::from_vec(data, shape.clone());
            let output = input.relu();
            
            assert_eq!(output.shape.dimensions(), shape,
                      "ReLU should preserve input shape {:?}", shape);
        }
    }

    #[test]
    fn test_relu_forward_multiple_calls() {
        // Test that multiple calls with same input give same output
        let mut input = Tensor::from_vec(vec![0.5, -0.5, 1.0], vec![3]);
        
        let output1 = input.relu();
        let output2 = input.relu();
        
        let result1 = output1.item();
        let result2 = output2.item();
        
        for i in 0..3 {
            assert!(approx_equal(result1[[i]], result2[[i]], 1e-10), 
                   "Multiple ReLU calls should give same result");
        }
    }

    #[test]
    fn test_relu_forward_edge_cases() {
        // Test edge cases around zero
        let edge_values = vec![-1e-10, -1e-6, 0.0, 1e-6, 1e-10];
        let mut input = Tensor::from_vec(edge_values, vec![5]);
        let output = input.relu();
        
        let result = output.item();
        
        // Very small negative values should become zero
        assert!(approx_equal(result[[0]], 0.0, 1e-12));
        assert!(approx_equal(result[[1]], 0.0, 1e-12));
        
        // Zero should stay zero
        assert!(approx_equal(result[[2]], 0.0, 1e-12));
        
        // Very small positive values should be preserved
        assert!(approx_equal(result[[3]], 1e-6, 1e-12));
        assert!(approx_equal(result[[4]], 1e-10, 1e-12));
    }

    // ========== BACKWARD PASS TESTS ==========

    #[test]
    fn test_relu_backward_single_positive() {
        // Test backward pass for single positive value
        let mut input = Tensor::from_vec(vec![2.0], vec![1]);
        input.set_requires_grad(true);
        
        let output = input.relu();
        let loss = output.sum(vec![0], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        // Gradient should be 1.0 for positive input
        assert!(approx_equal(grad[[0]], 1.0, 1e-6));
    }

    #[test]
    fn test_relu_backward_single_negative() {
        // Test backward pass for single negative value
        let mut input = Tensor::from_vec(vec![-2.0], vec![1]);
        input.set_requires_grad(true);
        
        let output = input.relu();
        let loss = output.sum(vec![0], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        // Gradient should be 0.0 for negative input
        assert!(approx_equal(grad[[0]], 0.0, 1e-6));
    }

    #[test]
    fn test_relu_backward_zero() {
        // Test backward pass for zero input
        let mut input = Tensor::from_vec(vec![0.0], vec![1]);
        input.set_requires_grad(true);
        
        let output = input.relu();
        let loss = output.sum(vec![0], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        // Gradient should be 0.0 for zero input (following PyTorch convention)
        assert!(approx_equal(grad[[0]], 0.0, 1e-6));
    }

    #[test]
    fn test_relu_backward_mixed_values() {
        // Test backward pass for mixed positive/negative values
        let mut input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        input.set_requires_grad(true);
        
        let output = input.relu();
        let loss = output.sum(vec![0], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        let expected_grads = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        
        for i in 0..5 {
            assert!(approx_equal(grad[[i]], expected_grads[i], 1e-6),
                   "Gradient for input {} should be {}, got {}", 
                   vec![-2.0, -1.0, 0.0, 1.0, 2.0][i], expected_grads[i], grad[[i]]);
        }
    }

    #[test]
    fn test_relu_backward_matrix() {
        // Test backward pass for 2x3 matrix
        let mut input = Tensor::from_vec(
            vec![-1.0, 0.0, 1.0, -0.5, 0.5, 2.0], 
            vec![2, 3]
        );
        input.set_requires_grad(true);
        
        let output = input.relu();
        let loss = output.sum(vec![0, 1], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        
        // Check gradients: 1.0 for positive inputs, 0.0 for negative/zero
        assert!(approx_equal(grad[[0, 0]], 0.0, 1e-6));  // -1.0 -> 0.0
        assert!(approx_equal(grad[[0, 1]], 0.0, 1e-6));  // 0.0 -> 0.0
        assert!(approx_equal(grad[[0, 2]], 1.0, 1e-6));  // 1.0 -> 1.0
        assert!(approx_equal(grad[[1, 0]], 0.0, 1e-6));  // -0.5 -> 0.0
        assert!(approx_equal(grad[[1, 1]], 1.0, 1e-6));  // 0.5 -> 1.0
        assert!(approx_equal(grad[[1, 2]], 1.0, 1e-6));  // 2.0 -> 1.0
    }

    #[test]
    fn test_relu_backward_batch() {
        // Test backward pass for batch processing
        let mut input = Tensor::from_vec(
            vec![
                1.0, -1.0, 0.0,  // Batch 1
                -0.5, 0.5, 2.0,  // Batch 2
            ],
            vec![2, 3]
        );
        input.set_requires_grad(true);
        
        let output = input.relu();
        let loss = output.sum(vec![0, 1], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        
        // Check batch 1 gradients
        assert!(approx_equal(grad[[0, 0]], 1.0, 1e-6));  // 1.0 -> 1.0
        assert!(approx_equal(grad[[0, 1]], 0.0, 1e-6));  // -1.0 -> 0.0
        assert!(approx_equal(grad[[0, 2]], 0.0, 1e-6));  // 0.0 -> 0.0
        
        // Check batch 2 gradients
        assert!(approx_equal(grad[[1, 0]], 0.0, 1e-6));  // -0.5 -> 0.0
        assert!(approx_equal(grad[[1, 1]], 1.0, 1e-6));  // 0.5 -> 1.0
        assert!(approx_equal(grad[[1, 2]], 1.0, 1e-6));  // 2.0 -> 1.0
    }

    #[test]
    fn test_relu_backward_chain_rule() {
        // Test ReLU in a computational chain to verify chain rule
        let mut input = Tensor::from_vec(vec![2.0, -1.0, 3.0], vec![3]);
        input.set_requires_grad(true);
        
        // Create a chain: input -> ReLU -> multiply by 2 -> sum
        let relu_output = input.relu();
        let scaled = relu_output * 2.0;
        let loss = scaled.sum(vec![0], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        
        // Expected gradients: d(loss)/d(input) = d(loss)/d(scaled) * d(scaled)/d(relu) * d(relu)/d(input)
        // For input[0] = 2.0: 1 * 2 * 1 = 2.0 (positive input)
        // For input[1] = -1.0: 1 * 2 * 0 = 0.0 (negative input) 
        // For input[2] = 3.0: 1 * 2 * 1 = 2.0 (positive input)
        assert!(approx_equal(grad[[0]], 2.0, 1e-6));
        assert!(approx_equal(grad[[1]], 0.0, 1e-6));
        assert!(approx_equal(grad[[2]], 2.0, 1e-6));
    }

    #[test]
    fn test_relu_backward_gradient_properties() {
        // Test key properties of ReLU gradients
        let mut input = Tensor::from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0], vec![5]);
        input.set_requires_grad(true);
        
        let output = input.relu();
        let loss = output.sum(vec![0], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        
        // Property 1: Gradients are either 0 or 1
        for i in 0..5 {
            let grad_val = grad[[i]];
            assert!(grad_val == 0.0 || grad_val == 1.0, 
                   "ReLU gradient should be 0 or 1, got {}", grad_val);
        }
        
        // Property 2: Gradient is 0 for negative inputs
        assert!(approx_equal(grad[[0]], 0.0, 1e-6)); // input = -5.0
        assert!(approx_equal(grad[[1]], 0.0, 1e-6)); // input = -1.0
        
        // Property 3: Gradient is 0 for zero input
        assert!(approx_equal(grad[[2]], 0.0, 1e-6)); // input = 0.0
        
        // Property 4: Gradient is 1 for positive inputs
        assert!(approx_equal(grad[[3]], 1.0, 1e-6)); // input = 1.0
        assert!(approx_equal(grad[[4]], 1.0, 1e-6)); // input = 5.0
    }

    #[test]
    fn test_relu_backward_large_values() {
        // Test backward pass with large magnitude values
        let mut input = Tensor::from_vec(vec![-1000.0, -100.0, 100.0, 1000.0], vec![4]);
        input.set_requires_grad(true);
        
        let output = input.relu();
        let loss = output.sum(vec![0], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        
        // Large negative values should have gradient 0
        assert!(approx_equal(grad[[0]], 0.0, 1e-6));
        assert!(approx_equal(grad[[1]], 0.0, 1e-6));
        
        // Large positive values should have gradient 1
        assert!(approx_equal(grad[[2]], 1.0, 1e-6));
        assert!(approx_equal(grad[[3]], 1.0, 1e-6));
    }

    #[test]
    fn test_relu_backward_different_upstream_gradients() {
        // Test ReLU backward with different upstream gradient values
        let mut input = Tensor::from_vec(vec![1.0, -1.0, 2.0], vec![3]);
        input.set_requires_grad(true);
        
        let output = input.relu();
        // Create upstream gradient by multiplying by different values
        let weighted = output * Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);
        let loss = weighted.sum(vec![0], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        
        // Expected: upstream_grad * relu_derivative
        // For input[0] = 1.0: 2.0 * 1.0 = 2.0
        // For input[1] = -1.0: 3.0 * 0.0 = 0.0
        // For input[2] = 2.0: 4.0 * 1.0 = 4.0
        assert!(approx_equal(grad[[0]], 2.0, 1e-6));
        assert!(approx_equal(grad[[1]], 0.0, 1e-6));
        assert!(approx_equal(grad[[2]], 4.0, 1e-6));
    }

    #[test]
    fn test_relu_backward_shape_preservation() {
        // Test that gradient shape matches input shape
        let shapes = vec![
            vec![1],
            vec![5],
            vec![2, 3],
            vec![2, 3, 4],
        ];
        
        for shape in shapes {
            let size: usize = shape.iter().product();
            let data: Vec<f32> = (0..size).map(|i| i as f32 - (size/2) as f32).collect();
            
            let mut input = Tensor::from_vec(data, shape.clone());
            input.set_requires_grad(true);
            
            let output = input.relu();
            let dims_to_sum: Vec<usize> = (0..shape.len()).collect();
            let loss = output.sum(dims_to_sum, true);
            
            zero_all_grads();
            loss.backward();
            
            let grad = input.grad();
            assert_eq!(grad.shape().to_vec(), shape,
                      "Gradient shape should match input shape {:?}", shape);
        }
    }

    #[test]
    fn test_relu_backward_edge_case_values() {
        // Test backward pass for edge case values around zero
        let mut input = Tensor::from_vec(vec![-1e-10, -1e-6, 0.0, 1e-6, 1e-10], vec![5]);
        input.set_requires_grad(true);
        
        let output = input.relu();
        let loss = output.sum(vec![0], true);
        
        zero_all_grads();
        loss.backward();
        
        let grad = input.grad();
        
        // Very small negative values should have gradient 0
        assert!(approx_equal(grad[[0]], 0.0, 1e-12));
        assert!(approx_equal(grad[[1]], 0.0, 1e-12));
        
        // Zero should have gradient 0
        assert!(approx_equal(grad[[2]], 0.0, 1e-12));
        
        // Very small positive values should have gradient 1
        assert!(approx_equal(grad[[3]], 1.0, 1e-12));
        assert!(approx_equal(grad[[4]], 1.0, 1e-12));
    }
}