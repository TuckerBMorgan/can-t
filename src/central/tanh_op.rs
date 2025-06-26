use super::{BackproagationPacket, Operation, Tensor};

impl Tensor {
    /// Applies the hyperbolic tangent function element-wise to the tensor
    /// Returns a new tensor with the same shape where each element x is replaced with tanh(x)
    pub fn tanh(&self) -> Tensor {
        panic!("Implement tanh");
    }
}

/// Handles calculating and passing back the gradient of a tanh operation
/// The derivative of tanh(x) is 1 - tanh²(x)
pub fn backward_for_tanh(backprop_packet: BackproagationPacket) {
    if let Operation::Tanh(source_id) = backprop_packet.operation {
        panic!("Implement tanh backwards");
    } else {
        panic!("Wrong operation for backward tanh");
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Shape, Tensor, Operation};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    fn tanh_forward_test() {
        // Test with known values
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![0.0, 1.0, -1.0],
            Operation::Nop,
        );

        let result = input.tanh();

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![3]);

        // Check values
        let result_data = result.item();
        assert!(approx_equal(result_data[[0]], 0.0_f32.tanh(), 1e-6)); // tanh(0) ≈ 0
        assert!(approx_equal(result_data[[1]], 1.0_f32.tanh(), 1e-6)); // tanh(1) ≈ 0.7616
        assert!(approx_equal(result_data[[2]], (-1.0_f32).tanh(), 1e-6)); // tanh(-1) ≈ -0.7616
    }

    #[test]
    fn tanh_2d_test() {
        // Test with 2D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2]),
            vec![0.0, 2.0, -2.0, 0.5],
            Operation::Nop,
        );

        let result = input.tanh();

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![2, 2]);

        // Check some values
        let result_data = result.item();
        assert!(approx_equal(result_data[[0, 0]], 0.0_f32.tanh(), 1e-6));
        assert!(approx_equal(result_data[[0, 1]], 2.0_f32.tanh(), 1e-6));
        assert!(approx_equal(result_data[[1, 0]], (-2.0_f32).tanh(), 1e-6));
        assert!(approx_equal(result_data[[1, 1]], 0.5_f32.tanh(), 1e-6));
    }

    #[test]
    fn tanh_3d_test() {
        // Test with 3D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 1, 2]),
            vec![1.0, -1.0, 0.0, 3.0],
            Operation::Nop,
        );

        let result = input.tanh();

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![2, 1, 2]);

        // Check that all values are in [-1, 1] range
        let result_data = result.item();
        for val in result_data.iter() {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn tanh_4d_test() {
        // Test with 4D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![1, 2, 1, 2]),
            vec![0.0, 1.0, -0.5, 2.0],
            Operation::Nop,
        );

        let result = input.tanh();

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![1, 2, 1, 2]);

        // Check that tanh is applied correctly
        let result_data = result.item();
        let input_data = input.item();
        
        for i in 0..2 {
            for j in 0..2 {
                let expected = input_data[[0, i, 0, j]].tanh();
                assert!(approx_equal(result_data[[0, i, 0, j]], expected, 1e-6));
            }
        }
    }

    #[test]
    fn tanh_extreme_values_test() {
        // Test with extreme values to check numerical stability
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4]),
            vec![100.0, -100.0, 0.0001, -0.0001],
            Operation::Nop,
        );

        let result = input.tanh();
        let result_data = result.item();

        // tanh(100) should be very close to 1
        assert!(approx_equal(result_data[[0]], 1.0, 1e-6));
        
        // tanh(-100) should be very close to -1
        assert!(approx_equal(result_data[[1]], -1.0, 1e-6));
        
        // tanh(small values) should be approximately equal to the input
        assert!(approx_equal(result_data[[2]], 0.0001, 1e-4));
        assert!(approx_equal(result_data[[3]], -0.0001, 1e-4));
    }

    #[test]
    fn tanh_gradient_test() {
        // Test gradient computation with a simple case
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![1]),
            vec![0.5],
            Operation::Nop,
        );

        let output = input.tanh();
        
        // Manually trigger backward pass would require more setup
        // For now, just verify the forward pass works correctly
        let result_data = output.item();
        assert!(approx_equal(result_data[[0]], 0.5_f32.tanh(), 1e-6));
    }
}