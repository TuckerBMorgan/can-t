use crate::central::get_equation;

use super::{BackproagationPacket, Operation, Tensor};

impl Tensor {
    /// Applies the hyperbolic tangent function element-wise to the tensor
    /// Returns a new tensor with the same shape where each element x is replaced with tanh(x)
    pub fn tanh(&self) -> Tensor {
        let result: Vec<f32> = get_equation()
            .get_data_flat_buffer(self.id)
            .iter()
            .map(|x| x.tanh())
            .collect();
        let tensor = Tensor::create_tensor_data_and_shape_and_operation(
            self.shape,
            result,
            Operation::Tanh(self.id),
        );
        return tensor;
    }
}

/// Handles calculating and passing back the gradient of a tanh operation
/// The derivative of tanh(x) is 1 - tanh²(x)
pub fn backward_for_tanh(backprop_packet: BackproagationPacket) {
    if let Operation::Tanh(source_id) = backprop_packet.operation {
        let in_gradient = backprop_packet
            .equation
            .get_grad_flat_buffer(backprop_packet.incoming_grad);
        let source_data = backprop_packet.equation.get_data_flat_buffer(source_id);

        // TODO: update this to use the platform specific acceleration, if this is taking to long in the future
        let updated: Vec<f32> = in_gradient
            .iter()
            .zip(source_data.iter())
            .map(|(grad, &x)| grad * (1.0 - x.tanh().powf(2.0)))
            .collect();

        backprop_packet.equation.add_tensor_grad(source_id, updated);
    } else {
        panic!("Wrong operation for backward tanh");
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Operation, Shape, Tensor};

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
    fn tanh_simple_backward_test() {
        // Test basic backward pass with scalar
        let a = Tensor::element(Shape::new(vec![1]), 0.5);
        let result = a.tanh();
        result.backward();

        // tanh'(x) = 1 - tanh²(x)
        // tanh'(0.5) = 1 - tanh²(0.5)
        let expected_grad = 1.0 - (0.5_f32.tanh().powf(2.0));
        let actual_grad = a.grad()[0];

        assert!(approx_equal(actual_grad, expected_grad, 1e-6));
    }

    #[test]
    fn tanh_multiple_values_backward_test() {
        // Test backward pass with multiple values
        let input_data = vec![0.0, 1.0, -1.0, 2.0];
        let a = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4]),
            input_data.clone(),
            Operation::Nop,
        );

        let result = a.tanh();
        let loss = result.sum(vec![0], true);
        loss.backward();

        let gradients = a.grad();

        // Check each gradient: tanh'(x) = 1 - tanh²(x)
        for (i, &input_val) in input_data.iter().enumerate() {
            let input_val: f32 = input_val;
            let expected_grad = 1.0 - (input_val.tanh().powf(2.0));
            assert!(approx_equal(gradients[i], expected_grad, 1e-6));
        }
    }

    #[test]
    fn tanh_2d_backward_test() {
        // Test backward pass with 2D tensor - sum over individual elements
        let a = Tensor::element(Shape::new(vec![2, 2]), 0.5);
        let result = a.tanh();
        let loss = result.sum(vec![0, 1], true); // Keep dimensions as [1, 1]
        loss.backward();

        let gradients = a.grad();
        let expected_grad = 1.0 - (0.5_f32.tanh().powf(2.0));

        // All gradients should be the same since all inputs are 0.5
        for &actual_grad in gradients.iter() {
            assert!(approx_equal(actual_grad, expected_grad, 1e-6));
        }
    }

    #[test]
    fn tanh_chained_operations_backward_test() {
        // Test tanh in a chain of operations
        let a = Tensor::element(Shape::new(vec![1]), 2.0);
        let b = Tensor::element(Shape::new(vec![1]), 3.0);

        // Chain: (a + b).tanh()
        let sum = a + b; // sum = 5.0
        let result = sum.tanh(); // tanh(5.0)
        result.backward();

        // Both a and b should have same gradient since they're added
        // d/da tanh(a + b) = tanh'(a + b) * 1 = 1 - tanh²(5.0)
        let expected_grad = 1.0 - (5.0_f32.tanh().powf(2.0));

        assert!(approx_equal(a.grad()[0], expected_grad, 1e-6));
        assert!(approx_equal(b.grad()[0], expected_grad, 1e-6));
    }

    #[test]
    fn tanh_extreme_values_backward_test() {
        // Test gradient computation with extreme values
        let input_data = vec![10.0, -10.0, 0.0];
        let a = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            input_data.clone(),
            Operation::Nop,
        );

        let result = a.tanh();
        let loss = result.sum(vec![0], true);
        loss.backward();

        let gradients = a.grad();

        // For extreme values, tanh'(x) ≈ 0 because tanh(x) ≈ ±1
        // For x = 0, tanh'(0) = 1
        assert!(approx_equal(
            gradients[0],
            1.0 - (10.0_f32.tanh().powf(2.0)),
            1e-6
        )); // Very small
        assert!(approx_equal(
            gradients[1],
            1.0 - ((-10.0_f32).tanh().powf(2.0)),
            1e-6
        )); // Very small  
        assert!(approx_equal(gradients[2], 1.0, 1e-6)); // tanh'(0) = 1
    }
}
