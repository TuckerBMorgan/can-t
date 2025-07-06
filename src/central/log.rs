use crate::central::*;

impl Tensor {
    /// Returns the natural log(ln, not log10) of each element
    /// it uses natural log because that is what pytorch does when you call .log on a tensor
    pub fn log(&self) -> Tensor {
        let data =  get_equation()
            .get_item(self.id)
            .clone()
            .map(|x| x.ln());

        let result = Tensor::create_tensor_data_and_shape_and_operation(self.shape, data.into_raw_vec(), Operation::Log(self.id));
        return result;
    }
}

pub fn backwards_for_log(backprop_packet: BackproagationPacket) {
    if let Operation::Log(source_id) = backprop_packet.operation {
        
        // The derivative of log is nice and simple, just 1 / x
        let mut input = backprop_packet.equation.get_item(source_id);
        input.map_inplace(|x|*x = 1.0 / (*x));

        // And then mul the incoming grad
        let incoming_grad = backprop_packet.equation.get_grad(backprop_packet.incoming_grad);
        let contribution = incoming_grad * input;
        backprop_packet.equation.add_tensor_grad(source_id, contribution.to_owned().into_raw_vec());
    }
    else {
        panic!("Wrong operation called for backwards_for_log");
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Shape, Tensor, Operation};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    // ========== FORWARD PASS TESTS ==========

    #[test]
    fn log_simple_test() {
        // Very simple test to check basic log functionality
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2]),
            vec![1.0, 2.0],
            Operation::Nop,
        );

        let result = input.log();

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![2]);
        
        let result_data = result.item();
        // Manual calculation: ln([1,2])
        assert!(approx_equal(result_data[[0]], 0.0, 1e-6)); // ln(1) = 0
        assert!(approx_equal(result_data[[1]], 0.693147, 1e-5)); // ln(2) ≈ 0.693
    }

    #[test]
    fn log_2d_test() {
        // Test log of 2D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.log();

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![2, 3]);
        
        let result_data = result.item();
        // Check each element
        assert!(approx_equal(result_data[[0, 0]], 1.0_f32.ln(), 1e-6));
        assert!(approx_equal(result_data[[0, 1]], 2.0_f32.ln(), 1e-6));
        assert!(approx_equal(result_data[[0, 2]], 3.0_f32.ln(), 1e-6));
        assert!(approx_equal(result_data[[1, 0]], 4.0_f32.ln(), 1e-6));
        assert!(approx_equal(result_data[[1, 1]], 5.0_f32.ln(), 1e-6));
        assert!(approx_equal(result_data[[1, 2]], 6.0_f32.ln(), 1e-6));
    }

    #[test]
    fn log_3d_test() {
        // Test log with 3D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.log();

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![2, 2, 2]);
        
        let result_data = result.item();
        // Check some elements
        assert!(approx_equal(result_data[[0, 0, 0]], 1.0_f32.ln(), 1e-6));
        assert!(approx_equal(result_data[[0, 0, 1]], 2.0_f32.ln(), 1e-6));
        assert!(approx_equal(result_data[[1, 1, 1]], 8.0_f32.ln(), 1e-6));
    }

    #[test]
    fn log_4d_test() {
        // Test log with 4D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![1, 2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.log();

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![1, 2, 2, 2]);
        
        let result_data = result.item();
        // Check some elements
        assert!(approx_equal(result_data[[0, 0, 0, 0]], 1.0_f32.ln(), 1e-6));
        assert!(approx_equal(result_data[[0, 1, 1, 1]], 8.0_f32.ln(), 1e-6));
    }

    #[test]
    fn log_special_values_test() {
        // Test log with special mathematical values
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4]),
            vec![1.0, std::f32::consts::E, 0.5, 10.0],
            Operation::Nop,
        );

        let result = input.log();
        let result_data = result.item();
        
        // ln(1) = 0
        assert!(approx_equal(result_data[[0]], 0.0, 1e-6));
        // ln(e) = 1
        assert!(approx_equal(result_data[[1]], 1.0, 1e-6));
        // ln(0.5) = -ln(2)
        assert!(approx_equal(result_data[[2]], -2.0_f32.ln(), 1e-6));
        // ln(10)
        assert!(approx_equal(result_data[[3]], 10.0_f32.ln(), 1e-6));
    }

    #[test]
    fn log_small_values_test() {
        // Test log with small positive values
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![0.1, 0.01, 0.001],
            Operation::Nop,
        );

        let result = input.log();
        let result_data = result.item();
        
        // All should be negative (log of values < 1)
        assert!(result_data[[0]] < 0.0);
        assert!(result_data[[1]] < 0.0);
        assert!(result_data[[2]] < 0.0);
        
        // Check specific values
        assert!(approx_equal(result_data[[0]], 0.1_f32.ln(), 1e-6));
        assert!(approx_equal(result_data[[1]], 0.01_f32.ln(), 1e-6));
        assert!(approx_equal(result_data[[2]], 0.001_f32.ln(), 1e-6));
    }

    #[test]
    fn log_large_values_test() {
        // Test log with large values
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![100.0, 1000.0, 10000.0],
            Operation::Nop,
        );

        let result = input.log();
        let result_data = result.item();
        
        // All should be positive (log of values > 1)
        assert!(result_data[[0]] > 0.0);
        assert!(result_data[[1]] > 0.0);
        assert!(result_data[[2]] > 0.0);
        
        // Check specific values
        assert!(approx_equal(result_data[[0]], 100.0_f32.ln(), 1e-5));
        assert!(approx_equal(result_data[[1]], 1000.0_f32.ln(), 1e-5));
        assert!(approx_equal(result_data[[2]], 10000.0_f32.ln(), 1e-5));
    }

    // ========== BACKWARD PASS TESTS ==========

    #[test]
    fn log_1d_backward_test() {
        // Test backward pass for 1D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![1.0, 2.0, 3.0],
            Operation::Nop,
        );

        let result = input.log();
        let loss = result.sum(vec![0], true);
        loss.backward();

        let gradients = input.grad();
        
        // Gradient of ln(x) is 1/x
        assert!(approx_equal(gradients[[0]], 1.0 / 1.0, 1e-6)); // 1/1 = 1
        assert!(approx_equal(gradients[[1]], 1.0 / 2.0, 1e-6)); // 1/2 = 0.5
        assert!(approx_equal(gradients[[2]], 1.0 / 3.0, 1e-6)); // 1/3 ≈ 0.333
        
        // All gradients should be finite
        for &grad in gradients.iter() {
            assert!(grad.is_finite());
        }
    }

    #[test]
    fn log_2d_backward_test() {
        // Test backward pass for 2D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2]),
            vec![1.0, 2.0, 4.0, 8.0],
            Operation::Nop,
        );

        let result = input.log();
        let loss = result.sum(vec![0, 1], true);
        loss.backward();

        let gradients = input.grad();
        
        // Check each gradient = 1/input_value
        assert!(approx_equal(gradients[[0, 0]], 1.0 / 1.0, 1e-6));
        assert!(approx_equal(gradients[[0, 1]], 1.0 / 2.0, 1e-6));
        assert!(approx_equal(gradients[[1, 0]], 1.0 / 4.0, 1e-6));
        assert!(approx_equal(gradients[[1, 1]], 1.0 / 8.0, 1e-6));
        
        // All gradients should be finite
        for &grad in gradients.iter() {
            assert!(grad.is_finite());
        }
    }

    #[test]
    fn log_3d_backward_test() {
        // Test backward pass for 3D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.log();
        let loss = result.sum(vec![0, 1, 2], true);
        loss.backward();

        let gradients = input.grad();
        
        // Check some gradients
        assert!(approx_equal(gradients[[0, 0, 0]], 1.0 / 1.0, 1e-6));
        assert!(approx_equal(gradients[[0, 0, 1]], 1.0 / 2.0, 1e-6));
        assert!(approx_equal(gradients[[1, 1, 1]], 1.0 / 8.0, 1e-6));
        
        // All gradients should be finite
        for &grad in gradients.iter() {
            assert!(grad.is_finite());
        }
    }

    #[test]
    fn log_chained_operations_backward_test() {
        // Test log in chain with other operations
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![1.0, 2.0, 4.0],
            Operation::Nop,
        );
        
        let scale = Tensor::element(Shape::new(vec![3]), 2.0);
        
        // Chain: input.log() * scale
        let log_result = input.log();
        let scaled = log_result * scale;
        let loss = scaled.sum(vec![0], true);
        loss.backward();
        
        let input_gradients = input.grad();
        let scale_gradients = scale.grad();
        
        // Input gradients should be (1/x) * scale
        assert!(approx_equal(input_gradients[[0]], 2.0 / 1.0, 1e-6)); // 2/1 = 2
        assert!(approx_equal(input_gradients[[1]], 2.0 / 2.0, 1e-6)); // 2/2 = 1
        assert!(approx_equal(input_gradients[[2]], 2.0 / 4.0, 1e-6)); // 2/4 = 0.5
        
        // Scale gradients should equal log output values
        let log_data = log_result.item();
        for i in 0..3 {
            assert!(approx_equal(scale_gradients[[i]], log_data[[i]], 1e-6));
        }
    }

    #[test]
    fn log_small_input_backward_test() {
        // Test backward pass with small inputs (large gradients)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![0.1, 0.01, 0.001],
            Operation::Nop,
        );

        let result = input.log();
        let loss = result.sum(vec![0], true);
        loss.backward();

        let gradients = input.grad();
        
        // Gradients should be 1/x (large for small x)
        assert!(approx_equal(gradients[[0]], 1.0 / 0.1, 1e-5));   // 10
        assert!(approx_equal(gradients[[1]], 1.0 / 0.01, 1e-4));  // 100
        assert!(approx_equal(gradients[[2]], 1.0 / 0.001, 1e-3)); // 1000
        
        // All gradients should be finite and positive
        for &grad in gradients.iter() {
            assert!(grad.is_finite());
            assert!(grad > 0.0);
        }
    }

    #[test]
    fn log_large_input_backward_test() {
        // Test backward pass with large inputs (small gradients)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![10.0, 100.0, 1000.0],
            Operation::Nop,
        );

        let result = input.log();
        let loss = result.sum(vec![0], true);
        loss.backward();

        let gradients = input.grad();
        
        // Gradients should be 1/x (small for large x)
        assert!(approx_equal(gradients[[0]], 1.0 / 10.0, 1e-6));   // 0.1
        assert!(approx_equal(gradients[[1]], 1.0 / 100.0, 1e-6));  // 0.01
        assert!(approx_equal(gradients[[2]], 1.0 / 1000.0, 1e-6)); // 0.001
        
        // All gradients should be finite and positive
        for &grad in gradients.iter() {
            assert!(grad.is_finite());
            assert!(grad > 0.0);
        }
    }

    #[test]
    fn log_numerical_gradient_test() {
        // Test backward pass using numerical differentiation
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![1.5, 2.5, 3.5],
            Operation::Nop,
        );

        // Compute analytical gradient
        let result = input.log();
        let loss = result.sum(vec![0], true);
        loss.backward();
        let analytical_grad = input.grad();

        // Compute numerical gradient
        let h = 1e-4;
        let mut numerical_grad = vec![0.0; 3];
        
        for i in 0..3 {
            // Forward difference
            let mut input_plus = vec![1.5, 2.5, 3.5];
            input_plus[i] += h;
            let input_plus_tensor = Tensor::create_tensor_data_and_shape_and_operation(
                Shape::new(vec![3]),
                input_plus,
                Operation::Nop,
            );
            let result_plus = input_plus_tensor.log().sum(vec![0], true);
            
            let mut input_minus = vec![1.5, 2.5, 3.5];
            input_minus[i] -= h;
            let input_minus_tensor = Tensor::create_tensor_data_and_shape_and_operation(
                Shape::new(vec![3]),
                input_minus,
                Operation::Nop,
            );
            let result_minus = input_minus_tensor.log().sum(vec![0], true);
            
            numerical_grad[i] = (result_plus.item()[[0]] - result_minus.item()[[0]]) / (2.0 * h);
        }

        // Compare analytical and numerical gradients
        for i in 0..3 {
            assert!(approx_equal(analytical_grad[[i]], numerical_grad[i], 1e-3));
        }
    }

    #[test]
    fn log_mathematical_properties_test() {
        // Test mathematical properties of log
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![1.0, std::f32::consts::E, std::f32::consts::E.powi(2)],
            Operation::Nop,
        );

        let result = input.log();
        let result_data = result.item();
        
        // ln(1) = 0
        assert!(approx_equal(result_data[[0]], 0.0, 1e-6));
        // ln(e) = 1
        assert!(approx_equal(result_data[[1]], 1.0, 1e-6));
        // ln(e^2) = 2
        assert!(approx_equal(result_data[[2]], 2.0, 1e-6));
    }
}