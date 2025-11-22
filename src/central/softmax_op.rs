use crate::central::*;
use ndarray::prelude::*;

/// Internal function for calculation the softmax of a array along a particular axis
/// Breaking it out into its own function, as both the forward and backwards step use it
/// Arugments
/// 'source' : the matrix that is being looked at
/// 'axis' : which axis I am working with
fn softmax_internal(souce: &mut ArrayD<f32>, axis: usize) -> ArrayD<f32> {
    // Get the max
    let max_values = souce.map_axis(Axis(axis), |row| {
        *row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    });
    let shape = souce.shape();
    let number_of_indices = shape.len();

    // Get the shape back
    let mut new_shape = vec![];
    for i in 0..number_of_indices {
        if i != axis {
            new_shape.push(shape[i]);
        } else {
            new_shape.push(1);
        }
    }

    // the main math portion of softmax
    //First we shift everything down by the max value of the axis
    // a way to think of this is that now the highest value will be 0
    // and everything else will be some negative number
    let result = max_values.into_shape(new_shape.clone()).unwrap();
    let result = result.broadcast(shape).unwrap();
    let mut shifted = souce.to_owned() - result;

    // Which when we take the exp of it then will mean that the largest number is 1
    // and everything else will be a fraction between 0 and 1
    shifted.map_inplace(|x| *x = x.exp());

    // Invert this so can turn a div to mul to avoid some computer nonsense
    let mut summed = shifted.sum_axis(Axis(axis));
    summed.map_inplace(|x| *x = x.powf(-1.0));

    // Broadcast the inverted sum back to original shape
    let inverted_sum_reshaped = summed.into_shape(new_shape).unwrap();
    let inverted_sum_broadcast = inverted_sum_reshaped.broadcast(shape).unwrap();

    // Multiply shifted (exp values) by broadcasted inverted sum
    // this will turn them all into the percetnage of the total that each head represents
    let result = shifted * inverted_sum_broadcast;
    return result;
}

impl Tensor {
    /// Computes the softmax function along specified axis
    /// # Arguments
    /// * `axis` - Axis along which to compute the softmax
    pub fn softmax(&self, axis: usize) -> Tensor {
        // Convert to Tensor and return
        return Tensor::create_tensor_data_and_shape_and_operation(
            self.shape,
            softmax_internal(&mut self.item(), axis)
                .to_owned()
                .into_raw_vec(),
            Operation::Softmax(self.id, axis),
        );
    }
}

/// Handles calculating and passing back the gradient of a softmax operation
/// The gradient computation involves the Jacobian of softmax
/// For softmax S_i = exp(x_i) / Σ exp(x_j), the gradient is:
/// ∂S_i/∂x_j = S_i * (δ_ij - S_j) where δ_ij is Kronecker delta
pub fn backward_for_softmax(backprop_packet: BackproagationPacket) {
    if let Operation::Softmax(source_id, axis) = backprop_packet.operation {
        // 1. Recompute softmax output (or retrieve if stored)
        let source_shape = backprop_packet.equation.get_tensor_shape(source_id);
        let softmax_output =
            softmax_internal(&mut backprop_packet.equation.get_item(source_id), axis);

        // 2. Get incoming gradients
        let incoming_grad = backprop_packet
            .equation
            .get_grad(backprop_packet.incoming_grad);

        // 3. Element-wise multiply and sum along axis
        let dot_product = incoming_grad.clone() * softmax_output.clone();
        let summed_dot = dot_product.sum_axis(Axis(axis));

        // 4. Broadcast back (same pattern as forward pass)
        let mut new_shape = source_shape.dimensions();
        new_shape[axis] = 1;
        let reshaped_dot = summed_dot.into_shape(new_shape).unwrap();
        let broadcasted_dot = reshaped_dot.broadcast(source_shape.dimensions()).unwrap();

        // 5. Apply gradient formula
        let gradients = softmax_output * (incoming_grad - broadcasted_dot);

        // 6. Add to gradients
        backprop_packet
            .equation
            .add_tensor_grad(source_id, gradients.into_raw_vec());
    } else {
        panic!("Wrong operation for backward softmax");
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Operation, Shape, Tensor};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    fn assert_softmax_properties(tensor: &Tensor, _axis: usize) {
        let data = tensor.item();

        // Check that all values are non-negative
        for val in data.iter() {
            assert!(*val >= 0.0, "Softmax values must be non-negative");
        }

        // For testing sum=1 along axis, we need to handle different shapes
        // For now, just check values are in valid range [0,1]
        for val in data.iter() {
            assert!(*val <= 1.0, "Softmax values must be <= 1.0");
        }
    }

    // ========== FORWARD PASS TESTS ==========

    #[test]
    fn softmax_1d_test() {
        // Test softmax of 1D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4]),
            vec![1.0, 2.0, 3.0, 4.0],
            Operation::Nop,
        );

        let result = input.softmax(0);

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![4]);

        // Check softmax properties
        assert_softmax_properties(&result, 0);

        let result_data = result.item();
        // Manual calculation: exp([1,2,3,4]) / sum(exp([1,2,3,4]))
        // exp values: [2.718, 7.389, 20.086, 54.598]
        // sum ≈ 84.791, so softmax ≈ [0.032, 0.087, 0.237, 0.644]
        assert!(approx_equal(result_data[[0]], 0.032058, 1e-5));
        assert!(approx_equal(result_data[[1]], 0.087144, 1e-5));
        assert!(approx_equal(result_data[[2]], 0.236883, 1e-5));
        assert!(approx_equal(result_data[[3]], 0.643914, 1e-5));

        // Check sum equals 1
        let sum: f32 = result_data.iter().sum();
        assert!(approx_equal(sum, 1.0, 1e-6));
    }

    #[test]
    fn softmax_2d_axis0_test() {
        // Test softmax along axis 0 (rows)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.softmax(0);

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![2, 3]);

        assert_softmax_properties(&result, 0);

        let result_data = result.item();
        // For each column, softmax over rows
        // Column 0: softmax([1,4]) = [exp(1)/(exp(1)+exp(4)), exp(4)/(exp(1)+exp(4))]
        assert!(approx_equal(result_data[[0, 0]], 0.047426, 1e-5)); // exp(1)/(exp(1)+exp(4))
        assert!(approx_equal(result_data[[1, 0]], 0.952574, 1e-5)); // exp(4)/(exp(1)+exp(4))

        // Each column should sum to 1
        for col in 0..3 {
            let col_sum = result_data[[0, col]] + result_data[[1, col]];
            assert!(approx_equal(col_sum, 1.0, 1e-6));
        }
    }

    #[test]
    fn softmax_2d_axis1_test() {
        // Test softmax along axis 1 (columns)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.softmax(1);

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![2, 3]);

        assert_softmax_properties(&result, 1);

        let result_data = result.item();
        // For each row, softmax over columns
        // Row 0: softmax([1,2,3]) ≈ [0.090, 0.245, 0.665]
        assert!(approx_equal(result_data[[0, 0]], 0.090031, 1e-5));
        assert!(approx_equal(result_data[[0, 1]], 0.244728, 1e-5));
        assert!(approx_equal(result_data[[0, 2]], 0.665241, 1e-5));

        // Each row should sum to 1
        for row in 0..2 {
            let row_sum = result_data[[row, 0]] + result_data[[row, 1]] + result_data[[row, 2]];
            assert!(approx_equal(row_sum, 1.0, 1e-6));
        }
    }

    #[test]
    fn softmax_3d_test() {
        // Test softmax with 3D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.softmax(2);

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![2, 2, 2]);

        assert_softmax_properties(&result, 2);

        let result_data = result.item();
        // For each [i,j,:] slice, softmax over last dimension
        // [0,0,:]: softmax([1,2]) ≈ [0.269, 0.731]
        assert!(approx_equal(result_data[[0, 0, 0]], 0.268941, 1e-5));
        assert!(approx_equal(result_data[[0, 0, 1]], 0.731059, 1e-5));

        // Check each slice sums to 1
        for i in 0..2 {
            for j in 0..2 {
                let slice_sum = result_data[[i, j, 0]] + result_data[[i, j, 1]];
                assert!(approx_equal(slice_sum, 1.0, 1e-6));
            }
        }
    }

    #[test]
    fn softmax_4d_test() {
        // Test softmax with 4D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![1, 2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.softmax(3);

        // Check shape is preserved
        assert_eq!(result.shape.dimensions(), vec![1, 2, 2, 2]);

        assert_softmax_properties(&result, 3);

        let result_data = result.item();
        // For each [0,i,j,:] slice, softmax over last dimension
        for i in 0..2 {
            for j in 0..2 {
                let slice_sum = result_data[[0, i, j, 0]] + result_data[[0, i, j, 1]];
                assert!(approx_equal(slice_sum, 1.0, 1e-6));
            }
        }
    }

    #[test]
    fn softmax_numerical_stability_test() {
        // Test with large values to check numerical stability
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![1000.0, 1001.0, 1002.0],
            Operation::Nop,
        );

        let result = input.softmax(0);

        assert_softmax_properties(&result, 0);

        let result_data = result.item();
        // Even with large values, should still get valid probabilities
        let sum: f32 = result_data.iter().sum();
        assert!(approx_equal(sum, 1.0, 1e-6));

        // Largest input should have highest probability
        assert!(result_data[[2]] > result_data[[1]]);
        assert!(result_data[[1]] > result_data[[0]]);
    }

    #[test]
    fn softmax_zero_input_test() {
        // Test with zero inputs
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![0.0, 0.0, 0.0],
            Operation::Nop,
        );

        let result = input.softmax(0);

        let result_data = result.item();
        // All zeros should give uniform distribution
        assert!(approx_equal(result_data[[0]], 1.0 / 3.0, 1e-6));
        assert!(approx_equal(result_data[[1]], 1.0 / 3.0, 1e-6));
        assert!(approx_equal(result_data[[2]], 1.0 / 3.0, 1e-6));
    }
    /*
        #[test]
        #[should_panic(expected = "axis 2 is out of bounds")]
        fn softmax_out_of_bounds_axis_test() {
            let input = Tensor::create_tensor_data_and_shape_and_operation(
                Shape::new(vec![2, 3]),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                Operation::Nop,
            );

            input.softmax(2); // Should panic - axis 2 doesn't exist for 2D tensor
        }
    */
    // ========== BACKWARD PASS TESTS ==========

    #[test]
    fn softmax_1d_backward_test() {
        // Test backward pass for 1D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![1.0, 2.0, 3.0],
            Operation::Nop,
        );

        let result = input.softmax(0);
        let loss = result.sum(vec![0], true);
        loss.backward();

        let gradients = input.grad();

        // For softmax, the gradient should satisfy: sum of gradients = 0
        let grad_sum: f32 = gradients.iter().sum();
        assert!(approx_equal(grad_sum, 0.0, 1e-6));

        // Each gradient should be reasonable (not NaN or infinity)
        for &grad in gradients.iter() {
            assert!(grad.is_finite());
        }
    }

    #[test]
    fn softmax_2d_axis0_backward_test() {
        // Test backward pass for 2D tensor along axis 0
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.softmax(0);
        let loss = result.sum(vec![0, 1], true);
        loss.backward();

        let gradients = input.grad();

        // Gradients should be finite
        for &grad in gradients.iter() {
            assert!(grad.is_finite());
        }

        // For each column, gradients should sum to 0
        for col in 0..3 {
            let col_grad_sum = gradients[[0, col]] + gradients[[1, col]];
            assert!(approx_equal(col_grad_sum, 0.0, 1e-6));
        }
    }

    #[test]
    fn softmax_2d_axis1_backward_test() {
        // Test backward pass for 2D tensor along axis 1
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.softmax(1);
        let loss = result.sum(vec![0, 1], true);
        loss.backward();

        let gradients = input.grad();

        // Gradients should be finite
        for &grad in gradients.iter() {
            assert!(grad.is_finite());
        }

        // For each row, gradients should sum to 0
        for row in 0..2 {
            let row_grad_sum: f32 = (0..3).map(|col| gradients[[row, col]]).sum();
            assert!(approx_equal(row_grad_sum, 0.0, 1e-6));
        }
    }

    #[test]
    fn softmax_3d_backward_test() {
        // Test backward pass for 3D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.softmax(2);
        let loss = result.sum(vec![0, 1, 2], true);
        loss.backward();

        let gradients = input.grad();

        // Gradients should be finite
        for &grad in gradients.iter() {
            assert!(grad.is_finite());
        }

        // For each [i,j,:] slice, gradients should sum to 0
        for i in 0..2 {
            for j in 0..2 {
                let slice_grad_sum = gradients[[i, j, 0]] + gradients[[i, j, 1]];
                assert!(approx_equal(slice_grad_sum, 0.0, 1e-6));
            }
        }
    }

    #[test]
    fn softmax_4d_backward_test() {
        // Test backward pass for 4D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![1, 2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.softmax(3);
        let loss = result.sum(vec![0, 1, 2, 3], true);
        loss.backward();

        let gradients = input.grad();

        // Gradients should be finite
        for &grad in gradients.iter() {
            assert!(grad.is_finite());
        }

        // For each [0,i,j,:] slice, gradients should sum to 0
        for i in 0..2 {
            for j in 0..2 {
                let slice_grad_sum = gradients[[0, i, j, 0]] + gradients[[0, i, j, 1]];
                assert!(approx_equal(slice_grad_sum, 0.0, 1e-6));
            }
        }
    }

    #[test]
    fn softmax_chained_operations_backward_test() {
        // Test softmax in chain with other operations
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![1.0, 2.0, 3.0],
            Operation::Nop,
        );

        let scale = Tensor::element(Shape::new(vec![3]), 2.0);

        // Chain: input.softmax() * scale
        let softmax_result = input.softmax(0);
        let scaled = softmax_result * scale;
        let loss = scaled.sum(vec![0], true);
        loss.backward();

        let input_gradients = input.grad();
        let scale_gradients = scale.grad();

        // Input gradients should sum to 0 (softmax property)
        let grad_sum: f32 = input_gradients.iter().sum();
        assert!(approx_equal(grad_sum, 0.0, 1e-6));

        // Scale gradients should equal softmax output values
        let softmax_data = softmax_result.item();
        for i in 0..3 {
            assert!(approx_equal(scale_gradients[[i]], softmax_data[[i]], 1e-6));
        }
    }

    #[test]
    fn softmax_numerical_gradient_test() {
        // Test backward pass using numerical differentiation
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![1.0, 2.0, 3.0],
            Operation::Nop,
        );

        // Compute analytical gradient
        let result = input.softmax(0);
        let loss = result.sum(vec![0], true);
        loss.backward();
        let analytical_grad = input.grad();

        // Compute numerical gradient
        let h = 1e-4;
        let mut numerical_grad = vec![0.0; 3];

        for i in 0..3 {
            // Forward difference
            let mut input_plus = vec![1.0, 2.0, 3.0];
            input_plus[i] += h;
            let input_plus_tensor = Tensor::create_tensor_data_and_shape_and_operation(
                Shape::new(vec![3]),
                input_plus,
                Operation::Nop,
            );
            let result_plus = input_plus_tensor.softmax(0).sum(vec![0], true);

            let mut input_minus = vec![1.0, 2.0, 3.0];
            input_minus[i] -= h;
            let input_minus_tensor = Tensor::create_tensor_data_and_shape_and_operation(
                Shape::new(vec![3]),
                input_minus,
                Operation::Nop,
            );
            let result_minus = input_minus_tensor.softmax(0).sum(vec![0], true);

            numerical_grad[i] = (result_plus.item()[[0]] - result_minus.item()[[0]]) / (2.0 * h);
        }

        // Compare analytical and numerical gradients
        for i in 0..3 {
            assert!(approx_equal(analytical_grad[[i]], numerical_grad[i], 1e-2));
        }
    }
}
