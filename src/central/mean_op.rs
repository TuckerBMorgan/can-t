use std::collections::HashSet;
use crate::{central::*, utils::padding_dimenions_to_four};
use ndarray::prelude::*;

impl Tensor {
    /// Computes the mean along specified axes
    /// # Arguments
    /// * 'axes' - Vector of axis indices along which to compute the mean
    pub fn mean(&self, axes: Vec<usize>) -> Tensor {

        // make sure each dimension is unique 
        for (i, a) in axes.iter().enumerate() {
            for (ii, b) in axes.iter().enumerate() {
                if i != ii && a == b {
                    panic!("dimensions for mean must be unique");
                }
            }
        }

        // Do some basic setup
        let mut tensor = get_equation().get_item(self.id);
        let self_shape = self.shape;
        let mut total_count = 1;

        // Sum each dimension, while counting up what 
        for (index, this_axis) in axes.iter().enumerate() {
            total_count *= self_shape.dimensions()[*this_axis];
            tensor = tensor.sum_axis(Axis(*this_axis - index));
        }

        // Divide the result by count to turn it into a mean
        let result_data: Vec<f32> = tensor.iter().map(|&x| x / total_count as f32).collect();

        // Copy which axes we summed over, so we can use it for back prop later
        // we are using the -1 as a flag values to know what are unused valyes
        // since each operation is stored in an enum, it needs to have a known value
        let mut coppied_axes = [-1, -1, -1, -1];
        for (i, axes) in axes.iter().enumerate() {
            coppied_axes[i] = *axes as isize;
        }

        // With Ndarray if you have just a single element, the shape will be empty
        // which is not the same as cant
        let output_shape = if tensor.shape().is_empty() {
            vec![1]  // Convert scalar to 1D tensor with 1 element
        } else {
            tensor.shape().to_vec()
        };


        return Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(output_shape),
            result_data,
            Operation::Mean(self.id, coppied_axes, axes.len())
        );
    }
}

/// Handles calculating and passing back the gradient of a mean operation
/// The gradient is distributed equally among all elements that contributed to the mean
pub fn backward_for_mean(backprop_packet: BackproagationPacket) {
    if let Operation::Mean(source_id, axes_array, num_axes) = backprop_packet.operation {

        // Get the data we will be working with
        let get_item = backprop_packet.equation.get_grad(backprop_packet.incoming_grad);
        let original_shape = backprop_packet.equation.get_tensor_shape(source_id);

        // We need to remake the shape we are working with into one that we can broadcast
        // this means adding 1s to the dimensions we reduced during the mean operation
        // easist way to do this, is get the original shape and 1 out the shapes
        let mut reshape_shape = original_shape.dimensions();
        for index in 0..num_axes {
            reshape_shape[axes_array[index] as usize] = 1;
        }
        // then reshape it
        let get_item = get_item.into_shape(reshape_shape).unwrap();

        // Then broadcast it back to the original shape, this undoes part of the mean opeartion
        let result = get_item.broadcast(original_shape.dimensions()).unwrap();

        // We need to get out count for the next operation
        let mut n = 1;
        for i in 0..num_axes {
            if axes_array[i] != -1 {  // -1 is your "unused" flag
                n *= original_shape.dimensions()[axes_array[i] as usize];
            }
        }
        // we then scale the grad down by 1/count, since each contributed "equally" output
        let scaled_grad: Vec<f32> = result.iter().map(|&g| g / n as f32).collect();

        backprop_packet.equation.add_tensor_grad(source_id, scaled_grad);


    } else {
        panic!("Wrong operation for backward mean");
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Shape, Tensor, Operation};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    fn mean_1d_test() {
        // Test mean of 1D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4]),
            vec![1.0, 2.0, 3.0, 4.0],
            Operation::Nop,
        );

        let result = input.mean(vec![0]);

        // Result should be scalar-like (1D tensor with 1 element)
        assert_eq!(result.shape.dimensions(), vec![1]);
        
        let result_data = result.item();
        assert!(approx_equal(result_data[[0]], 2.5, 1e-6)); // (1+2+3+4)/4 = 2.5
    }

    #[test]
    fn mean_2d_axis0_test() {
        // Test mean along axis 0 (rows)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.mean(vec![0]);

        // Result should be [3] (mean of each column)
        assert_eq!(result.shape.dimensions(), vec![3]);
        
        let result_data = result.item();
        assert!(approx_equal(result_data[[0]], 2.5, 1e-6)); // (1+4)/2 = 2.5
        assert!(approx_equal(result_data[[1]], 3.5, 1e-6)); // (2+5)/2 = 3.5
        assert!(approx_equal(result_data[[2]], 4.5, 1e-6)); // (3+6)/2 = 4.5
    }

    #[test]
    fn mean_2d_axis1_test() {
        // Test mean along axis 1 (columns)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.mean(vec![1]);

        // Result should be [2] (mean of each row)
        assert_eq!(result.shape.dimensions(), vec![2]);
        
        let result_data = result.item();
        assert!(approx_equal(result_data[[0]], 2.0, 1e-6)); // (1+2+3)/3 = 2.0
        assert!(approx_equal(result_data[[1]], 5.0, 1e-6)); // (4+5+6)/3 = 5.0
    }

    #[test]
    fn mean_2d_all_axes_test() {
        // Test mean along all axes
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.mean(vec![0, 1]);

        // Result should be scalar-like (1D tensor with 1 element)
        assert_eq!(result.shape.dimensions(), vec![1]);
        
        let result_data = result.item();
        assert!(approx_equal(result_data[[0]], 3.5, 1e-6)); // (1+2+3+4+5+6)/6 = 3.5
    }

    #[test]
    fn mean_3d_test() {
        // Test mean with 3D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.mean(vec![0]);

        // Result should be [2, 2]
        assert_eq!(result.shape.dimensions(), vec![2, 2]);
        
        let result_data = result.item();
        assert!(approx_equal(result_data[[0, 0]], 3.0, 1e-6)); // (1+5)/2
        assert!(approx_equal(result_data[[0, 1]], 4.0, 1e-6)); // (2+6)/2
        assert!(approx_equal(result_data[[1, 0]], 5.0, 1e-6)); // (3+7)/2
        assert!(approx_equal(result_data[[1, 1]], 6.0, 1e-6)); // (4+8)/2
    }

    #[test]
    fn mean_4d_test() {
        // Test mean with 4D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![1, 2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.mean(vec![1]);

        // Result should be [1, 2, 2]
        assert_eq!(result.shape.dimensions(), vec![1, 2, 2]);
        
        let result_data = result.item();
        assert!(approx_equal(result_data[[0, 0, 0]], 3.0, 1e-6)); // (1+5)/2
        assert!(approx_equal(result_data[[0, 0, 1]], 4.0, 1e-6)); // (2+6)/2
        assert!(approx_equal(result_data[[0, 1, 0]], 5.0, 1e-6)); // (3+7)/2
        assert!(approx_equal(result_data[[0, 1, 1]], 6.0, 1e-6)); // (4+8)/2
    }

    #[test]
    #[should_panic(expected = "dimensions for mean must be unique")]
    fn mean_duplicate_axes_test() {
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        input.mean(vec![0, 0]); // Should panic due to duplicate axes
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn mean_out_of_bounds_axis_test() {
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        input.mean(vec![2]); // Should panic - axis 2 doesn't exist for 2D tensor
    }

    #[test]
    fn mean_empty_axes_test() {
        // Test with empty axes vector (should return copy of original)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.mean(vec![]);

        // Result should have same shape
        assert_eq!(result.shape.dimensions(), vec![2, 3]);
        
        // Result should be identical to input
        let result_data = result.item();
        let input_data = input.item();
        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_equal(result_data[[i, j]], input_data[[i, j]], 1e-6));
            }
        }
    }

    // ========== BACKWARD PASS TESTS ==========

    #[test]
    fn mean_simple_backward_test() {
        // Test basic backward pass with 1D tensor
        let a = Tensor::element(Shape::new(vec![4]), 2.0);
        let result = a.mean(vec![0]);
        result.backward();
        
        // For mean over all elements, each element gets gradient / count
        // mean(x1, x2, x3, x4) = (x1 + x2 + x3 + x4) / 4
        // d/dx_i mean = 1/4 = 0.25
        let expected_grad = 1.0 / 4.0;
        let gradients = a.grad();
        
        for &actual_grad in gradients.iter() {
            assert!(approx_equal(actual_grad, expected_grad, 1e-6));
        }
    }

    #[test]
    fn mean_1d_backward_test() {
        // Test backward pass with 1D tensor with different values
        let a = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4]),
            vec![1.0, 2.0, 3.0, 4.0],
            Operation::Nop,
        );
        
        let result = a.mean(vec![0]);
        result.backward();
        
        // Each element contributes equally to the mean
        // Gradient for each element = 1/4 = 0.25
        let expected_grad = 1.0 / 4.0;
        let gradients = a.grad();
        
        for &actual_grad in gradients.iter() {
            assert!(approx_equal(actual_grad, expected_grad, 1e-6));
        }
    }

    #[test]
    fn mean_2d_axis0_backward_test() {
        // Test backward pass with 2D tensor, mean along axis 0 (rows)
        let a = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );
        
        let result = a.mean(vec![0]);  // Mean along rows: [2,3] -> [3]
        let loss = result.sum(vec![0], true);  // Sum to scalar for backward
        loss.backward();
        
        // For mean along axis 0: each element contributes to one output element
        // Gradient = 1/num_rows = 1/2 = 0.5 for each element
        let expected_grad = 1.0 / 2.0;
        let gradients = a.grad();
        
        for &actual_grad in gradients.iter() {
            assert!(approx_equal(actual_grad, expected_grad, 1e-6));
        }
    }

    #[test]
    fn mean_2d_axis1_backward_test() {
        // Test backward pass with 2D tensor, mean along axis 1 (columns)
        let a = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );
        
        let result = a.mean(vec![1]);  // Mean along columns: [2,3] -> [2]
        let loss = result.sum(vec![0], true);  // Sum to scalar for backward
        loss.backward();
        
        // For mean along axis 1: each element contributes to one output element
        // Gradient = 1/num_cols = 1/3 ≈ 0.333... for each element
        let expected_grad = 1.0 / 3.0;
        let gradients = a.grad();
        
        for &actual_grad in gradients.iter() {
            assert!(approx_equal(actual_grad, expected_grad, 1e-6));
        }
    }

    #[test]
    fn mean_2d_all_axes_backward_test() {
        // Test backward pass with 2D tensor, mean along all axes
        let a = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );
        
        let result = a.mean(vec![0, 1]);  // Global mean: [2,3] -> [1]
        result.backward();
        
        // For global mean: each element contributes equally
        // Gradient = 1/total_elements = 1/6 ≈ 0.1667
        let expected_grad = 1.0 / 6.0;
        let gradients = a.grad();
        
        for &actual_grad in gradients.iter() {
            assert!(approx_equal(actual_grad, expected_grad, 1e-6));
        }
    }

    #[test]
    fn mean_3d_backward_test() {
        // Test backward pass with 3D tensor
        let a = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );
        
        let result = a.mean(vec![0]);  // Mean along first axis: [2,2,2] -> [2,2]
        let loss = result.sum(vec![0, 1], true);  // Sum to scalar for backward
        loss.backward();
        
        // For mean along axis 0: gradient = 1/size_of_axis0 = 1/2 = 0.5
        let expected_grad = 1.0 / 2.0;
        let gradients = a.grad();
        
        for &actual_grad in gradients.iter() {
            assert!(approx_equal(actual_grad, expected_grad, 1e-6));
        }
    }

    #[test]
    fn mean_chained_operations_backward_test() {
        // Test mean in a chain of operations
        let a = Tensor::element(Shape::new(vec![4]), 3.0);
        let b = Tensor::element(Shape::new(vec![4]), 2.0);
        
        // Chain: (a + b).mean()
        let sum = a + b;  // Each element = 5.0
        let result = sum.mean(vec![0]);  // mean = 5.0
        result.backward();
        
        // d/da mean(a + b) = d/da (1/n * sum(a + b)) = 1/n * 1 = 1/4
        // d/db mean(a + b) = d/db (1/n * sum(a + b)) = 1/n * 1 = 1/4
        let expected_grad = 1.0 / 4.0;
        
        let a_gradients = a.grad();
        let b_gradients = b.grad();
        
        for &actual_grad in a_gradients.iter() {
            assert!(approx_equal(actual_grad, expected_grad, 1e-6));
        }
        
        for &actual_grad in b_gradients.iter() {
            assert!(approx_equal(actual_grad, expected_grad, 1e-6));
        }
    }

    #[test]
    fn mean_multiple_axes_backward_test() {
        // Test backward pass with multiple axes
        let a = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3, 4]),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
            ],
            Operation::Nop,
        );
        
        let result = a.mean(vec![1, 2]);  // Mean along axes 1 and 2: [2,3,4] -> [2]
        let loss = result.sum(vec![0], true);  // Sum to scalar for backward
        loss.backward();
        
        // For mean along axes 1 and 2: gradient = 1/(size_axis1 * size_axis2) = 1/(3*4) = 1/12
        let expected_grad = 1.0 / 12.0;
        let gradients = a.grad();
        
        for &actual_grad in gradients.iter() {
            assert!(approx_equal(actual_grad, expected_grad, 1e-6));
        }
    }

    #[test]
    fn mean_nested_operations_backward_test() {
        // Test mean with nested operations for complex gradients
        let a = Tensor::element(Shape::new(vec![6]), 2.0);
        
        // Complex chain: mean(a * 3).mean() where first mean goes [6] -> [2], second [2] -> [1]
        let scaled = a * Tensor::element(Shape::new(vec![6]), 3.0);  // Each element = 6.0
        let first_mean = scaled.mean(vec![0]);  // Global mean = 6.0
        let second_mean = first_mean.mean(vec![0]); // Mean of single element = 6.0
        second_mean.backward();
        
        // d/da mean(mean(a * 3)) = d/da mean(a * 3) * d/d(mean(a*3)) mean(mean(a*3))
        // = (1/6 * 3) * 1 = 3/6 = 0.5
        let expected_grad = 0.5;
        let gradients = a.grad();
        
        for &actual_grad in gradients.iter() {
            assert!(approx_equal(actual_grad, expected_grad, 1e-6));
        }
    }
}