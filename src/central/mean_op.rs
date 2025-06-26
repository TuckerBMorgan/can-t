use std::collections::HashSet;
use crate::central::*;
use ndarray::prelude::*;

impl Tensor {
    /// Computes the mean along specified axes
    /// # Arguments
    /// * 'axes' - Vector of axis indices along which to compute the mean
    pub fn mean(&self, mut axes: Vec<usize>) -> Tensor {
        panic!("Implement mean op")
    }
}

/// Handles calculating and passing back the gradient of a mean operation
/// The gradient is distributed equally among all elements that contributed to the mean
pub fn backward_for_mean(backprop_packet: BackproagationPacket) {
    if let Operation::Mean(source_id, axes_array, num_axes) = backprop_packet.operation {
        panic!("Imple,ent backwarsd for mean");
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
    #[should_panic(expected = "axis 2 is out of bounds")]
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
}