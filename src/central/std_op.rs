use std::collections::HashSet;
use crate::central::*;
use ndarray::prelude::*;

impl Tensor {
    /// Computes the standard deviation along specified axes
    /// # Arguments
    /// * 'axes' - Vector of axis indices along which to compute the standard deviation
    pub fn std(&self, mut axes: Vec<usize>) -> Tensor {
        panic!("Implemet std fo");
    }
}

/// Handles calculating and passing back the gradient of a std operation
/// The gradient computation involves the derivative of standard deviation
pub fn backward_for_std(backprop_packet: BackproagationPacket) {
    if let Operation::Std(source_id, axes_array, num_axes) = backprop_packet.operation {
        panic!("Implemet std backwards");
    } else {
        panic!("Wrong operation for backward std");
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Shape, Tensor, Operation};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    fn std_1d_test() {
        // Test std of 1D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4]),
            vec![1.0, 2.0, 3.0, 4.0],
            Operation::Nop,
        );

        let result = input.std(vec![0]);

        // Result should be scalar-like (1D tensor with 1 element)
        assert_eq!(result.shape.dimensions(), vec![1]);
        
        let result_data = result.item();
        // Standard deviation of [1,2,3,4] = sqrt(((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2)/4)
        // = sqrt((2.25 + 0.25 + 0.25 + 2.25)/4) = sqrt(5/4) = sqrt(1.25) ≈ 1.118
        assert!(approx_equal(result_data[[0]], 1.118034, 1e-5));
    }

    #[test]
    fn std_2d_axis0_test() {
        // Test std along axis 0 (rows)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.std(vec![0]);

        // Result should be [3] (std of each column)
        assert_eq!(result.shape.dimensions(), vec![3]);
        
        let result_data = result.item();
        // std of column 0: [1,4] -> std = sqrt(((1-2.5)^2 + (4-2.5)^2)/2) = sqrt(4.5/2) = 1.5
        assert!(approx_equal(result_data[[0]], 1.5, 1e-6));
        // std of column 1: [2,5] -> std = sqrt(((2-3.5)^2 + (5-3.5)^2)/2) = sqrt(4.5/2) = 1.5
        assert!(approx_equal(result_data[[1]], 1.5, 1e-6));
        // std of column 2: [3,6] -> std = sqrt(((3-4.5)^2 + (6-4.5)^2)/2) = sqrt(4.5/2) = 1.5
        assert!(approx_equal(result_data[[2]], 1.5, 1e-6));
    }

    #[test]
    fn std_2d_axis1_test() {
        // Test std along axis 1 (columns)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.std(vec![1]);

        // Result should be [2] (std of each row)
        assert_eq!(result.shape.dimensions(), vec![2]);
        
        let result_data = result.item();
        // std of row 0: [1,2,3] -> mean=2, std = sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2)/3) = sqrt(2/3) ≈ 0.816
        assert!(approx_equal(result_data[[0]], 0.816497, 1e-5));
        // std of row 1: [4,5,6] -> mean=5, std = sqrt(((4-5)^2 + (5-5)^2 + (6-5)^2)/3) = sqrt(2/3) ≈ 0.816
        assert!(approx_equal(result_data[[1]], 0.816497, 1e-5));
    }

    #[test]
    fn std_2d_all_axes_test() {
        // Test std along all axes
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.std(vec![0, 1]);

        // Result should be scalar-like (1D tensor with 1 element)
        assert_eq!(result.shape.dimensions(), vec![1]);
        
        let result_data = result.item();
        // std of all elements [1,2,3,4,5,6] -> mean=3.5, variance=2.916667, std ≈ 1.708
        assert!(approx_equal(result_data[[0]], 1.708, 1e-3));
    }

    #[test]
    fn std_3d_test() {
        // Test std with 3D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.std(vec![0]);

        // Result should be [2, 2]
        assert_eq!(result.shape.dimensions(), vec![2, 2]);
        
        let result_data = result.item();
        // Each position should have std between corresponding elements across first dimension
        assert!(approx_equal(result_data[[0, 0]], 2.0, 1e-6)); // std([1,5]) = 2
        assert!(approx_equal(result_data[[0, 1]], 2.0, 1e-6)); // std([2,6]) = 2
        assert!(approx_equal(result_data[[1, 0]], 2.0, 1e-6)); // std([3,7]) = 2
        assert!(approx_equal(result_data[[1, 1]], 2.0, 1e-6)); // std([4,8]) = 2
    }

    #[test]
    fn std_4d_test() {
        // Test std with 4D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![1, 2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.std(vec![1]);

        // Result should be [1, 2, 2]
        assert_eq!(result.shape.dimensions(), vec![1, 2, 2]);
        
        let result_data = result.item();
        // Each position should have std between corresponding elements across dimension 1
        assert!(approx_equal(result_data[[0, 0, 0]], 2.0, 1e-6)); // std([1,5]) = 2
        assert!(approx_equal(result_data[[0, 0, 1]], 2.0, 1e-6)); // std([2,6]) = 2
        assert!(approx_equal(result_data[[0, 1, 0]], 2.0, 1e-6)); // std([3,7]) = 2
        assert!(approx_equal(result_data[[0, 1, 1]], 2.0, 1e-6)); // std([4,8]) = 2
    }

    #[test]
    #[should_panic(expected = "dimensions for std must be unique")]
    fn std_duplicate_axes_test() {
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        input.std(vec![0, 0]); // Should panic due to duplicate axes
    }

    #[test]
    #[should_panic(expected = "axis 2 is out of bounds")]
    fn std_out_of_bounds_axis_test() {
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        input.std(vec![2]); // Should panic - axis 2 doesn't exist for 2D tensor
    }

    #[test]
    fn std_empty_axes_test() {
        // Test with empty axes vector (should return copy of original)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.std(vec![]);

        // Result should have same shape
        assert_eq!(result.shape.dimensions(), vec![2, 3]);
        
        // Result should be all zeros since no std computation was done
        let result_data = result.item();
        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_equal(result_data[[i, j]], 0.0, 1e-6));
            }
        }
    }
}