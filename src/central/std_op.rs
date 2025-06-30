use std::collections::HashSet;
use crate::central::*;
use ndarray::prelude::*;

impl Tensor {
    /// Computes the standard deviation along specified axes
    /// # Arguments
    /// * 'axes' - Vector of axis indices along which to compute the standard deviation
    pub fn std(&self, axes: Vec<usize>) -> Tensor {

        // make sure each dimension is unique 
        let (data, output_shape) = {
            // make sure we only have unique dimensions
            let equation = get_equation();
            for (i, a) in axes.iter().enumerate() {
                for (ii, b) in axes.iter().enumerate() {
                    if i != ii && a == b {
                        panic!("dimensions for std must be unique");
                    }
                }
            }

            // Do some basic setup
            let mut tensor = equation.get_item(self.id);
            let self_shape = self.shape;
            let mut total_count = 1;

            // Sum each dimension, while counting up what 
            for (index, this_axis) in axes.iter().enumerate() {
                total_count *= self_shape.dimensions()[*this_axis];
                tensor = tensor.sum_axis(Axis(*this_axis - index));
            }

            // Divide the result by count to turn it into a mean
            let mean: Vec<f32> = tensor.iter().map(|&x| x / total_count as f32).collect();
            let local = equation.get_data_flat_buffer(self.id);

            // Now is the parat thats are unique to std calculations
            let original_shape = equation.get_tensor_shape(self.id);
            let mut dimensions = original_shape.dimensions();
            for axes in &axes {
                dimensions[*axes] = 1;
            }

            // sum_axis removed the axes, we want to add them back so that broadcast below works
            let mean_as_ndarray = ArrayD::from_shape_vec(dimensions, mean).unwrap();

            // Broadcast the mean back to the original shape, so we can do the (x - mean)^2 part of std calculation(variance)
            let broadcasted = mean_as_ndarray.broadcast(equation.get_tensor_shape(self.id).as_ndarray_shape()).unwrap().to_owned();
            let shifted = equation.sub_vector(local, &broadcasted.into_raw_vec());
            let powered : Vec<f32> = shifted.iter().map(|x|x*x).collect();            


            let mut as_ndarray = ArrayD::from_shape_vec(self.shape.dimensions(), powered).unwrap();
            
            // Find our std
            for (index, this_axis) in axes.iter().enumerate() {
                as_ndarray = as_ndarray.sum_axis(Axis(*this_axis - index));
            }
            let test = as_ndarray.map(|x|(x / total_count as f32).sqrt());

            // Fix up the shape (this is a ndarray vs cant issue)
            let output_shape = if as_ndarray.shape().is_empty() {
                vec![1]  // Convert scalar to 1D tensor with 1 element
            } else {
                as_ndarray.shape().to_vec()
            };
            

            (test, output_shape) 
        };
        let mut copied_axes = [0, 0, 0, 0];
        for (i, axis) in axes.iter().enumerate() {
            copied_axes[i] = *axis;
        }

        return Tensor::create_tensor_data_and_shape_and_operation(Shape::new(output_shape), data.to_owned().into_raw_vec(), Operation::Std(self.id, copied_axes, axes.len()));
    }
}

/// Handles calculating and passing back the gradient of a std operation
/// The gradient computation involves the derivative of standard deviation
/// For σ = sqrt(Σ(x - μ)² / N), the gradient is: ∂σ/∂x_i = (1/σ) * (1/N) * (x_i - μ)
pub fn backward_for_std(backprop_packet: BackproagationPacket) {
    if let Operation::Std(source_id, axes_array, num_axes) = backprop_packet.operation {
        panic!("implement backwards for std");
    } else {
        panic!("Wrong operation for backward std");
    }
}

// Helper function to convert flat index to multi-dimensional index
fn unflatten_index(flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut multi_idx = vec![0; shape.len()];
    let mut remaining = flat_idx;
    
    for i in (0..shape.len()).rev() {
        let stride = shape[i+1..].iter().product::<usize>();
        multi_idx[i] = remaining / stride;
        remaining %= stride;
    }
    
    multi_idx
}

// Helper function to remove an axis from a multi-dimensional index
fn remove_axis_from_index(multi_idx: &[usize], axis: usize) -> Vec<usize> {
    let mut result = Vec::new();
    for (i, &idx) in multi_idx.iter().enumerate() {
        if i != axis {
            result.push(idx);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use crate::central::{Shape, Tensor, Operation, get_equation};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    fn sub_test_check() {
        let a = vec![5.0, 4.0, 3.0, 2.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let result = get_equation().sub_vector(&a, &b);
        println!("Result: {:?}", result);  
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
        println!("{:?}", result_data[[0]]);
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
    #[should_panic(expected = "index out of bounds: the len is 2 but the index is 2")]
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

    // ========== BACKWARD PASS TESTS ==========
    // These tests will work once backward_for_std is properly implemented

    #[test]
    fn std_1d_backward_test() {
        // Test backward pass for 1D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4]),
            vec![1.0, 2.0, 3.0, 4.0],
            Operation::Nop,
        );

        let result = input.std(vec![0]);
        let loss = result.sum(vec![], true); // Sum to scalar for backward
        loss.backward();

        let gradients = input.grad();
        
        // For std([1,2,3,4]), mean = 2.5, std ≈ 1.118
        // Gradient formula: (1/σ) * (1/N) * (x_i - μ)
        let mean = 2.5;
        let std_val = 1.118034; // From forward test
        let n = 4.0;
        
        for i in 0..4 {
            let expected = (1.0 / std_val) * (1.0 / n) * (input.item()[[i]] - mean);
            assert!(approx_equal(gradients[[i]], expected, 1e-5));
        }
    }

    #[test]
    fn std_2d_axis0_backward_test() {
        // Test backward pass for 2D tensor along axis 0
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.std(vec![0]);
        let loss = result.sum(vec![0], true); // Sum all to scalar
        loss.backward();

        let gradients = input.grad();
        
        // For each column, we computed std([row1_val, row2_val])
        // Column 0: std([1,4]) = 1.5, mean = 2.5
        // Column 1: std([2,5]) = 1.5, mean = 3.5  
        // Column 2: std([3,6]) = 1.5, mean = 4.5
        
        let std_val = 1.5;
        let n = 2.0;
        
        // Check gradients for each element
        assert!(approx_equal(gradients[[0, 0]], (1.0/std_val) * (1.0/n) * (1.0 - 2.5), 1e-5));
        assert!(approx_equal(gradients[[0, 1]], (1.0/std_val) * (1.0/n) * (2.0 - 3.5), 1e-5));
        assert!(approx_equal(gradients[[0, 2]], (1.0/std_val) * (1.0/n) * (3.0 - 4.5), 1e-5));
        assert!(approx_equal(gradients[[1, 0]], (1.0/std_val) * (1.0/n) * (4.0 - 2.5), 1e-5));
        assert!(approx_equal(gradients[[1, 1]], (1.0/std_val) * (1.0/n) * (5.0 - 3.5), 1e-5));
        assert!(approx_equal(gradients[[1, 2]], (1.0/std_val) * (1.0/n) * (6.0 - 4.5), 1e-5));
    }

    #[test]
    fn std_2d_axis1_backward_test() {
        // Test backward pass for 2D tensor along axis 1
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.std(vec![1]);
        let loss = result.sum(vec![0], true); // Sum all to scalar
        loss.backward();

        let gradients = input.grad();
        
        // Row 0: std([1,2,3]) ≈ 0.816, mean = 2.0
        // Row 1: std([4,5,6]) ≈ 0.816, mean = 5.0
        
        let std_val = 0.816497;
        let n = 3.0;
        
        // Check gradients for row 0
        assert!(approx_equal(gradients[[0, 0]], (1.0/std_val) * (1.0/n) * (1.0 - 2.0), 1e-4));
        assert!(approx_equal(gradients[[0, 1]], (1.0/std_val) * (1.0/n) * (2.0 - 2.0), 1e-4));
        assert!(approx_equal(gradients[[0, 2]], (1.0/std_val) * (1.0/n) * (3.0 - 2.0), 1e-4));
        
        // Check gradients for row 1
        assert!(approx_equal(gradients[[1, 0]], (1.0/std_val) * (1.0/n) * (4.0 - 5.0), 1e-4));
        assert!(approx_equal(gradients[[1, 1]], (1.0/std_val) * (1.0/n) * (5.0 - 5.0), 1e-4));
        assert!(approx_equal(gradients[[1, 2]], (1.0/std_val) * (1.0/n) * (6.0 - 5.0), 1e-4));
    }

    #[test]
    fn std_2d_all_axes_backward_test() {
        // Test backward pass for 2D tensor along all axes
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Operation::Nop,
        );

        let result = input.std(vec![0, 1]);
        let loss = result.sum(vec![], true); // Sum to scalar
        loss.backward();

        let gradients = input.grad();
        
        // Global std of [1,2,3,4,5,6]: mean = 3.5, std ≈ 1.708
        let mean = 3.5;
        let std_val = 1.7078252; // From forward test
        let n = 6.0;
        
        // All elements should have gradient according to same formula
        let input_data = input.item();
        for i in 0..2 {
            for j in 0..3 {
                let expected = (1.0 / std_val) * (1.0 / n) * (input_data[[i, j]] - mean);
                assert!(approx_equal(gradients[[i, j]], expected, 1e-4));
            }
        }
    }

    #[test]
    fn std_3d_backward_test() {
        // Test backward pass for 3D tensor
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Operation::Nop,
        );

        let result = input.std(vec![0]);
        let loss = result.sum(vec![0, 1], true); // Sum all to scalar
        loss.backward();

        let gradients = input.grad();
        
        // For each position [i,j], we compute std across first dimension
        // Position [0,0]: std([1,5]) = 2.0, mean = 3.0
        // Position [0,1]: std([2,6]) = 2.0, mean = 4.0
        // Position [1,0]: std([3,7]) = 2.0, mean = 5.0
        // Position [1,1]: std([4,8]) = 2.0, mean = 6.0
        
        let std_val = 2.0;
        let n = 2.0;
        
        // Check gradients for first layer (index 0 in first dimension)
        assert!(approx_equal(gradients[[0, 0, 0]], (1.0/std_val) * (1.0/n) * (1.0 - 3.0), 1e-5));
        assert!(approx_equal(gradients[[0, 0, 1]], (1.0/std_val) * (1.0/n) * (2.0 - 4.0), 1e-5));
        assert!(approx_equal(gradients[[0, 1, 0]], (1.0/std_val) * (1.0/n) * (3.0 - 5.0), 1e-5));
        assert!(approx_equal(gradients[[0, 1, 1]], (1.0/std_val) * (1.0/n) * (4.0 - 6.0), 1e-5));
        
        // Check gradients for second layer (index 1 in first dimension)
        assert!(approx_equal(gradients[[1, 0, 0]], (1.0/std_val) * (1.0/n) * (5.0 - 3.0), 1e-5));
        assert!(approx_equal(gradients[[1, 0, 1]], (1.0/std_val) * (1.0/n) * (6.0 - 4.0), 1e-5));
        assert!(approx_equal(gradients[[1, 1, 0]], (1.0/std_val) * (1.0/n) * (7.0 - 5.0), 1e-5));
        assert!(approx_equal(gradients[[1, 1, 1]], (1.0/std_val) * (1.0/n) * (8.0 - 6.0), 1e-5));
    }

    #[test]
    fn std_chained_operations_backward_test() {
        // Test std in chain with other operations
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![2.0, 4.0, 6.0],
            Operation::Nop,
        );
        
        let scale = Tensor::element(Shape::new(vec![1]), 3.0);
        
        // Chain: input.std() * scale
        let std_result = input.std(vec![0]);
        let scaled = std_result * scale;
        let loss = scaled.sum(vec![], true);
        loss.backward();
        
        let input_gradients = input.grad();
        let scale_gradients = scale.grad();
        
        // Input std: [2,4,6] -> mean=4, std=1.632993
        let mean = 4.0;
        let std_val = 1.632993;
        let n = 3.0;
        let scale_val = 3.0;
        
        // Input gradients should be std gradients * scale
        for i in 0..3 {
            let expected = scale_val * (1.0 / std_val) * (1.0 / n) * (input.item()[[i]] - mean);
            assert!(approx_equal(input_gradients[[i]], expected, 1e-4));
        }
        
        // Scale gradient should be the std value
        assert!(approx_equal(scale_gradients[[0]], std_val, 1e-4));
    }

    #[test]
    fn std_zero_variance_backward_test() {
        // Test backward pass when std is zero (constant values)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![5.0, 5.0, 5.0],
            Operation::Nop,
        );

        let result = input.std(vec![0]);
        let loss = result.sum(vec![], true);
        loss.backward();

        let gradients = input.grad();
        
        // When all values are the same, std = 0, so gradients should be 0
        // (we handle division by zero in implementation)
        for i in 0..3 {
            assert!(approx_equal(gradients[[i]], 0.0, 1e-6));
        }
    }

    #[test]
    fn std_empty_axes_backward_test() {
        // Test backward pass with empty axes (no std computed)
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2]),
            vec![1.0, 2.0, 3.0, 4.0],
            Operation::Nop,
        );

        let result = input.std(vec![]);
        let loss = result.sum(vec![0, 1], true);
        loss.backward();

        let gradients = input.grad();
        
        // No std was computed, so all gradients should be zero
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx_equal(gradients[[i, j]], 0.0, 1e-6));
            }
        }
    }

    #[test]
    fn std_numerical_gradient_test() {
        // Test backward pass using numerical differentiation
        // This test verifies the analytical gradient implementation once it's done
        let input = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![1.0, 3.0, 5.0],
            Operation::Nop,
        );

        // Compute analytical gradient
        let result = input.std(vec![0]);
        let loss = result.sum(vec![], true);
        loss.backward();
        let analytical_grad = input.grad();

        // Compute numerical gradient
        let h = 1e-4;
        let mut numerical_grad = vec![0.0; 3];
        
        for i in 0..3 {
            // Forward difference
            let mut input_plus = vec![1.0, 3.0, 5.0];
            input_plus[i] += h;
            let input_plus_tensor = Tensor::create_tensor_data_and_shape_and_operation(
                Shape::new(vec![3]),
                input_plus,
                Operation::Nop,
            );
            let result_plus = input_plus_tensor.std(vec![0]).sum(vec![], true);
            
            let mut input_minus = vec![1.0, 3.0, 5.0];
            input_minus[i] -= h;
            let input_minus_tensor = Tensor::create_tensor_data_and_shape_and_operation(
                Shape::new(vec![3]),
                input_minus,
                Operation::Nop,
            );
            let result_minus = input_minus_tensor.std(vec![0]).sum(vec![], true);
            
            numerical_grad[i] = (result_plus.item()[[0]] - result_minus.item()[[0]]) / (2.0 * h);
        }

        // Compare analytical and numerical gradients
        for i in 0..3 {
            assert!(approx_equal(analytical_grad[[i]], numerical_grad[i], 1e-2));
        }
    }
}