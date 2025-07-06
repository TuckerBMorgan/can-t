use crate::central::{get_equation, BackproagationPacket, Operation, Shape, Tensor};

impl Tensor {

    // helper function to actually swap the data of two axes
    fn swap_axes(data: &[f32], shape: &Vec<usize>, axis1: usize, axis2: usize) -> Vec<f32> {
        let mut output_data = vec![0.0; data.len()];
        let ndims = shape.len();
        
        // Calculate strides for both original and swapped dimensions
        let mut orig_strides = vec![1; ndims];
        let mut swap_strides = vec![1; ndims];
        
        for i in (0..ndims-1).rev() {
            orig_strides[i] = orig_strides[i + 1] * shape[i + 1];
            swap_strides[i] = swap_strides[i + 1] * shape[if i + 1 == axis1 { axis2 } 
                                                         else if i + 1 == axis2 { axis1 } 
                                                         else { i + 1 }];
        }
    
        // For each element in the input array
        for i in 0..data.len() {
            let old_idx = i;
            let mut new_idx = 0;
            
            // Convert to coordinates
            for d in 0..ndims {
                let coord = (old_idx / orig_strides[d]) % shape[d];
                let target_dim = if d == axis1 { axis2 }
                               else if d == axis2 { axis1 }
                               else { d };
                new_idx += coord * swap_strides[target_dim];
            }
            
            output_data[new_idx] = data[i];
        }
        
        output_data
    }

    /// Swaps two provides axis for a tensor
    /// # Arguments
    /// 'first_index' : first index we are swapping
    /// 'second_index' : second index we are swapping
    pub fn transpose(&self, first_index: usize, second_index: usize) -> Tensor {
        let data =  {
            let equation = get_equation();
            let data_as_flat_buffer = equation.get_data_flat_buffer(self.id);
            let shape = equation.get_tensor_shape(self.id).dimensions();
            let swapped_data = Tensor::swap_axes(data_as_flat_buffer, &shape, first_index, second_index);
            swapped_data
        };
        let mut shape = self.shape.dimensions();
        let hold = shape[first_index];
        shape[first_index] = shape[second_index];
        shape[second_index] = hold;
        let tensor = Tensor::create_tensor_data_and_shape_and_operation(Shape::new(shape), data, super::Operation::Transpose(self.id, first_index, second_index));

        return tensor;
    }

}


pub fn backwards_for_transpose(packet: BackproagationPacket) {
    if let Operation::Transpose(source, first, second) = packet.operation {
        // Get the incoming gradient (already transposed)
        let incoming_grad = packet.equation.get_grad_flat_buffer(packet.incoming_grad);
        
        // Get the shape of the transposed tensor (current gradient shape)
        let transposed_shape = packet.equation.get_tensor_shape(packet.incoming_grad).dimensions();
        
        // Apply the same transpose operation to the gradient to "undo" it
        // Since transpose is its own inverse: transpose(transpose(x)) = x
        let source_grad = Tensor::swap_axes(incoming_grad, &transposed_shape, first, second);
        
        // Add the gradient back to the source tensor
        packet.equation.add_tensor_grad(source, source_grad);
    }
    else {
        panic!("Wrong operation called for backwards for transpose");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::central::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_transpose_2d_simple() {
        // Test basic 2D matrix transpose
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let original = Tensor::from_vec(data, vec![2, 3]);
        
        let transposed = original.transpose(0, 1);
        
        // Check shape is swapped
        assert_eq!(transposed.shape.dimensions(), vec![3, 2]);
        
        // Check data is correctly transposed
        let result = transposed.item();
        assert!(approx_equal(result[[0, 0]], 1.0, 1e-6)); // (0,0) -> (0,0)
        assert!(approx_equal(result[[1, 0]], 2.0, 1e-6)); // (0,1) -> (1,0)
        assert!(approx_equal(result[[2, 0]], 3.0, 1e-6)); // (0,2) -> (2,0)
        assert!(approx_equal(result[[0, 1]], 4.0, 1e-6)); // (1,0) -> (0,1)
        assert!(approx_equal(result[[1, 1]], 5.0, 1e-6)); // (1,1) -> (1,1)
        assert!(approx_equal(result[[2, 1]], 6.0, 1e-6)); // (1,2) -> (2,1)
    }

    #[test]
    fn test_transpose_3d_axes_01() {
        // Test 3D tensor transpose of axes 0 and 1
        let data = vec![
            1.0, 2.0,    // [0,0,:]
            3.0, 4.0,    // [0,1,:]
            5.0, 6.0,    // [1,0,:]
            7.0, 8.0,    // [1,1,:]
        ];
        let original = Tensor::from_vec(data, vec![2, 2, 2]);
        
        let transposed = original.transpose(0, 1);
        
        // Check shape: [2,2,2] -> [2,2,2] (swapping dims 0 and 1)
        assert_eq!(transposed.shape.dimensions(), vec![2, 2, 2]);
        
        let result = transposed.item();
        // Original [0,0,:] -> Transposed [0,0,:]
        assert!(approx_equal(result[[0, 0, 0]], 1.0, 1e-6));
        assert!(approx_equal(result[[0, 0, 1]], 2.0, 1e-6));
        
        // Original [0,1,:] -> Transposed [1,0,:]
        assert!(approx_equal(result[[1, 0, 0]], 3.0, 1e-6));
        assert!(approx_equal(result[[1, 0, 1]], 4.0, 1e-6));
        
        // Original [1,0,:] -> Transposed [0,1,:]
        assert!(approx_equal(result[[0, 1, 0]], 5.0, 1e-6));
        assert!(approx_equal(result[[0, 1, 1]], 6.0, 1e-6));
        
        // Original [1,1,:] -> Transposed [1,1,:]
        assert!(approx_equal(result[[1, 1, 0]], 7.0, 1e-6));
        assert!(approx_equal(result[[1, 1, 1]], 8.0, 1e-6));
    }

    #[test]
    fn test_transpose_3d_axes_02() {
        // Test 3D tensor transpose of axes 0 and 2
        let data = vec![
            1.0, 2.0,    // [0,0,:]
            3.0, 4.0,    // [0,1,:]
            5.0, 6.0,    // [1,0,:]
            7.0, 8.0,    // [1,1,:]
        ];
        let original = Tensor::from_vec(data, vec![2, 2, 2]);
        
        let transposed = original.transpose(0, 2);
        
        // Check shape: [2,2,2] -> [2,2,2] (swapping dims 0 and 2)
        assert_eq!(transposed.shape.dimensions(), vec![2, 2, 2]);
        
        let result = transposed.item();
        // Original [0,0,0] -> Transposed [0,0,0]
        assert!(approx_equal(result[[0, 0, 0]], 1.0, 1e-6));
        
        // Original [0,0,1] -> Transposed [1,0,0]
        assert!(approx_equal(result[[1, 0, 0]], 2.0, 1e-6));
        
        // Original [0,1,0] -> Transposed [0,1,0]
        assert!(approx_equal(result[[0, 1, 0]], 3.0, 1e-6));
        
        // Original [0,1,1] -> Transposed [1,1,0]
        assert!(approx_equal(result[[1, 1, 0]], 4.0, 1e-6));
        
        // Original [1,0,0] -> Transposed [0,0,1]
        assert!(approx_equal(result[[0, 0, 1]], 5.0, 1e-6));
        
        // Original [1,0,1] -> Transposed [1,0,1]
        assert!(approx_equal(result[[1, 0, 1]], 6.0, 1e-6));
        
        // Original [1,1,0] -> Transposed [0,1,1]
        assert!(approx_equal(result[[0, 1, 1]], 7.0, 1e-6));
        
        // Original [1,1,1] -> Transposed [1,1,1]
        assert!(approx_equal(result[[1, 1, 1]], 8.0, 1e-6));
    }

    #[test]
    fn test_transpose_3d_axes_12() {
        // Test 3D tensor transpose of axes 1 and 2
        let data = vec![
            1.0, 2.0,    // [0,0,:]
            3.0, 4.0,    // [0,1,:]
            5.0, 6.0,    // [1,0,:]
            7.0, 8.0,    // [1,1,:]
        ];
        let original = Tensor::from_vec(data, vec![2, 2, 2]);
        
        let transposed = original.transpose(1, 2);
        
        // Check shape: [2,2,2] -> [2,2,2] (swapping dims 1 and 2)
        assert_eq!(transposed.shape.dimensions(), vec![2, 2, 2]);
        
        let result = transposed.item();
        // Original [0,0,:] -> Transposed [0,:,0]
        assert!(approx_equal(result[[0, 0, 0]], 1.0, 1e-6));
        assert!(approx_equal(result[[0, 1, 0]], 2.0, 1e-6));
        
        // Original [0,1,:] -> Transposed [0,:,1]
        assert!(approx_equal(result[[0, 0, 1]], 3.0, 1e-6));
        assert!(approx_equal(result[[0, 1, 1]], 4.0, 1e-6));
        
        // Original [1,0,:] -> Transposed [1,:,0]
        assert!(approx_equal(result[[1, 0, 0]], 5.0, 1e-6));
        assert!(approx_equal(result[[1, 1, 0]], 6.0, 1e-6));
        
        // Original [1,1,:] -> Transposed [1,:,1]
        assert!(approx_equal(result[[1, 0, 1]], 7.0, 1e-6));
        assert!(approx_equal(result[[1, 1, 1]], 8.0, 1e-6));
    }

    #[test]
    fn test_transpose_4d() {
        // Test 4D tensor transpose
        let data = (0..16).map(|i| i as f32).collect::<Vec<f32>>();
        let original = Tensor::from_vec(data, vec![2, 2, 2, 2]);
        
        let transposed = original.transpose(0, 3);
        
        // Check shape: [2,2,2,2] -> [2,2,2,2] (swapping dims 0 and 3)
        assert_eq!(transposed.shape.dimensions(), vec![2, 2, 2, 2]);
        
        let result = transposed.item();
        // Original [0,0,0,0] -> Transposed [0,0,0,0]
        assert!(approx_equal(result[[0, 0, 0, 0]], 0.0, 1e-6));
        
        // Original [0,0,0,1] -> Transposed [1,0,0,0]
        assert!(approx_equal(result[[1, 0, 0, 0]], 1.0, 1e-6));
        
        // Original [1,0,0,0] -> Transposed [0,0,0,1]
        assert!(approx_equal(result[[0, 0, 0, 1]], 8.0, 1e-6));
        
        // Original [1,1,1,1] -> Transposed [1,1,1,1]
        assert!(approx_equal(result[[1, 1, 1, 1]], 15.0, 1e-6));
    }

    #[test]
    fn test_transpose_identity() {
        // Test that transposing the same axis twice returns to original
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let original = Tensor::from_vec(data.clone(), vec![2, 3]);
        
        let double_transposed = original.transpose(0, 1).transpose(0, 1);
        
        // Check shape is back to original
        assert_eq!(double_transposed.shape.dimensions(), vec![2, 3]);
        
        // Check data is back to original
        let result = double_transposed.item();
        let orig_result = original.item();
        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_equal(result[[i, j]], orig_result[[i, j]], 1e-6));
            }
        }
    }

    #[test]
    fn test_transpose_single_element() {
        // Test transpose on single element tensor
        let original = Tensor::from_vec(vec![42.0], vec![1, 1]);
        let transposed = original.transpose(0, 1);
        
        assert_eq!(transposed.shape.dimensions(), vec![1, 1]);
        assert!(approx_equal(transposed.item()[[0, 0]], 42.0, 1e-6));
    }

    #[test]
    fn test_transpose_vector_to_column() {
        // Test transpose of row vector to column vector
        let data = vec![1.0, 2.0, 3.0];
        let row_vector = Tensor::from_vec(data, vec![1, 3]);
        
        let col_vector = row_vector.transpose(0, 1);
        
        assert_eq!(col_vector.shape.dimensions(), vec![3, 1]);
        let result = col_vector.item();
        assert!(approx_equal(result[[0, 0]], 1.0, 1e-6));
        assert!(approx_equal(result[[1, 0]], 2.0, 1e-6));
        assert!(approx_equal(result[[2, 0]], 3.0, 1e-6));
    }

    #[test]
    fn test_transpose_large_matrix() {
        // Test transpose on a larger matrix
        let data = (0..20).map(|i| i as f32).collect::<Vec<f32>>();
        let original = Tensor::from_vec(data, vec![4, 5]);
        
        let transposed = original.transpose(0, 1);
        
        assert_eq!(transposed.shape.dimensions(), vec![5, 4]);
        
        let result = transposed.item();
        let orig_result = original.item();
        
        // Check a few key elements
        assert!(approx_equal(result[[0, 0]], orig_result[[0, 0]], 1e-6)); // (0,0) -> (0,0)
        assert!(approx_equal(result[[1, 0]], orig_result[[0, 1]], 1e-6)); // (0,1) -> (1,0)
        assert!(approx_equal(result[[4, 3]], orig_result[[3, 4]], 1e-6)); // (3,4) -> (4,3)
    }

    #[test]
    fn test_transpose_non_square_3d() {
        // Test transpose on non-square 3D tensor
        let data = (0..24).map(|i| i as f32).collect::<Vec<f32>>();
        let original = Tensor::from_vec(data, vec![2, 3, 4]);
        
        let transposed = original.transpose(1, 2);
        
        assert_eq!(transposed.shape.dimensions(), vec![2, 4, 3]);
        
        let result = transposed.item();
        let orig_result = original.item();
        
        // Check that [0,1,2] -> [0,2,1]
        assert!(approx_equal(result[[0, 2, 1]], orig_result[[0, 1, 2]], 1e-6));
    }

    #[test]
    fn test_transpose_backward_2d() {
        // Test backward pass for 2D transpose
        let mut x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        x.set_requires_grad(true);
        
        let y = x.transpose(0, 1);
        let loss = y.sum(vec![0, 1], true);
        
        zero_all_grads();
        loss.backward();
        
        let x_grad = x.grad();
        // All gradients should be 1.0 since we just summed all elements
        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_equal(x_grad[[i, j]], 1.0, 1e-6));
            }
        }
    }

    #[test] 
    fn test_transpose_backward_3d() {
        // Test backward pass for 3D transpose
        let mut x = Tensor::from_vec((0..8).map(|i| i as f32).collect(), vec![2, 2, 2]);
        x.set_requires_grad(true);
        
        let y = x.transpose(0, 2);
        let loss = (y * 2.0).sum(vec![0, 1, 2], true);
        
        zero_all_grads();
        loss.backward();
        
        let x_grad = x.grad();
        // All gradients should be 2.0 since we multiplied by 2 then summed
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    assert!(approx_equal(x_grad[[i, j, k]], 2.0, 1e-6));
                }
            }
        }
    }

    #[test]
    fn test_transpose_chain_backward() {
        // Test backward pass through a chain of operations including transpose
        let mut x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        x.set_requires_grad(true);
        
        let y = x.transpose(0, 1); // [2,2] -> [2,2]
        let z = y * 3.0;           // Scale by 3
        let w = z.transpose(0, 1); // Transpose back
        let loss = w.sum(vec![0, 1], true);
        
        zero_all_grads();
        loss.backward();
        
        let x_grad = x.grad();
        // All gradients should be 3.0 (scale factor flows back through both transposes)
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx_equal(x_grad[[i, j]], 3.0, 1e-6));
            }
        }
    }

    #[test]
    fn test_transpose_matmul_backward() {
        // Test transpose in a more realistic scenario with matrix multiplication
        let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut b = Tensor::from_vec(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);
        
        // Compute a @ b^T
        let b_t = b.transpose(0, 1);
        let result = a << b_t;
        let loss = result.sum(vec![0, 1], true);
        
        zero_all_grads();
        loss.backward();
        
        let a_grad = a.grad();
        let b_grad = b.grad();
        
        // Verify gradients are reasonable (non-zero and finite)
        for i in 0..2 {
            for j in 0..2 {
                assert!(!a_grad[[i, j]].is_nan());
                assert!(a_grad[[i, j]].is_finite());
                assert!(!b_grad[[i, j]].is_nan());
                assert!(b_grad[[i, j]].is_finite());
            }
        }
        
        // Specific gradient checks for this operation
        // For C = A @ B^T, we have ∂L/∂A = (∂L/∂C) @ B
        // Since ∂L/∂C = [[1,1],[1,1]] and B = [[0.5,1.5],[2.5,3.5]]
        // ∂L/∂A = [[1,1],[1,1]] @ [[0.5,1.5],[2.5,3.5]] = [[3.0,5.0],[3.0,5.0]]
        let expected_a_grad_row = 0.5 + 2.5; // b[0,0] + b[1,0] = 3.0
        let expected_a_grad_col = 1.5 + 3.5; // b[0,1] + b[1,1] = 5.0
        
        assert!(approx_equal(a_grad[[0, 0]], expected_a_grad_row, 1e-5));
        assert!(approx_equal(a_grad[[0, 1]], expected_a_grad_col, 1e-5));
        assert!(approx_equal(a_grad[[1, 0]], expected_a_grad_row, 1e-5));
        assert!(approx_equal(a_grad[[1, 1]], expected_a_grad_col, 1e-5));
    }
}