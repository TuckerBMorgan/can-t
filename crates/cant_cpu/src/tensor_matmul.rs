/// Takes two tensors as flat buffers, and preforms matrix multiplication upon them
/// It makes an assumption that all of them are 4d.
/// This is to make things simpler, it is up to the caller to render the shape correctly
/// how does this make it simplier? well if I have a matrix of size [5, 10]
/// if I reshape it to [1, 1, 5, 10] I have not actually increased the number of elements
/// and in a system of batch matrix multlpication you can imagine that matrix that does not have a explict leading
/// third or fourth dimension has an implict one
/// * 'a' : The first vector
/// * 'b' : the second vector
/// 

use ndarray::prelude::*;
// Checks if the tensor shapes are compatible for batched matrix multiplication
fn valid_shape(a: [usize;4], b:[usize;4]) {
    // Ensure outer batch dimensions match
    assert!(a[0] == b[0], "{:?} {:?}", a, b);
    // Ensure inner batch dimensions match
    assert!(a[1] == b[1], "{:?} {:?}", a, b);
    // Ensure the inner dimensions of A and B are compatible for matrix multiplication
    assert!(a[3] == b[2], "{:?} {:?}", a, b);
}

// Performs batched 2D matrix multiplication over 4D tensors
// Arguments
// 'a' - a flat buffer of the data in the first tensor
// 'a_shape' - a 4 element array that represents the matrix dimensions for a
// 'b' - a flat buffer of the data in the second data
// 'b_shape' - a 4 element array that represents the matrix dimensions for b
pub fn tensor_matmul(a: &[f32], a_shape: [usize;4], b: &[f32], b_shape:[usize;4]) -> Vec<f32> {

    // Validate that the input shapes are compatible
    valid_shape(a_shape, b_shape);
    
    // Output vector to store the result of the batched matrix multiplications
    // TODO: precalucate how big the result will be, and slab allocate that 
    let mut output : Vec<f32> = vec![];

    // Iterate over the outer batch dimension
    for outer_batch_dimension in 0..a_shape[0] {
        // Iterate over the inner batch dimension
        for inner_batch_dimension in 0..b_shape[1] {

            // Calculate the offset into the flat input tensor `a` for the current batch
            let a_outer_offset_start = outer_batch_dimension * (a_shape[1] * a_shape[2] * a_shape[3]);
            let a_inner_offset_start = inner_batch_dimension * (a_shape[2] * a_shape[3]);
            let a_offset_start = a_outer_offset_start + a_inner_offset_start;

            // Extract the relevant sub-matrix from `a` and reshape it into a 2D matrix
            let a_sub_matrix = &a[a_offset_start..a_offset_start + (a_shape[2] * a_shape[3])];
            let a_tensor = Array2::from_shape_vec([a_shape[2], a_shape[3]], a_sub_matrix.to_vec()).unwrap();

            // Same logic applied to tensor `b`
            let b_outer_offset_start = outer_batch_dimension * (b_shape[1] * b_shape[2] * b_shape[3]);
            let b_inner_offset_start = inner_batch_dimension * (b_shape[2] * b_shape[3]);
            let b_offset_start = b_outer_offset_start + b_inner_offset_start;

            // Extract and reshape sub-matrix from `b`
            let b_sub_matrix = &b[b_offset_start..b_offset_start + (b_shape[2] * b_shape[3])];
            let b_tensor = Array2::from_shape_vec([b_shape[2], b_shape[3]], b_sub_matrix.to_vec()).unwrap();

            // Perform 2D matrix multiplication and flatten the result into the output vector
            let result = a_tensor.dot(&b_tensor);
            output.extend(result.into_raw_vec());
        }
    }

    return output;
}