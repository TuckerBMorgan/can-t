use crate::*;
use metal::*;
use std::mem;

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

use crate::MAX_DIMS;

fn valid_shape(a: [usize; MAX_DIMS], b: [usize; MAX_DIMS]) {
    // batch dims must match exactly (no broadcasting here)
    for i in 0..(MAX_DIMS - 2) {
        assert!(a[i] == b[i], "Batch dims mismatch for matmul: {:?} vs {:?}", a, b);
    }
    // inner matmul dims must align: (.., M, K) x (.., K, N)
    assert!(
        a[MAX_DIMS - 1] == b[MAX_DIMS - 2],
        "Inner dims mismatch for matmul: {:?} vs {:?}", a, b
    );
}

pub fn tensor_matmul(a: &[f32], a_shape: [usize; MAX_DIMS], b: &[f32], b_shape: [usize; MAX_DIMS]) -> Vec<f32> {
    objc::rc::autoreleasepool(|| {
        // Validate that the input shapes are compatible
        valid_shape(a_shape, b_shape);

        // the shape of the resultant matrix
        // we already know that is it a valid shape
        let result_shape = [a_shape[0], b_shape[1], a_shape[2], b_shape[3]];

        // Pull out the function we need
        let function = METAL_LIBRARY
            .get_function("batchedMatMul", None)
            .expect("Function not found");

        // Start to setup our compute pipeline
        let queue = METAL_DEVICE.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Create and copy over our buffers on the metal device
        let a_buffer = METAL_DEVICE.new_buffer_with_data(
            a.as_ptr() as *const _,
            (a.len() * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buffer = METAL_DEVICE.new_buffer_with_data(
            b.as_ptr() as *const _,
            (b.len() * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let mut result_shape_total = 1;
        for d in result_shape {
            result_shape_total *= d;
        }
        let mut c_data = vec![0.0; result_shape_total];
        let c_buffer = METAL_DEVICE.new_buffer_with_data(
            c_data.as_mut_ptr() as *const _,
            (result_shape_total * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // M, N, K, B2
        let mut dimensions = [
            a_shape[2] as u32,
            b_shape[3] as u32,
            a_shape[3] as u32,
            a_shape[1] as u32,
        ];
        let dimensions_buffer = METAL_DEVICE.new_buffer_with_data(
            dimensions.as_mut_ptr() as *const _,
            (4 * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Setup the compute grid
        let compute_pipeline = METAL_DEVICE
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        encoder.set_compute_pipeline_state(&compute_pipeline);
        encoder.set_buffer(0, Some(&a_buffer), 0);
        encoder.set_buffer(1, Some(&b_buffer), 0);
        encoder.set_buffer(2, Some(&c_buffer), 0);
        encoder.set_buffer(3, Some(&dimensions_buffer), 0);

        let b1 = a_shape[0];
        let b2 = b_shape[1];
        let n = dimensions[1];
        let m = dimensions[0];

        let tile_x = 8;
        let tile_y = 8;

        let threads_per_group = MTLSize::new(tile_x, tile_y, 1);

        let num_batches = (b1 * b2) as u64;
        let thread_groups_x = (n as u64 + tile_x - 1) / tile_x;
        let thread_groups_y = (m as u64 + tile_y - 1) / tile_y;
        let thread_groups_z = num_batches;

        let thread_groups = MTLSize::new(thread_groups_x, thread_groups_y, thread_groups_z);

        // Kick off the shader call
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy the data back
        let c_ptr = c_buffer.contents() as *const f32;
        let c_slice = unsafe { std::slice::from_raw_parts(c_ptr, c_data.len()) };

        c_slice.to_vec()
    })
}
