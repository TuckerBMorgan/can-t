use crate::*;
use metal::*;
use std::mem;

pub fn tensor_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    objc::rc::autoreleasepool(|| {
        // Pull out the function we need
        let function = METAL_LIBRARY
            .get_function("add_arrays", None)
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
        let mut c_data = vec![0.0; a.len()];
        let c_buffer = METAL_DEVICE.new_buffer_with_data(
            c_data.as_mut_ptr() as *const _,
            (c_data.len() * mem::size_of::<f32>()) as u64,
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
        let threads_per_group =
            MTLSize::new(compute_pipeline.max_total_threads_per_threadgroup(), 1, 1);
        let thread_groups = MTLSize::new(a.len() as u64, 1, 1);

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
