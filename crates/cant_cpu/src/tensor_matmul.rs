use crate::MAX_DIMS;
use ndarray::Array2;
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
use std::{char::MAX, cmp::max};

fn valid_shape(a: [usize; MAX_DIMS], b: [usize; MAX_DIMS]) {
    // batch dims must match exactly (no broadcasting here)
    for i in 0..(MAX_DIMS - 2) {
        assert!(
            a[i] == b[i],
            "Batch dims mismatch for matmul: {:?} vs {:?}",
            a,
            b
        );
    }
    // inner matmul dims must align: (.., M, K) x (.., K, N)
    assert!(
        a[MAX_DIMS - 1] == b[MAX_DIMS - 2],
        "Inner dims mismatch for matmul: {:?} vs {:?}",
        a,
        b
    );
}

fn product(slice: &[usize]) -> usize {
    slice
        .iter()
        .copied()
        .fold(1usize, |acc, x| acc.saturating_mul(x))
}

pub fn loop_count(shape: [usize; MAX_DIMS]) -> usize {
    // product of batch dims only
    product(&shape[..(MAX_DIMS - 2)])
}

pub fn tensor_matmul(
    a: &[f32],
    a_shape: [usize; MAX_DIMS],
    b: &[f32],
    b_shape: [usize; MAX_DIMS],
) -> Vec<f32> {
    valid_shape(a_shape, b_shape);

    let batches = loop_count(a_shape);
    let m = a_shape[MAX_DIMS - 2];
    let k = a_shape[MAX_DIMS - 1];
    let _k_check = b_shape[MAX_DIMS - 2]; // equals k by valid_shape
    let n = b_shape[MAX_DIMS - 1];

    // strides per batch (elements per single matrix)
    let a_stride = m * k;
    let b_stride = k * n;

    // sanity-check buffer sizes
    assert!(
        a.len() == batches * a_stride,
        "A buffer size mismatch: len={}, expected={}",
        a.len(),
        batches * a_stride
    );
    assert!(
        b.len() == batches * b_stride,
        "B buffer size mismatch: len={}, expected={}",
        b.len(),
        batches * b_stride
    );

    // allocate output once
    let mut out = Vec::with_capacity(batches * m * n);

    for batch in 0..batches {
        let a_offset = batch * a_stride;
        let b_offset = batch * b_stride;

        let a_sub = &a[a_offset..a_offset + a_stride];
        let b_sub = &b[b_offset..b_offset + b_stride];

        // Build 2D views for this batch
        let a_mat = Array2::from_shape_vec([m, k], a_sub.to_vec()).unwrap();
        let b_mat = Array2::from_shape_vec([k, n], b_sub.to_vec()).unwrap();

        let c = a_mat.dot(&b_mat); // (m, n)
        out.extend(c.into_raw_vec());
    }

    out
}
