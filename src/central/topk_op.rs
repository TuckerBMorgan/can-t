use super::{BackproagationPacket, Operation};
use crate::central::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use core::cmp::Ordering;

// BinaryHeap in rust by default is a MaxHeap
// If you wrap the data in core::cmp::Reverse it will turn it into a MinHeap
// however, f32 is not techinally comparable, it lacks Ord
// we are going to write our own version here
// it is ok for the most part
#[derive(Clone, Copy, Default, Debug)]
pub struct RevF32(pub f32);

impl PartialEq for RevF32 {
    fn eq(&self, other: &Self) -> bool {
        // bitwise equality so +0.0/-0.0 and different NaNs are distinct as in total_cmp
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for RevF32 {}

impl PartialOrd for RevF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // reverse the total order
        Some(other.0.total_cmp(&self.0))
    }
}

impl Ord for RevF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        // reverse the total order
        other.0.total_cmp(&self.0)
    }
}

impl Tensor {
    /// Returns (values, indices), both sized k * num_cols, laid out column-major:
    /// for each "column" c in 0..num_cols, the k results live at offsets c + j * num_cols
    /// for j in 0..k (descending by value).
    fn batch_with_indices(
        data: &[f32],
        k: usize,
        dimension_length: usize,
        tensor_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        if k == 0 || tensor_size == 0 || dimension_length == 0 {
            return (Vec::new(), Vec::new());
        }

        assert!(tensor_size % dimension_length == 0);
        let num_cols = tensor_size / dimension_length;

        // Allocate outputs
        let mut top_k_values  = vec![0.0f32; num_cols * k];
        let mut top_k_indices = vec![0.0f32; num_cols * k];

        // For each column independently
        for col in 0..num_cols {
            // Min-heap of size at most k
            let mut min_heap: BinaryHeap<(RevF32, usize)> = BinaryHeap::with_capacity(k);

            for row in 0..dimension_length {
                let v = data[row * num_cols + col];

                if min_heap.len() < k {
                    min_heap.push((RevF32(v), row));
                } else if v > min_heap.peek().unwrap().0 .0 {
                    let _ = min_heap.pop();
                    min_heap.push((RevF32(v), row));
                }
            }

            // Pop back in reverse so final order is descending
            for j in (0..k).rev() {
                let (RevF32(val), idx) = min_heap.pop().unwrap();
                let out_offset = col + j * num_cols;
                top_k_values[out_offset]  = val;
                top_k_indices[out_offset] = idx as f32; // consider int tensor if supported
            }
        }

        (top_k_values, top_k_indices)
    }

    pub fn topk(
        &self,
        k: usize,
        dimension: usize,
        _largest: bool,
        _sorted: bool,
    ) -> (Tensor, Tensor) {
        if k == 0 {
            panic!("topk: k must be > 0");
        }

        let data = get_equation().get_data_flat_buffer(self.id).to_vec();

        let dims = self.shape.dimensions();
        let dim_len = dims[dimension];
        if k > dim_len {
            panic!("topk: k ({}) > dimension length ({})", k, dim_len);
        }

        // shape for the outputs: same as input but with axis=dimension set to k
        let mut out_dims = dims.clone();
        out_dims[dimension] = k;
        let out_shape = Shape::new(out_dims);

        // collapse batches (leading product before `dimension`)
        let mut collapse_batch = 1;
        for i in 0..dimension {
            collapse_batch *= dims[i];
        }

        // trailing product starting at `dimension`
        let mut tensor_size = 1;
        for i in dimension..self.shape.number_of_dimension() {
            tensor_size *= dims[i];
        }

        let mut all_values  = Vec::new();
        let mut all_indices = Vec::new();
        all_values.reserve(collapse_batch * k * (tensor_size / dim_len));
        all_indices.reserve(collapse_batch * k * (tensor_size / dim_len));

        for b in 0..collapse_batch {
            let offset = tensor_size * b;
            let data_subset = &data[offset..offset + tensor_size];

            let (vals, inds) = Tensor::batch_with_indices(
                data_subset,
                k,
                dim_len,
                tensor_size,
            );

            all_values.extend(vals);
            all_indices.extend(inds);
        }

        let values_tensor  =
            Tensor::create_tensor_data_and_shape_and_operation(out_shape.clone(), all_values, Operation::Nop);

        let indices_tensor =
            Tensor::create_tensor_data_and_shape_and_operation(out_shape, all_indices,
                Operation::Topk(self.id, values_tensor.id, k, dimension, true, true)
            );

        // Return (values, indices) to match the doc comment.
        (values_tensor, indices_tensor)
    }
}

pub fn backward_for_topk(backprop_backet: BackproagationPacket) {
    if let  Operation::Topk(input_id, indices_id, k, dimension, _, _) = backprop_backet.operation {

        // Shapes and buffers
        let dims = get_equation().get_tensor_shape(input_id).dimensions().to_vec();
        let nd = dims.len();
        let dim_len = dims[dimension];

        // Sanity
        assert!(k > 0 && k <= dim_len);

        // Products per your forward
        let mut collapse_batch = 1;
        for i in 0..dimension {
            collapse_batch *= dims[i];
        }

        let mut tensor_size = 1;
        for i in dimension..nd {
            tensor_size *= dims[i];
        }

        assert!(tensor_size % dim_len == 0);
        let num_cols = tensor_size / dim_len;

        // Read saved indices and upstream grad for values output.
        // Notes:
        // - In your forward you returned (indices, values).
        // - If you switch to (values, indices), adjust which grad you fetch here.
        let indices: Vec<f32> = get_equation().get_data_flat_buffer(indices_id).to_vec();
        let grad_values: Vec<f32> = get_equation().get_grad_flat_buffer(backprop_backet.incoming_grad).to_vec();
        // Prepare grad for input
        let mut grad_input = vec![0.0f32; collapse_batch * tensor_size];

        // Scatter-add
        for b in 0..collapse_batch {
            let in_base  = b * tensor_size;
            let out_base = b * (k * num_cols);

            for col in 0..num_cols {
                for j in 0..k {
                    let out_off = out_base + (col + j * num_cols);
                    // indices were stored as row indices along `dimension`
                    let row = indices[out_off] as usize;

                    // Safety: if you ever allow very large dims, prefer integer indices tensor.
                    debug_assert!(row < dim_len);

                    let g = grad_values[out_off];
                    let in_off = in_base + (row * num_cols + col);

                    // Accumulate in case your top-k implementation ever allows duplicates
                    grad_input[in_off] += g;
                }
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= eps,
                "mismatch at {}: got {}, expected {} (eps={})",
                i, x, y, eps
            );
        }
    }

    #[test]
    fn k_zero_returns_empty() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let (vals, idxs) = Tensor::batch_with_indices(&data, 0, 2, 4);
        assert!(vals.is_empty());
        assert!(idxs.is_empty());
    }

    #[test]
    fn tensor_size_zero_returns_empty() {
        let data: [f32; 0] = [];
        let (vals, idxs) = Tensor::batch_with_indices(&data, 3, 2, 0);
        assert!(vals.is_empty());
        assert!(idxs.is_empty());
    }

    #[test]
    fn dimension_length_zero_returns_empty() {
        let data: [f32; 0] = [];
        let (vals, idxs) = Tensor::batch_with_indices(&data, 3, 0, 0);
        assert!(vals.is_empty());
        assert!(idxs.is_empty());
    }

    #[test]
    fn basic_single_column_k1() {
        // One column (num_cols = tensor_size / dimension_length = 5/5 = 1)
        let data = [0.5, 2.0, -1.0, 3.5, 1.0];
        let (vals, idxs) = Tensor::batch_with_indices(&data, 1, 5, 5);
        approx_eq(&vals, &[3.5], 1e-6);
        approx_eq(&idxs, &[3.0], 0.0);
    }

    #[test]
    fn two_columns_k2_descending_and_indices() {
        // Matrix (rows=3, cols=2):
        // row0: [1, 2]
        // row1: [5, 4]
        // row2: [3, 6]
        // Stored as data[row * num_cols + col]
        let data = [1.0, 2.0, 5.0, 4.0, 3.0, 6.0];
        let k = 2;
        let dimension_length = 3;
        let tensor_size = 6; // 3 rows * 2 cols
        let (vals, idxs) = Tensor::batch_with_indices(&data, k, dimension_length, tensor_size);

        // Expected layout: out_offset = col + j * num_cols, j from k-1..0
        // col0 top2: [5, 3] with indices [1, 2]
        // col1 top2: [6, 4] with indices [2, 1]
        approx_eq(&vals, &[5.0, 6.0, 3.0, 4.0], 1e-6);
        approx_eq(&idxs, &[1.0, 2.0, 2.0, 1.0], 0.0);
    }

    #[test]
    fn k_equals_dimension_length_returns_all_sorted_per_column() {
        // rows=3, cols=2, k = rows
        // col0: [7, -1, 0] -> sorted desc [7, 0, -1] idx [0, 2, 1]
        // col1: [1,  9, 3] -> sorted desc [9, 3, 1] idx [1, 2, 0]
        let data = [7.0, 1.0, -1.0, 9.0, 0.0, 3.0];
        let (vals, idxs) = Tensor::batch_with_indices(&data, 3, 3, 6);
        approx_eq(&vals, &[7.0, 9.0, 0.0, 3.0, -1.0, 1.0], 1e-6);
        approx_eq(&idxs, &[0.0, 1.0, 2.0, 2.0, 1.0, 0.0], 0.0);
    }

    #[test]
    fn negative_values_are_handled() {
        // One column with negatives
        // values: [-1, -2, 0, -0.5], k=2 -> [0, -0.5], idx [2, 3]
        let data = [-1.0, -2.0, 0.0, -0.5];
        let (vals, idxs) = Tensor::batch_with_indices(&data, 2, 4, 4);
        approx_eq(&vals, &[0.0, -0.5], 1e-6);
        approx_eq(&idxs, &[2.0, 3.0], 0.0);
    }

    #[test]
    fn multiple_columns_mixed_values() {
        // rows=4, cols=3, k=2
        // Build column-wise:
        // col0: [0, 10, 3, 7] -> top2 [10, 7], idx [1, 3]
        // col1: [5,  4, 9, 1] -> top2 [9, 5],  idx [2, 0]
        // col2: [8, -2, 6, 6] -> top2 [8, 6],  idx [0, 2]  (tie at 6—row2 first seen vs row3 equal; row2 wins)
        let num_cols = 3usize;
        let rows = 4usize;
        let data = [
            // row0
            0.0, 5.0, 8.0,
            // row1
            10.0, 4.0, -2.0,
            // row2
            3.0, 9.0, 6.0,
            // row3
            7.0, 1.0, 6.0,
        ];
        let (vals, idxs) = Tensor::batch_with_indices(&data, 2, rows, rows * num_cols);
        // out layout: [c0_top1, c1_top1, c2_top1, c0_top2, c1_top2, c2_top2]
        approx_eq(&vals, &[10.0, 9.0, 8.0, 7.0, 5.0, 6.0], 1e-6);
        approx_eq(&idxs, &[1.0, 2.0, 0.0, 3.0, 0.0, 2.0], 0.0);
    }

    #[test]
    #[should_panic]
    fn panics_when_tensor_size_not_multiple_of_dimension_length() {
        // tensor_size % dimension_length != 0 triggers assert
        let data = [1.0, 2.0, 3.0];
        let _ = Tensor::batch_with_indices(&data, 1, 2, 3);
    }

    #[test]
    #[should_panic]
    fn panics_when_k_exceeds_dimension_length() {
        // With k > rows, the pop loop will underflow the heap.
        let data = [1.0, 2.0, 3.0];
        let _ = Tensor::batch_with_indices(&data, 4, 3, 3);
    }
}
#[cfg(test)]
mod topk_tests {
    use super::*;
    use crate::central::*; // for get_equation

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= eps,
                "mismatch at {}: got {}, expected {} (eps={})",
                i, x, y, eps
            );
        }
    }

    fn tensor_from(data: Vec<f32>, dims: &[usize]) -> Tensor {
        let shape = Shape::new(dims.to_vec());
        Tensor::create_tensor_data_and_shape_and_operation(shape, data, Operation::Nop)
    }

    fn read(id: TensorID) -> Vec<f32> {
        get_equation().get_data_flat_buffer(id).to_vec()
    }

    #[test]
    #[should_panic(expected = "topk: k must be > 0")]
    fn topk_panics_when_k_zero() {
        let x = tensor_from(vec![1.0, 2.0, 3.0], &[3]);
        let _ = x.topk(0, 0, true, true);
    }

    #[test]
    #[should_panic(expected = "topk: k (4) > dimension length (3)")]
    fn topk_panics_when_k_exceeds_axis() {
        let x = tensor_from(vec![1.0, 2.0, 3.0], &[3]);
        let _ = x.topk(4, 0, true, true);
    }

    #[test]
    fn topk_1d_k1() {
        // Input: [0.5, 2.0, -1.0, 3.5, 1.0], k=1 along dim 0
        let x = tensor_from(vec![0.5, 2.0, -1.0, 3.5, 1.0], &[5]);
        let (vals_t, idxs_t) = x.topk(1, 0, true, true);

        // shape should be [1]
        assert_eq!(vals_t.shape.dimensions(), vec![1]);
        assert_eq!(idxs_t.shape.dimensions(), vec![1]);

        let vals = read(vals_t.id);
        let idxs = read(idxs_t.id);
        approx_eq(&vals, &[3.5], 1e-6);
        approx_eq(&idxs, &[3.0], 0.0);
    }

    #[test]
    fn topk_2d_dim0_k2() {
        // rows=3, cols=2, dim=0 (over rows)
        // Matrix:
        // r0: [1, 2]
        // r1: [5, 4]
        // r2: [3, 6]
        let x = tensor_from(vec![1.0, 2.0, 5.0, 4.0, 3.0, 6.0], &[3, 2]);
        let (vals_t, idxs_t) = x.topk(2, 0, true, true);

        // out shape: [k, cols] = [2, 2]
        assert_eq!(vals_t.shape.dimensions(), vec![2, 2]);
        assert_eq!(idxs_t.shape.dimensions(), vec![2, 2]);

        let vals = read(vals_t.id);
        let idxs = read(idxs_t.id);
        // Column-major per-column layout from batch_with_indices:
        // col0 top2: [5, 3] -> idx [1, 2]
        // col1 top2: [6, 4] -> idx [2, 1]
        approx_eq(&vals, &[5.0, 6.0, 3.0, 4.0], 1e-6);
        approx_eq(&idxs, &[1.0, 2.0, 2.0, 1.0], 0.0);
    }

    #[test]
    fn topk_2d_dim1_k2() {
        // rows=3, cols=4; dim=1 (over cols) → per-row top-k
        // r0: [1,  7,  3,  2] -> top2 [7,3] idx [1,2]
        // r1: [9,  0,  5,  4] -> top2 [9,5] idx [0,2]
        // r2: [6,  6, -1, 10] -> top2 [10,6] idx [3,0] (tie of 6 keeps first seen col 0)
        let data = vec![
            1.0, 7.0, 3.0, 2.0,
            9.0, 0.0, 5.0, 4.0,
            6.0, 6.0, -1.0, 10.0,
        ];
        let x = tensor_from(data, &[3, 4]);
        let (vals_t, idxs_t) = x.topk(2, 1, true, true);

        // out shape: [rows, k] = [3, 2]
        assert_eq!(vals_t.shape.dimensions(), vec![3, 2]);
        assert_eq!(idxs_t.shape.dimensions(), vec![3, 2]);

        // Layout per row (num_cols=1 within each slice, so just stacked rows):
        let vals = read(vals_t.id);
        let idxs = read(idxs_t.id);
        // Row0: [7,3], Row1: [9,5], Row2: [10,6] — flattened row-major
        approx_eq(&vals, &[7.0, 3.0, 9.0, 5.0, 10.0, 6.0], 1e-6);
        approx_eq(&idxs, &[1.0, 2.0, 0.0, 2.0, 3.0, 0.0], 0.0);
    }

    #[test]
    fn topk_3d_dim1_k1_batches_across_leading() {
        // Shape [B=2, R=3, C=2], k=1 along dim=1 (R)
        // Batch 0 (b=0):
        // R0: [1, 8]
        // R1: [5, 3]
        // R2: [7, 4]
        // top1 per column: col0->7(idx2), col1->8(idx0)
        //
        // Batch 1 (b=1):
        // R0: [9, -1]
        // R1: [0,  6]
        // R2: [2,  6]
        // top1 per column: col0->9(idx0), col1->6(idx1) (tie with idx2, first seen wins)
        let data = vec![
            // b0
            1.0, 8.0,
            5.0, 3.0,
            7.0, 4.0,
            // b1
            9.0, -1.0,
            0.0,  6.0,
            2.0,  6.0,
        ];
        let x = tensor_from(data, &[2, 3, 2]);
        let (vals_t, idxs_t) = x.topk(1, 1, true, true);

        // Out shape: [2, 1, 2]
        assert_eq!(vals_t.shape.dimensions(), vec![2, 1, 2]);
        assert_eq!(idxs_t.shape.dimensions(), vec![2, 1, 2]);

        // Flattened results are batch-concatenated, and within each batch column-major per batch_with_indices.
        let vals = read(vals_t.id);
        let idxs = read(idxs_t.id);
        // b0: [7, 8], b1: [9, 6]
        approx_eq(&vals, &[7.0, 8.0, 9.0, 6.0], 1e-6);
        // b0 idx: [2, 0], b1 idx: [0, 1]
        approx_eq(&idxs, &[2.0, 0.0, 0.0, 1.0], 0.0);
    }

    #[test]
    fn topk_descending_and_indices_correct() {
        // Verify sorted descending within each column slice and indices match source.
        // rows=4, cols=3, dim=0, k=3
        // col0: [0,10,3,7] -> [10,7,3] idx [1,3,2]
        // col1: [5, 4,9,1] -> [ 9,5,4] idx [2,0,1]
        // col2: [8,-2,6,6] -> [ 8,6,6] idx [0,2,3]
        let data = vec![
            0.0, 5.0, 8.0,
            10.0, 4.0, -2.0,
            3.0, 9.0, 6.0,
            7.0, 1.0, 6.0,
        ];
        let x = tensor_from(data, &[4, 3]);
        let (vals_t, idxs_t) = x.topk(3, 0, true, true);
        let vals = read(vals_t.id);
        let idxs = read(idxs_t.id);

        approx_eq(&vals, &[10.0, 9.0, 8.0, 7.0, 5.0, 6.0, 3.0, 4.0, 6.0], 1e-6);
        approx_eq(&idxs, &[ 1.0, 2.0, 0.0, 3.0, 0.0, 2.0, 2.0, 1.0, 3.0], 0.0);
    }

    #[test]
    fn topk_ties_keep_first_seen() {
        // One row, dim=1 (over cols), ties of equal value
        // row: [5, 5, 4], k=2 -> values [5,5], indices [0,1] (first seen wins due to heap behavior)
        let x = tensor_from(vec![5.0, 5.0, 4.0], &[1, 3]);
        let (vals_t, idxs_t) = x.topk(2, 1, true, true);
        let vals = read(vals_t.id);
        let idxs = read(idxs_t.id);
        approx_eq(&vals, &[5.0, 5.0], 1e-6);
        approx_eq(&idxs, &[0.0, 1.0], 0.0);
    }

    #[test]
    fn topk_handles_negatives() {
        // 1D, negatives; k=2
        let x = tensor_from(vec![-1.0, -2.0, 0.0, -0.5], &[4]);
        let (vals_t, idxs_t) = x.topk(2, 0, true, true);
        let vals = read(vals_t.id);
        let idxs = read(idxs_t.id);
        approx_eq(&vals, &[0.0, -0.5], 1e-6);
        approx_eq(&idxs, &[2.0, 3.0], 0.0);
    }

    #[test]
    fn topk_output_shape_matches_replaced_axis() {
        // Input [2,3,4], k=2 along dim=2 -> output [2,3,2]
        let x = tensor_from((0..24).map(|v| v as f32).collect(), &[2, 3, 4]);
        let (vals_t, idxs_t) = x.topk(2, 2, true, true);
        assert_eq!(vals_t.shape.dimensions(), vec![2, 3, 2]);
        assert_eq!(idxs_t.shape.dimensions(), vec![2, 3, 2]);
    }
}

