use core::f32;

//use itertools::izip; // optional, just nice for zipping
use super::{BackproagationPacket, Operation, Tensor};
use crate::central::*;

impl Tensor {
    pub fn max(&self, dimension: usize, keep_dim: bool) -> Tensor {
        let data = get_equation().get_item(self.id); // ndarray::ArrayD<f32> (I’m assuming)
        let shape = data.shape().to_vec();
        let dim = dimension;

        let axis_len = shape[dim];
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();

        let raw = data
            .as_slice()
            .expect("max: input must be contiguous for flat indexing");

        let out_len = outer * inner;
        let mut out_vals = Vec::with_capacity(out_len);
        let mut flat_indices = Vec::with_capacity(out_len);

        // Iterate over all “lanes” produced by reducing axis dim
        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut best_val = f32::MIN;
                let mut best_flat = 0usize;

                for a in 0..axis_len {
                    let flat = ((outer_idx * axis_len + a) * inner) + inner_idx;
                    let v = raw[flat];
                    if v > best_val {
                        best_val = v;
                        best_flat = flat;
                    }
                }

                out_vals.push(best_val);
                flat_indices.push(best_flat); // offset into *input* flat buffer
            }
        }

        // Build the reduced shape (without dim)
        let mut reduced_shape = shape.clone();
        reduced_shape.remove(dim);

        // Handle keep_dim: insert axis of length 1
        let out_shape = if keep_dim {
            let mut s = reduced_shape.clone();
            s.insert(dim, 1);
            s
        } else {
            reduced_shape.clone()
        };
        let flat_indices: Vec<f32> = flat_indices.iter().map(|x| *x as f32).collect();
        let flat_indices_len = flat_indices.len();
        let indicies = Tensor::from_vec(flat_indices, vec![flat_indices_len]);

        Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(out_shape),
            out_vals,
            Operation::Max(self.id, indicies.id, dimension, keep_dim),
        )
    }
}

pub fn backwards_for_max(packet: BackproagationPacket) {
    if let Operation::Max(source, indices, _dimension, _keep_dim) = packet.operation {
        let shape = packet.equation.get_tensor_shape(source);

        let indices_data = packet.equation.get_data_flat_buffer(indices);
        let incoming_grad = packet.equation.get_grad_flat_buffer(packet.incoming_grad);

        let mut grad_array = vec![0.0f32; shape.total_size()];

        for (indiex, grad) in indices_data.iter().zip(incoming_grad) {
            let as_usize = *indiex as usize;
            grad_array[as_usize] += grad;
        }

        packet.equation.add_tensor_grad(source, grad_array);
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Tensor, zero_all_grads};

    fn approx_equal(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6
    }

    #[test]
    fn max_reduces_dimension_without_keepdim() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![1.0, 5.0, 2.0, -1.0, 4.0, 10.0], vec![2, 3]);

        let reduced = data.max(0, false);
        let array = reduced.item();

        assert_eq!(array.shape(), &[3]);
        assert!(approx_equal(array[[0]], 1.0));
        assert!(approx_equal(array[[1]], 5.0));
        assert!(approx_equal(array[[2]], 10.0));
    }

    #[test]
    fn max_keepdim_true_preserves_axis_location() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![1.0, 5.0, 2.0, -1.0, 4.0, 10.0], vec![2, 3]);

        let reduced = data.max(1, true);
        let array = reduced.item();

        assert_eq!(array.shape(), &[2, 1]);
        assert!(approx_equal(array[[0, 0]], 5.0));
        assert!(approx_equal(array[[1, 0]], 10.0));
    }

    #[test]
    fn max_reduces_last_dimension_of_three_d_tensor() {
        zero_all_grads();
        let data = Tensor::from_vec((1..=12).map(|v| v as f32).collect(), vec![2, 2, 3]);

        let reduced = data.max(2, false);
        let array = reduced.item();

        assert_eq!(array.shape(), &[2, 2]);
        assert!(approx_equal(array[[0, 0]], 3.0));
        assert!(approx_equal(array[[0, 1]], 6.0));
        assert!(approx_equal(array[[1, 0]], 9.0));
        assert!(approx_equal(array[[1, 1]], 12.0));
    }

    #[test]
    fn max_handles_negative_values_across_dimension() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![-3.0, -7.0, -2.0, -8.0, -10.0, -5.0], vec![3, 2]);

        let reduced = data.max(1, false);
        let array = reduced.item();

        assert_eq!(array.shape(), &[3]);
        assert!(approx_equal(array[[0]], -3.0));
        assert!(approx_equal(array[[1]], -2.0));
        assert!(approx_equal(array[[2]], -5.0));
    }

    #[test]
    #[should_panic]
    fn max_panics_when_dimension_is_invalid() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let _ = data.max(3, false);
    }

    #[test]
    fn max_backward_dimension_one_routes_grad_to_row_maxima() {
        zero_all_grads();
        let mut data = Tensor::from_vec(vec![1.0, 7.0, 2.0, -3.0, 0.0, -10.0], vec![2, 3]);
        data.set_requires_grad(true);

        let reduced = data.max(1, false);
        let loss = reduced.sum(vec![0], true);
        loss.backward();

        let grads = data.grad();
        assert!(approx_equal(grads[[0, 0]], 0.0));
        assert!(approx_equal(grads[[0, 1]], 1.0));
        assert!(approx_equal(grads[[0, 2]], 0.0));
        assert!(approx_equal(grads[[1, 0]], 0.0));
        assert!(approx_equal(grads[[1, 1]], 1.0));
        assert!(approx_equal(grads[[1, 2]], 0.0));
    }

    #[test]
    fn max_backward_dimension_zero_keepdim_tracks_column_maxima() {
        zero_all_grads();
        let mut data = Tensor::from_vec(vec![1.0, 9.0, 4.0, 5.0, 2.0, 6.0], vec![2, 3]);
        data.set_requires_grad(true);

        let reduced = data.max(0, true);
        let loss = reduced.sum(vec![0, 1], true);
        loss.backward();

        let grads = data.grad();
        assert!(approx_equal(grads[[0, 0]], 0.0));
        assert!(approx_equal(grads[[0, 1]], 1.0));
        assert!(approx_equal(grads[[0, 2]], 0.0));
        assert!(approx_equal(grads[[1, 0]], 1.0));
        assert!(approx_equal(grads[[1, 1]], 0.0));
        assert!(approx_equal(grads[[1, 2]], 1.0));
    }

    #[test]
    fn max_backward_three_dimensional_inputs_propagate_along_reduced_axis() {
        zero_all_grads();
        let mut data = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, // batch 0, row 0
                6.0, 4.0, 5.0, // batch 0, row 1
                7.0, 8.0, 1.0, // batch 1, row 0
                0.0, 9.0, 2.0, // batch 1, row 1
            ],
            vec![2, 2, 3],
        );
        data.set_requires_grad(true);

        let reduced = data.max(2, false);
        let loss = reduced.sum(vec![0, 1], true);
        loss.backward();

        let grads = data.grad();
        assert!(approx_equal(grads[[0, 0, 0]], 0.0));
        assert!(approx_equal(grads[[0, 0, 1]], 0.0));
        assert!(approx_equal(grads[[0, 0, 2]], 1.0));
        assert!(approx_equal(grads[[0, 1, 0]], 1.0));
        assert!(approx_equal(grads[[0, 1, 1]], 0.0));
        assert!(approx_equal(grads[[0, 1, 2]], 0.0));
        assert!(approx_equal(grads[[1, 0, 0]], 0.0));
        assert!(approx_equal(grads[[1, 0, 1]], 1.0));
        assert!(approx_equal(grads[[1, 0, 2]], 0.0));
        assert!(approx_equal(grads[[1, 1, 0]], 0.0));
        assert!(approx_equal(grads[[1, 1, 1]], 1.0));
        assert!(approx_equal(grads[[1, 1, 2]], 0.0));
    }
}
