use super::{BackproagationPacket, Operation, Tensor};
use crate::central::*;

pub fn linear_index(point: &[usize], dims: &[usize]) -> Option<usize> {
    if point.len() != dims.len() {
        return None;
    }

    let n = dims.len();
    let mut idx: usize = 0;
    let mut stride: usize = 1;

    // Row-major: accumulate from the last axis to the first.
    for i in (0..n).rev() {
        let p = point[i];
        let d = dims[i];
        if p >= d {
            return None; // out of bounds
        }
        idx = idx.checked_add(p.checked_mul(stride)?)?;
        stride = stride.checked_mul(d)?;
    }

    Some(idx)
}

impl Tensor {
    pub fn gather(&self, dimension: usize, index: Tensor) -> Tensor {
        let look_ups = index.shape.generate_all_positions();

        let offsets = get_equation().get_data_flat_buffer(index.id).to_vec();
        let input_data = get_equation().get_data_flat_buffer(self.id).to_vec();
        let mut return_vec = vec![0.0; index.shape.total_size()];

        for p in look_ups {
            let qualified_offset: usize = linear_index(&p, &index.shape.dimensions()).unwrap();
            let new_index: usize = offsets[qualified_offset] as usize;

            let mut data_index = p.clone();
            data_index[dimension] = new_index;
            // Calculate the offset
            let data_qualified_offset: usize =
                linear_index(&data_index, &self.shape.dimensions()).unwrap();
            return_vec[qualified_offset] = input_data[data_qualified_offset];
        }

        return Tensor::create_tensor_data_and_shape_and_operation(
            index.shape,
            return_vec,
            Operation::Gather(self.id, index.id, dimension),
        );
    }
}

pub fn backward_for_gather(packet: BackproagationPacket) {
    if let Operation::Gather(source, index, dimension) = packet.operation {
        let incoming_grad = packet
            .equation
            .get_grad_flat_buffer(packet.incoming_grad)
            .to_vec();
        let source_tensor_size = packet.equation.get_tensor_shape(source);
        let mut new_grad = vec![0.0; source_tensor_size.total_size()];
        let index_shape = packet.equation.get_tensor_shape(index);
        let offsets = packet.equation.get_data_flat_buffer(index).to_vec();

        let look_ups = index_shape.generate_all_positions();

        for p in look_ups {
            let qualified_offset: usize = linear_index(&p, &index_shape.dimensions()).unwrap();
            let new_index = offsets[qualified_offset] as usize;
            let mut data_index = p.clone();
            data_index[dimension] = new_index;
            // Calculate the offset
            let data_qualified_offset: usize =
                linear_index(&data_index, &source_tensor_size.dimensions()).unwrap();
            // Since the same index can show up more then once, we need to do an ADD here and not a SET
            new_grad[data_qualified_offset] += incoming_grad[qualified_offset];
        }

        packet.equation.add_tensor_grad(source, new_grad);
    }
}

#[cfg(test)]
mod tests {
    use super::linear_index;
    use crate::central::{Tensor, zero_all_grads};
    use std::usize;

    fn approx_equal(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6
    }

    #[test]
    fn linear_index_respects_row_major_layout() {
        assert_eq!(linear_index(&[1, 2], &[2, 3]), Some(5));
    }

    #[test]
    fn linear_index_rejects_dimension_mismatch() {
        assert_eq!(linear_index(&[0, 0], &[2]), None);
    }

    #[test]
    fn linear_index_rejects_out_of_bounds_points() {
        assert_eq!(linear_index(&[1, 3], &[2, 3]), None);
    }

    #[test]
    fn linear_index_refuses_overflowing_multiplication() {
        let dims = [usize::MAX, 2];
        let point = [usize::MAX - 1, 1];
        assert_eq!(linear_index(&point, &dims), None);
    }

    #[test]
    fn linear_index_returns_expected_value_for_three_dimensions() {
        assert_eq!(linear_index(&[1, 2, 3], &[2, 3, 4]), Some(23));
    }

    #[test]
    fn linear_index_detects_stride_overflow() {
        let half_max = (usize::MAX / 2) + 1;
        let dims = [half_max, half_max];
        let point = [1, 1];
        assert_eq!(linear_index(&point, &dims), None);
    }

    #[test]
    fn gather_swaps_rows_when_index_targets_dimension_zero() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let index = Tensor::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0], vec![2, 3]);

        let gathered = data.gather(0, index);
        let result = gathered.item();

        assert!(approx_equal(result[[0, 0]], 4.0));
        assert!(approx_equal(result[[0, 1]], 5.0));
        assert!(approx_equal(result[[0, 2]], 6.0));
        assert!(approx_equal(result[[1, 0]], 1.0));
        assert!(approx_equal(result[[1, 1]], 2.0));
        assert!(approx_equal(result[[1, 2]], 3.0));
    }

    #[test]
    fn gather_shuffles_columns_when_dimension_is_one() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0], vec![2, 3]);
        let index = Tensor::from_vec(vec![2.0, 1.0, 0.0, 0.0, 0.0, 2.0], vec![2, 3]);

        let gathered = data.gather(1, index);
        let result = gathered.item();

        assert!(approx_equal(result[[0, 0]], 12.0));
        assert!(approx_equal(result[[0, 1]], 11.0));
        assert!(approx_equal(result[[0, 2]], 10.0));
        assert!(approx_equal(result[[1, 0]], 20.0));
        assert!(approx_equal(result[[1, 1]], 20.0));
        assert!(approx_equal(result[[1, 2]], 22.0));
    }

    #[test]
    fn gather_preserves_shape_when_index_matches_input_rank() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![7.0; 9], vec![3, 3]);
        let index = Tensor::from_vec(vec![0.0; 9], vec![3, 3]);

        let gathered = data.gather(1, index);
        let result = gathered.item();

        assert_eq!(result.shape(), &[3, 3]);
        for value in result.iter() {
            assert!(approx_equal(*value, 7.0));
        }
    }

    #[test]
    fn gather_handles_one_dimensional_inputs() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0], vec![5]);
        let index = Tensor::from_vec(vec![4.0, 3.0, 0.0, 1.0, 2.0], vec![5]);

        let gathered = data.gather(0, index);
        let result = gathered.item();
        let expected = vec![1.0, 2.0, 5.0, 4.0, 3.0];

        assert_eq!(result.shape(), &[5]);
        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!(approx_equal(*actual, *expected));
        }
    }

    #[test]
    fn gather_handles_middle_dimension_on_three_dimensional_input() {
        zero_all_grads();
        let data = Tensor::from_vec((0..12).map(|v| v as f32).collect(), vec![2, 3, 2]);
        let index = Tensor::from_vec(
            vec![
                1.0, 1.0, // (0, 0, :)
                2.0, 2.0, // (0, 1, :)
                0.0, 0.0, // (0, 2, :)
                2.0, 2.0, // (1, 0, :)
                0.0, 0.0, // (1, 1, :)
                1.0, 1.0, // (1, 2, :)
            ],
            vec![2, 3, 2],
        );

        let gathered = data.gather(1, index);
        let result = gathered.item();
        let expected = vec![
            2.0, 3.0, // pulled from row 1
            4.0, 5.0, // pulled from row 2
            0.0, 1.0, // pulled from row 0
            10.0, 11.0, // pulled from row 2
            6.0, 7.0, // pulled from row 0
            8.0, 9.0, // pulled from row 1
        ];

        assert_eq!(result.shape(), &[2, 3, 2]);
        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!(approx_equal(*actual, *expected));
        }
    }

    #[test]
    fn gather_backward_dimension_two_accumulates_counts() {
        zero_all_grads();
        let mut data = Tensor::from_vec((1..=12).map(|v| v as f32).collect(), vec![2, 2, 3]);
        data.set_requires_grad(true);
        let index = Tensor::from_vec(
            vec![
                0.0, 1.0, 2.0, // (0, 0, :)
                0.0, 0.0, 0.0, // (0, 1, :)
                2.0, 2.0, 2.0, // (1, 0, :)
                1.0, 0.0, 1.0, // (1, 1, :)
            ],
            vec![2, 2, 3],
        );

        let gathered = data.gather(2, index);
        let loss = gathered.sum(vec![0, 1, 2], true);
        loss.backward();

        let grads = data.grad();
        assert!(approx_equal(grads[[0, 0, 0]], 1.0));
        assert!(approx_equal(grads[[0, 0, 1]], 1.0));
        assert!(approx_equal(grads[[0, 0, 2]], 1.0));
        assert!(approx_equal(grads[[0, 1, 0]], 3.0));
        assert!(approx_equal(grads[[0, 1, 1]], 0.0));
        assert!(approx_equal(grads[[0, 1, 2]], 0.0));
        assert!(approx_equal(grads[[1, 0, 0]], 0.0));
        assert!(approx_equal(grads[[1, 0, 1]], 0.0));
        assert!(approx_equal(grads[[1, 0, 2]], 3.0));
        assert!(approx_equal(grads[[1, 1, 0]], 1.0));
        assert!(approx_equal(grads[[1, 1, 1]], 2.0));
        assert!(approx_equal(grads[[1, 1, 2]], 0.0));
    }

    #[test]
    fn gather_backward_leaves_unselected_elements_at_zero() {
        zero_all_grads();
        let mut data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        data.set_requires_grad(true);
        let index = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);

        let gathered = data.gather(1, index);
        let loss = gathered.sum(vec![0, 1], true);
        loss.backward();

        let grads = data.grad();
        assert!(approx_equal(grads[[0, 0]], 2.0));
        assert!(approx_equal(grads[[0, 1]], 0.0));
        assert!(approx_equal(grads[[1, 0]], 2.0));
        assert!(approx_equal(grads[[1, 1]], 0.0));
    }

    #[test]
    fn gather_backward_one_dimensional_tensor_accumulates_duplicates() {
        zero_all_grads();
        let mut data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        data.set_requires_grad(true);
        let index = Tensor::from_vec(vec![1.0, 1.0, 3.0, 3.0], vec![4]);

        let gathered = data.gather(0, index);
        let loss = gathered.sum(vec![0], true);
        loss.backward();

        let grads = data.grad();
        assert!(approx_equal(grads[[0]], 0.0));
        assert!(approx_equal(grads[[1]], 2.0));
        assert!(approx_equal(grads[[2]], 0.0));
        assert!(approx_equal(grads[[3]], 2.0));
    }

    #[test]
    #[should_panic]
    fn gather_panics_when_dimension_out_of_range() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let index = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        let _ = data.gather(2, index);
    }

    #[test]
    #[should_panic]
    fn gather_panics_when_index_rank_differs_from_input() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let index = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2, 1]);
        let _ = data.gather(0, index);
    }

    #[test]
    fn gather_respects_row_indices() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let index = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![3, 2]);

        let gathered = data.gather(0, index);
        let result = gathered.item();

        assert_eq!(result.shape(), &[3, 2]);
        assert!(approx_equal(result[[0, 0]], 3.0));
        assert!(approx_equal(result[[0, 1]], 2.0));
        assert!(approx_equal(result[[1, 0]], 1.0));
        assert!(approx_equal(result[[1, 1]], 2.0));
        assert!(approx_equal(result[[2, 0]], 1.0));
        assert!(approx_equal(result[[2, 1]], 2.0));
    }

    #[test]
    fn gather_respects_column_indices() {
        zero_all_grads();
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let index = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![3, 2]);

        let gathered = data.gather(1, index);
        let result = gathered.item();

        assert_eq!(result.shape(), &[3, 2]);
        assert!(approx_equal(result[[0, 0]], 2.0));
        assert!(approx_equal(result[[0, 1]], 1.0));
        assert!(approx_equal(result[[1, 0]], 3.0));
        assert!(approx_equal(result[[1, 1]], 3.0));
        assert!(approx_equal(result[[2, 0]], 5.0));
        assert!(approx_equal(result[[2, 1]], 5.0));
    }

    #[test]
    fn gather_handles_three_dimensional_inputs() {
        zero_all_grads();
        let data = Tensor::from_vec((1..=12).map(|v| v as f32).collect(), vec![2, 2, 3]);
        let index = Tensor::from_vec(
            vec![0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            vec![2, 2, 3],
        );

        let gathered = data.gather(2, index);
        let result = gathered.item();
        let expected = vec![
            1.0, 2.0, 3.0, 6.0, 5.0, 4.0, 8.0, 8.0, 8.0, 10.0, 10.0, 10.0,
        ];

        assert_eq!(result.shape(), &[2, 2, 3]);
        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!(approx_equal(*actual, *expected));
        }
    }

    #[test]
    fn gather_backward_row_dimension_scatter_adds() {
        zero_all_grads();
        let mut data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        data.set_requires_grad(true);
        let index = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]);

        let gathered = data.gather(0, index);
        let loss = gathered.sum(vec![0, 1], true);
        loss.backward();
        println!("{:?}", loss.item());

        let grads = data.grad();
        assert!(approx_equal(grads[[0, 0]], 1.0));
        assert!(approx_equal(grads[[0, 1]], 1.0));
        assert!(approx_equal(grads[[1, 0]], 1.0));
        assert!(approx_equal(grads[[1, 1]], 1.0));
    }

    #[test]
    fn gather_backward_accumulates_duplicate_indices() {
        zero_all_grads();
        let mut data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        data.set_requires_grad(true);
        let index = Tensor::from_vec(vec![0.0, 0.0, 2.0, 1.0, 1.0, 1.0], vec![2, 3]);

        let gathered = data.gather(1, index);
        let loss = gathered.sum(vec![0, 1], true);
        loss.backward();

        let grads = data.grad();
        assert!(approx_equal(grads[[0, 0]], 2.0));
        assert!(approx_equal(grads[[0, 1]], 0.0));
        assert!(approx_equal(grads[[0, 2]], 1.0));
        assert!(approx_equal(grads[[1, 0]], 0.0));
        assert!(approx_equal(grads[[1, 1]], 3.0));
        assert!(approx_equal(grads[[1, 2]], 0.0));
    }
}
