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
        let mut return_vec = vec![0.0;index.shape.total_size()];

        for p in look_ups {
            let qualified_offset : usize = linear_index(&p, &self.shape.dimensions()).unwrap();
            let new_index = offsets[qualified_offset] as usize;
            let mut data_index = p.clone();
            data_index[dimension] = new_index;
            // Calculate the offset
            let data_qualified_offset : usize = linear_index(&data_index, &self.shape.dimensions()).unwrap();
            return_vec[qualified_offset] = input_data[data_qualified_offset];
        }
        
        return Tensor::create_tensor_data_and_shape_and_operation(index.shape, return_vec, Operation::Gather(self.id, index.id, dimension));
    }
}


pub fn backward_for_gather(packet: BackproagationPacket)  {
    if let Operation::Gather(source, index, dimension) = packet.operation {
        let incoming_grad = packet.equation.get_grad_flat_buffer(packet.incoming_grad).to_vec();
        let source_tensor_size = packet.equation.get_tensor_shape(source);
        let mut new_grad = vec![0.0;source_tensor_size.total_size()];
        let index_shape = packet.equation.get_tensor_shape(index);
        let offsets = packet.equation.get_data_flat_buffer(index).to_vec();

        let look_ups = index_shape.generate_all_positions();

        for p in look_ups {
            let qualified_offset : usize = linear_index(&p, &source_tensor_size.dimensions()).unwrap();
            let new_index = offsets[qualified_offset] as usize;
            let mut data_index = p.clone();
            data_index[dimension] = new_index;
            // Calculate the offset
            let data_qualified_offset : usize = linear_index(&data_index, &source_tensor_size.dimensions()).unwrap();
            // Since the same index can show up more then once, we need to do an ADD here and not a SET
            new_grad[data_qualified_offset] += incoming_grad[qualified_offset];
        }

        packet.equation.add_tensor_grad(source, new_grad);
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{zero_all_grads, Tensor};

    fn approx_equal(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6
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
        let expected = vec![1.0, 2.0, 3.0, 6.0, 5.0, 4.0, 8.0, 8.0, 8.0, 10.0, 10.0, 10.0];

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
        let index = Tensor::from_vec(
            vec![0.0, 0.0, 2.0,
                 1.0, 1.0, 1.0],
            vec![2, 3],
        );

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
