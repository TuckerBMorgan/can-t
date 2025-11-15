use crate::central::MAX_DIMS;

use super::{BackproagationPacket, Operation, Shape, Tensor, get_equation};
use std::collections::HashSet;

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let rank = dims.len();
    if rank == 0 {
        return vec![];
    }

    let mut strides = vec![1usize; rank];
    for axis in (0..rank - 1).rev() {
        strides[axis] = strides[axis + 1] * dims[axis + 1];
    }

    strides
}

pub(crate) fn permute_flat_data(data: &[f32], dims: &[usize], permutation: &[usize]) -> Vec<f32> {
    assert_eq!(
        dims.len(),
        permutation.len(),
        "permute_flat_data: permutation length must match rank"
    );

    let total_size = data.len();
    let rank = dims.len();

    let output_dims: Vec<usize> = permutation.iter().map(|&axis| dims[axis]).collect();
    let input_strides = compute_strides(dims);
    let output_strides = compute_strides(&output_dims);

    let mut result = vec![0.0f32; total_size];
    let mut coordinates = vec![0usize; rank];

    for (flat_index, value) in data.iter().enumerate() {
        let mut remainder = flat_index;
        for axis in 0..rank {
            let stride = input_strides[axis];
            let coord = if stride == 0 { 0 } else { remainder / stride };
            coordinates[axis] = coord;
            remainder -= coord * stride;
        }

        let mut new_flat_index = 0usize;
        for (axis, &source_axis) in permutation.iter().enumerate() {
            let coord_value = coordinates[source_axis];
            let stride = output_strides.get(axis).copied().unwrap_or(1);
            new_flat_index += coord_value * stride;
        }

        result[new_flat_index] = *value;
    }

    result
}

pub(crate) fn invert_permutation(permutation: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0usize; permutation.len()];
    for (position, &axis) in permutation.iter().enumerate() {
        inverse[axis] = position;
    }
    inverse
}

impl Tensor {
    fn all_unique(values: &[usize]) -> bool {
        let mut seen = HashSet::with_capacity(values.len());
        values.iter().all(|&v| seen.insert(v))
    }
    /// Reorders the tensor axes according to the provided permutation.
    pub fn permute(&self, permutation: Vec<usize>) -> Tensor {
        assert!(
            Tensor::all_unique(&permutation),
            "All axes in a permutation must be unique {:?}",
            permutation
        );

        let (source_dims, source_data) = {
            let equation = get_equation();
            let shape = equation.get_tensor_shape(self.id);
            let dims = shape.dimensions();
            let data = equation.get_data_flat_buffer(self.id).to_vec();
            (dims, data)
        };

        assert!(
            permutation.len() == source_dims.len(),
            "permute: permutation length {} must match tensor rank {}",
            permutation.len(),
            source_dims.len()
        );

        let output_dims: Vec<usize> = permutation.iter().map(|&axis| source_dims[axis]).collect();
        let output_shape = Shape::new(output_dims);
        let permuted_data = permute_flat_data(&source_data, &source_dims, &permutation);

        let mut stored_permutation = [0usize; MAX_DIMS];
        for (index, value) in permutation.iter().enumerate() {
            stored_permutation[index] = *value;
        }

        Tensor::create_tensor_data_and_shape_and_operation(
            output_shape,
            permuted_data,
            Operation::Permute(self.id, stored_permutation, permutation.len()),
        )
    }
}

pub fn backward_for_permute(packet: BackproagationPacket) {
    if let Operation::Permute(source, indices, count) = packet.operation {
        let permutation: Vec<usize> = indices.iter().copied().take(count).collect();
        let inverse = invert_permutation(&permutation);

        let incoming_shape = packet
            .equation
            .get_tensor_shape(packet.incoming_grad)
            .dimensions();

        let incoming_grad = packet
            .equation
            .get_grad_flat_buffer(packet.incoming_grad)
            .to_vec();

        let restored = permute_flat_data(&incoming_grad, &incoming_shape, &inverse);
        packet.equation.add_tensor_grad(source, restored);
    } else {
        panic!("Wrong operation called for backward_for_permute");
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Tensor, zero_all_grads};

    #[test]
    fn permute_forward_swap_axes() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data.clone(), vec![2, 3]);

        let permuted = tensor.permute(vec![1, 0]);
        assert_eq!(permuted.shape.dimensions(), vec![3, 2]);

        let result = permuted.item().into_raw_vec();
        let expected = vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn permute_forward_three_axes() {
        let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
        let tensor = Tensor::from_vec(data.clone(), vec![2, 3, 4]);

        let permuted = tensor.permute(vec![2, 0, 1]);
        assert_eq!(permuted.shape.dimensions(), vec![4, 2, 3]);

        let result = permuted.item().into_raw_vec();
        let mut expected = Vec::new();
        for axis2 in 0..4 {
            for axis0 in 0..2 {
                for axis1 in 0..3 {
                    let index = axis0 * (3 * 4) + axis1 * 4 + axis2;
                    expected.push(data[index]);
                }
            }
        }
        assert_eq!(result, expected);
    }

    #[test]
    fn permute_backward_propagates_grad() {
        let data = (0..12).map(|x| x as f32).collect::<Vec<f32>>();
        let mut tensor = Tensor::from_vec(data, vec![3, 4]);
        tensor.set_requires_grad(true);

        let permuted = tensor.permute(vec![1, 0]);
        let loss = permuted.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        let grad = tensor.grad();
        for i in 0..3 {
            for j in 0..4 {
                assert!((grad[[i, j]] - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[should_panic]
    fn permute_invalid_length_panics() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let _ = tensor.permute(vec![0]);
    }
}
