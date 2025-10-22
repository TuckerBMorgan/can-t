use super::permute::{invert_permutation, permute_flat_data};
use super::{BackproagationPacket, Operation, Shape, Tensor, get_equation};

fn build_move_permutation(origin: usize, destination: usize, rank: usize) -> Vec<usize> {
    assert!(origin < rank, "movedim: origin index out of range");
    assert!(
        destination < rank,
        "movedim: destination index out of range"
    );

    let mut axes: Vec<usize> = (0..rank).collect();
    let axis = axes.remove(origin);
    axes.insert(destination, axis);
    axes
}

impl Tensor {
    /// Moves the dimension at `original_location` to `new_destination`.
    pub fn movedim(&self, original_location: usize, new_destination: usize) -> Tensor {
        let (source_dims, source_data) = {
            let equation = get_equation();
            let shape = equation.get_tensor_shape(self.id);
            let dims = shape.dimensions();
            let data = equation.get_data_flat_buffer(self.id).to_vec();
            (dims, data)
        };

        let rank = source_dims.len();
        let permutation = build_move_permutation(original_location, new_destination, rank);

        let mut output_dims = source_dims.clone();
        let moved_dimension = output_dims.remove(original_location);
        output_dims.insert(new_destination, moved_dimension);
        let output_shape = Shape::new(output_dims);

        let permuted_data = permute_flat_data(&source_data, &source_dims, &permutation);

        Tensor::create_tensor_data_and_shape_and_operation(
            output_shape,
            permuted_data,
            Operation::MoveDim(self.id, original_location, new_destination),
        )
    }
}

pub fn backward_for_movedim(packet: BackproagationPacket) {
    if let Operation::MoveDim(source, original, destination) = packet.operation {
        let source_shape = packet.equation.get_tensor_shape(source);
        let rank = source_shape.number_of_dimension();
        let permutation = build_move_permutation(original, destination, rank);
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
        panic!("Wrong operation called for backward_for_movedim");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::central::{Tensor, zero_all_grads};

    #[test]
    fn movedim_forward_move_last_to_front() {
        let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
        let tensor = Tensor::from_vec(data.clone(), vec![2, 3, 4]);

        let moved = tensor.movedim(2, 0);
        assert_eq!(moved.shape.dimensions(), vec![4, 2, 3]);

        let result = moved.item().into_raw_vec();
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
    fn movedim_forward_move_middle_to_end() {
        let data = (0..12).map(|x| x as f32).collect::<Vec<f32>>();
        let tensor = Tensor::from_vec(data.clone(), vec![2, 3, 2]);

        let moved = tensor.movedim(1, 2);
        assert_eq!(moved.shape.dimensions(), vec![2, 2, 3]);

        let mut expected = Vec::new();
        for axis0 in 0..2 {
            for axis2 in 0..2 {
                for axis1 in 0..3 {
                    let index = axis0 * (3 * 2) + axis1 * 2 + axis2;
                    expected.push(data[index]);
                }
            }
        }
        assert_eq!(moved.item().into_raw_vec(), expected);
    }

    #[test]
    fn movedim_backward_propagates_grad() {
        let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
        let mut tensor = Tensor::from_vec(data, vec![2, 3, 4]);
        tensor.set_requires_grad(true);

        let moved = tensor.movedim(2, 0);
        let loss = moved.sum(vec![0, 1, 2], true);

        zero_all_grads();
        loss.backward();

        let grad = tensor.grad();
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert!((grad[[i, j, k]] - 1.0).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn movedim_invalid_origin_panics() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let _ = tensor.movedim(2, 0);
    }
}
