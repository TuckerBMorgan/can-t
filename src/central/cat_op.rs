use crate::central::*;

use super::{BackproagationPacket, Operation, Tensor};

impl Tensor {
    pub fn cat(&self, other: Tensor, dimension: usize) -> Tensor {
        assert!(
            dimension < self.shape.number_of_dimension(),
            "Trying to cat a tensor without the right number of dimensions",
        );
        assert!(
            self.shape.number_of_dimension() == other.shape.number_of_dimension(),
            "Cannot cat tensors with differing ranks",
        );

        let left_dims = self.shape.dimensions();
        let right_dims = other.shape.dimensions();

        for (idx, (lhs, rhs)) in left_dims.iter().zip(right_dims.iter()).enumerate() {
            if idx != dimension {
                assert!(
                    lhs == rhs,
                    "Mismatch in dimension {} while concatenating",
                    idx,
                );
            }
        }

        let mut new_dims = left_dims.clone();
        new_dims[dimension] += right_dims[dimension];
        let new_shape = Shape::new(new_dims);

        // The tensors are stored in a flattened layout. To concatenate efficiently we
        // compute how many contiguous "blocks" we need to copy from each input.
        // number_of_takes represents how many of these blocks exist when looking at
        // all dimensions preceding the concatenation axis.
        let mut number_of_takes = 1usize;
        for dim in &left_dims[..dimension] {
            number_of_takes *= *dim;
        }

        let mut trailing_product = 1usize;
        if dimension + 1 < left_dims.len() {
            for dim in &left_dims[dimension + 1..] {
                trailing_product *= *dim;
            }
        }

        // Each block we copy from the left/right tensor spans the size of the
        // concatenated axis multiplied by the size of all trailing dimensions.
        let left_take = left_dims[dimension] * trailing_product;
        let right_take = right_dims[dimension] * trailing_product;

        let left_data = get_equation().get_data_flat_buffer(self.id).to_vec();
        let right_data = get_equation().get_data_flat_buffer(other.id).to_vec();
        let mut output_data = Vec::with_capacity(new_shape.total_size());

        for block in 0..number_of_takes {
            let left_start = block * left_take;
            output_data.extend_from_slice(&left_data[left_start..left_start + left_take]);

            let right_start = block * right_take;
            output_data.extend_from_slice(&right_data[right_start..right_start + right_take]);
        }

        Tensor::create_tensor_data_and_shape_and_operation(
            new_shape,
            output_data,
            Operation::Cat(self.id, other.id, dimension),
        )
    }
}

pub fn backwards_for_cat(packet: BackproagationPacket) {
    if let Operation::Cat(left, right, dimension) = packet.operation {
        let left_shape = packet.equation.get_tensor_shape(left);
        let right_shape = packet.equation.get_tensor_shape(right);

        let left_dims = left_shape.dimensions();
        let right_dims = right_shape.dimensions();

        // Recreate the same bookkeeping as the forward pass so we can peel the
        // gradient buffer back into left/right sections.
        let mut number_of_takes = 1usize;
        for dim in &left_dims[..dimension] {
            number_of_takes *= *dim;
        }

        let mut trailing_product = 1usize;
        if dimension + 1 < left_dims.len() {
            for dim in &left_dims[dimension + 1..] {
                trailing_product *= *dim;
            }
        }

        let left_take = left_dims[dimension] * trailing_product;
        let right_take = right_dims[dimension] * trailing_product;
        let combined_take = left_take + right_take;

        let incoming_grad = packet
            .equation
            .get_grad_flat_buffer(packet.incoming_grad)
            .to_vec();

        let mut left_grad = Vec::with_capacity(left_take * number_of_takes);
        let mut right_grad = Vec::with_capacity(right_take * number_of_takes);

        for block in 0..number_of_takes {
            let block_start = block * combined_take;
            let left_end = block_start + left_take;
            left_grad.extend_from_slice(&incoming_grad[block_start..left_end]);

            let right_end = left_end + right_take;
            right_grad.extend_from_slice(&incoming_grad[left_end..right_end]);
        }

        packet.equation.add_tensor_grad(left, left_grad);
        packet.equation.add_tensor_grad(right, right_grad);
    } else {
        panic!("backwards_for_cat called with wrong Operation");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn cat_forward_appends_along_axis() {
        let left = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let right = Tensor::from_vec(vec![5.0, 6.0], vec![1, 2]);
        let result = left.cat(right, 0);

        assert_eq!(result.shape.dimensions(), vec![3, 2]);
        let data = result.item().into_raw_vec();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn cat_backward_splits_gradients() {
        zero_all_grads();
        let mut left = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        left.set_requires_grad(true);
        let mut right = Tensor::from_vec(vec![5.0, 6.0], vec![1, 2]);
        right.set_requires_grad(true);

        let result = left.cat(right, 0);
        let loss = result.sum(vec![0, 1], true);
        loss.backward();

        for value in left.grad().iter() {
            assert!(approx_equal(*value, 1.0, 1e-6));
        }

        for value in right.grad().iter() {
            assert!(approx_equal(*value, 1.0, 1e-6));
        }
    }
}
