use crate::central::*;
use ndarray::{Axis, Slice};

use super::{BackproagationPacket, Operation, Tensor};

impl Tensor {
    pub fn chunk(&self, dimensions: usize, chunks: usize) -> Vec<Tensor> {
        assert!(chunks > 0, "Cannot chunk into zero pieces");
        let current_dimensions = self.shape.dimensions();
        assert!(
            dimensions < current_dimensions.len(),
            "Chunk dimension out of bounds"
        );

        let axis_length = current_dimensions[dimensions];
        assert!(
            chunks <= axis_length,
            "Cannot chunk dimension of length {} into {} pieces without empty chunks",
            axis_length,
            chunks
        );

        let base = axis_length / chunks;
        let remainder = axis_length % chunks;

        let data = self.item();
        let mut start_index = 0usize;
        let mut outputs = Vec::with_capacity(chunks);

        for chunk_idx in 0..chunks {
            let extra = if chunk_idx < remainder { 1 } else { 0 };
            let chunk_size = base + extra;

            let mut new_shape = current_dimensions.clone();
            new_shape[dimensions] = chunk_size;

            let chunk_slice = data
                .slice_axis(
                    Axis(dimensions),
                    Slice::from(start_index..start_index + chunk_size),
                )
                .to_owned();

            let tensor = Tensor::create_tensor_data_and_shape_and_operation(
                Shape::new(new_shape),
                chunk_slice.into_raw_vec(),
                Operation::Chunk(self.id, start_index, dimensions),
            );

            outputs.push(tensor);
            start_index += chunk_size;
        }

        outputs
    }
}

pub fn backward_for_chunk(packet: BackproagationPacket) {
    if let Operation::Chunk(from, start_index, dimension) = packet.operation {
        let source_shape = packet.equation.get_tensor_shape(from);
        let chunk_shape = packet.equation.get_tensor_shape(packet.incoming_grad);

        let chunk_grad = packet
            .equation
            .get_grad_flat_buffer(packet.incoming_grad)
            .to_vec();

        let mut accumulated = ndarray::ArrayD::from_shape_vec(
            source_shape.as_ndarray_shape(),
            vec![0.0; source_shape.total_size()],
        )
        .unwrap();

        let chunk_array =
            ndarray::ArrayD::from_shape_vec(chunk_shape.as_ndarray_shape(), chunk_grad).unwrap();

        let end_index = start_index + chunk_shape.dimensions()[dimension];
        accumulated
            .slice_axis_mut(Axis(dimension), Slice::from(start_index..end_index))
            .assign(&chunk_array);

        packet
            .equation
            .add_tensor_grad(from, accumulated.into_raw_vec());
    } else {
        panic!("Wrong operation for backward chunk");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn chunk_forward_splits_evenly() {
        let tensor = Tensor::from_vec((0..6).map(|v| v as f32).collect(), vec![3, 2]);
        let chunks = tensor.chunk(0, 3);

        assert_eq!(chunks.len(), 3);
        for (idx, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.shape.dimensions(), vec![1, 2]);
            let data = chunk.item().into_raw_vec();
            let expected_row = idx as f32;
            for (col, value) in data.iter().enumerate() {
                assert!(approx_equal(*value, expected_row * 2.0 + col as f32, 1e-6));
            }
        }
    }

    #[test]
    fn chunk_backward_injects_gradients_in_correct_slice() {
        zero_all_grads();
        let mut tensor = Tensor::from_vec((0..6).map(|v| v as f32).collect(), vec![3, 2]);
        tensor.set_requires_grad(true);

        let chunks = tensor.chunk(0, 3);
        let loss = chunks[1].sum(vec![0, 1], true);
        loss.backward();

        let grad = tensor.grad();
        for row in 0..3 {
            for col in 0..2 {
                let expected = if row == 1 { 1.0 } else { 0.0 };
                assert!(approx_equal(grad[[row, col]], expected, 1e-6));
            }
        }
    }
}
