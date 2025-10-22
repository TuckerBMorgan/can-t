use super::{BackproagationPacket, Operation, Shape, Tensor, get_equation};
impl Tensor {
    // Interesting things I have noticed
    // first and second dimensions seem to end up being sorted
    // and have to be unique
    pub fn diagonal(
        &self,
        offset: usize,
        first_dimension: usize,
        second_dimension: usize,
    ) -> Tensor {
        assert!(
            first_dimension != second_dimension,
            "first_dimension and second_dimension must be unique"
        );

        let (result_shape, result_data) = {
            let equation = get_equation();
            let shape = equation.get_tensor_shape(self.id);
            let dims = shape.dimensions();

            assert!(
                first_dimension < dims.len() && second_dimension < dims.len(),
                "dimension index out of bounds"
            );

            let first_dimension_size = dims[first_dimension];
            let second_dimension_size = dims[second_dimension];

            assert!(
                second_dimension_size > offset,
                "offset {} is out of bounds for dimension {} with size {}",
                offset,
                second_dimension,
                second_dimension_size
            );

            let diagonal_length = usize::min(first_dimension_size, second_dimension_size - offset);
            assert!(
                diagonal_length > 0,
                "requested diagonal produces zero length output"
            );

            let mut other_axes: Vec<usize> = (0..dims.len())
                .filter(|axis: &usize| *axis != first_dimension && *axis != second_dimension)
                .collect();
            other_axes.sort();

            let mut output_shape: Vec<usize> = other_axes.iter().map(|axis| dims[*axis]).collect();
            output_shape.push(diagonal_length);

            let mut strides = vec![1usize; dims.len()];
            if dims.len() > 1 {
                for i in (0..dims.len() - 1).rev() {
                    strides[i] = strides[i + 1] * dims[i + 1];
                }
            }

            let data = equation.get_data_flat_buffer(self.id);

            let total_output_size = output_shape.iter().product();
            let mut result = Vec::with_capacity(total_output_size);
            let mut index_buffer = vec![0usize; dims.len()];

            if other_axes.is_empty() {
                for diagonal_index in 0..diagonal_length {
                    index_buffer[first_dimension] = diagonal_index;
                    index_buffer[second_dimension] = diagonal_index + offset;
                    let flat_index = index_buffer
                        .iter()
                        .zip(strides.iter())
                        .map(|(idx, stride)| idx * stride)
                        .sum::<usize>();
                    result.push(data[flat_index]);
                }
            } else {
                let mut counters = vec![0usize; other_axes.len()];
                loop {
                    for (counter_index, axis) in other_axes.iter().enumerate() {
                        index_buffer[*axis] = counters[counter_index];
                    }

                    for diagonal_index in 0..diagonal_length {
                        index_buffer[first_dimension] = diagonal_index;
                        index_buffer[second_dimension] = diagonal_index + offset;
                        let flat_index = index_buffer
                            .iter()
                            .zip(strides.iter())
                            .map(|(idx, stride)| idx * stride)
                            .sum::<usize>();
                        result.push(data[flat_index]);
                    }

                    let mut incremented = false;
                    for axis_index in (0..other_axes.len()).rev() {
                        counters[axis_index] += 1;
                        if counters[axis_index] < dims[other_axes[axis_index]] {
                            incremented = true;
                            break;
                        } else {
                            counters[axis_index] = 0;
                        }
                    }

                    if !incremented {
                        break;
                    }
                }
            }

            (output_shape, result)
        };

        Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(result_shape),
            result_data,
            Operation::Diagonal(self.id, offset, first_dimension, second_dimension),
        )
    }
}

pub fn backwards_for_diagonal(backprop_backet: BackproagationPacket) {
    if let Operation::Diagonal(source, offset, first_dimension, second_dimension) =
        backprop_backet.operation
    {
        let equation = backprop_backet.equation;

        let source_shape = equation.get_tensor_shape(source);
        let dims = source_shape.dimensions();

        assert!(first_dimension < dims.len() && second_dimension < dims.len());
        assert!(first_dimension != second_dimension);

        let first_dimension_size = dims[first_dimension];
        let second_dimension_size = dims[second_dimension];
        assert!(offset < second_dimension_size);

        let diagonal_length = usize::min(first_dimension_size, second_dimension_size - offset);

        let mut other_axes: Vec<usize> = (0..dims.len())
            .filter(|axis| *axis != first_dimension && *axis != second_dimension)
            .collect();
        other_axes.sort();

        let mut expected_output_shape: Vec<usize> =
            other_axes.iter().map(|axis| dims[*axis]).collect();
        expected_output_shape.push(diagonal_length);

        let incoming_shape = equation
            .get_tensor_shape(backprop_backet.incoming_grad)
            .dimensions();
        assert_eq!(
            incoming_shape, expected_output_shape,
            "incoming gradient shape {:?} did not match expected {:?}",
            incoming_shape, expected_output_shape
        );

        let incoming_grad = equation
            .get_grad_flat_buffer(backprop_backet.incoming_grad)
            .to_vec();

        let mut source_grad = vec![0.0f32; source_shape.total_size()];

        let mut strides = vec![1usize; dims.len()];
        if dims.len() > 1 {
            for i in (0..dims.len() - 1).rev() {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
        }

        let mut index_buffer = vec![0usize; dims.len()];
        let mut grad_cursor = 0usize;

        if other_axes.is_empty() {
            for diagonal_index in 0..diagonal_length {
                index_buffer[first_dimension] = diagonal_index;
                index_buffer[second_dimension] = diagonal_index + offset;
                let flat_index = index_buffer
                    .iter()
                    .zip(strides.iter())
                    .map(|(idx, stride)| idx * stride)
                    .sum::<usize>();
                source_grad[flat_index] += incoming_grad[grad_cursor];
                grad_cursor += 1;
            }
        } else {
            let mut counters = vec![0usize; other_axes.len()];
            loop {
                for (counter_index, axis) in other_axes.iter().enumerate() {
                    index_buffer[*axis] = counters[counter_index];
                }

                for diagonal_index in 0..diagonal_length {
                    index_buffer[first_dimension] = diagonal_index;
                    index_buffer[second_dimension] = diagonal_index + offset;
                    let flat_index = index_buffer
                        .iter()
                        .zip(strides.iter())
                        .map(|(idx, stride)| idx * stride)
                        .sum::<usize>();
                    source_grad[flat_index] += incoming_grad[grad_cursor];
                    grad_cursor += 1;
                }

                let mut incremented = false;
                for axis_index in (0..other_axes.len()).rev() {
                    counters[axis_index] += 1;
                    if counters[axis_index] < dims[other_axes[axis_index]] {
                        incremented = true;
                        break;
                    } else {
                        counters[axis_index] = 0;
                    }
                }

                if !incremented {
                    break;
                }
            }
        }

        assert_eq!(grad_cursor, incoming_grad.len());

        equation.add_tensor_grad(source, source_grad);
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Tensor, zero_all_grads};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    fn diagonal_forward_2d_offset() {
        let data = (0..20).map(|x| x as f32).collect::<Vec<f32>>();
        let tensor = Tensor::from_vec(data.clone(), vec![4, 5]);

        let diagonal = tensor.diagonal(1, 0, 1);

        assert_eq!(diagonal.shape.dimensions(), vec![4]);

        let result = diagonal.item().into_raw_vec();
        let expected = vec![1.0, 7.0, 13.0, 19.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn diagonal_forward_zero_offset() {
        let tensor = Tensor::from_vec((0..9).map(|x| x as f32).collect(), vec![3, 3]);

        let diagonal = tensor.diagonal(0, 0, 1);

        assert_eq!(diagonal.shape.dimensions(), vec![3]);
        let result = diagonal.item().into_raw_vec();
        assert_eq!(result, vec![0.0, 4.0, 8.0]);
    }

    #[test]
    fn diagonal_forward_3d() {
        let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
        let tensor = Tensor::from_vec(data, vec![2, 3, 4]);

        let diagonal = tensor.diagonal(0, 1, 2);

        assert_eq!(diagonal.shape.dimensions(), vec![2, 3]);

        let result = diagonal.item().into_raw_vec();
        let expected = vec![0.0, 5.0, 10.0, 12.0, 17.0, 22.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn diagonal_forward_batched_axes() {
        let tensor = Tensor::from_vec((0..18).map(|x| x as f32).collect(), vec![2, 3, 3]);

        let diagonal = tensor.diagonal(0, 1, 2);

        assert_eq!(diagonal.shape.dimensions(), vec![2, 3]);
        let result = diagonal.item().into_raw_vec();
        assert_eq!(result, vec![0.0, 4.0, 8.0, 9.0, 13.0, 17.0]);
    }

    #[test]
    fn diagonal_backward() {
        let data = (0..20).map(|x| x as f32).collect::<Vec<f32>>();
        let mut tensor = Tensor::from_vec(data, vec![4, 5]);
        tensor.set_requires_grad(true);

        let diagonal = tensor.diagonal(1, 0, 1);
        let loss = diagonal.sum(vec![0], true);

        zero_all_grads();
        loss.backward();

        let grad = tensor.grad();
        for row in 0..4 {
            for col in 0..5 {
                let expected = if col == row + 1 { 1.0 } else { 0.0 };
                assert!(approx_equal(grad[[row, col]], expected, 1e-6));
            }
        }
    }

    #[test]
    fn diagonal_backward_batched() {
        let data = (0..40).map(|x| x as f32).collect::<Vec<f32>>();
        let mut tensor = Tensor::from_vec(data, vec![2, 4, 5]);
        tensor.set_requires_grad(true);

        let diagonal = tensor.diagonal(1, 1, 2);
        let loss = diagonal.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        let grad = tensor.grad();
        for batch in 0..2 {
            for row in 0..4 {
                for col in 0..5 {
                    let expected = if col == row + 1 { 1.0 } else { 0.0 };
                    assert!(approx_equal(grad[[batch, row, col]], expected, 1e-6));
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "offset")]
    fn diagonal_invalid_offset_panics() {
        let tensor = Tensor::from_vec((0..9).map(|x| x as f32).collect(), vec![3, 3]);
        let _ = tensor.diagonal(3, 0, 1);
    }
}
