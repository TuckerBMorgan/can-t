use crate::central::{Shape, Tensor, TensorID, equation, get_equation};

use super::{BackproagationPacket, Operation};

impl Tensor {
    pub fn select(&self, indices_id: TensorID) -> Tensor {
        // We need to make sure that we the referenece to get_equation before we allocate the tensor(which has a call to get_equation)
        let (final_shape, result_data) = {
            // The shape is always shape of indices + self.shape[1:](all but the leading dimension, in a 1d dimenion we lose it fully)

            // Grab all the data we are going to need
            let equation = get_equation();

            // The shape and data of the indices we are selecting with
            let indices_shape = equation.get_tensor_shape(indices_id);
            let indices = equation.get_data_flat_buffer(indices_id);

            // Convert the indices into a usize, so we can use them as indices
            let as_indices: Vec<usize> = indices.iter().map(|x| *x as usize).collect();

            // The shape and data of the tensor we are selecting from
            let shape = equation.get_tensor_shape(self.id);
            let as_flat_buffer = equation.get_data_flat_buffer(self.id);

            // Dump the dimensions for indices into the vec we we use for final shape
            let shape_dimensions = shape.dimensions();
            let mut final_shape = vec![];
            for dim in indices_shape.dimensions() {
                final_shape.push(dim);
            }

            // Calculate the size of the thing we are selecting
            // ex: [2, 3, 4]
            // total_shape_of_single_select == 3 * 4
            let mut total_shape_of_single_select = 1;
            if shape_dimensions.len() != 1 {
                // We are only selecting the outer most dimension, so so we skip that when we calcuate this shape
                for dim in shape_dimensions.iter().skip(1) {
                    total_shape_of_single_select *= *dim;
                    // And while we are here we might as well polish off our shape
                    final_shape.push(*dim);
                }
            }

            // Preform the selection
            let mut result: Vec<&[f32]> = vec![];
            for index in as_indices {
                let anchor_point = index * total_shape_of_single_select;
                let final_element = anchor_point + total_shape_of_single_select;
                let subarray = &as_flat_buffer[anchor_point..final_element];
                result.push(subarray);
            }

            // Dump it all into the array we are using
            let mut result_data: Vec<f32> = vec![0.0; total_shape_of_single_select * indices.len()];
            let mut current_index = 0;
            for element in result {
                for single in element {
                    result_data[current_index] = *single;
                    current_index += 1;
                }
            }

            (final_shape, result_data)
        };
        let tensor = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(final_shape),
            result_data,
            Operation::Select(self.id, indices_id),
        );
        return tensor;
    }
}

/// Handles calculating and passing back the gradient of a select operation
/// For select operations, gradients are accumulated back to the source tensor
/// at the positions specified by the indices tensor
pub fn backward_for_select(backprop_packet: BackproagationPacket) {
    if let Operation::Select(source_id, indices_id) = backprop_packet.operation {
        // step 1 allocate a slab memory the size of the source_id grad
        // step 2 addset sections of it with the right grad
        // step 3 add that whole thing to the original grad
        let original_shape = backprop_packet.equation.get_tensor_shape(source_id);
        let total_size = original_shape.total_size();
        let mut grad_slab = vec![0.0; total_size];
        let indices = backprop_packet.equation.get_data_flat_buffer(indices_id);
        let incoming_grad_slab = backprop_packet
            .equation
            .get_grad_flat_buffer(backprop_packet.incoming_grad);

        // Caluclate how much of the information we need to copy over
        let mut total_shape_of_single_select = 1;
        if original_shape.dimensions().len() != 1 {
            // We are only selecting the outer most dimension, so so we skip that when we calcuate this shape
            for dim in original_shape.dimensions().iter().skip(1) {
                total_shape_of_single_select *= *dim;
            }
        }

        let indices: Vec<usize> = indices.iter().map(|x| *x as usize).collect();
        for (index, indices) in indices.iter().enumerate() {
            // Index is the index of the element within in indices
            // Indices is the index from the source id that we selected
            // So we want to look up incoming_grad[index] and then set

            let anchor_point = index * total_shape_of_single_select;
            let origin_anchor_point = *indices * total_shape_of_single_select;
            for i in 0..total_shape_of_single_select {
                grad_slab[origin_anchor_point + i] += incoming_grad_slab[anchor_point + i];
            }
        }

        backprop_packet
            .equation
            .add_tensor_grad(source_id, grad_slab);
    } else {
        panic!("Wrong operation for backward select");
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Shape, Tensor};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    fn basic_select_test() {
        // Create a simple embedding matrix [3, 2]
        let embedding = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            crate::central::Operation::Nop,
        );

        // Create indices tensor [2] with indices [0, 2]
        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2]),
            vec![0.0, 2.0],
            crate::central::Operation::Nop,
        );

        // Select using select method
        let result = embedding.select(indices.id);

        // Check result shape should be [2, 2]
        assert_eq!(result.shape.dimensions(), vec![2, 2]);

        // Check result values
        let result_data = result.item();
        // Expected: row 0 [1.0, 2.0] and row 2 [5.0, 6.0] from embedding

        // Check first selected row (index 0): [1.0, 2.0]
        assert!(approx_equal(result_data[[0, 0]], 1.0, 1e-6));
        assert!(approx_equal(result_data[[0, 1]], 2.0, 1e-6));

        // Check second selected row (index 2): [5.0, 6.0]
        assert!(approx_equal(result_data[[1, 0]], 5.0, 1e-6));
        assert!(approx_equal(result_data[[1, 1]], 6.0, 1e-6));
    }

    #[test]
    fn select_shape_test() {
        // Test with different shapes
        let embedding = Tensor::zeros(Shape::new(vec![10, 5])); // vocab_size=10, embedding_dim=5
        let indices = Tensor::zeros(Shape::new(vec![3, 4])); // batch_size=3, seq_len=4

        let result = embedding.select(indices.id);

        // Result should be [3, 4, 5]
        assert_eq!(result.shape.dimensions(), vec![3, 4, 5]);
    }

    #[test]
    fn select_direct_test() {
        // Test the select method directly
        let embedding = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            crate::central::Operation::Nop,
        );

        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![1]),
            vec![1.0],
            crate::central::Operation::Nop,
        );

        let result = embedding.select(indices.id);

        // Should get row 1: [4.0, 5.0, 6.0], result shape should be [1, 3]
        let result_data = result.item();
        assert_eq!(result.shape.dimensions(), vec![1, 3]);
        assert!(approx_equal(result_data[[0, 0]], 4.0, 1e-6));
        assert!(approx_equal(result_data[[0, 1]], 5.0, 1e-6));
        assert!(approx_equal(result_data[[0, 2]], 6.0, 1e-6));
    }

    // ========== MULTI-DIMENSIONAL SELECT TESTS ==========

    #[test]
    fn select_batch_embedding_test() {
        // Test batch embedding lookup like in batch_norm_simple_test
        // Embedding matrix: [vocab_size=5, embedding_dim=3]
        let embedding = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![5, 3]),
            vec![
                1.0, 2.0, 3.0, // Token 0 embedding
                4.0, 5.0, 6.0, // Token 1 embedding
                7.0, 8.0, 9.0, // Token 2 embedding
                10.0, 11.0, 12.0, // Token 3 embedding
                13.0, 14.0, 15.0, // Token 4 embedding
            ],
            crate::central::Operation::Nop,
        );

        // Batch indices: [batch_size=2, seq_len=3]
        // Like the pattern in batch_norm_simple_test: [BATCH_SIZE, 3]
        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![
                0.0, 1.0, 2.0, // Batch 0: tokens [0, 1, 2]
                3.0, 4.0, 0.0, // Batch 1: tokens [3, 4, 0]
            ],
            crate::central::Operation::Nop,
        );

        let result = embedding.select(indices.id);

        // Result shape should be [2, 3, 3] (batch_size, seq_len, embedding_dim)
        assert_eq!(result.shape.dimensions(), vec![2, 3, 3]);

        let result_data = result.item();

        // Check batch 0, position 0 (token 0): [1.0, 2.0, 3.0]
        assert!(approx_equal(result_data[[0, 0, 0]], 1.0, 1e-6));
        assert!(approx_equal(result_data[[0, 0, 1]], 2.0, 1e-6));
        assert!(approx_equal(result_data[[0, 0, 2]], 3.0, 1e-6));

        // Check batch 0, position 1 (token 1): [4.0, 5.0, 6.0]
        assert!(approx_equal(result_data[[0, 1, 0]], 4.0, 1e-6));
        assert!(approx_equal(result_data[[0, 1, 1]], 5.0, 1e-6));
        assert!(approx_equal(result_data[[0, 1, 2]], 6.0, 1e-6));

        // Check batch 1, position 0 (token 3): [10.0, 11.0, 12.0]
        assert!(approx_equal(result_data[[1, 0, 0]], 10.0, 1e-6));
        assert!(approx_equal(result_data[[1, 0, 1]], 11.0, 1e-6));
        assert!(approx_equal(result_data[[1, 0, 2]], 12.0, 1e-6));

        // Check batch 1, position 2 (token 0 again): [1.0, 2.0, 3.0]
        assert!(approx_equal(result_data[[1, 2, 0]], 1.0, 1e-6));
        assert!(approx_equal(result_data[[1, 2, 1]], 2.0, 1e-6));
        assert!(approx_equal(result_data[[1, 2, 2]], 3.0, 1e-6));
    }

    #[test]
    fn select_3d_indices_test() {
        // Test with 3D indices tensor
        let source = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            crate::central::Operation::Nop,
        );

        // 3D indices: [2, 2, 2]
        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2, 2]),
            vec![
                0.0, 1.0, // [0, 0, :] -> rows 0, 1
                2.0, 3.0, // [0, 1, :] -> rows 2, 3
                1.0, 0.0, // [1, 0, :] -> rows 1, 0
                3.0, 2.0, // [1, 1, :] -> rows 3, 2
            ],
            crate::central::Operation::Nop,
        );

        let result = source.select(indices.id);

        // Result shape should be [2, 2, 2, 2] (preserves indices shape + adds source's last dim)
        assert_eq!(result.shape.dimensions(), vec![2, 2, 2, 2]);

        let result_data = result.item();

        // Check [0, 0, 0, :] (index 0): [1.0, 2.0]
        assert!(approx_equal(result_data[[0, 0, 0, 0]], 1.0, 1e-6));
        assert!(approx_equal(result_data[[0, 0, 0, 1]], 2.0, 1e-6));

        // Check [0, 0, 1, :] (index 1): [3.0, 4.0]
        assert!(approx_equal(result_data[[0, 0, 1, 0]], 3.0, 1e-6));
        assert!(approx_equal(result_data[[0, 0, 1, 1]], 4.0, 1e-6));

        // Check [1, 1, 0, :] (index 3): [7.0, 8.0]
        assert!(approx_equal(result_data[[1, 1, 0, 0]], 7.0, 1e-6));
        assert!(approx_equal(result_data[[1, 1, 0, 1]], 8.0, 1e-6));
    }

    #[test]
    fn select_large_vocabulary_test() {
        // Test with larger embedding matrix like real NLP models
        let vocab_size = 8;
        let embedding_dim = 4;
        let batch_size = 3;
        let seq_len = 2;

        // Create embedding matrix [8, 4]
        let mut embedding_data = Vec::new();
        for i in 0..vocab_size {
            for j in 0..embedding_dim {
                embedding_data.push((i * embedding_dim + j) as f32);
            }
        }

        let embedding = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![vocab_size, embedding_dim]),
            embedding_data,
            crate::central::Operation::Nop,
        );

        // Batch indices [3, 2]
        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![batch_size, seq_len]),
            vec![
                0.0, 7.0, // Batch 0: first and last tokens
                3.0, 4.0, // Batch 1: middle tokens
                1.0, 6.0, // Batch 2: mixed tokens
            ],
            crate::central::Operation::Nop,
        );

        let result = embedding.select(indices.id);

        // Result shape should be [3, 2, 4]
        assert_eq!(result.shape.dimensions(), vec![3, 2, 4]);

        let result_data = result.item();

        // Check batch 0, pos 0 (token 0): [0, 1, 2, 3]
        for j in 0..embedding_dim {
            assert!(approx_equal(result_data[[0, 0, j]], j as f32, 1e-6));
        }

        // Check batch 0, pos 1 (token 7): [28, 29, 30, 31]
        for j in 0..embedding_dim {
            assert!(approx_equal(
                result_data[[0, 1, j]],
                (7 * embedding_dim + j) as f32,
                1e-6
            ));
        }

        // Check batch 1, pos 0 (token 3): [12, 13, 14, 15]
        for j in 0..embedding_dim {
            assert!(approx_equal(
                result_data[[1, 0, j]],
                (3 * embedding_dim + j) as f32,
                1e-6
            ));
        }
    }

    // ========== BACKWARD PASS TESTS ==========

    #[test]
    fn select_simple_backward_test() {
        // Test basic backward pass with select operation
        let embedding = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            crate::central::Operation::Nop,
        );

        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2]),
            vec![0.0, 2.0], // Select rows 0 and 2
            crate::central::Operation::Nop,
        );

        let result = embedding.select(indices.id);
        let loss = result.sum(vec![0, 1], true); // Sum all to scalar
        loss.backward();

        let gradients = embedding.grad();

        // Row 0 was selected once -> gradient = 1.0 for each element
        assert!(approx_equal(gradients[[0, 0]], 1.0, 1e-6));
        assert!(approx_equal(gradients[[0, 1]], 1.0, 1e-6));

        // Row 1 was never selected -> gradient = 0.0 for each element
        assert!(approx_equal(gradients[[1, 0]], 0.0, 1e-6));
        assert!(approx_equal(gradients[[1, 1]], 0.0, 1e-6));

        // Row 2 was selected once -> gradient = 1.0 for each element
        assert!(approx_equal(gradients[[2, 0]], 1.0, 1e-6));
        assert!(approx_equal(gradients[[2, 1]], 1.0, 1e-6));
    }

    #[test]
    fn select_repeated_indices_backward_test() {
        // Test backward pass when same index is selected multiple times
        let embedding = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            crate::central::Operation::Nop,
        );

        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3]),
            vec![0.0, 0.0, 1.0], // Select row 0 twice, row 1 once
            crate::central::Operation::Nop,
        );

        let result = embedding.select(indices.id);
        let loss = result.sum(vec![0, 1], true); // Sum all to scalar
        loss.backward();

        let gradients = embedding.grad();

        // Row 0 was selected twice -> gradients should accumulate = 2.0 each
        assert!(approx_equal(gradients[[0, 0]], 2.0, 1e-6));
        assert!(approx_equal(gradients[[0, 1]], 2.0, 1e-6));

        // Row 1 was selected once -> gradient = 1.0 each
        assert!(approx_equal(gradients[[1, 0]], 1.0, 1e-6));
        assert!(approx_equal(gradients[[1, 1]], 1.0, 1e-6));

        // Row 2 was never selected -> gradient = 0.0 each
        assert!(approx_equal(gradients[[2, 0]], 0.0, 1e-6));
        assert!(approx_equal(gradients[[2, 1]], 0.0, 1e-6));
    }

    #[test]
    fn select_batch_backward_test() {
        // Test backward pass with batch-like indices (2D)
        let embedding = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            crate::central::Operation::Nop,
        );

        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2]),
            vec![
                0.0, 1.0, // Batch 0: select rows 0, 1
                2.0, 0.0, // Batch 1: select rows 2, 0 (row 0 selected again)
            ],
            crate::central::Operation::Nop,
        );

        let result = embedding.select(indices.id); // Shape: [2, 2, 2]
        let loss = result.sum(vec![0, 1, 2], true); // Sum all to scalar
        loss.backward();

        let gradients = embedding.grad();

        // Row 0 selected twice (batch 0 pos 0, batch 1 pos 1) -> gradient = 2.0 each
        assert!(approx_equal(gradients[[0, 0]], 2.0, 1e-6));
        assert!(approx_equal(gradients[[0, 1]], 2.0, 1e-6));

        // Row 1 selected once (batch 0 pos 1) -> gradient = 1.0 each
        assert!(approx_equal(gradients[[1, 0]], 1.0, 1e-6));
        assert!(approx_equal(gradients[[1, 1]], 1.0, 1e-6));

        // Row 2 selected once (batch 1 pos 0) -> gradient = 1.0 each
        assert!(approx_equal(gradients[[2, 0]], 1.0, 1e-6));
        assert!(approx_equal(gradients[[2, 1]], 1.0, 1e-6));

        // Row 3 never selected -> gradient = 0.0 each
        assert!(approx_equal(gradients[[3, 0]], 0.0, 1e-6));
        assert!(approx_equal(gradients[[3, 1]], 0.0, 1e-6));
    }

    #[test]
    fn select_chained_operations_backward_test() {
        // Test select in chain with other operations
        let embedding = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            crate::central::Operation::Nop,
        );

        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2]),
            vec![0.0, 2.0],
            crate::central::Operation::Nop,
        );

        let scale = Tensor::element(Shape::new(vec![2, 2]), 2.0);

        // Chain: embedding.select(indices) * scale
        let selected = embedding.select(indices.id); // Shape: [2, 2]
        let scaled = selected * scale; // Element-wise multiply by 2.0
        let loss = scaled.sum(vec![0, 1], true); // Sum to scalar
        loss.backward();

        let embedding_gradients = embedding.grad();
        let scale_gradients = scale.grad();

        // Embedding: Each selected element gradient = 2.0 (from scale factor)
        // Row 0 selected -> gradient = 2.0 each
        assert!(approx_equal(embedding_gradients[[0, 0]], 2.0, 1e-6));
        assert!(approx_equal(embedding_gradients[[0, 1]], 2.0, 1e-6));

        // Row 1 not selected -> gradient = 0.0 each
        assert!(approx_equal(embedding_gradients[[1, 0]], 0.0, 1e-6));
        assert!(approx_equal(embedding_gradients[[1, 1]], 0.0, 1e-6));

        // Row 2 selected -> gradient = 2.0 each
        assert!(approx_equal(embedding_gradients[[2, 0]], 2.0, 1e-6));
        assert!(approx_equal(embedding_gradients[[2, 1]], 2.0, 1e-6));

        // Scale tensor: gradient = selected values element-wise
        // selected tensor: [[1.0, 2.0], [5.0, 6.0]]
        // So scale gradients should be: [[1.0, 2.0], [5.0, 6.0]]
        assert!(approx_equal(scale_gradients[[0, 0]], 1.0, 1e-6)); // First selected embedding value
        assert!(approx_equal(scale_gradients[[0, 1]], 2.0, 1e-6)); // Second selected embedding value
        assert!(approx_equal(scale_gradients[[1, 0]], 5.0, 1e-6)); // Third selected embedding value  
        assert!(approx_equal(scale_gradients[[1, 1]], 6.0, 1e-6)); // Fourth selected embedding value
    }

    #[test]
    fn select_complex_batch_backward_test() {
        // Test complex batch scenario like in real transformer models
        let vocab_size = 6;
        let embedding_dim = 3;
        let batch_size = 2;
        let seq_len = 3;

        // Create embedding matrix
        let mut embedding_data = Vec::new();
        for i in 0..vocab_size * embedding_dim {
            embedding_data.push(i as f32 + 1.0); // Values 1.0 to 18.0
        }

        let embedding = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![vocab_size, embedding_dim]),
            embedding_data,
            crate::central::Operation::Nop,
        );

        // Batch indices with some repetition
        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![batch_size, seq_len]),
            vec![
                0.0, 1.0, 0.0, // Batch 0: [0, 1, 0] - token 0 appears twice
                2.0, 3.0, 1.0, // Batch 1: [2, 3, 1] - token 1 appears again
            ],
            crate::central::Operation::Nop,
        );

        let result = embedding.select(indices.id); // Shape: [2, 3, 3]
        let loss = result.sum(vec![0, 1, 2], true); // Sum all to scalar
        loss.backward();

        let gradients = embedding.grad();

        // Token 0 appears 2 times -> each element gets gradient 2.0
        assert!(approx_equal(gradients[[0, 0]], 2.0, 1e-6)); // embedding[0][0]
        assert!(approx_equal(gradients[[0, 1]], 2.0, 1e-6)); // embedding[0][1]
        assert!(approx_equal(gradients[[0, 2]], 2.0, 1e-6)); // embedding[0][2]

        // Token 1 appears 2 times -> each element gets gradient 2.0
        assert!(approx_equal(gradients[[1, 0]], 2.0, 1e-6)); // embedding[1][0]
        assert!(approx_equal(gradients[[1, 1]], 2.0, 1e-6)); // embedding[1][1]
        assert!(approx_equal(gradients[[1, 2]], 2.0, 1e-6)); // embedding[1][2]

        // Token 2 appears 1 time -> each element gets gradient 1.0
        assert!(approx_equal(gradients[[2, 0]], 1.0, 1e-6)); // embedding[2][0]
        assert!(approx_equal(gradients[[2, 1]], 1.0, 1e-6)); // embedding[2][1]
        assert!(approx_equal(gradients[[2, 2]], 1.0, 1e-6)); // embedding[2][2]

        // Token 3 appears 1 time -> each element gets gradient 1.0
        assert!(approx_equal(gradients[[3, 0]], 1.0, 1e-6)); // embedding[3][0]
        assert!(approx_equal(gradients[[3, 1]], 1.0, 1e-6)); // embedding[3][1]
        assert!(approx_equal(gradients[[3, 2]], 1.0, 1e-6)); // embedding[3][2]

        // Tokens 4 and 5 never selected -> gradient 0.0
        assert!(approx_equal(gradients[[4, 0]], 0.0, 1e-6)); // embedding[4][0]
        assert!(approx_equal(gradients[[4, 1]], 0.0, 1e-6)); // embedding[4][1]
        assert!(approx_equal(gradients[[4, 2]], 0.0, 1e-6)); // embedding[4][2]
        assert!(approx_equal(gradients[[5, 0]], 0.0, 1e-6)); // embedding[5][0]
        assert!(approx_equal(gradients[[5, 1]], 0.0, 1e-6)); // embedding[5][1]
        assert!(approx_equal(gradients[[5, 2]], 0.0, 1e-6)); // embedding[5][2]
    }
}
