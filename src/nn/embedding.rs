use crate::central::*;
use crate::nn::*;
use crate::utils::GGUFFile;

pub struct Embedding {
    pub weights: Tensor,
}

impl Embedding {
    pub fn new(vocab_size: usize, embeding_dimensions: usize) -> Embedding {
        let mut weights = Tensor::randn(Shape::new(vec![vocab_size, embeding_dimensions]));
        weights.set_requires_grad(true);
        weights.set_keep_alive(true);
        Embedding { weights }
    }

    pub fn from_tensor(mut weights: Tensor) -> Embedding {
        weights.set_keep_alive(true);
        Embedding { weights }
    }
}

impl Layer for Embedding {
    fn forward(&mut self, inputs: Tensor) -> Tensor {
        self.weights.select(inputs.id)
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        vec![self.weights.id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_embedding_creation() {
        let embedding = Embedding::new(100, 64);

        // Check weight shape is [vocab_size, embed_dim]
        assert_eq!(embedding.weights.shape.dimensions(), vec![100, 64]);

        // Check parameters returns the weight tensor ID
        let params = embedding.get_parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], embedding.weights.id);
    }

    #[test]
    fn test_embedding_forward_single_token() {
        let mut embedding = Embedding::new(5, 3);

        // Set known weights for predictable testing
        // Weight matrix: [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]]
        let weight_data = (0..15).map(|i| i as f32).collect();
        embedding.weights = Tensor::from_vec(weight_data, vec![5, 3]);

        // Select token 2 (should get row 2 of the weight matrix)
        let token_indices = Tensor::from_vec(vec![2.0], vec![1]);
        let result = embedding.forward(token_indices);

        // Expected: row 2 of weight matrix = [6, 7, 8]
        assert_eq!(result.shape.dimensions(), vec![1, 3]);
        let output = result.item();
        assert!(approx_equal(output[[0, 0]], 6.0, 1e-6));
        assert!(approx_equal(output[[0, 1]], 7.0, 1e-6));
        assert!(approx_equal(output[[0, 2]], 8.0, 1e-6));
    }

    #[test]
    fn test_embedding_forward_multiple_tokens() {
        let mut embedding = Embedding::new(4, 2);

        // Simple weight matrix: [[0,1], [2,3], [4,5], [6,7]]
        let weight_data = (0..8).map(|i| i as f32).collect();
        embedding.weights = Tensor::from_vec(weight_data, vec![4, 2]);

        // Select tokens [1, 0, 3]
        let token_indices = Tensor::from_vec(vec![1.0, 0.0, 3.0], vec![3]);
        let result = embedding.forward(token_indices);

        // Expected shape: [3, 2]
        assert_eq!(result.shape.dimensions(), vec![3, 2]);
        let output = result.item();

        // Token 1: row 1 = [2, 3]
        assert!(approx_equal(output[[0, 0]], 2.0, 1e-6));
        assert!(approx_equal(output[[0, 1]], 3.0, 1e-6));

        // Token 0: row 0 = [0, 1]
        assert!(approx_equal(output[[1, 0]], 0.0, 1e-6));
        assert!(approx_equal(output[[1, 1]], 1.0, 1e-6));

        // Token 3: row 3 = [6, 7]
        assert!(approx_equal(output[[2, 0]], 6.0, 1e-6));
        assert!(approx_equal(output[[2, 1]], 7.0, 1e-6));
    }

    #[test]
    fn test_embedding_forward_batch() {
        let mut embedding = Embedding::new(6, 2);

        // Simple weight matrix: [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11]]
        let weight_data = (0..12).map(|i| i as f32).collect();
        embedding.weights = Tensor::from_vec(weight_data, vec![6, 2]);

        // Batch of sequences: [[0, 1], [2, 5]] - shape [2, 2]
        let token_indices = Tensor::from_vec(vec![0.0, 1.0, 2.0, 5.0], vec![2, 2]);
        let result = embedding.forward(token_indices);

        // Expected shape: [2, 2, 2] (batch_size, seq_len, embed_dim)
        assert_eq!(result.shape.dimensions(), vec![2, 2, 2]);
        let output = result.item();

        // Batch 0, Token 0 (index 0): [0, 1]
        assert!(approx_equal(output[[0, 0, 0]], 0.0, 1e-6));
        assert!(approx_equal(output[[0, 0, 1]], 1.0, 1e-6));

        // Batch 0, Token 1 (index 1): [2, 3]
        assert!(approx_equal(output[[0, 1, 0]], 2.0, 1e-6));
        assert!(approx_equal(output[[0, 1, 1]], 3.0, 1e-6));

        // Batch 1, Token 0 (index 2): [4, 5]
        assert!(approx_equal(output[[1, 0, 0]], 4.0, 1e-6));
        assert!(approx_equal(output[[1, 0, 1]], 5.0, 1e-6));

        // Batch 1, Token 1 (index 5): [10, 11]
        assert!(approx_equal(output[[1, 1, 0]], 10.0, 1e-6));
        assert!(approx_equal(output[[1, 1, 1]], 11.0, 1e-6));
    }

    #[test]
    fn test_embedding_backward_pass() {
        use crate::central::zero_all_grads;

        let mut embedding = Embedding::new(4, 3);

        // Set known weights and enable gradients
        let weight_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        embedding.weights = Tensor::from_vec(weight_data, vec![4, 3]);
        embedding.weights.set_requires_grad(true);

        // Forward pass: select tokens [0, 2, 0] (token 0 appears twice)
        let token_indices = Tensor::from_vec(vec![0.0, 2.0, 0.0], vec![3]);
        let result = embedding.forward(token_indices);

        // Compute loss and backward pass
        let loss = result.sum(vec![0, 1], true);
        zero_all_grads();
        loss.backward();

        // Check that gradients exist and have correct shape
        let weight_grad = embedding.weights.grad();
        assert_eq!(weight_grad.shape(), &[4, 3]);

        // Token 0 was selected twice, so its gradient should be 2.0 for each dimension
        assert!(approx_equal(weight_grad[[0, 0]], 2.0, 1e-6));
        assert!(approx_equal(weight_grad[[0, 1]], 2.0, 1e-6));
        assert!(approx_equal(weight_grad[[0, 2]], 2.0, 1e-6));

        // Token 1 was never selected, so gradient should be 0
        assert!(approx_equal(weight_grad[[1, 0]], 0.0, 1e-6));
        assert!(approx_equal(weight_grad[[1, 1]], 0.0, 1e-6));
        assert!(approx_equal(weight_grad[[1, 2]], 0.0, 1e-6));

        // Token 2 was selected once, so gradient should be 1.0 for each dimension
        assert!(approx_equal(weight_grad[[2, 0]], 1.0, 1e-6));
        assert!(approx_equal(weight_grad[[2, 1]], 1.0, 1e-6));
        assert!(approx_equal(weight_grad[[2, 2]], 1.0, 1e-6));

        // Token 3 was never selected, so gradient should be 0
        assert!(approx_equal(weight_grad[[3, 0]], 0.0, 1e-6));
        assert!(approx_equal(weight_grad[[3, 1]], 0.0, 1e-6));
        assert!(approx_equal(weight_grad[[3, 2]], 0.0, 1e-6));
    }

    #[test]
    fn test_embedding_weight_initialization() {
        let embedding = Embedding::new(50, 32);
        let weight_data = embedding.weights.item();
        let weight_values: Vec<f32> = weight_data.iter().copied().collect();

        // Check that weights are not all zeros (should be random from Tensor::new)
        let all_zeros = weight_values.iter().all(|&x| x == 0.0);
        assert!(!all_zeros, "Embedding weights should not all be zero");

        // Check that we have reasonable diversity in values
        let unique_values: std::collections::HashSet<_> = weight_values
            .iter()
            .map(|&x| (x * 1000.0) as i32) // Round to 3 decimal places for uniqueness check
            .collect();
        assert!(
            unique_values.len() > 10,
            "Should have diverse weight values"
        );
    }

    #[test]
    fn test_embedding_different_input_shapes() {
        let mut embedding = Embedding::new(8, 4);

        // Set simple weights for testing
        let weight_data = (0..32).map(|i| i as f32).collect();
        embedding.weights = Tensor::from_vec(weight_data, vec![8, 4]);

        // Test 1D input (single sequence)
        let input_1d = Tensor::from_vec(vec![1.0, 3.0, 5.0], vec![3]);
        let output_1d = embedding.forward(input_1d);
        assert_eq!(output_1d.shape.dimensions(), vec![3, 4]);

        // Test 2D input (batch of sequences)
        let input_2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let output_2d = embedding.forward(input_2d);
        assert_eq!(output_2d.shape.dimensions(), vec![2, 2, 4]);

        // Test 3D input (batch of sequences with extra dimension)
        let input_3d =
            Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 2, 2]);
        let output_3d = embedding.forward(input_3d);
        assert_eq!(output_3d.shape.dimensions(), vec![2, 2, 2, 4]);
    }

    #[test]
    fn test_embedding_consistent_output() {
        let mut embedding = Embedding::new(10, 6);

        // Set fixed weights
        let weight_data = (0..60).map(|i| (i as f32) * 0.1).collect();
        embedding.weights = Tensor::from_vec(weight_data, vec![10, 6]);

        // Multiple calls with same input should give same output
        let token_indices = Tensor::from_vec(vec![3.0, 7.0, 1.0], vec![3]);

        let result1 = embedding.forward(token_indices.clone());
        let result2 = embedding.forward(token_indices);

        let output1 = result1.item();
        let output2 = result2.item();

        // Results should be identical
        for i in 0..3 {
            for j in 0..6 {
                assert!(approx_equal(output1[[i, j]], output2[[i, j]], 1e-10));
            }
        }
    }

    #[test]
    fn test_embedding_gpt2_sized() {
        // Test with GPT-2 like dimensions
        let vocab_size = 50257; // GPT-2 vocab size
        let embed_dim = 768; // GPT-2 small embedding dimension

        let embedding = Embedding::new(vocab_size, embed_dim);

        // Check dimensions
        assert_eq!(
            embedding.weights.shape.dimensions(),
            vec![vocab_size, embed_dim]
        );

        // Test forward pass with typical sequence
        let mut embedding = embedding;
        let seq_len = 128;
        let batch_size = 2;

        // Create sample token sequence
        let token_data: Vec<f32> = (0..(batch_size * seq_len))
            .map(|i| (i % vocab_size) as f32)
            .collect();
        let token_indices = Tensor::from_vec(token_data, vec![batch_size, seq_len]);

        let result = embedding.forward(token_indices);

        // Should output [batch_size, seq_len, embed_dim]
        assert_eq!(
            result.shape.dimensions(),
            vec![batch_size, seq_len, embed_dim]
        );
    }

    #[test]
    fn test_embedding_edge_cases() {
        // Test with minimal dimensions
        let mut embedding = Embedding::new(2, 1);
        embedding.weights = Tensor::from_vec(vec![10.0, 20.0], vec![2, 1]);

        // Single token
        let single_token = Tensor::from_vec(vec![1.0], vec![1]);
        let result = embedding.forward(single_token);
        assert_eq!(result.shape.dimensions(), vec![1, 1]);

        let output = result.item();
        assert!(approx_equal(output[[0, 0]], 20.0, 1e-6));

        // Empty-like input (this will depend on how select handles edge cases)
        // We'll test the smallest meaningful case instead
        let zero_token = Tensor::from_vec(vec![0.0], vec![1]);
        let result_zero = embedding.forward(zero_token);
        let output_zero = result_zero.item();
        assert!(approx_equal(output_zero[[0, 0]], 10.0, 1e-6));
    }

    #[test]
    fn test_embedding_backward_gradient_accumulation() {
        use crate::central::zero_all_grads;

        let mut embedding = Embedding::new(3, 2);

        // Set known weights: [[1,2], [3,4], [5,6]]
        embedding.weights = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        embedding.weights.set_requires_grad(true);

        // Test that gradients accumulate correctly when same token appears multiple times
        let token_indices = Tensor::from_vec(vec![1.0, 1.0, 2.0], vec![3]); // Token 1 appears twice
        let result = embedding.forward(token_indices);

        // Create a loss that gives different gradients to each position
        // Multiply by [1, 2, 3] so each occurrence contributes differently
        let multiplier = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let weighted_result = result * multiplier;
        let loss = weighted_result.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        let weight_grad = embedding.weights.grad();

        // Token 0: never selected, gradient should be 0
        assert!(approx_equal(weight_grad[[0, 0]], 0.0, 1e-6));
        assert!(approx_equal(weight_grad[[0, 1]], 0.0, 1e-6));

        // Token 1: selected at positions 0 and 1 with multipliers 1 and 2
        // Total gradient contribution: 1*1 + 2*1 = 3 for each dimension
        assert!(approx_equal(weight_grad[[1, 0]], 3.0, 1e-6));
        assert!(approx_equal(weight_grad[[1, 1]], 3.0, 1e-6));

        // Token 2: selected at position 2 with multiplier 3
        // Total gradient contribution: 3*1 = 3 for each dimension
        assert!(approx_equal(weight_grad[[2, 0]], 3.0, 1e-6));
        assert!(approx_equal(weight_grad[[2, 1]], 3.0, 1e-6));
    }

    #[test]
    fn test_embedding_backward_different_losses() {
        use crate::central::zero_all_grads;

        let mut embedding = Embedding::new(4, 3);

        // Set simple weights for easy verification
        let weight_data: Vec<f32> = (0..12).map(|i| (i + 1) as f32).collect();
        embedding.weights = Tensor::from_vec(weight_data, vec![4, 3]);
        embedding.weights.set_requires_grad(true);

        // Test different loss functions give different gradients
        let token_indices = Tensor::from_vec(vec![0.0, 2.0], vec![2]);

        // Test 1: Sum loss
        let result1 = embedding.forward(token_indices.clone());
        let loss1 = result1.sum(vec![0, 1], true);

        zero_all_grads();
        loss1.backward();
        let grad1 = embedding.weights.grad();

        // Save gradients for comparison
        let grad1_token0 = [grad1[[0, 0]], grad1[[0, 1]], grad1[[0, 2]]];
        let grad1_token2 = [grad1[[2, 0]], grad1[[2, 1]], grad1[[2, 2]]];

        // Test 2: Mean loss (should give different gradients)
        let result2 = embedding.forward(token_indices);
        let loss2 = result2.mean(vec![0, 1]);

        zero_all_grads();
        loss2.backward();
        let grad2 = embedding.weights.grad();

        // Mean loss should give different gradients than sum loss
        assert!(!approx_equal(grad2[[0, 0]], grad1_token0[0], 1e-6));
        assert!(!approx_equal(grad2[[2, 0]], grad1_token2[0], 1e-6));

        // But the gradients should still be non-zero and finite
        assert!(grad2[[0, 0]].is_finite() && grad2[[0, 0]] != 0.0);
        assert!(grad2[[2, 0]].is_finite() && grad2[[2, 0]] != 0.0);
    }

    #[test]
    fn test_embedding_backward_batch_processing() {
        use crate::central::zero_all_grads;

        let mut embedding = Embedding::new(5, 2);

        // Set weights: [[1,2], [3,4], [5,6], [7,8], [9,10]]
        embedding.weights = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![5, 2],
        );
        embedding.weights.set_requires_grad(true);

        // Batch input: [[0, 1], [2, 0]] - batch_size=2, seq_len=2
        // Token 0 appears at positions (0,0) and (1,1)
        // Token 1 appears at position (0,1)
        // Token 2 appears at position (1,0)
        let token_indices = Tensor::from_vec(vec![0.0, 1.0, 2.0, 0.0], vec![2, 2]);
        let result = embedding.forward(token_indices);

        // Simple sum loss
        let loss = result.sum(vec![0, 1, 2], true);

        zero_all_grads();
        loss.backward();

        let weight_grad = embedding.weights.grad();

        // Token 0: appears twice, gradient should be 2.0 for each dimension
        assert!(approx_equal(weight_grad[[0, 0]], 2.0, 1e-6));
        assert!(approx_equal(weight_grad[[0, 1]], 2.0, 1e-6));

        // Token 1: appears once, gradient should be 1.0 for each dimension
        assert!(approx_equal(weight_grad[[1, 0]], 1.0, 1e-6));
        assert!(approx_equal(weight_grad[[1, 1]], 1.0, 1e-6));

        // Token 2: appears once, gradient should be 1.0 for each dimension
        assert!(approx_equal(weight_grad[[2, 0]], 1.0, 1e-6));
        assert!(approx_equal(weight_grad[[2, 1]], 1.0, 1e-6));

        // Tokens 3 and 4: never selected, gradients should be 0
        assert!(approx_equal(weight_grad[[3, 0]], 0.0, 1e-6));
        assert!(approx_equal(weight_grad[[3, 1]], 0.0, 1e-6));
        assert!(approx_equal(weight_grad[[4, 0]], 0.0, 1e-6));
        assert!(approx_equal(weight_grad[[4, 1]], 0.0, 1e-6));
    }

    #[test]
    fn test_embedding_backward_chain_rule() {
        use crate::central::zero_all_grads;

        let mut embedding = Embedding::new(3, 4);

        // Set known weights
        let weight_data: Vec<f32> = (0..12).map(|i| (i + 1) as f32 * 0.1).collect();
        embedding.weights = Tensor::from_vec(weight_data, vec![3, 4]);
        embedding.weights.set_requires_grad(true);

        // Forward through embedding then through a simple transformation
        let token_indices = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let embeddings = embedding.forward(token_indices);

        // Apply a simple linear transformation: multiply by 2 and add 1
        let transformed = embeddings * 2.0 + 1.0;
        let loss = transformed.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        let weight_grad = embedding.weights.grad();

        // Since we multiplied by 2, the gradients should be 2.0 for each selected token
        // Token 0: not selected, gradient should be 0
        for j in 0..4 {
            assert!(approx_equal(weight_grad[[0, j]], 0.0, 1e-6));
        }

        // Token 1: selected once, gradient should be 2.0 due to chain rule
        for j in 0..4 {
            assert!(approx_equal(weight_grad[[1, j]], 2.0, 1e-6));
        }

        // Token 2: selected once, gradient should be 2.0 due to chain rule
        for j in 0..4 {
            assert!(approx_equal(weight_grad[[2, j]], 2.0, 1e-6));
        }
    }
}
