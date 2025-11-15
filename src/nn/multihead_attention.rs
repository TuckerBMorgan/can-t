use crate::central::*;
use crate::nn::*;
use crate::utils::GGUFFile;

pub struct MultiHeadAttention {
    // Parameters
    pub number_of_heads: usize,
    pub head_dimension: usize,
    pub scale: f32,

    // learnable parameters
    query_projection: Linear,
    key_projection: Linear,
    value_projection: Linear,
    out_projections: Linear,

    // Attention
    scaled_dot_project_attention: ScaledDotProductAttention,

    // Mask
    mask: Option<Tensor>,
}

impl MultiHeadAttention {
    pub fn new(
        embed_dim: usize,
        number_of_heads: usize,
        mask: Option<Tensor>,
    ) -> MultiHeadAttention {
        assert_eq!(embed_dim % number_of_heads, 0);
        let head_dimension = embed_dim / number_of_heads;
        let scale = 1.0 / (head_dimension as f32).sqrt();

        MultiHeadAttention {
            number_of_heads,
            head_dimension,
            scale,
            query_projection: Linear::new(embed_dim, embed_dim, true),
            key_projection: Linear::new(embed_dim, embed_dim, true),
            value_projection: Linear::new(embed_dim, embed_dim, true),

            out_projections: Linear::new(embed_dim, embed_dim, true),

            scaled_dot_project_attention: ScaledDotProductAttention::new(scale),
            mask,
        }
    }

    pub fn from_gguf_file(
        gguf_file: &mut GGUFFile,
        block_count: i64,
        config: &GPT2Config,
    ) -> MultiHeadAttention {
        // ---- output projection (c_proj) ----
        let attn_output_weight = format!("blk.{}.attn_output.weight", block_count);
        let attn_output_bias = format!("blk.{}.attn_output.bias", block_count);
        let output_project =
            Linear::from_gguf_file(attn_output_weight, Some(attn_output_bias), gguf_file);

        // ---- fused QKV weight & bias (c_attn) ----
        let qkv: Tensor =
            Tensor::from_gguf_file(format!("blk.{}.attn_qkv.weight", block_count), gguf_file);
        let b: Tensor =
            Tensor::from_gguf_file(format!("blk.{}.attn_qkv.bias", block_count), gguf_file);

        let qkv_item = qkv.item();
        let b_item = b.item();
        assert!(b_item.ndim() == 1, "qkv bias should be 1-D");

        let total = b_item.shape()[0];
        assert!(total % 3 == 0, "qkv bias length not divisible by 3");
        let embed_dim = total / 3;

        // Decide split axis by matching out_features == bias length:
        // - If qkv.shape()[0] == 3*embed_dim  -> rows are out_features -> split axis 0 (HF Conv1D style)
        // - If qkv.shape()[1] == 3*embed_dim  -> cols are out_features -> split axis 1
        // - Else it's an unexpected layout.
        let shape = qkv_item.shape();
        let (q_mat, k_mat, v_mat) = if shape[0] == total {
            // Split rows: [3*E, E_in] -> [E, E_in] x3
            let (q_arr, kv_arr) = qkv_item.view().split_at(ndarray::Axis(0), embed_dim);
            let (k_arr, v_arr) = kv_arr.split_at(ndarray::Axis(0), embed_dim);
            (
                Tensor::from_vec(q_arr.to_owned().into_raw_vec(), q_arr.shape().to_vec()),
                Tensor::from_vec(k_arr.to_owned().into_raw_vec(), k_arr.shape().to_vec()),
                Tensor::from_vec(v_arr.to_owned().into_raw_vec(), v_arr.shape().to_vec()),
            )
        } else if shape.len() == 2 && shape[1] == total {
            // Split cols: [E_in, 3*E] -> [E_in, E] x3
            let (q_arr, kv_arr) = qkv_item.view().split_at(ndarray::Axis(1), embed_dim);
            let (k_arr, v_arr) = kv_arr.split_at(ndarray::Axis(1), embed_dim);
            (
                Tensor::from_vec(q_arr.to_owned().into_raw_vec(), q_arr.shape().to_vec()),
                Tensor::from_vec(k_arr.to_owned().into_raw_vec(), k_arr.shape().to_vec()),
                Tensor::from_vec(v_arr.to_owned().into_raw_vec(), v_arr.shape().to_vec()),
            )
        } else {
            panic!(
                "Unexpected QKV weight shape {:?} vs bias len {}",
                shape, total
            );
        };

        // Split bias the straightforward way (flat vector of length 3*E)
        let (q_b, kv_b) = b_item.view().split_at(ndarray::Axis(0), embed_dim);
        let (k_b, v_b) = kv_b.split_at(ndarray::Axis(0), embed_dim);

        let mut q_bias = Tensor::from_vec(q_b.to_owned().into_raw_vec(), vec![embed_dim]);
        q_bias.set_requires_grad(true);
        q_bias.set_keep_alive(true);
        let mut k_bias = Tensor::from_vec(k_b.to_owned().into_raw_vec(), vec![embed_dim]);
        k_bias.set_requires_grad(true);
        k_bias.set_keep_alive(true);
        let mut v_bias = Tensor::from_vec(v_b.to_owned().into_raw_vec(), vec![embed_dim]);
        v_bias.set_keep_alive(true);
        v_bias.set_requires_grad(true);

        // ---- heads & scaling ----
        assert!(
            embed_dim % config.number_of_heads == 0,
            "embed_dim must be divisible by n_heads"
        );
        let head_dimension = embed_dim / config.number_of_heads;

        // ❗ Correct scale is 1 / sqrt(head_dim), not sqrt(n_heads)
        let scale = 1.0f32 / (head_dimension as f32).sqrt();
        let mut q_mat = q_mat.transpose(0, 1);
        q_mat.set_requires_grad(true);
        q_mat.set_keep_alive(true);
        let mut k_mat = k_mat.transpose(0, 1);
        k_mat.set_requires_grad(true);
        k_mat.set_keep_alive(true);
        let mut v_mat = v_mat.transpose(0, 1);
        v_mat.set_requires_grad(true);
        v_mat.set_keep_alive(true);

        MultiHeadAttention {
            number_of_heads: config.number_of_heads,
            head_dimension,
            scale, // keep if you use it outside SDPA
            query_projection: Linear::from_tensors(q_mat, Some(q_bias)),
            key_projection: Linear::from_tensors(k_mat, Some(k_bias)),
            value_projection: Linear::from_tensors(v_mat, Some(v_bias)),
            out_projections: output_project,
            scaled_dot_project_attention: ScaledDotProductAttention::new(scale),
            mask: None,
        }
    }

    pub fn set_mask(&mut self, mask: Tensor) {
        self.mask = Some(mask);
    }
}

impl Layer for MultiHeadAttention {
    fn forward(&mut self, inputs: Tensor) -> Tensor {
        // Break out the dimensions to make out life easier
        let batch_size = inputs.shape.dimensions()[0];
        let sequence_length = inputs.shape.dimensions()[1];
        let embeding_dimensions = inputs.shape.dimensions()[2];

        // Calculate the Q, K, V
        let query = self.query_projection.forward(inputs);
        let key = self.key_projection.forward(inputs);
        let value = self.value_projection.forward(inputs);

        let query = query.reshape(Shape::new(vec![
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        ]));
        let key = key.reshape(Shape::new(vec![
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        ]));
        let value = value.reshape(Shape::new(vec![
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        ]));

        let query = query.transpose(1, 2);
        let key = key.transpose(1, 2);
        let value = value.transpose(1, 2);
        let attended = self
            .scaled_dot_project_attention
            .forward(query, key, value, self.mask);

        let attended = attended.transpose(1, 2);
        let attended = attended.reshape(Shape::new(vec![
            batch_size,
            sequence_length,
            embeding_dimensions,
        ]));

        self.out_projections.forward(attended)
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multihead_attention_constructor() {
        let embed_dim = 128;
        let num_heads = 8;
        let attention = MultiHeadAttention::new(embed_dim, num_heads, None);

        assert_eq!(attention.number_of_heads, 8);
        assert_eq!(attention.head_dimension, 16);
        assert_eq!(attention.scale, 1.0 / (16.0_f32).sqrt());
    }

    #[test]
    fn test_multihead_attention_forward_basic() {
        let embed_dim = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 8;

        let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);

        // Create input tensor [batch_size, seq_len, embed_dim]
        let input = Tensor::from_vec(
            vec![1.0; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );

        let result = attention.forward(input);

        // Output should have same shape as input
        assert_eq!(result.shape.dimensions(), &[batch_size, seq_len, embed_dim]);
    }

    #[test]
    fn test_multihead_attention_forward_different_sizes() {
        // Test various embed_dim and num_heads combinations
        let test_cases = vec![
            (128, 8),  // head_dim = 16
            (256, 16), // head_dim = 16
            (512, 8),  // head_dim = 64
            (64, 4),   // head_dim = 16
        ];

        for (embed_dim, num_heads) in test_cases {
            let batch_size = 1;
            let seq_len = 4;

            let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);
            let input = Tensor::from_vec(
                vec![1.0; batch_size * seq_len * embed_dim],
                vec![batch_size, seq_len, embed_dim],
            );

            let result = attention.forward(input);
            assert_eq!(result.shape.dimensions(), &[batch_size, seq_len, embed_dim]);
        }
    }

    #[test]
    fn test_multihead_attention_forward_with_mask() {
        let embed_dim = 64;
        let num_heads = 4;
        let batch_size = 1;
        let seq_len = 4;

        // Create a causal mask (lower triangular)
        let mask_data = vec![
            0.0, 1.0, 1.0, 1.0, // First position can only attend to itself
            0.0, 0.0, 1.0, 1.0, // Second position can attend to first two
            0.0, 0.0, 0.0, 1.0, // Third position can attend to first three
            0.0, 0.0, 0.0, 0.0, // Fourth position can attend to all
        ];
        let mask = Tensor::from_vec(mask_data, vec![seq_len, seq_len]);

        let mut attention = MultiHeadAttention::new(embed_dim, num_heads, Some(mask));
        let input = Tensor::from_vec(
            vec![1.0; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );

        let result = attention.forward(input);
        assert_eq!(result.shape.dimensions(), &[batch_size, seq_len, embed_dim]);
    }

    #[test]
    fn test_multihead_attention_forward_batch_processing() {
        let embed_dim = 128;
        let num_heads = 8;
        let batch_size = 4;
        let seq_len = 16;

        let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);
        let input = Tensor::from_vec(
            vec![1.0; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );

        let result = attention.forward(input);
        assert_eq!(result.shape.dimensions(), &[batch_size, seq_len, embed_dim]);
    }

    #[test]
    fn test_multihead_attention_forward_varying_inputs() {
        let embed_dim = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 6;

        let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);

        // Test with different input values
        let input1 = Tensor::from_vec(
            vec![1.0; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );
        let input2 = Tensor::from_vec(
            vec![2.0; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );

        let result1 = attention.forward(input1);
        let result2 = attention.forward(input2);

        // Both should have correct output shape
        assert_eq!(
            result1.shape.dimensions(),
            &[batch_size, seq_len, embed_dim]
        );
        assert_eq!(
            result2.shape.dimensions(),
            &[batch_size, seq_len, embed_dim]
        );
    }

    #[test]
    fn test_multihead_attention_forward_sequence_lengths() {
        let embed_dim = 64;
        let num_heads = 4;
        let batch_size = 2;

        // Test different sequence lengths
        let seq_lengths = vec![1, 4, 8, 16, 32];

        for seq_len in seq_lengths {
            let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);
            let input = Tensor::from_vec(
                vec![1.0; batch_size * seq_len * embed_dim],
                vec![batch_size, seq_len, embed_dim],
            );

            let result = attention.forward(input);
            assert_eq!(result.shape.dimensions(), &[batch_size, seq_len, embed_dim]);
        }
    }

    #[test]
    #[should_panic]
    fn test_multihead_attention_invalid_embed_dim() {
        // embed_dim not divisible by num_heads should panic
        let _attention = MultiHeadAttention::new(65, 4, None);
    }

    #[test]
    fn test_multihead_attention_head_dimension_calculation() {
        let test_cases = vec![
            (128, 8, 16),  // 128/8 = 16
            (256, 16, 16), // 256/16 = 16
            (512, 8, 64),  // 512/8 = 64
            (96, 6, 16),   // 96/6 = 16
        ];

        for (embed_dim, num_heads, expected_head_dim) in test_cases {
            let attention = MultiHeadAttention::new(embed_dim, num_heads, None);
            assert_eq!(attention.head_dimension, expected_head_dim);
            assert_eq!(attention.scale, 1.0 / (expected_head_dim as f32).sqrt());
        }
    }

    #[test]
    fn test_multihead_attention_backward_basic() {
        use crate::central::zero_all_grads;

        // Use very small dimensions to avoid overflow issues
        let embed_dim = 8;
        let num_heads = 2;
        let batch_size = 1;
        let seq_len = 2;

        let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);

        // Create input tensor with requires_grad = true
        let mut input = Tensor::from_vec(
            vec![0.1; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );
        input.set_requires_grad(true);

        let result = attention.forward(input.clone());
        let loss = result.sum(vec![0, 1, 2], true);

        zero_all_grads();
        loss.backward();

        // Check that gradients exist and have correct shape
        let input_grad = input.grad();
        assert_eq!(input_grad.shape(), &[batch_size, seq_len, embed_dim]);
    }

    #[test]
    fn test_multihead_attention_backward_shape_only() {
        use crate::central::zero_all_grads;

        // Test different configurations but only verify shapes, not gradient values
        let test_cases = vec![
            (8, 2, 1, 2),  // Very small case
            (16, 4, 1, 2), // Slightly larger
        ];

        for (embed_dim, num_heads, batch_size, seq_len) in test_cases {
            let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);
            let mut input = Tensor::from_vec(
                vec![0.1; batch_size * seq_len * embed_dim],
                vec![batch_size, seq_len, embed_dim],
            );
            input.set_requires_grad(true);

            let result = attention.forward(input.clone());
            let loss = result.sum(vec![0, 1, 2], true);

            zero_all_grads();
            loss.backward();

            let input_grad = input.grad();

            // Just verify shape correctness
            assert_eq!(input_grad.shape(), &[batch_size, seq_len, embed_dim]);
        }
    }

    #[test]
    fn test_multihead_attention_backward_without_crash() {
        use crate::central::zero_all_grads;

        // Minimal test just to ensure backward pass doesn't crash
        let embed_dim = 4;
        let num_heads = 2;
        let batch_size = 1;
        let seq_len = 1;

        let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);
        let mut input = Tensor::from_vec(
            vec![0.1; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );
        input.set_requires_grad(true);

        let result = attention.forward(input.clone());
        let loss = result.sum(vec![0, 1, 2], true);

        zero_all_grads();
        loss.backward();

        // Test passes if we get here without panicking
        assert!(true, "Backward pass completed without crashing");
    }

    #[test]
    fn test_multihead_attention_backward_step_up_1() {
        use crate::central::zero_all_grads;

        // Gradually increase size - step 1
        let embed_dim = 12;
        let num_heads = 3;
        let batch_size = 1;
        let seq_len = 2;

        let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);
        let mut input = Tensor::from_vec(
            vec![0.1; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );
        input.set_requires_grad(true);

        let result = attention.forward(input.clone());
        let loss = result.sum(vec![0, 1, 2], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();
        assert_eq!(input_grad.shape(), &[batch_size, seq_len, embed_dim]);
    }

    #[test]
    fn test_multihead_attention_backward_step_up_2() {
        use crate::central::zero_all_grads;

        // Gradually increase size - step 2
        let embed_dim = 16;
        let num_heads = 4;
        let batch_size = 1;
        let seq_len = 2;

        let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);
        let mut input = Tensor::from_vec(
            vec![0.1; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );
        input.set_requires_grad(true);

        let result = attention.forward(input.clone());
        let loss = result.sum(vec![0, 1, 2], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();
        assert_eq!(input_grad.shape(), &[batch_size, seq_len, embed_dim]);
    }

    #[test]
    fn test_multihead_attention_backward_step_up_3() {
        use crate::central::zero_all_grads;

        // Gradually increase size - step 3: larger seq_len
        let embed_dim = 16;
        let num_heads = 4;
        let batch_size = 1;
        let seq_len = 3;

        let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);
        let mut input = Tensor::from_vec(
            vec![0.1; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );
        input.set_requires_grad(true);

        let result = attention.forward(input.clone());
        let loss = result.sum(vec![0, 1, 2], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();
        assert_eq!(input_grad.shape(), &[batch_size, seq_len, embed_dim]);
    }

    #[test]
    fn test_multihead_attention_backward_step_up_4() {
        use crate::central::zero_all_grads;

        // Gradually increase size - step 4: add batch dimension
        let embed_dim = 16;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 3;

        let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);
        let mut input = Tensor::from_vec(
            vec![0.1; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );
        input.set_requires_grad(true);

        let result = attention.forward(input.clone());
        let loss = result.sum(vec![0, 1, 2], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();
        assert_eq!(input_grad.shape(), &[batch_size, seq_len, embed_dim]);
    }

    #[test]
    fn test_multihead_attention_backward_step_up_5() {
        use crate::central::zero_all_grads;

        // Gradually increase size - step 5: larger embed_dim
        let embed_dim = 32;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 3;

        let mut attention = MultiHeadAttention::new(embed_dim, num_heads, None);
        let mut input = Tensor::from_vec(
            vec![0.1; batch_size * seq_len * embed_dim],
            vec![batch_size, seq_len, embed_dim],
        );
        input.set_requires_grad(true);

        let result = attention.forward(input.clone());
        let loss = result.sum(vec![0, 1, 2], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();
        assert_eq!(input_grad.shape(), &[batch_size, seq_len, embed_dim]);
    }
}
