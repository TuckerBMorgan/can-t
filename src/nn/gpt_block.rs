
use crate::central::*;
use crate::nn::*;
#[derive(Copy, Clone)]
pub struct GPT2Config {
    pub vocab_size: usize,
    pub embedding_dimensions: usize,
    pub number_of_layers: usize,
    pub number_of_heads: usize,
    pub number_of_positions: usize,
    pub dropout: f32,
    pub layer_norm_epsilon: f32
}

impl GPT2Config {
    pub fn gpt2_small() -> GPT2Config {
        GPT2Config {
            vocab_size: 50257,
            embedding_dimensions: 768,
            number_of_layers: 12,
            number_of_heads: 12,
            number_of_positions: 1024,
            dropout: 0.1, 
            layer_norm_epsilon: 1e-5
        }
    }
}


pub fn causal_mask(seq_len: usize) -> Tensor {
    let mut mask_data = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = 1.0; // Mask future positions
        }
    }
    // Create mask with shape [1, 1, seq_len, seq_len] for proper broadcasting
    // across batch and head dimensions in multi-head attention
    let mask = Tensor::from_vec(mask_data, vec![seq_len, seq_len]);
    mask.reshape(Shape::new(vec![1, 1, seq_len, seq_len]))
}

pub struct GPT2Block {
    layer_norm_1: LayerNorm,
    attention: MultiHeadAttention,
    layer_norm_2: LayerNorm,
    multi_layer_perceptron: Sequential
}

impl GPT2Block {
    pub fn new(config: GPT2Config) -> GPT2Block {
        let multi_layer_perceptron = Sequential::new(vec![
            Box::new(Linear::new(config.embedding_dimensions, 4 * config.embedding_dimensions, true)),
            Box::new(GELU::new()),
            Box::new(Linear::new(4 * config.embedding_dimensions, config.embedding_dimensions, true))
        ]);

        GPT2Block {
            layer_norm_1: LayerNorm::new(config.embedding_dimensions),
            attention: MultiHeadAttention::new(config.embedding_dimensions, config.number_of_heads, None),
            layer_norm_2: LayerNorm::new(config.embedding_dimensions),
            multi_layer_perceptron
        }
    }
}

impl Layer for GPT2Block {
    fn forward(&mut self, inputs: Tensor) -> Tensor {
        // Get the first layer norm out of the way
        let layer_norm_1 = self.layer_norm_1.forward(inputs);
        // Create the mask for this pass of the network
        let dimensions = inputs.shape.dimensions();
        let seq_len = dimensions[1];
        let mask = causal_mask(seq_len);
        self.attention.set_mask(mask);
 
        // attention and residual
        let attended = self.attention.forward(layer_norm_1);
        let inputs = inputs + attended;

        // Second layer norm and the MLP
        let layer_norm_2 = self.layer_norm_2.forward(inputs);
        let mlp: Tensor = self.multi_layer_perceptron.forward(layer_norm_2);
        // second residual
        inputs + mlp
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        let mut parameters =  vec![];
        parameters.extend(self.layer_norm_1.get_parameters());
        parameters.extend(self.layer_norm_2.get_parameters());
        parameters.extend(self.attention.get_parameters());
        parameters.extend(self.multi_layer_perceptron.get_parameters());
        return parameters;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_gpt2_config_creation() {
        let config = GPT2Config::gpt2_small();
        
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.embedding_dimensions, 768);
        assert_eq!(config.number_of_layers, 12);
        assert_eq!(config.number_of_heads, 12);
        assert_eq!(config.number_of_positions, 1024);
        assert!(approx_equal(config.dropout, 0.1, 1e-6));
        assert!(approx_equal(config.layer_norm_epsilon, 1e-5, 1e-10));
    }

    #[test]
    fn test_causal_mask_generation() {
        // Test small mask
        let mask = causal_mask(3);
        assert_eq!(mask.shape.dimensions(), vec![1, 1, 3, 3]);
        
        let mask_data = mask.item();
        
        // Lower triangle should be 0 (unmasked) - accessing with 4D indices
        assert!(approx_equal(mask_data[[0, 0, 0, 0]], 0.0, 1e-6));
        assert!(approx_equal(mask_data[[0, 0, 1, 0]], 0.0, 1e-6));
        assert!(approx_equal(mask_data[[0, 0, 1, 1]], 0.0, 1e-6));
        assert!(approx_equal(mask_data[[0, 0, 2, 0]], 0.0, 1e-6));
        assert!(approx_equal(mask_data[[0, 0, 2, 1]], 0.0, 1e-6));
        assert!(approx_equal(mask_data[[0, 0, 2, 2]], 0.0, 1e-6));
        
        // Upper triangle should be 1 (masked) - accessing with 4D indices
        assert!(approx_equal(mask_data[[0, 0, 0, 1]], 1.0, 1e-6));
        assert!(approx_equal(mask_data[[0, 0, 0, 2]], 1.0, 1e-6));
        assert!(approx_equal(mask_data[[0, 0, 1, 2]], 1.0, 1e-6));
    }

    #[test]
    fn test_gpt2_block_creation() {
        let config = GPT2Config::gpt2_small();
        let block = GPT2Block::new(config);
        
        // Check that all components are initialized
        let params = block.get_parameters();
        assert!(params.len() > 0, "Block should have parameters");
        
        // Verify we have parameters from all components
        // LayerNorm (2) + MultiHeadAttention + MLP (2 linear layers)
        assert!(params.len() >= 4, "Should have parameters from all components");
    }

    #[test]
    fn test_gpt2_block_forward_pass_shapes() {
        let config = GPT2Config {
            vocab_size: 1000,
            embedding_dimensions: 64,
            number_of_layers: 2,
            number_of_heads: 4,
            number_of_positions: 128,
            dropout: 0.1,
            layer_norm_epsilon: 1e-5
        };
        
        let mut block = GPT2Block::new(config);
        
        // Test single sequence
        let batch_size = 1;
        let seq_len = 8;
        let embed_dim = 64;
        
        let input_data: Vec<f32> = (0..(batch_size * seq_len * embed_dim))
            .map(|i| (i as f32) * 0.01)
            .collect();
        let input = Tensor::from_vec(input_data, vec![batch_size, seq_len, embed_dim]);
        
        let output = block.forward(input);
        
        // Output should maintain input shape
        assert_eq!(output.shape.dimensions(), vec![batch_size, seq_len, embed_dim]);
    }

    #[test]
    fn test_gpt2_block_forward_pass_batch() {
        let config = GPT2Config {
            vocab_size: 500,
            embedding_dimensions: 32,
            number_of_layers: 1,
            number_of_heads: 2,
            number_of_positions: 64,
            dropout: 0.0,
            layer_norm_epsilon: 1e-5
        };
        
        let mut block = GPT2Block::new(config);
        
        // Test batch processing
        let batch_size = 3;
        let seq_len = 5;
        let embed_dim = 32;
        
        let input_data: Vec<f32> = (0..(batch_size * seq_len * embed_dim))
            .map(|i| ((i % 100) as f32) * 0.02 - 1.0) // Range from -1 to 1
            .collect();
        let input = Tensor::from_vec(input_data, vec![batch_size, seq_len, embed_dim]);
        
        let output = block.forward(input);
        
        // Output should maintain batch input shape
        assert_eq!(output.shape.dimensions(), vec![batch_size, seq_len, embed_dim]);
        
        // Check that output values are finite and reasonable
        let output_data = output.item();
        for value in output_data.iter() {
            assert!(value.is_finite(), "Output should be finite");
            assert!(value.abs() < 100.0, "Output should be reasonable magnitude");
        }
    }

    #[test]
    fn test_gpt2_block_residual_connections() {
        let config = GPT2Config {
            vocab_size: 100,
            embedding_dimensions: 16,
            number_of_layers: 1,
            number_of_heads: 2,
            number_of_positions: 32,
            dropout: 0.0,
            layer_norm_epsilon: 1e-5
        };
        
        let mut block = GPT2Block::new(config);
        
        let batch_size = 1;
        let seq_len = 4;
        let embed_dim = 16;
        
        // Create input with known values
        let input_data = vec![1.0; batch_size * seq_len * embed_dim];
        let input = Tensor::from_vec(input_data, vec![batch_size, seq_len, embed_dim]);
        
        let output = block.forward(input.clone());
        
        // Due to residual connections, output should be related to input
        // but modified by attention and MLP
        assert_ne!(output.item(), input.item(), "Output should be different from input");
        
        // Output should have same shape
        assert_eq!(output.shape.dimensions(), input.shape.dimensions());
    }

    #[test]
    fn test_gpt2_block_different_sequence_lengths() {
        let config = GPT2Config {
            vocab_size: 200,
            embedding_dimensions: 24,
            number_of_layers: 1,
            number_of_heads: 3,
            number_of_positions: 16,
            dropout: 0.0,
            layer_norm_epsilon: 1e-5
        };
        
        let mut block = GPT2Block::new(config);
        
        // Test different sequence lengths
        let embed_dim = 24;
        let test_seq_lens = vec![1, 3, 7];
        
        for seq_len in test_seq_lens {
            let input_data: Vec<f32> = (0..(seq_len * embed_dim))
                .map(|i| (i as f32) * 0.1)
                .collect();
            let input = Tensor::from_vec(input_data, vec![1, seq_len, embed_dim]);
            
            let output = block.forward(input);
            assert_eq!(output.shape.dimensions(), vec![1, seq_len, embed_dim]);
        }
    }

    #[test]
    fn test_gpt2_block_parameter_count() {
        let config = GPT2Config {
            vocab_size: 1000,
            embedding_dimensions: 48,
            number_of_layers: 1,
            number_of_heads: 4,
            number_of_positions: 64,
            dropout: 0.1,
            layer_norm_epsilon: 1e-5
        };
        
        let block = GPT2Block::new(config);
        let params = block.get_parameters();
        
        // Should have parameters from:
        // - LayerNorm 1 (weight + bias = 2 tensors)
        // - LayerNorm 2 (weight + bias = 2 tensors) 
        // - MultiHeadAttention (query, key, value, output projections)
        // - MLP (2 linear layers with weights + biases = 4 tensors)
        
        assert!(params.len() >= 8, "Should have at least 8 parameter tensors");
        
        // All parameter IDs should be unique
        let mut unique_params = std::collections::HashSet::new();
        for param in &params {
            assert!(unique_params.insert(*param), "All parameter IDs should be unique");
        }
    }

    #[test]
    fn test_gpt2_block_consistency() {
        let config = GPT2Config {
            vocab_size: 300,
            embedding_dimensions: 20,
            number_of_layers: 1,
            number_of_heads: 2,
            number_of_positions: 8,
            dropout: 0.0,
            layer_norm_epsilon: 1e-5
        };
        
        let mut block1 = GPT2Block::new(config.clone());
        let mut block2 = GPT2Block::new(config);
        
        let batch_size = 1;
        let seq_len = 3;
        let embed_dim = 20;
        
        let input_data = vec![0.5; batch_size * seq_len * embed_dim];
        let input = Tensor::from_vec(input_data, vec![batch_size, seq_len, embed_dim]);
        
        let output1 = block1.forward(input.clone());
        let output2 = block2.forward(input);
        
        // Different blocks should produce different outputs (due to different random weights)
        assert_ne!(output1.item(), output2.item(), "Different blocks should produce different outputs");
        
        // But both should have correct shapes
        assert_eq!(output1.shape.dimensions(), vec![batch_size, seq_len, embed_dim]);
        assert_eq!(output2.shape.dimensions(), vec![batch_size, seq_len, embed_dim]);
    }
}