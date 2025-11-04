use cant::central::*;
use cant::nn::*;
use cant::utils::GGUFFile;

pub struct ModelConfig {
    number_of_hidden_layers: usize,
    number_of_experts: usize,
    number_of_experts_per_token: usize,
    vocab_size: usize,
    hidden_layer_size: usize,
    intermediate_size: usize,
    swiglu_limit: f32,
    head_dimenesions: usize,
    number_of_attention_heads: usize,
    number_of_key_value_heads: usize,
    sliding_window: usize,
    initial_content_length: usize,
    rope_theta: f32,
    rope_scaling_factor: f32,
    rope_ntk_alpha: f32,
    rope_ntk_beta: f32,
    world_size: usize,
}

struct MLPBlock {
    number_of_experts: usize,
    number_of_experts_per_token: usize,
    swiglu_limit: f32,
    norm: RMSNorm,
    gate: Linear,
    mlp1_weight: Tensor,
    mlp1_bias: Tensor,
    mlp2_weight: Tensor,
    mlp2_bias: Tensor,
}

impl MLPBlock {
    pub fn new(config: &ModelConfig) -> MLPBlock {
        MLPBlock {
            number_of_experts: config.number_of_experts,
            number_of_experts_per_token: config.number_of_experts_per_token,
            swiglu_limit: config.swiglu_limit,
            // world_limit??
            norm: RMSNorm::new(config.hidden_layer_size),
            gate: Linear::new(config.number_of_experts, config.hidden_layer_size, true),
            mlp1_weight: Tensor::new(Shape::new(vec![
                config.number_of_experts,
                config.intermediate_size * 2,
                config.hidden_layer_size,
            ])),
            mlp1_bias: Tensor::new(Shape::new(vec![
                config.number_of_experts,
                config.intermediate_size * 2 / config.world_size,
            ])),
            mlp2_weight: Tensor::new(Shape::new(vec![
                config.number_of_experts,
                config.hidden_layer_size,
                config.intermediate_size / config.world_size,
            ])),
            mlp2_bias: Tensor::new(Shape::new(vec![
                config.number_of_experts,
                config.hidden_layer_size,
            ])),
        }
    }
}

impl Layer for MLPBlock {
    fn forward(&mut self, inputs: Tensor) -> Tensor {
        let t = self.norm.forward(inputs);
        let g = self.gate.forward(inputs);

        panic!("")
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        vec![]
    }
}

pub fn main() {}
