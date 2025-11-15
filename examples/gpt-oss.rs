use cant::central::*;
use cant::nn::*;
use cant::utils::GGUFFile;

pub struct GPTOSSModelConfig {
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

pub struct AttentionBlockGPTOSS {
    head_dimensions: usize,
    number_of_attention_heads: usize,
    number_of_key_value_heads: usize,
    sliding_window: usize,
    sinks: Tensor,
    norm: RMSNorm,
    query_key_value_dimensions: usize,
    query_key_value: Linear,
    out_projection: Linear,
    sm_scale: f32,
    rope: RotaryEmbedding,
}

impl AttentionBlockGPTOSS {}

struct MLPBlockGPTOSS {
    number_of_experts: usize,
    number_of_experts_per_token: usize,
    swiglu_limit: f32,
    norm: RMSNorm,
    gate: Linear,
    mlp1_weight: Tensor,
    mlp1_bias: Tensor,
    mlp2_weight: Tensor,
    mlp2_bias: Tensor,
    swiglu: Swiglu,
}

impl MLPBlockGPTOSS {
    pub fn new(config: &GPTOSSModelConfig) -> MLPBlockGPTOSS {
        MLPBlockGPTOSS {
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
            swiglu: Swiglu::new(1.702, config.swiglu_limit),
        }
    }
}

impl Layer for MLPBlockGPTOSS {
    fn forward(&mut self, inputs: Tensor) -> Tensor {
        let t = self.norm.forward(inputs);
        let g = self.gate.forward(inputs);
        let g_last_dimension = g.shape.dimensions().len() - 1;
        let experts = g.topk(
            self.number_of_experts_per_token,
            g_last_dimension,
            true,
            true,
        );
        let expert_weights = experts.0.softmax(3);
        let expert_indices = experts.1;
        let mlp1_weights = self.mlp1_weight.select(expert_indices.id);
        let mlp1_bias = self.mlp1_bias.select(expert_indices.id);
        let t = Tensor::einsum("beck,bk->bec", vec![mlp1_weights, t]) + mlp1_bias;
        let t = self.swiglu.forward(t);

        let mlp2_weights = self.mlp2_weight.select(expert_indices.id);
        let mlp2_bias = self.mlp2_bias.select(expert_indices.id);
        let t = Tensor::einsum("beck,bek->bec", vec![mlp2_weights, t]);
        // There is a world size thing here

        let t = t + mlp2_bias;

        let t = Tensor::einsum("bec,be->bc", vec![t, expert_weights]);
        return inputs + t;
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        vec![]
    }
}

pub struct TransformerBlock {
    layer_index: usize,
    attention_layer: MultiHeadAttention,
    mlp: MLPBlockGPTOSS,
}
pub fn main() {}
