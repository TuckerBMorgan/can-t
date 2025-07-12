use crate::central::*;
use crate::nn::*;


pub struct MultiHeadAttention {
    // Parameters
    pub number_of_heads: usize,
    pub head_dimension: usize,
    pub scale: f32,
    
    // learnable parameters
    query_projection: Linear,
    key_projection: Linear,
    value_projection: Linear,

    // Attention
    scaled_dot_project_attention: ScaledDotProductAttention
}