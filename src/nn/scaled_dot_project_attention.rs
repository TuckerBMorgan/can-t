use crate::central::*;
use crate::nn::*;


pub struct ScaledDotProductAttention {
    scale: f32,
    dropout: f32
}

impl ScaledDotProductAttention {
    pub fn forward(&self, query: Tensor, key: Tensor, value: Tensor, mask: Option<Tensor>) -> Tensor {
        let key_length = key.shape.dimensions().len();
        let scores = (query << key.transpose(key_length - 2, key_length - 1)) / self.scale;
        let scores = match mask {
            Some(mask) => scores,
            None => scores
        };

        let weights = scores.softmax(scores.shape.dimensions().len()-1);
        return weights << value;
    }
}