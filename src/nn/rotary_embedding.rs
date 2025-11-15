use crate::central::*;

const DEFAULT_ROTARY_BASE: f32 = 10000.0;

/// Implements rotary positional embeddings (RoPE) as used in GPT-style transformers.
/// Follows the semantics of the GPT-OSS reference, while caching cosine/sine tables on demand.
pub struct RotaryEmbedding {
    head_dim: usize,
    base: f32,
    _max_seq_len_cached: usize,
    scaling_factor: f32,
    initial_context_length: usize,
    ntk_beta: f32,
    ntk_alpha: f32,
    _cos_cache: Option<Tensor>,
    _sin_cache: Option<Tensor>,
}

impl RotaryEmbedding {
    /// Constructs a rotary embedding with the default GPT base (10000.0).
    pub fn new(head_dim: usize) -> Self {
        Self::with_base(head_dim, DEFAULT_ROTARY_BASE)
    }

    /// Constructs a rotary embedding with an explicit base.
    pub fn with_base(head_dim: usize, base: f32) -> Self {
        assert!(
            head_dim % 2 == 0,
            "RotaryEmbedding expects an even head dimension"
        );

        RotaryEmbedding {
            head_dim,
            base,
            scaling_factor: 1.0,
            _max_seq_len_cached: 0,
            initial_context_length: 1,
            ntk_beta: 1.0,
            ntk_alpha: 1.0,
            _cos_cache: None,
            _sin_cache: None,
        }
    }

    /// Applies rotary embeddings to query/key tensors (place-holder implementation).
    /// This mirrors the GPT-OSS API, but depends on tensor slicing and interleaving helpers
    /// that are not yet implemented in this repository. The function will panic until those
    /// helpers are in place.
    pub fn rotary_forward(&mut self, query: Tensor, key: Tensor) -> (Tensor, Tensor) {
        let number_of_tokens = query.shape.dimensions()[0];
        let (cos, sin) = self.compute_cos_sin(number_of_tokens);

        let query_shape = query.shape;
        let keep_dim = query_shape.dimensions()[1];
        let query = query.reshape(Shape::new(vec![number_of_tokens, keep_dim, self.head_dim]));
        let query = self.apply_rotary_embedding(query, cos, sin);
        let query = query.reshape(query_shape);

        let key_shape = key.shape;
        let keep_dim = key_shape.dimensions()[1];
        let key = key.reshape(Shape::new(vec![number_of_tokens, keep_dim, self.head_dim]));
        let key = self.apply_rotary_embedding(key, cos, sin);
        let key = key.reshape(key_shape);

        return (query, key);
    }

    fn build_concentration_and_inv_freq(&self) -> (Tensor, Tensor) {
        let mut freq = Vec::new();
        for i in (0..self.head_dim).step_by(2) {
            let exponent = (i as f32) / (self.head_dim as f32);
            freq.push(self.base.powf(exponent));
        }
        let freq_len = freq.len();
        let freq = Tensor::from_vec(freq, vec![freq_len]);
        // Common pathway
        if self.scaling_factor > 1.0 {
            let concentration = 0.1 * self.scaling_factor.ln() + 1.0;
            let d_half = self.head_dim / 2;
            let low = d_half as f32
                * (self.initial_context_length as f32 / (self.ntk_beta * 2.0 * 3.14)).ln()
                / self.base.ln();
            let high = d_half as f32
                * (self.initial_context_length as f32 / (self.ntk_alpha * 2.0 * 3.14)).ln()
                / self.base.ln();
            let ramp = (Tensor::arange(0, d_half, 1)) - low / (high / low);
            let mask = Tensor::element(Shape::new(vec![1]), 1.0) - ramp.clamp(0.0, 1.0);
            let interpolation =
                Tensor::element(Shape::new(vec![1]), 1.0) / (self.scaling_factor * freq);
            let extrapolation = Tensor::element(Shape::new(vec![1]), 1.0) / freq;
            let inverted_frequency = interpolation * (1.0 - mask) + extrapolation * mask;
            return (
                Tensor::element(Shape::new(vec![1]), concentration),
                inverted_frequency,
            );
        }

        return (
            Tensor::element(Shape::new(vec![1]), 1.0),
            Tensor::element(Shape::new(vec![1]), 1.0) / freq,
        );
    }

    /// Placeholder for gathering per-position cosine/sine slices.
    fn compute_cos_sin(&self, number_of_tokens: usize) -> (Tensor, Tensor) {
        let (concentration, inv_freq) = self.build_concentration_and_inv_freq();
        let t = Tensor::arange(0, number_of_tokens, 1);
        let result = t.unsqueeze(-1) << inv_freq.unsqueeze(0);
        let cos = result.cos() * concentration;
        let sin = result.sin() * concentration;
        return (cos, sin);
    }

    fn apply_rotary_embedding(&self, input: Tensor, cos: Tensor, sin: Tensor) -> Tensor {
        let cos = cos.unsqueeze(-2);
        let sin = sin.unsqueeze(-2);
        let chunks = input.chunk(input.shape.number_of_dimension() - 1, 2);

        let o1 = chunks[0] * cos - chunks[1] * sin;
        let o2 = chunks[1] * cos + chunks[0] * sin;
        return o1.cat(o2, cos.shape.number_of_dimension() - 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_cos_sin_returns_outer_product_shape() {
        let rotary = RotaryEmbedding::new(8);
        let (cos, sin) = rotary.compute_cos_sin(4);
        assert_eq!(cos.shape.dimensions(), vec![4, 4]);
        assert_eq!(sin.shape.dimensions(), vec![4, 4]);
    }

    #[test]
    fn apply_rotary_embedding_preserves_input_shape() {
        let rotary = RotaryEmbedding::new(8);
        let (cos, sin) = rotary.compute_cos_sin(4);
        let input = Tensor::from_vec((0..32).map(|v| v as f32).collect(), vec![4, 1, 8]);
        let result = rotary.apply_rotary_embedding(input, cos, sin);
        assert_eq!(result.shape.dimensions(), vec![4, 1, 8]);
    }

    #[test]
    fn apply_rotary_embedding_identity_cos_sin_no_change() {
        let rotary = RotaryEmbedding::new(4);
        let cos = Tensor::from_vec(vec![1.0, 1.0], vec![1, 2]);
        let sin = Tensor::from_vec(vec![0.0, 0.0], vec![1, 2]);
        let original_values = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec(original_values.clone(), vec![1, 1, 4]);
        let result = rotary.apply_rotary_embedding(input, cos, sin);
        assert_eq!(result.item().into_raw_vec(), original_values);
    }

    #[test]
    fn apply_rotary_embedding_quadrature_rotation() {
        let rotary = RotaryEmbedding::new(4);
        let cos = Tensor::from_vec(vec![0.0, 0.0], vec![1, 2]);
        let sin = Tensor::from_vec(vec![1.0, 1.0], vec![1, 2]);
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
        let result = rotary.apply_rotary_embedding(input, cos, sin);
        assert_eq!(result.item().into_raw_vec(), vec![-3.0, -4.0, 1.0, 2.0]);
    }
}
