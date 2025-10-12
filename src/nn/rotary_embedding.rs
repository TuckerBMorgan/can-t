use crate::central::*;
use crate::nn::*;

const DEFAULT_ROTARY_BASE: f32 = 10000.0;

/// Implements rotary positional embeddings (RoPE) as used in GPT-style transformers.
/// Follows the semantics of the GPT-OSS reference, while caching cosine/sine tables on demand.
pub struct RotaryEmbedding {
    head_dim: usize,
    base: f32,
    max_seq_len_cached: usize,
    scaling_factor: f32,
    inv_freq: Vec<f32>,
    initial_context_length: usize,
    ntk_beta: f32,
    ntk_alpha: f32,
    cos_cache: Option<Tensor>,
    sin_cache: Option<Tensor>,
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
        let inv_freq = Self::build_inv_freq(head_dim, base);
        RotaryEmbedding {
            head_dim,
            base,
            scaling_factor: 1.0,
            max_seq_len_cached: 0,
            initial_context_length: 1,
            ntk_beta: 1.0,
            ntk_alpha: 1.0,
            inv_freq,
            cos_cache: None,
            sin_cache: None,
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
        /*
        let query = query.reshape();
        let query =
        */
        panic!("")
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

    /// Builds the inverse frequency vector used to parameterize the sinusoid frequencies.
    fn build_inv_freq(head_dim: usize, base: f32) -> Vec<f32> {
        let half_dim = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let exponent = (2 * i) as f32 / head_dim as f32;
            inv_freq.push(base.powf(-exponent));
        }
        inv_freq
    }
    /// Placeholder for gathering per-position cosine/sine slices.
    fn compute_cos_sin(&self, number_of_tokens: usize) -> (Tensor, Tensor) {
        let (concentration, inv_freq) = self.build_concentration_and_inv_freq();
        let t = Tensor::arange(0, number_of_tokens, 1);
        let result = t << inv_freq;
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
