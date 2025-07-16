
use crate::central::*;
use crate::nn::*;
use crate::utils::GGUFFile;
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



pub fn sinusoidal_position_encoding(seq_len: usize, d_model: usize) -> Tensor {
    let mut pos_encoding = vec![0.0; seq_len * d_model];
    
    for pos in 0..seq_len {
        for i in (0..d_model).step_by(2) {
            let angle = pos as f32 / 10000.0_f32.powf(i as f32 / d_model as f32);
            pos_encoding[pos * d_model + i] = angle.sin();
            if i + 1 < d_model {
                pos_encoding[pos * d_model + i + 1] = angle.cos();
            }
        }
    }
    
    Tensor::from_vec(pos_encoding, vec![seq_len, d_model])
}

pub struct GPT2 {
    config: GPT2Config,
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<GPT2Block>,
    final_layer_norm: LayerNorm,
    final_head: Linear
}

impl GPT2 {
    pub fn new(config: GPT2Config) -> GPT2 {
        let blocks = (0..config.number_of_layers).map(|_|GPT2Block::new(config)).collect();

        GPT2 {
            wte: Embedding::new(config.vocab_size, config.embedding_dimensions),
            wpe: Embedding::new(config.number_of_positions, config.embedding_dimensions),
            blocks,
            final_layer_norm: LayerNorm::new(config.embedding_dimensions),
            final_head: Linear::new(config.embedding_dimensions, config.vocab_size, true),
            config
        }
    }

    pub fn from_gguf_file(gguf_file: &mut GGUFFile) {

        let wpe_tensor = Tensor::from_gguf_file(String::from("token_embd.weight"), gguf_file);
        let wte_tensor = Tensor::from_gguf_file(String::from("position_embd.weight"), gguf_file);
        let wpe = Embedding::from_tensor(wpe_tensor);
        let wte = Embedding::from_tensor(wte_tensor);

        let block_count_value = gguf_file.get_value(String::from("gpt2.block_count"));
        let block_count = block_count_value.as_i64().unwrap();

        for i in 0..block_count {
            let block = GPT2Block::from_gguf_file(gguf_file, i);
        }

        panic!("Sdasd");
    }
}

impl Model for GPT2 {

    fn forward(&mut self, input: Tensor) -> Tensor {

        let seq_length = input.shape.dimensions()[1];
        let position_ids = sinusoidal_position_encoding(seq_length, self.config.embedding_dimensions);        
        let token_embeddings = self.wte.forward(input);
        let positional_embeddings = self.wpe.forward(position_ids);
        let mut hidden_states = token_embeddings + positional_embeddings;

        for block in &mut self.blocks {
            hidden_states = block.forward(hidden_states);
        }

        hidden_states = self.final_layer_norm.forward(hidden_states);

        self.final_head.forward(hidden_states)

    }

    fn get_parameters(&self) -> Vec<TensorID> {
        let mut ids = vec![];

        ids.extend(self.wte.get_parameters());
        ids.extend(self.wpe.get_parameters());
        for block in &self.blocks {
            ids.extend(block.get_parameters());
        }

        ids.extend(self.final_layer_norm.get_parameters());
        ids.extend(self.final_head.get_parameters());
        ids
    }
}


#[cfg(test)]
mod tests {
    use crate::nn::{GPT2Config, GPT2};
    use crate::central::*;
    use crate::utils::GGUFFile;

    #[test]
    fn basic_test() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/gpt2/Gpt2-124M-F16.gguf"));
        let mut gpt2 = GPT2::from_gguf_file(&mut gguf_file);
        let mut gpt2 = GPT2::new(GPT2Config::gpt2_small());
    }
}