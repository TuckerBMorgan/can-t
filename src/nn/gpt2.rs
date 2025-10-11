use crate::central::*;
use crate::nn::*;
use crate::utils::GGUFFile;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer};

#[derive(Copy, Clone)]
pub struct GPT2Config {
    pub vocab_size: usize, // The number of tokens, from tokenization of the corpus
    pub embedding_dimensions: usize, // the size of the learned representation of the vocab
    pub number_of_layers: usize, // The number of blocks
    pub number_of_heads: usize, // How many heads per Transofmer
    pub number_of_positions: usize, //
    pub dropout: f32,      // Rate of dropout(note used)
    pub layer_norm_epsilon: f32, // how much we add to the denomnators of part of layer norm to avoid a div by 0
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
            layer_norm_epsilon: 1e-5,
        }
    }

    pub fn from_gguf_file(gguf_file: &mut GGUFFile) -> GPT2Config {
        println!(
            "{:?}",
            gguf_file
                .get_value(String::from("tokenizer.ggml.tokens"))
                .as_array()
                .unwrap()
        );
        GPT2Config {
            vocab_size: 50257,
            embedding_dimensions: gguf_file
                .get_value(String::from("gpt2.embedding_length"))
                .as_u64()
                .unwrap() as usize,
            number_of_layers: gguf_file
                .get_value(String::from("gpt2.block_count"))
                .as_u64()
                .unwrap() as usize,
            number_of_heads: gguf_file
                .get_value(String::from("gpt2.attention.head_count"))
                .as_u64()
                .unwrap() as usize,
            number_of_positions: 1024, //gguf_file.get_value(String::from("vocab_size")).as_i64().unwrap() as usize,
            dropout: 0.1,
            layer_norm_epsilon: 1e-5,
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
    final_head: Linear,
}

impl GPT2 {
    pub fn new(config: GPT2Config) -> GPT2 {
        let blocks = (0..config.number_of_layers)
            .map(|_| GPT2Block::new(config))
            .collect();

        GPT2 {
            wte: Embedding::new(config.vocab_size, config.embedding_dimensions),
            wpe: Embedding::new(config.number_of_positions, config.embedding_dimensions),
            blocks,
            final_layer_norm: LayerNorm::new(config.embedding_dimensions),
            final_head: Linear::new(config.embedding_dimensions, config.vocab_size, true),
            config,
        }
    }

    pub fn from_gguf_file(gguf_file: &mut GGUFFile) -> GPT2 {
        let gpt_config = GPT2Config::gpt2_small();
        let wpe_tensor = Tensor::from_gguf_file(String::from("position_embd.weight"), gguf_file);
        let wte_tensor = Tensor::from_gguf_file(String::from("token_embd.weight"), gguf_file);
        let wpe: Embedding = Embedding::from_tensor(wpe_tensor);
        let wte = Embedding::from_tensor(wte_tensor);

        let block_count_value = gguf_file.get_value(String::from("gpt2.block_count"));
        let block_count = block_count_value.as_i64().unwrap();

        let mut blocks = vec![];

        for i in 0..block_count {
            let block = GPT2Block::from_gguf_file(gguf_file, i, &gpt_config);
            blocks.push(block);
        }

        let final_layer_norm = LayerNorm::from_gguf_file(
            gguf_file,
            String::from("output_norm.weight"),
            String::from("output_norm.bias"),
        );

        let wte_weights_reshapes_for_weight_tying = wte.weights.reshape(Shape::new(vec![
            gpt_config.vocab_size,
            gpt_config.embedding_dimensions,
        ]));
        let wte_weights_reshapes_for_weight_tying =
            wte_weights_reshapes_for_weight_tying.transpose(0, 1);
        let final_head = Linear::from_tensors(wte_weights_reshapes_for_weight_tying, None);

        GPT2 {
            config: gpt_config,
            wpe,
            wte,
            blocks,
            final_layer_norm,
            final_head,
        }
    }
}

impl Model for GPT2 {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // --- position ids: GPT-2 expects integer indices starting at 0 ---
        let seq_length = input.shape.dimensions()[1];
        // If your Tensor::from_vec needs i64 (HF uses int64), use that; adjust if your type differs.
        let position_ids: Vec<f32> = (0..seq_length).map(|x| x as f32).collect();
        let position_ids = Tensor::from_vec(position_ids, vec![1, seq_length]);

        // --- embeddings ---
        let token_embeddings = self.wte.forward(input);

        let positional_embeddings = self.wpe.forward(position_ids);

        // Sum embeddings (HF calls the tensor after dropout 'hidden_states'; in eval dropout is identity)
        let mut hidden_states = token_embeddings + positional_embeddings;

        // --- transformer blocks ---
        for (i, block) in self.blocks.iter_mut().enumerate() {
            hidden_states = block.forward(hidden_states);
        }

        hidden_states = self.final_layer_norm.forward(hidden_states);
        let logits = self.final_head.forward(hidden_states);
        logits
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
    use crate::central::*;
    use crate::nn::{GPT2, GPT2Config, Model};
    use crate::utils::GGUFFile;
    use ndarray::Axis;

    fn decode_gpt2_tokens(s: &str) -> String {
        s.replace("Ġ", " ").replace("Ċ", "\n")
    }

    #[test]
    fn basic_test() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/gpt2/Gpt2-124M-F16.gguf"));
        let mut gpt2 = GPT2::from_gguf_file(&mut gguf_file);
        let test_text = "A";

        let bpe_builder = BPE::from_file(
            "./data/tokenizers/gpt2/vocab.json",
            "./data/tokenizers/gpt2/merges.txt",
        );
        let bpe = bpe_builder
            //.unk_token("[UNK]".into())
            .build()
            .unwrap();

        let tokenizer = Tokenizer::new(bpe);

        let mut test_text = String::from("A");

        for _ in 0..30 {
            let encoding = tokenizer.encode(test_text.clone(), false).unwrap();
            let input = Tensor::from_vec(
                encoding
                    .get_ids()
                    .to_vec()
                    .iter()
                    .map(|x| *x as f32)
                    .collect(),
                vec![1, encoding.get_ids().to_vec().len()],
            );

            let output = gpt2.forward(input);

            let arr = output.item();
            // Get the index of the max along the last axis

            let test = arr.rows();
            let mut indices = vec![];
            for row in test {
                let mut index = 0;
                let mut current = std::f32::NEG_INFINITY;

                for (i, element) in row.iter().enumerate() {
                    if *element > current {
                        current = *element;
                        index = i; // correct 0-based index
                    }
                }
                indices.push(index as u32);
            }

            println!(
                "{:?}",
                decode_gpt2_tokens(&tokenizer.decode(&indices, true).unwrap())
            );
            let a = [indices[&indices.len() - 1]];
            test_text = test_text.to_owned() + &String::from(tokenizer.decode(&a, true).unwrap());
            get_equation().garbage_collect();
        }
    }

    use tokenizers::models::bpe::BPE;
    use tokenizers::tokenizer::{EncodeInput, Result, Tokenizer};

    #[test]
    fn tokenizer_test() -> Result<()> {
        let bpe_builder = BPE::from_file(
            "./data/tokenizers/gpt2/vocab.json",
            "./data/tokenizers/gpt2/merges.txt",
        );
        let bpe = bpe_builder
            .dropout(0.1)
            //.unk_token("[UNK]".into())
            .build()?;

        let mut tokenizer = Tokenizer::new(bpe);

        let encoding = tokenizer.encode("Hey there!", false)?;
        println!("{:?}", encoding.get_tokens());
        println!("{:?}", encoding.get_ids());
        println!("{:?}", tokenizer.get_vocab(false).len());
        Ok(())
    }
}
