
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
    pub dropout: f32, // Rate of dropout(note used)
    pub layer_norm_epsilon: f32 // how much we add to the denomnators of part of layer norm to avoid a div by 0
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

    pub fn from_gguf_file(gguf_file: &mut  GGUFFile) -> GPT2Config {
        println!("{:?}", gguf_file.get_value(String::from("tokenizer.ggml.tokens")).as_array().unwrap());
        GPT2Config {
            vocab_size: 50257,
            embedding_dimensions: gguf_file.get_value(String::from("gpt2.embedding_length")).as_u64().unwrap() as usize,
            number_of_layers: gguf_file.get_value(String::from("gpt2.block_count")).as_u64().unwrap() as usize,
            number_of_heads: gguf_file.get_value(String::from("gpt2.attention.head_count")).as_u64().unwrap() as usize,
            number_of_positions: 1024,//gguf_file.get_value(String::from("vocab_size")).as_i64().unwrap() as usize,
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

    pub fn from_gguf_file(gguf_file: &mut GGUFFile) -> GPT2{
        let gpt_config = GPT2Config::gpt2_small();
        let wpe_tensor = Tensor::from_gguf_file(String::from("position_embd.weight"), gguf_file);
        let wte_tensor = Tensor::from_gguf_file(String::from("token_embd.weight"), gguf_file);
        let wpe = Embedding::from_tensor(wpe_tensor);
        let wte = Embedding::from_tensor(wte_tensor);

        let block_count_value = gguf_file.get_value(String::from("gpt2.block_count"));
        let block_count = block_count_value.as_i64().unwrap();

        let mut blocks  = vec![];

        for i in 0..block_count {
            let block = GPT2Block::from_gguf_file(gguf_file, i, &gpt_config);
            blocks.push(block);
        }
        
        let final_layer_norm = LayerNorm::from_gguf_file(gguf_file, String::from("output_norm.weight"), String::from("output_norm.bias"));

        let wte_weights_reshapes_for_weight_tying = wte.weights.reshape(Shape::new(vec![gpt_config.embedding_dimensions, gpt_config.vocab_size]));
        let final_head = Linear::from_tensors(wte_weights_reshapes_for_weight_tying, None);
        
        GPT2 {
            config: gpt_config,
            wpe,
            wte,
            blocks,
            final_layer_norm,
            final_head
        }
    }
}

impl Model for GPT2 {

    fn forward(&mut self, input: Tensor) -> Tensor {

        let seq_length = input.shape.dimensions()[1];
        let positional_ids = (0..seq_length).map(|x|x as f32).collect();
        let positional_ids = Tensor::from_vec(positional_ids, vec![1, seq_length]);
        let token_embeddings = self.wte.forward(input);
        let positional_embeddings = self.wpe.forward(positional_ids);

        let mut hidden_states = token_embeddings + positional_embeddings;


        for block in &mut self.blocks {
            hidden_states = block.forward(hidden_states);
        }

        hidden_states = self.final_layer_norm.forward(hidden_states);
        println!("{:?}", hidden_states.item());
        panic!("");
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
    use crate::nn::{GPT2Config, Model, GPT2};
    use crate::central::*;
    use crate::utils::GGUFFile;
    use ndarray::Axis;
    
    #[test]
    fn basic_test() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/gpt2/Gpt2-124M-F16.gguf"));
        let mut gpt2 = GPT2::from_gguf_file(&mut gguf_file);
        let test_text = "I play the Game Boy Game";

        let bpe_builder = BPE::from_file("./data/tokenizers/gpt2/vocab.json", "./data/tokenizers/gpt2/merges.txt");
        let bpe = bpe_builder
            //.unk_token("[UNK]".into())
            .build().unwrap();

        let tokenizer = Tokenizer::new(bpe);
        let encoding = tokenizer.encode(test_text, false).unwrap();
        let input = Tensor::from_vec(encoding.get_ids().to_vec().iter().map(|x|*x as f32).collect(), vec![1, encoding.get_ids().to_vec().len()]);

        let output = gpt2.forward(input);



        let arr = output.item();
        // Get the index of the max along the last axis

        let test = arr.rows();
        let mut indices = vec![];
        for row in test {
            let mut index = 0;
            let mut current = std::f32::NEG_INFINITY;
            let mut counter = 0;
            for element in row {
                counter += 1;
                if *element > current {
                    current = *element;
                    index = counter;
                }
            }
            println!("{:?}", current);
            indices.push(index);
        }
        println!("{:?}", indices)

    }

    use tokenizers::tokenizer::{Result, Tokenizer, EncodeInput};
    use tokenizers::models::bpe::BPE;

    #[test]
    fn tokenizer_test() -> Result<()> {
        let bpe_builder = BPE::from_file("./data/tokenizers/gpt2/vocab.json", "./data/tokenizers/gpt2/merges.txt");
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