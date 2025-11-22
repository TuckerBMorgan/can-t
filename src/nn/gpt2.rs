use crate::central::*;
use crate::nn::*;
use crate::utils::GGUFFile;

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

pub struct GPT2 {
    _config: GPT2Config,
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
            _config: config,
        }
    }

    pub fn from_gguf_file(gguf_file: &mut GGUFFile) -> GPT2 {
        let gpt_config = GPT2Config::gpt2_small();
        let mut wpe_tensor =
            Tensor::from_gguf_file(String::from("position_embd.weight"), gguf_file);

        let mut wte_tensor = Tensor::from_gguf_file(String::from("token_embd.weight"), gguf_file);
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

        let mut wte_weights_reshapes_for_weight_tying = wte.weights.reshape(Shape::new(vec![
            gpt_config.vocab_size,
            gpt_config.embedding_dimensions,
        ]));
        wte_weights_reshapes_for_weight_tying.set_requires_grad(true);
        wte_weights_reshapes_for_weight_tying.set_keep_alive(true);
        println!("{:?}", wte_weights_reshapes_for_weight_tying.id);
        let mut wte_weights_reshapes_for_weight_tying =
            wte_weights_reshapes_for_weight_tying.transpose(0, 1);
        wte_weights_reshapes_for_weight_tying.set_keep_alive(true);
        wte_weights_reshapes_for_weight_tying.set_requires_grad(true);
        println!("{:?}", wte_weights_reshapes_for_weight_tying.id);
        let final_head = Linear::from_tensors(wte_weights_reshapes_for_weight_tying, None);

        GPT2 {
            _config: gpt_config,
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
        for (_i, block) in self.blocks.iter_mut().enumerate() {
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
    use crate::nn::{GPT2, Model};
    use crate::utils::GGUFFile;

    fn decode_gpt2_tokens(s: &str) -> String {
        s.replace("Ġ", " ").replace("Ċ", "\n")
    }

    #[test]
    fn basic_test() {
        // Load model
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/gpt2/Gpt2-124M-F16.gguf"));
        let mut gpt2 = GPT2::from_gguf_file(&mut gguf_file);

        // Build tokenizer
        let bpe = BPE::from_file(
            "./data/tokenizers/gpt2/vocab.json",
            "./data/tokenizers/gpt2/merges.txt",
        )
        .build()
        .unwrap();

        let tokenizer = Tokenizer::new(bpe);

        // For now, keep the original hard-coded IDs
        let mut token_ids: Vec<f32> = vec![32.0, 13.0, 198.0, 198.0];

        // Generate 5 tokens
        for _ in 0..5 {
            let input = Tensor::from_vec(token_ids.clone(), vec![1, token_ids.len()]);

            let output = gpt2.forward(input);
            let arr = output.item();
            let rows = arr.rows();

            // Greedy decoding: argmax over last dimension for each row
            let mut indices = vec![];
            for row in rows {
                let (max_idx, _) = row.iter().enumerate().fold(
                    (0usize, f32::NEG_INFINITY),
                    |(best_i, best_v), (i, &v)| {
                        if v > best_v { (i, v) } else { (best_i, best_v) }
                    },
                );
                indices.push(max_idx as u32);
            }

            // Take the last predicted token
            let last_token_id = *indices.last().expect("indices should not be empty");
            token_ids.push(last_token_id as f32);

            println!("{:?}", token_ids);

            // Decode just the last token and append to running text
            let decoded = tokenizer.decode(&[last_token_id], true).unwrap();

            get_equation().garbage_collect();
        }

        // Optionally assert something about `test_text` here
        // e.g.:
        // assert!(test_text.len() > " A".len());
    }

    fn shifted_batches(data: &[u32], batch_size: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
        let n = data.len();

        // Need at least batch_size + 1 elements to make one (x, y) pair
        if batch_size == 0 || n < batch_size + 1 {
            return Vec::new();
        }

        let mut result = Vec::new();

        // Last valid start index so that both x and y slices fit
        let last_start = n - batch_size - 1;

        for start in 0..=last_start {
            let x = data[start..start + batch_size].to_vec();
            let y = data[start + 1..start + 1 + batch_size].to_vec();
            result.push((x, y));
        }

        result
    }

    fn single_run(input: Vec<f32>, expected_output: Vec<f32>, model: &mut GPT2, vocab_size: usize) {
        let input_len = input.len();
        let input = Tensor::from_vec(input, vec![1, input_len]);
        println!("{:?}", input.id);
        println!("Forward pass");
        let output = model.forward(input);

        let output_onehots = Tensor::zeros(Shape::new(vec![expected_output.len(), vocab_size]));
        for i in 0..expected_output.len() {
            output_onehots.set_index(Indexable::Double(i, expected_output[i] as usize), 1.0);
        }
        println!("Calculating loss");
        let loss = output.cross_entropy_loss(output_onehots);
        zero_all_grads();
        println!("Backwards pass");
        loss.backward();
        println!("Updating parameters");
        update_parameters(-0.01);
        get_equation().garbage_collect();
        validate_tensor_store();
    }

    #[test]
    fn basic_retrain_test() {
        let BATCH_SIZE = 16;
        // Load model
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/gpt2/Gpt2-124M-F16.gguf"));
        let mut gpt2 = GPT2::from_gguf_file(&mut gguf_file);

        // Build tokenizer
        let bpe = BPE::from_file(
            "./data/tokenizers/gpt2/vocab.json",
            "./data/tokenizers/gpt2/merges.txt",
        )
        .build()
        .unwrap();

        let tokenizer = Tokenizer::new(bpe);

        let test_data = r#"One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt.
Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt."
Together, they shared the needle and sewed the button on Lily's shirt. It was not difficult for them because they were sharing and helping each other. After they finished, Lily thanked her mom for sharing the needle and fixing her shirt. They both felt happy because they had shared and worked together.
<|endoftext|>"#;

        let tokens = tokenizer.encode(test_data, false).unwrap();

        let mut training_batches = shifted_batches(tokens.get_ids(), BATCH_SIZE);
        training_batches.shuffle(&mut thread_rng());
        let mut i = 0;
        let traing_batch_length = training_batches.len();
        for (a, b) in training_batches {
            println!("Starting run {:?}", i + 1);
            let a = a.iter().map(|x| *x as f32).collect();
            let b = b.iter().map(|x| *x as f32).collect();
            single_run(a, b, &mut gpt2, tokenizer.get_vocab_size(false));
            i += 1;
            let percent_done = i as f32 / traing_batch_length as f32;
            println!("Percent done {:?}%", percent_done * 100.0);
        }
        return;
        let read_for_cant: Vec<f32> = tokens.get_ids().iter().map(|x| return *x as f32).collect();

        // For now, keep the original hard-coded IDs
        let mut token_ids: Vec<f32> = vec![32.0, 13.0, 198.0, 198.0];

        // Generate 5 tokens
        for _ in 0..5 {
            let input = Tensor::from_vec(token_ids.clone(), vec![1, token_ids.len()]);

            let output = gpt2.forward(input);
            let arr = output.item();
            let rows = arr.rows();

            // Greedy decoding: argmax over last dimension for each row
            let mut indices = vec![];
            for row in rows {
                let (max_idx, _) = row.iter().enumerate().fold(
                    (0usize, f32::NEG_INFINITY),
                    |(best_i, best_v), (i, &v)| {
                        if v > best_v { (i, v) } else { (best_i, best_v) }
                    },
                );
                indices.push(max_idx as u32);
            }
            // Take the last predicted token
            let last_token_id = *indices.last().expect("indices should not be empty");
            token_ids.push(last_token_id as f32);

            let test_ytrue_onehot = Tensor::element(
                Shape::new(vec![BATCH_SIZE, tokenizer.get_vocab_size(true)]),
                0.0,
            );

            for b in 0..BATCH_SIZE {
                //    test_ytrue_onehot.set_index([b, ytr[b]].into(), 1.0);
            }

            println!("{:?}", token_ids);

            // Decode just the last token and append to running text
            let decoded = tokenizer.decode(&[last_token_id], true).unwrap();

            get_equation().garbage_collect();
        }

        // Optionally assert something about `test_text` here
        // e.g.:
        // assert!(test_text.len() > " A".len());
    }

    use rand::seq::SliceRandom;
    use rand::thread_rng;
    use tokenizers::models::bpe::BPE;
    use tokenizers::tokenizer::{Result, Tokenizer};

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

        let tokenizer = Tokenizer::new(bpe);

        let encoding = tokenizer.encode("Hey there!", false)?;
        println!("{:?}", encoding.get_tokens());
        println!("{:?}", encoding.get_ids());
        println!("{:?}", tokenizer.get_vocab(false).len());
        Ok(())
    }
}
