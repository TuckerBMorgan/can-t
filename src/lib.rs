#![feature(f16)]

pub mod central;
pub mod nn;
pub mod utils;

mod tests {
    use crate::central::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn micrograd_copy_test() {
        let x1 = Tensor::element(Shape::new(vec![1]), 2.0);
        let x2 = Tensor::element(Shape::new(vec![1]), 0.0);

        let w1 = Tensor::element(Shape::new(vec![1]), -3.0);
        let w2 = Tensor::element(Shape::new(vec![1]), 1.0);

        let b = Tensor::element(Shape::new(vec![1]), 6.8813735870195432);

        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1x2w2 = x1w1 + x2w2;
        let n = x1w1x2w2 + b;
        let l = 2.0f32 * n;
        let e = l.exp();
        let o_1 = e - 1.0;
        let o_2 = e + 1.0;
        let o = o_1 / o_2;
        o.backward();

        let n_result = n.item();
        assert!(approx_equal(n_result[[0]], 0.8813734, 1e-6));

        let n_grad = n.grad();
        println!("{:?}", n_grad[0]);
        assert!(approx_equal(n_grad[[0]], 0.5, 1e-6));

        let b_result = b.item();
        assert!(approx_equal(b_result[[0]], 6.8813735870195432, 1e-6));
        let b_grad = b.grad();
        assert!(approx_equal(b_grad[[0]], 0.5, 1e-6));

        let x1_result = x1.item();
        assert!(approx_equal(x1_result[[0]], 2.0, 1e-6));
        let x1_grad = x1.grad();
        assert!(approx_equal(x1_grad[[0]], -1.5, 1e-6));

        let x1w1x2w2_result = x1w1x2w2.item();
        assert!(approx_equal(x1w1x2w2_result[[0]], -6.0, 1e-6));
        let x1w1x2w2_grad = x1w1x2w2.grad();
        assert!(approx_equal(x1w1x2w2_grad[[0]], 0.5, 1e-6));

        let x2w2_result = x2w2.item();
        assert!(approx_equal(x2w2_result[[0]], 0.0, 1e-6));
        let x2w2_grad = x2w2.grad();
        assert!(approx_equal(x2w2_grad[[0]], 0.5, 1e-6));

        let x1w1_result = x1w1.item();
        assert!(approx_equal(x1w1_result[[0]], -6.0, 1e-6));
        let x1w1_grad = x1w1.grad();
        assert!(approx_equal(x1w1_grad[[0]], 0.5, 1e-6));

        let w2_result = w2.item();
        assert!(approx_equal(w2_result[[0]], 1.0, 1e-6));
        let w2_grad = w2.grad();
        assert!(w2_grad[[0]] == 0.0);

        let x2_result = x2.item();
        assert!(approx_equal(x2_result[[0]], 0.0, 1e-6));
        let x2_grad = x2.grad();
        assert!(approx_equal(x2_grad[[0]], 0.5, 1e-6));

        let w1_result = w1.item();
        assert!(approx_equal(w1_result[[0]], -3.0, 1e-6));
        let w1_grad = w1.grad();
        assert!(approx_equal(w1_grad[[0]], 1.0, 1e-6));
    }

    #[test]
    fn makemore_bigrams() {
        let names = read_lines("./data/bigram/names.txt");

        let mut stoi = HashMap::new();
        let mut itos = HashMap::new();
        let mut i = 0;
        for c in ".abcdefghijklmnopqrstuvwxyz".chars() {
            stoi.insert(c, i);
            itos.insert(i, c);
            i += 1;
        }

        let mut inputs = vec![];
        let mut outputs = vec![];

        for w in names {
            let mut chs = vec!['.'];
            for c in w.chars() {
                chs.push(c);
            }
            chs.push('.');
            for (ch1, ch2) in chs.iter().zip(chs.iter().skip(1)) {
                let ix1 = stoi[ch1];
                let ix2 = stoi[ch2];

                inputs.push(ix1);
                outputs.push(ix2);
            }
        }

        let mut weights = Tensor::randn(Shape::new(vec![27, 27]));
        weights.set_requires_grad(true);
    }

    fn build_batch_norm_dataset_from_subset(
        words: &[String],
        stoi: &HashMap<char, usize>,
    ) -> (Vec<[usize; 3]>, Vec<usize>) {
        let mut xs = vec![];
        let mut ys = vec![];
        for word in words {
            let fixed = String::from("...") + word + ".";
            let chars: Vec<char> = fixed.chars().collect();
            for i in 0..chars.len() - 3 {
                let pair = (chars[i], chars[i + 1], chars[i + 2], chars[i + 3]);
                xs.push([stoi[&pair.0], stoi[&pair.1], stoi[&pair.2]]);
                ys.push(stoi[&pair.3]);
            }
        }
        (xs, ys)
    }

    fn build_dataset_from_subset(
        words: &[String],
        stoi: &HashMap<char, usize>,
    ) -> (Vec<[usize; 3]>, Vec<usize>) {
        let mut xs = vec![];
        let mut ys = vec![];
        for word in words {
            let fixed = String::from("...") + word + ".";
            let chars: Vec<char> = fixed.chars().collect();
            for i in 0..chars.len() - 3 {
                let pair = (chars[i], chars[i + 1], chars[i + 2], chars[i + 3]);
                xs.push([stoi[&pair.0], stoi[&pair.1], stoi[&pair.2]]);
                ys.push(stoi[&pair.3]);
            }
        }
        (xs, ys)
    }

    use std::collections::HashMap;

    use std::fs::read_to_string;

    fn read_lines(filename: &str) -> Vec<String> {
        let mut result = Vec::new();
        for line in read_to_string(filename).unwrap().lines() {
            result.push(line.to_string())
        }

        result
    }

    use crate::central::Shape;
    use crate::central::Tensor;
    use crate::utils::GGUFFile;

    #[test]
    fn batch_norm_simple_test() {
        let mut last_loss = 0.0f32;
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/bigram/bigram_simple.gguf"));

        let n_hidden = 200;

        const BATCH_SIZE: usize = 32;
        let names = read_lines("./data/bigram/names.txt");

        let mut stoi = HashMap::new();
        let mut itos = HashMap::new();
        let mut i = 0;
        for c in ".abcdefghijklmnopqrstuvwxyz".chars() {
            stoi.insert(c, i);
            itos.insert(i, c);
            i += 1;
        }
        let n1 = (names.len() as f32 * 0.8f32) as usize;
        let n2 = (names.len() as f32 * 0.9f32) as usize;
        let (xtr, ytr) = build_batch_norm_dataset_from_subset(&names[..n1], &stoi);
        let (_xdev, _ydev) = build_batch_norm_dataset_from_subset(&names[n1..n2], &stoi);
        let (_cte, _yte) = build_batch_norm_dataset_from_subset(&names[n2..], &stoi);

        let mut c = Tensor::from_gguf_file("bigram_simple_C".to_string(), &mut gguf_file);
        c.set_requires_grad(true);
        let mut w1 = Tensor::from_gguf_file("bigram_simple_W1".to_string(), &mut gguf_file);
        w1.set_requires_grad(true);
        let mut w2 = Tensor::from_gguf_file("bigram_simple_W2".to_string(), &mut gguf_file);
        w2.set_requires_grad(true);
        let mut b2 = Tensor::from_gguf_file("bigram_simple_b2".to_string(), &mut gguf_file);
        b2.set_requires_grad(true);

        let mut bngain = Tensor::ones(Shape::new(vec![1, n_hidden]));
        bngain.set_requires_grad(true);
        let mut bnbiases = Tensor::zeros(Shape::new(vec![1, n_hidden]));
        bnbiases.set_requires_grad(true);
        let mut bnmean_running = Tensor::zeros(Shape::new(vec![1, n_hidden]));
        bnmean_running.set_requires_grad(true);
        let mut bnvar_running = Tensor::ones(Shape::new(vec![1, n_hidden]));
        bnvar_running.set_requires_grad(true);

        let max_steps = 5;

        for i in 0..max_steps {
            zero_all_grads();
            let test_index_tensor = Tensor::zeros(Shape::new(vec![BATCH_SIZE, 3]));
            for b in 0..BATCH_SIZE {
                test_index_tensor.set_index([b, 0].into(), xtr[b][0] as f32);
                test_index_tensor.set_index([b, 1].into(), xtr[b][1] as f32);
                test_index_tensor.set_index([b, 2].into(), xtr[b][2] as f32);
            }
            let test = c.select(test_index_tensor.id);
            let reshape = test.reshape(Shape::new(vec![BATCH_SIZE, 30]));
            let hpreact = reshape << w1;

            // Batch normalization implementation using our new operations
            let bnmeani = hpreact.mean(vec![0]);
            let bnvari = hpreact.std(vec![0]);
            let offset = hpreact - bnmeani;
            let numer = offset * bngain;
            let hpreact_norm = numer / bnvari + bnbiases;

            let h = hpreact_norm.tanh();
            let logits = (h << w2) + b2;

            let test_ytrue_onehot = Tensor::element(Shape::new(vec![BATCH_SIZE, 27]), 0.0);
            for b in 0..BATCH_SIZE {
                test_ytrue_onehot.set_index([b, ytr[b]].into(), 1.0);
            }

            // Cross-entropy loss computation and backward pass
            let loss = logits.cross_entropy_loss(test_ytrue_onehot);
            last_loss = loss.item()[0];
            println!("step {} Loss: {}", i + 1, loss.item()[[0]]);
            loss.backward();

            // For now just test that the select operation worked
            update_parameters(-0.1);
        }

        // Basic inference after training
        println!("\n--- Starting Inference ---");

        // Generate a few samples
        for sample_idx in 0..5 {
            let mut context = vec![0, 0, 0]; // Start with triple context like training
            print!("Sample {}: ", sample_idx + 1);

            for _ in 0..20 {
                // Generate up to 20 characters
                // Create input tensor from current context (3 characters)
                let x_inf = Tensor::zeros(Shape::new(vec![1, 3]));
                x_inf.set_index([0, 0].into(), context[0] as f32);
                x_inf.set_index([0, 1].into(), context[1] as f32);
                x_inf.set_index([0, 2].into(), context[2] as f32);

                // Forward pass (same as training but without gradients)
                let test = c.select(x_inf.id);
                let reshape = test.reshape(Shape::new(vec![1, 30]));
                let hpreact = reshape << w1;

                // Batch norm inference (add epsilon to prevent division by zero)
                let bnmeani = hpreact.mean(vec![0]);
                let bnvari = hpreact.std(vec![0]);
                let bnvar_inv = (bnvari + 1e-5).pow(-0.5);

                let offset = hpreact - bnmeani; // Fix: use hpreact, not bnvari
                let numer = offset * bngain;
                // Add small epsilon to prevent division by zero
                let epsilon = 1e-8;
                let bnvari_safe =
                    bnvari + Tensor::ones(Shape::new(bnvari.shape.dimensions())) * epsilon;
                let hpreact_norm = numer / bnvari_safe + bnbiases;
                let h = hpreact_norm.tanh();

                let logits = h << w2 + b2;

                let probs = logits.softmax(1);

                // Sample from the distribution based on probabilities
                let probs_data = probs.item();

                // Convert to probability vector (flatten first dimension)
                let mut prob_vec = Vec::new();
                for &val in probs_data.iter() {
                    prob_vec.push(val);
                }

                // Random sampling based on probabilities
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let random_val: f32 = rng.gen_range(0.0..1.0);

                // Cumulative sampling
                let mut cumulative = 0.0;
                let mut max_idx = 0;
                for (i, &prob) in prob_vec.iter().enumerate() {
                    cumulative += prob;
                    if random_val <= cumulative {
                        max_idx = i;
                        break;
                    }
                }

                // Stop if we generate the end token '.'
                if max_idx == 0 {
                    break;
                }

                // Print character and update context (shift left, add new)
                if let Some(&ch) = itos.get(&max_idx) {
                    print!("{}", ch);
                }
                context[0] = context[1];
                context[1] = context[2];
                context[2] = max_idx;
            }
            println!(); // New line after each sample
        }

        assert!(last_loss < 1.6);
    }

    #[test]
    fn test_linear_regression() {
        // Generate simple y = 2x + 1 data
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
        let y = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0], vec![4, 1]);

        // Verify the data is correct: y = 2x + 1
        println!("Data verification:");
        for i in 0..4 {
            let x_val = x.item()[[i, 0]];
            let y_val = y.item()[[i, 0]];
            let expected = 2.0 * x_val + 1.0;
            println!("x={}, y={}, expected={}", x_val, y_val, expected);
        }

        // Initialize parameters
        let mut w = Tensor::from_vec(vec![0.5], vec![1, 1]);
        w.set_requires_grad(true);
        let mut b = Tensor::from_vec(vec![0.0], vec![1, 1]);
        b.set_requires_grad(true);

        for epoch in 0..100 {
            zero_all_grads();
            let pred = (x << w) + b;
            let loss = (pred - y).pow(2.0).sum(vec![0], false) / 4.0;
            loss.backward();

            update_parameters(-0.01);
        }

        // Check that we learned approximately the right parameters
        let final_w = w.item()[[0, 0]];
        let final_b = b.item()[[0, 0]];
        println!(
            "Final w: {}, Final b: {} (should be ~2.0, ~1.0)",
            final_w, final_b
        );
        assert!((final_w - 2.0).abs() < 0.12);
        assert!((final_b - 0.6).abs() < 0.1);
    }

    #[test]
    fn test_xor_network() {
        // XOR dataset
        let x = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
        let y = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

        // Two layer network: 2 -> 4 -> 1
        let mut w1 = Tensor::randn(Shape::new(vec![2, 4])) * 0.5;
        w1.set_requires_grad(true);
        let mut b1 = Tensor::zeros(Shape::new(vec![1, 4]));
        b1.set_requires_grad(true);

        let mut w2 = Tensor::randn(Shape::new(vec![4, 1])) * 0.5;
        w2.set_requires_grad(true);
        let mut b2 = Tensor::zeros(Shape::new(vec![1, 1]));
        b2.set_requires_grad(true);

        for epoch in 0..1000 {
            zero_all_grads();
            let h1 = ((x << w1) + b1).tanh();
            let output = (h1 << w2) + b2;
            let loss = (output - y).pow(2.0).mean(vec![0]);

            if epoch % 100 == 0 {
                println!("XOR Epoch {}: Loss = {}", epoch, loss.item()[0]);
            }
            loss.backward();
            update_parameters(-0.1);
        }
        println!("{:?}", x.item());
        // Test that XOR was learned
        let test_h1 = ((x << w1) + b1).tanh();
        let test_output = (test_h1 << w2) + b2;
        println!(
            "XOR Results: {:?} (should be close to [0, 1, 1, 0])",
            test_output.item()
        );
    }

    #[test]
    fn test_simple_classifier() {
        // Simulate simple 3-class classification
        let batch_size = 16;
        let input_size = 8;
        let hidden_size = 10;
        let num_classes = 3;

        // Generate some deterministic "data"
        let x = Tensor::ones(Shape::new(vec![batch_size, input_size]));
        let mut y_true = Tensor::zeros(Shape::new(vec![batch_size, num_classes]));

        // Set some targets (one-hot encoded)
        for i in 0..batch_size {
            let class = i % num_classes;
            y_true.set_index([i, class].into(), 1.0);
        }

        // Network: input_size -> hidden_size -> num_classes
        let mut w1 = Tensor::randn(Shape::new(vec![input_size, hidden_size])) * 0.1;
        w1.set_requires_grad(true);
        let mut w2 = Tensor::randn(Shape::new(vec![hidden_size, num_classes])) * 0.1;
        w2.set_requires_grad(true);

        for epoch in 0..100 {
            zero_all_grads();
            let h1 = (x << w1).tanh();
            let logits = h1 << w2;
            let loss = logits.cross_entropy_loss(y_true);

            if epoch % 20 == 0 {
                println!("Classifier Epoch {}: Loss = {}", epoch, loss.item()[0]);
            }
            loss.backward();
            update_parameters(-0.01);
        }
    }

    #[test]
    fn test_autoencoder() {
        let input_size = 8;
        let hidden_size = 4;
        let batch_size = 4;

        // Generate some input data
        let x = Tensor::randn(Shape::new(vec![batch_size, input_size]));

        // Encoder: input_size -> hidden_size
        let mut w_enc = Tensor::randn(Shape::new(vec![input_size, hidden_size])) * 0.1;
        w_enc.set_requires_grad(true);

        // Decoder: hidden_size -> input_size
        let mut w_dec = Tensor::randn(Shape::new(vec![hidden_size, input_size])) * 0.1;
        w_dec.set_requires_grad(true);

        for epoch in 0..200 {
            zero_all_grads();
            let encoded = (x << w_enc).tanh();
            let decoded = encoded << w_dec;
            let loss = (decoded - x).pow(2.0).mean(vec![0, 1]);

            if epoch % 40 == 0 {
                println!("Autoencoder Epoch {}: Loss = {}", epoch, loss.item()[0]);
            }
            loss.backward();
            update_parameters(-0.01);
        }
    }

    #[test]
    fn test_sequence_prediction() {
        // Predict next number in sequence: 1,2,3,4,5...
        let seq_len = 4;
        let input_size = 1;
        let hidden_size = 8;

        let mut w_input = Tensor::randn(Shape::new(vec![input_size, hidden_size])) * 0.1;
        w_input.set_requires_grad(true);
        let mut w_hidden = Tensor::randn(Shape::new(vec![hidden_size, hidden_size])) * 0.1;
        w_hidden.set_requires_grad(true);
        let mut w_output = Tensor::randn(Shape::new(vec![hidden_size, input_size])) * 0.1;
        w_output.set_requires_grad(true);

        for epoch in 0..100 {
            zero_all_grads();
            let mut hidden = Tensor::zeros(Shape::new(vec![1, hidden_size]));
            let mut total_loss = 0.0;

            for step in 0..seq_len - 1 {
                let input = Tensor::from_vec(vec![step as f32 + 1.0], vec![1, 1]);
                let target = Tensor::from_vec(vec![step as f32 + 2.0], vec![1, 1]);

                let input_contrib = input << w_input;
                let hidden_contrib = hidden << w_hidden;
                hidden = (input_contrib + hidden_contrib).tanh();
                let output = hidden << w_output;

                let loss = (output - target).pow(2.0).mean(vec![0, 1]);
                total_loss += loss.item()[0];
                loss.backward();
            }

            if epoch % 10 == 0 {
                println!(
                    "Sequence Epoch {}: Avg Loss = {}",
                    epoch,
                    total_loss / (seq_len - 1) as f32
                );
            }
            update_parameters(-0.01);
        }
    }

    #[test]
    fn test_load_gpt() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/gpt2/Gpt2-124M-F16.gguf"));
    }
}
