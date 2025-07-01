mod central;
mod utils;

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

        let mut gguf_file =
        GGUFFile::new(String::from("./models/tests/bigram/bigram_simple.gguf"));

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

        let max_steps = 2;

        for _i in 0..max_steps {
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
            println!("Loss: {}", loss.item()[[0]]);
            loss.backward();
            
            // For now just test that the select operation worked
            println!("Select operation in view worked! Shape: {:?}", test.shape.dimensions());
          //  update_parameters(-0.01);
        }
        println!("w1 grad {:?}", w1.grad());
    }
 
}
