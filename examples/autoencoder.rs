use cant::central::*;
use cant::nn::{Layer, Linear, Model};
use mnist::{Mnist, MnistBuilder};
use ndarray::ArrayD;

// Convert u8 pixel values to normalized f32 tensors
fn preprocess_images(raw_images: Vec<u8>, num_samples: usize) -> Tensor {
    let normalized: Vec<f32> = raw_images
        .iter()
        .map(|&pixel| pixel as f32 / 255.0)
        .collect();

    Tensor::from_vec(normalized, vec![num_samples, 784])
}

// Convert digit labels to one-hot encoded vectors
fn preprocess_labels(raw_labels: Vec<u8>) -> Tensor {
    let num_samples = raw_labels.len();
    let mut one_hot = vec![0.0f32; num_samples * 10];

    for (i, &label) in raw_labels.iter().enumerate() {
        one_hot[i * 10 + label as usize] = 1.0;
    }

    Tensor::from_vec(one_hot, vec![num_samples, 10])
}

// Extract batch from tensor by copying rows
fn extract_batch(
    tensor: &ArrayD<f32>,
    features: usize,
    start_idx: usize,
    end_idx: usize,
) -> Tensor {
    let batch_size = end_idx - start_idx;
    let features = features;
    let data = tensor;

    // Pre-allocate the exact size needed
    let mut batch_data = Vec::with_capacity(batch_size * features);

    // Use slice copying for better performance
    for i in start_idx..end_idx {
        let row_start = i * features;
        let row_end = row_start + features;

        // Copy entire row at once (more efficient than element-by-element)
        if let Some(flat_data) = data.as_slice() {
            batch_data.extend_from_slice(&flat_data[row_start..row_end]);
        } else {
            // Fallback to element access
            for j in 0..features {
                batch_data.push(data[[i, j]]);
            }
        }
    }

    Tensor::from_vec(batch_data, vec![batch_size, features])
}

// Helper function to load and preprocess MNIST data
fn load_mnist_data() -> (Tensor, Tensor, Tensor, Tensor) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        //  .training_set_length(1000)  // Small subset for fast testing
        //   .test_set_length(200)
        .finalize();
    println!("Preprocess images, train");
    let train_images = preprocess_images(trn_img, 60000);
    println!("Preprocess labels, train");
    let train_labels = preprocess_labels(trn_lbl);
    println!("Preprocess images, test");
    let test_images = preprocess_images(tst_img, 10000);
    println!("Preprocess images, test");
    let test_labels = preprocess_labels(tst_lbl);

    (train_images, train_labels, test_images, test_labels)
}

struct Encoder {
    layers: Vec<Linear>,
}

impl Encoder {
    pub fn new() -> Encoder {
        let linear_1 = Linear::new(784, 784 / 2, true);
        let linear_2 = Linear::new(784 / 2, 784 / 4, true);
        let linear_3 = Linear::new(784 / 4, 784 / 8, true);
        let layers = vec![linear_1, linear_2, linear_3];
        Encoder { layers }
    }
}

impl Model for Encoder {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let x = self.layers[0].forward(input).relu();
        let x = self.layers[1].forward(x).relu();
        let x = self.layers[2].forward(x).relu();
        return x;
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        let mut param = vec![];
        param.extend(self.layers[0].get_parameters());
        param.extend(self.layers[1].get_parameters());
        param.extend(self.layers[2].get_parameters());
        return param;
    }
}
struct Decoder {
    layers: Vec<Linear>,
}

impl Decoder {
    pub fn new() -> Decoder {
        let linear_1 = Linear::new(784 / 8, 784 / 4, true);
        let linear_2 = Linear::new(784 / 4, 784 / 2, true);
        let linear_3 = Linear::new(784 / 2, 784, true);
        let layers = vec![linear_1, linear_2, linear_3];

        Decoder { layers }
    }
}

impl Model for Decoder {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let x = self.layers[0].forward(input).relu();
        let x = self.layers[1].forward(x).relu();
        let x = self.layers[2].forward(x).relu();
        return x;
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        let mut param = vec![];
        param.extend(self.layers[0].get_parameters());
        param.extend(self.layers[1].get_parameters());
        param.extend(self.layers[2].get_parameters());
        return param;
    }
}

struct Autoencoder {
    encoder: Encoder,
    decoder: Decoder,
}

impl Autoencoder {
    pub fn new() -> Autoencoder {
        let encoder = Encoder::new();
        let decoder = Decoder::new();

        Autoencoder { encoder, decoder }
    }
}

impl Model for Autoencoder {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let x = self.encoder.forward(input);
        return self.decoder.forward(x);
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        let mut params = vec![];

        params.extend(self.encoder.get_parameters());
        params.extend(self.decoder.get_parameters());

        return params;
    }
}

fn main() {
    let mut auto_encoder = Autoencoder::new();
    let data_set = load_mnist_data();
    let test = auto_encoder.forward(data_set.0);
}
