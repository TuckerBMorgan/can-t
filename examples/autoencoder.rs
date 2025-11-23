use core::num;

use cant::central::*;
use cant::nn::{Layer, Linear, Model};
use mnist::{Mnist, MnistBuilder};
use ndarray::ArrayD;
use rand::{Rng, thread_rng};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

use std::io;
use std::error::Error;

use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
    widgets::{Block, Borders, Paragraph, Chart, Axis, Dataset, GraphType},
    layout::{Layout, Constraint, Direction},
    text::Span,
};
use ratatui::backend::Backend;
use ratatui::Frame;


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
    let training_set_size = 1000;
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
          .training_set_length(training_set_size)  // Small subset for fast testing
        //   .test_set_length(200)
        .finalize();
    println!("Preprocess images, train");
    let train_images = preprocess_images(trn_img, training_set_size as usize);
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
        let linear_1 = Linear::new(784, 256, true);
        let linear_2 = Linear::new(256, 128, true);
        let linear_3 = Linear::new(128, 64, true);
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
        let linear_1 = Linear::new(64, 128, true);
        let linear_2 = Linear::new(128, 256, true);
        let linear_3 = Linear::new(256, 784, true);
        let layers = vec![linear_1, linear_2, linear_3];

        Decoder { layers }
    }
}

impl Model for Decoder {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let x = self.layers[0].forward(input).relu();
        let x = self.layers[1].forward(x).relu();
        let x = self.layers[2].forward(x).sigmoid();
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
fn lerp_array(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 0 {
        return vec![];
    }

    if steps == 1 {
        return vec![start];
    }

    let mut result = Vec::with_capacity(steps);

    for i in 0..steps {
        let t = i as f32 / (steps - 1) as f32;
        let value = start + t * (end - start);
        result.push(value);
    }

    result
}

fn image_to_ascii(image: &[f32]) -> String {
    // expects a single 28x28 image flattened to 784 elements
    const WIDTH: usize = 28;
    const HEIGHT: usize = 28;
    let shades: [char; 10] = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

    let mut s = String::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let v = image[y * WIDTH + x].clamp(0.0, 1.0);
            let idx = (v * (shades.len() as f32 - 1.0)).round() as usize;
            s.push(shades[idx]);
        }
        s.push('\n');
    }
    s
}

fn first_image_from_array2(arr: &ArrayD<f32>) -> Option<Vec<f32>> {
    if arr.ndim() != 2 {
        return None;
    }
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 784 {
        return None;
    }
    let cols = shape[1];

    if let Some(slice) = arr.as_slice() {
        if slice.len() < cols {
            return None;
        }
        Some(slice[0..cols].to_vec())
    } else {
        let mut v = Vec::with_capacity(cols);
        for j in 0..cols {
            v.push(arr[[0, j]]);
        }
        Some(v)
    }
}

fn draw_ui(
    f: &mut Frame,
    loss_history: &[(f64, f64)],
    original_ascii: &str,
    recon_ascii: &str,
) {
    let size = f.size();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints(
            [
                Constraint::Percentage(50), // loss chart
                Constraint::Percentage(50), // images
            ]
            .as_ref(),
        )
        .split(size);

    // ----- Loss chart -----
    let (min_loss, max_loss) = if loss_history.is_empty() {
        (0.0, 1.0)
    } else {
        let mut min = loss_history[0].1;
        let mut max = loss_history[0].1;
        for &(_, y) in loss_history.iter() {
            if y < min {
                min = y;
            }
            if y > max {
                max = y;
            }
        }
        if (max - min).abs() < 1e-6 {
            (min - 1.0, max + 1.0)
        } else {
            (min, max)
        }
    };

    let x_max = loss_history
        .last()
        .map(|(x, _)| *x)
        .unwrap_or(1.0)
        .max(1.0);

    let x_labels = vec![
        Span::raw("0"),
        Span::raw(format!("{:.0}", x_max)),
    ];

    let y_labels = vec![
        Span::raw(format!("{:.3}", min_loss)),
        Span::raw(format!("{:.3}", max_loss)),
    ];

    let datasets = vec![Dataset::default()
        .name("avg loss")
        .graph_type(GraphType::Line)
        .data(loss_history)];

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title("Average Loss"))
        .x_axis(
            Axis::default()
                .title("Step")
                .bounds([0.0, x_max])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("Loss")
                .bounds([min_loss, max_loss])
                .labels(y_labels),
        );

    f.render_widget(chart, chunks[0]);

    // ----- Images: original vs reconstruction -----
    let img_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(
            [
                Constraint::Percentage(50), // original
                Constraint::Percentage(50), // reconstruction
            ]
            .as_ref(),
        )
        .split(chunks[1]);

    let orig_text = if original_ascii.is_empty() {
        "Waiting for sample..."
    } else {
        original_ascii
    };

    let recon_text = if recon_ascii.is_empty() {
        "Waiting for sample..."
    } else {
        recon_ascii
    };

    let orig_para = Paragraph::new(orig_text)
        .block(Block::default().borders(Borders::ALL).title("Original"));
    let recon_para = Paragraph::new(recon_text)
        .block(Block::default().borders(Borders::ALL).title("Reconstruction"));

    f.render_widget(orig_para, img_chunks[0]);
    f.render_widget(recon_para, img_chunks[1]);
}

fn main() -> Result<(), Box<dyn Error>> {
    // ---- TUI setup ----
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // ---- Original training setup ----
    let batch_size = 32;
    let mut auto_encoder = Autoencoder::new();
    let (train_images, _, _, _) = load_mnist_data();

    let train_images_array_d = train_images.item();
    let train_features = train_images.shape.dimensions()[1];
    let epochs = 1000;
    let learning_rates = lerp_array(0.5, 0.01, epochs);

    // New: keep loss history and sample ascii images
    let mut loss_history: Vec<(f64, f64)> = Vec::new();
    let mut global_step: f64 = 0.0;
    let mut original_ascii = String::new();
    let mut recon_ascii = String::new();

    for epoch in 0..epochs {
        println!("Starting Epoch {:?}", epoch);
        let mut total_loss = 0.0;
        let num_samples = train_images.shape.dimensions()[0];
        let num_batches = (num_samples + batch_size - 1) / batch_size; // Ceiling division
        let mut offsets : Vec<usize> = (0..num_batches).collect();
        offsets.shuffle(&mut thread_rng());
        for batch_idx in 0..num_batches {
            let start_idx = offsets[batch_idx] * batch_size;
            let end_idx = ((offsets[batch_idx] + 1) * batch_size).min(num_samples);

            // Assuming you already had something like this:
            let batch_images = extract_batch(
                &train_images_array_d,
                train_features,
                start_idx,
                end_idx,
            );

            let test = auto_encoder.forward(batch_images.clone());

            let diff = (test.clone() - batch_images.clone()).pow(2.0).mean(vec![0, 1]);
            total_loss += diff.item()[0];

            // ---- New: update loss history more frequently (per batch) ----
            let avg_loss_so_far = total_loss / (batch_idx + 1) as f32;
            loss_history.push((global_step, avg_loss_so_far as f64));
            // Optional: keep history bounded
            if loss_history.len() > 500 {
                let excess = loss_history.len() - 500;
                loss_history.drain(0..excess);
            }
            global_step += 1.0;

            // ---- New: sample reconstructions occasionally ----
            if ((global_step as usize) % 50) == 0 {
                let test_arr = test.item();
                let input_arr = batch_images.item();

                if let (Some(out_vec), Some(inp_vec)) =
                    (first_image_from_array2(&test_arr), first_image_from_array2(&input_arr))
                {
                    recon_ascii = image_to_ascii(&out_vec);
                    original_ascii = image_to_ascii(&inp_vec);
                }
            }

            // ---- TUI draw ----
            terminal.draw(|f| {
                draw_ui(f, &loss_history, &original_ascii, &recon_ascii);
            })?;

            // ---- Backprop + update (original code) ----
            zero_all_grads();
            diff.backward();
            update_parameters(-learning_rates[epoch]);
            get_equation().garbage_collect();
        }

        println!("Epoch {} loss: {:?}", epoch, total_loss / num_batches as f32);
    }

    // ---- TUI cleanup ----
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
