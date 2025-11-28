use crate::{central::Tensor, utils::DataSet};
use rand::{thread_rng, Rng};

pub struct DataLoader {
    batch_size: usize,
    data_set: Box<dyn DataSet>,
}

impl DataLoader {
    pub fn new(data_set: Box<dyn DataSet>, batch_size: usize) -> DataLoader {
        DataLoader {
            data_set,
            batch_size,
        }
    }

    pub fn get_random_batch(&self, batch_size: usize) -> (Tensor, Tensor) {
        assert!(batch_size > 0);
        let number_of_samples = self.data_set.length();
        let mut rng = thread_rng();
        
        // Create empty tensors for input and label batches
        let mut batch_inputs: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut batch_labels: Vec<Tensor> = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = rng.gen_range(0..number_of_samples);
            let (x, y) = self.data_set.get_item(idx);
            batch_inputs.push(x);
            batch_labels.push(y);
        }
        let batch_inputs = batch_inputs[0].stack(batch_inputs[1..].to_vec(), 0);
        let batch_labels = batch_labels[0].stack(batch_labels[1..].to_vec(), 0);

        return (batch_inputs, batch_labels);
    }

    pub fn number_of_samples(&self) -> usize {
        return self.data_set.length();
    }
}
