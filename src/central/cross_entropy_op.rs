use crate::central::*;
use ndarray::prelude::*;

impl Tensor {
    /// Computes cross-entropy loss between logits and targets
    /// # Arguments
    /// * 'targets' - One-hot encoded target tensor with same shape as logits
    pub fn cross_entropy_loss(&self, targets: Tensor) -> Tensor {
        let softmax = self.softmax(1);
        let log_softmax = softmax.log();
        let loss = targets * log_softmax;
        let sum = loss.sum(vec![1], true);
        let mean = sum.mean(vec![0]);
        return -mean;
    }
}