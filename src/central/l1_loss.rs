use crate::central::*;
impl Tensor {
    pub fn l1_loss(&self, other: Tensor) -> Tensor {
        let diff = *self - other;
        return diff.mean(vec![diff.shape.dimensions().len() - 1]);
    }
}