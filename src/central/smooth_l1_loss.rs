use super::get_equation;
use crate::central::*;

const BETA: f32 = 1.0;
impl Tensor {
    pub fn smooth_l1_loss(&self, other: Tensor) -> Tensor {
        let local_data = get_equation().get_data_flat_buffer(self.id).to_vec();
        let other_data = get_equation().get_data_flat_buffer(other.id).to_vec();

        let diff = local_data
            .iter()
            .zip(other_data)
            .map(|(a, b)| *a - b)
            .map(|diff| {
                if diff < BETA {
                    return 0.5 * diff.powf(2.0) / BETA;
                } else {
                    return diff - 0.5 * BETA;
                }
            });

        let summed: f32 = diff.sum();

        let final_value = summed / local_data.len() as f32;

        panic!()
    }
}
