use super::{BackproagationPacket, Operation, Tensor};
use crate::central::*;

impl Tensor {
    pub fn max(&self, dimension: usize, keep_dim: bool) -> Tensor {
        let number_of_elements = self.shape.dimensions()[dimension];

        let mut permuted : Vec<usize> = (0..self.shape.number_of_dimension()).collect();
        
        permuted.remove(dimension);
        let total_batchs : usize = permuted.iter().product();
        permuted.push(dimension);
        let self_permuted = self.permute(permuted.clone());

        let as_flat = get_equation().get_data_flat_buffer(self_permuted.id).to_vec();
        let mut data = vec![];
        for b in 0..total_batchs {
            let offset_start = b * number_of_elements;
            let elements = &as_flat[offset_start..(offset_start + number_of_elements)];
            let max = elements.iter().map(|x|*x).reduce(f32::max).unwrap_or(0.0);
            data.push(max);
        }

        let mut new_dimensions = self.shape.dimensions();

        if keep_dim == false {
        //    new_dimensions.remove(dimension);
        }
        else {
            new_dimensions[dimension] = 1;
        }

        let mut inv_perm = vec![0; permuted.len()];
        for (i, &p) in permuted.iter().enumerate() {
            inv_perm[p] = i;
        }
        let maxxed = Tensor::create_tensor_data_and_shape_and_operation(Shape::new(new_dimensions), data, Operation::Max(self_permuted.id, dimension, keep_dim));
        return maxxed.permute(inv_perm);
    }
}

pub fn backwards_for_max(packet: BackproagationPacket) {
    if let Operation::Max(source, dimension, keep_dim) = packet.operation {
        let shape = get_equation().get_tensor_shape(source);
        let mut grad = vec![0.0f32;shape.total_size()];

        

    }
}