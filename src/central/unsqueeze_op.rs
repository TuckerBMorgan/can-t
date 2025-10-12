use crate::central::*;

impl Tensor {
    /// Reshapes the underlaying tensor into a new shape
    /// it preforms a full copy under the hood
    pub fn unsqueeze(&self, dimension: isize) -> Tensor {
        let new_shape = self.shape.unsqueeze(dimension);
        // Get the original data source
        // this is in effect a copy for large tensors, so while I am not happy with it, I am ok with for the moment
        let data = self.item().into_raw_vec();
        let new_tensor = Tensor::create_tensor_data_and_shape_and_operation(
            new_shape,
            data,
            Operation::Unsqueeze(self.id, dimension),
        );
        return new_tensor;
    }
}

pub fn backward_for_unsqueeze(packet: BackproagationPacket) {
    if let Operation::Unsqueeze(from, _shape) = packet.operation {
        // The reshape operation is simple, since at the end of the day the number and order of
        // the elements does not change
        let grad = packet
            .equation
            .get_grad_flat_buffer(packet.incoming_grad)
            .to_owned();

        packet.equation.add_tensor_grad(from, grad.to_vec());
    }
}
