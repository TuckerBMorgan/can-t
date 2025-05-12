
use super::{get_equation, shape, Shape};

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TensorID {
    pub id: usize
}

pub struct InternalTensor {
    id: TensorID, // the unique id for this tensor, it is used by the equation to look up the data
    shape: Shape, // the shape this tensor has
    data_start_index: usize, // where the equation.data this tensors data starts
    grad_start_index: usize // where in equation.grad this tensors grad starts
}

impl InternalTensor {
    pub fn new(id: TensorID, shape: Shape, data_start_index: usize, grad_start_index: usize) -> InternalTensor {
        InternalTensor {
            id,
            shape,
            data_start_index,
            grad_start_index
        }
    }
}

pub struct Tensor {
    id: TensorID, // The unique id for this tensor, ties it to the InternalTensor that can be used to look up the data
    shape: Shape // The shape of the tensor
}

impl Tensor {
    /// Allocates a new tensor of the provided shape, calls equation.allocate_tensor
    /// # Arugments
    /// * 'shape' - The shape of the allocated tensor
    pub fn new(shape: Shape) -> Tensor {
        let id = get_equation().allocate_tensor(shape);
        Tensor {
            id,
            shape
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::central::Shape;

    use super::Tensor;
    #[test]
    pub fn allocate_test() {
        let tensor = Tensor::new(Shape::new(vec![1, 2, 3]));
    }
}