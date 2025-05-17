
use ndarray::ArrayD;

use super::{get_equation, Shape, Operation};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TensorID {
    pub id: usize
}

pub struct InternalTensor {
    pub id: TensorID, // the unique id for this tensor, it is used by the equation to look up the data
    pub shape: Shape, // the shape this tensor has
    pub data_start_index: usize, // where the equation.data this tensors data starts
    pub grad_start_index: usize, // where in equation.grad this tensors grad starts
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
    pub id: TensorID, // The unique id for this tensor, ties it to the InternalTensor that can be used to look up the data
    pub shape: Shape, // The shape of the tensor
    opeartion: Operation // The operation that created this Tensor(Nop for basic allocations)
}

impl Tensor {
    /// Allocates a new tensor of the provided shape, calls equation.allocate_tensor
    /// # Arugments
    /// * 'shape' - The shape of the allocated tensor
    pub fn new(shape: Shape) -> Tensor {
        let id = get_equation().allocate_zero_tensor(shape);
        Tensor {
            id,
            shape,
            opeartion: Operation::Nop
        }
    }

    /// Allocates a new tensor with provided shape, all with 0s
    /// # Arugments
    /// * 'shape' - The shape of the allocated tensor
    pub fn zeros(shape: Shape) -> Tensor {
        let id = get_equation().allocate_zero_tensor(shape);
        Tensor {
            id,
            shape,
            opeartion: Operation::Nop
        }
    }

    /// Allocates a new tensor with provided shape, where each element is element
    /// # Arugments
    /// * 'shape' - The shape of the allocated tensor
    /// * 'element' - The value that will be set for each element
    pub fn element(shape: Shape, element: f32) -> Tensor {
        let id = get_equation().allocate_from_element(shape, element);
        Tensor {
            id,
            shape,
            opeartion: Operation::Nop
        }
    }

    /// Allocates a random tensor of provided shape
    /// # Arugments
    /// *'shape' - The shape of the random tensor
    pub fn randn(shape: Shape) -> Tensor {
        let id = get_equation().allocate_random_tesnor(shape);
        Tensor {
            id,
            shape,
            opeartion: Operation::Nop
        }
    }

    pub fn create_tensor_from_id_and_shape_and_operation(shape: Shape, tensor_id: TensorID, operation: Operation) -> Tensor {
        return Tensor {
            id: tensor_id,
            shape,
            opeartion: operation
        }
    }
    
    /// Returns the underlaying tensor as an Array
    pub fn item(&self) -> ArrayD<f32> {
        return get_equation().get_item(self.id);
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

    #[test]
    pub fn allocate_and_data_test() {
        let tensor = Tensor::new(Shape::new(vec![1, 2, 3]));
        let item = tensor.item();
        for datum in item {
            assert!(datum == 0.0);
        }
    }

    #[test]
    pub fn zeroes_test() {
        let tensor = Tensor::zeros(Shape::new(vec![1, 2, 3]));
        let item = tensor.item();
        for datum in item {
            assert!(datum == 0.0);
        }
    }

    #[test]
    pub fn element_test() {
        let tensor = Tensor::element(Shape::new(vec![1, 2, 3]), 42.0);
        let item = tensor.item();
        for datum in item {
            assert!(datum == 42.0);
        }
    }

    #[test]
    pub fn randn_test() {
        let tensor = Tensor::randn(Shape::new(vec![1, 2, 3]));
    }
}