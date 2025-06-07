use ndarray::ArrayD;

use crate::utils::GGUFFile;

use super::{Operation, Shape, get_equation, operation};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TensorID {
    pub id: usize,
}

/// A internal record keeping version of a Tensor, holds the pointers to the data and grad of the array
pub struct InternalTensor {
    pub id: TensorID, // the unique id for this tensor, it is used by the equation to look up the data
    pub shape: Shape, // the shape this tensor has
    pub data_start_index: usize, // where the equation.data this tensors data starts
    pub grad_start_index: usize, // where in equation.grad this tensors grad starts
    pub operation: Operation, // Ther operation that created this tensor, Nop for an allocation
}

impl InternalTensor {
    pub fn new(
        id: TensorID,
        shape: Shape,
        data_start_index: usize,
        grad_start_index: usize,
        operation: Operation,
    ) -> InternalTensor {
        InternalTensor {
            id,
            shape,
            data_start_index,
            grad_start_index,
            operation,
        }
    }

    /// Returns which Tensors created this operation
    pub fn dependencies(&self) -> Vec<TensorID> {
        match &self.operation {
            Operation::Nop => {
                return vec![];
            }
            Operation::Add(left, right) => {
                return vec![*left, *right];
            }
            Operation::BroadCast(from, _shape) => {
                return vec![*from];
            }
            Operation::Mul(left, right) => {
                return vec![*left, *right];
            }
            Operation::Sum(from, _, _, _) => {
                return vec![*from];
            },
            Operation::Pow(from, _power) => {
                return vec![*from];
            },
            Operation::Matmul(left, right, ) => {
                return vec![*left, *right];
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Tensor {
    pub id: TensorID, // The unique id for this tensor, ties it to the InternalTensor that can be used to look up the data
    pub shape: Shape, // The shape of the tensor
    operation: Operation, // The operation that created this Tensor(Nop for basic allocations)
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
            operation: Operation::Nop,
        }
    }

    // Loads a tensor from a GGUF file
    // # Arugments
    // * 'tensor_name' - The name of the tensor in the file
    // * 'gguf_file' - the data struct that wraps the file we will be loading
    pub fn from_gguf_file(tensor_name: String, gguf_file: &mut GGUFFile) -> Tensor {
        //    let id = get_equation().allocate_zero_tensor(shape);
        let data = gguf_file.get_weight_for_tensor(tensor_name.clone());
        let tensor_data = gguf_file.get_tensor(tensor_name);

        return  Tensor::create_tensor_data_and_shape_and_operation(Shape::new(tensor_data.dimensions), data, Operation::Nop);
    }

    /// Allocates a new tensor with provided shape, all with 0s
    /// # Arugments
    /// * 'shape' - The shape of the allocated tensor
    pub fn zeros(shape: Shape) -> Tensor {
        let id = get_equation().allocate_zero_tensor(shape);
        Tensor {
            id,
            shape,
            operation: Operation::Nop,
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
            operation: Operation::Nop,
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
            operation: Operation::Nop,
        }
    }

    /// Utility function for create a tensor with data, shape and operation
    /// # Arugments
    /// 'shape' - the shape of the tensor
    /// 'data' - the data
    pub fn create_tensor_data_and_shape_and_operation(
        shape: Shape,
        data: Vec<f32>,
        operation: Operation,
    ) -> Tensor {
        assert!(
            shape.total_size() == data.len(),
            "You cannot create a tensor with a shape of different size then data : shape size {} data length {}",
            shape.total_size(),
            data.len()
        );
        let id = get_equation().allocate_tensor(shape, data, operation);

        return Tensor {
            id,
            shape,
            operation: operation,
        };
    }

    /// Returns the underlaying tensor as an Array
    pub fn item(&self) -> ArrayD<f32> {
        return get_equation().get_item(self.id);
    }

    /// returns the underlaying grad of this tensor as an Array
    pub fn grad(&self) -> ArrayD<f32> {
        return get_equation().get_grad(self.id);
    }

    /// Broadcasts two tensors togethers to form a new broadcast shape
    /// # Aruguments
    /// 'shape' - The shape we want to broadcast to
    pub fn broadcast(&self, shape: Shape) -> Tensor {
        // We need to get the shape that we are going to brodcast from
        // as broadcast shape can be a combination of local shape
        // and then shape we are brocasting to
        // EX:
        // [1, 4, 3] bc [4, 1, 3] = [4, 3, 3]
        assert!(shape.can_broadcast(self.shape));

        let broad_cast_shape = shape.broadcast_shape(self.shape);
        // use the ndarry lib to do this, as I am sure I would fuck it up
        // And I don't know if there is a hardware accerlated way of doing this faster
        let data: Vec<f32> = self
            .item()
            .broadcast(broad_cast_shape.as_ndarray_shape())
            .unwrap()
            .iter()
            .map(|x| *x)
            .collect();

        return Tensor::create_tensor_data_and_shape_and_operation(
            broad_cast_shape,
            data,
            Operation::BroadCast(self.id, shape),
        );
    }

    /// sends this node backwards though the network, adding to the grad of every node that feeds into this one
    pub fn backward(&self) {
        assert!(
            self.shape.total_size() == 1,
            "You may only pass back a loss of size 1"
        );
        get_equation().backward(self.id);
    }


}

#[cfg(test)]
mod tests {
    use super::Tensor;
    use crate::central::Shape;

    #[test]
    pub fn allocate_test() {
        let _tensor = Tensor::new(Shape::new(vec![1, 2, 3]));
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
        let _tensor = Tensor::randn(Shape::new(vec![1, 2, 3]));
    }
}
