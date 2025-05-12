use std::collections::HashMap;

use super::{shape::*, InternalTensor, TensorID};



pub struct Equation {
    data: Vec<f32>, // All of the data in all of the tensors, kept in one place
    grad: Vec<f32>, // All of the grads for all of the tensors, kept in one place
    tensor_count: usize, // the number of tensors we have allocated, used for tensor ids
    tensor_record: HashMap<TensorID, InternalTensor> // A lookup table from the tensor id to a internal tensor \
        // (which has the information needed to find tensors in data and grad)
}

impl Equation {
    /// Create a new equation, DO NOT USE DIRECTLY
    /// use get_equation()
    pub fn new() -> Equation {
        Equation {
            data: vec![],
            grad: vec![],
            tensor_count: 0,
            tensor_record: HashMap::new()
        }
    }
    
    /// Allocates a tensor, and returns the ID of it, so it can be looked up later
    /// # Arguments
    /// * 'shape' - the shape of the tensor. shape.total_size * 2 memory will be allocated(data and grad)
    pub fn allocate_tensor(&mut self, shape: Shape) -> TensorID {

        let id = self.allocate_tensor_id();
        let total_size = shape.total_size();

        // Allocate our data
        let data = vec![0.0;total_size];
        let grad = vec![0.0;total_size];
        let data_start = self.data.len();
        let grad_stat = self.grad.len();

        //Do some record keeping
        let internal_tensor = InternalTensor::new(id, shape, data_start, grad_stat);
        self.tensor_record.insert(id, internal_tensor);

        // Extend the vectors by the right length, with 0 init data
        self.data.extend(data);
        self.grad.extend(grad);

        return id;
    }

    /// Allocated a tensor id, this is unique id for each tensor, used to look in up later
    fn allocate_tensor_id(&mut self) -> TensorID {
        self.tensor_count += 1;
        return TensorID {
            id: self.tensor_count
        };
    }
}