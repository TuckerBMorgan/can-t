use std::collections::HashMap;

use ndarray::ArrayD;
use rand_distr::{Distribution, Normal};
use super::{shape::*, InternalTensor, TensorID};

//use cant_cpu::prelude::*;
use cant_metal::prelude::*;

macro_rules! extract_tensor_data {
    ($record:expr, $key:expr, $data:expr) => {{
        let internal_tensor = $record.get(&$key).unwrap();
        let shape = internal_tensor.shape;
        &$data[internal_tensor.data_start_index..(internal_tensor.data_start_index + shape.total_size())]
    }};
}

pub struct Equation {
    data: Vec<f32>, // All of the data in all of the tensors, kept in one place
    grad: Vec<f32>, // All of the grads for all of the tensors, kept in one place
    tensor_count: usize, // the number of tensors we have allocated, used for tensor ids
    tensor_record: HashMap<TensorID, InternalTensor>, // A lookup table from the tensor id to a internal tensor \
        // (which has the information needed to find tensors in data and grad),
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
    /// * 'data' - the data that backs the tensor. It is a flat buffer for simplicty of storagee
    pub fn allocate_tensor(&mut self, shape: Shape, data: Vec<f32>) -> TensorID {

        let id = self.allocate_tensor_id();
        let total_size = shape.total_size();

        // Allocate our data
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

    /// Allocates a tensor, and returns the ID of it, so it can be looked up later, is all zeroes
    /// # Arguments
    /// * 'shape' - the shape of the tensor. shape.total_size * 2 memory will be allocated(data and grad)
    pub fn allocate_zero_tensor(&mut self, shape: Shape) -> TensorID {
        let zero = vec![0.0;shape.total_size()];
        return self.allocate_tensor(shape, zero);
    }

    /// Allocates a tensor, and returns the ID of it, so it can be looked up later, will be filled with element
    /// # Arguments
    /// * 'shape' - the shape of the tensor. shape.total_size * 2 memory will be allocated(data and grad)
    /// * 'element' - the value that will be filled in all positions
    pub fn allocate_from_element(&mut self, shape: Shape, element: f32) -> TensorID {
        let data = vec![element; shape.total_size()];
        return self.allocate_tensor(shape, data);
    }

    pub fn allocate_random_tesnor(&mut self, shape: Shape) -> TensorID{
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.01).unwrap();
        let data: Vec<f32> = (0..shape.total_size())
            .map(|_| normal.sample(&mut rng))
            .collect();
        return self.allocate_tensor(shape, data);
    }

    /// Allocated a tensor id, this is unique id for each tensor, used to look in up later
    fn allocate_tensor_id(&mut self) -> TensorID {
        self.tensor_count += 1;
        return TensorID {
            id: self.tensor_count
        };
    }

    /// Takes two tensors as flat buffers, adds them together at an elementwise level and returns the result
    /// # Arguments
    /// * 'a' : The first tensor
    /// * 'b' : the second tensor
    pub fn add_tensors(&mut self, a: TensorID, b: TensorID) -> TensorID {
        
        // Get the left side of the add
        let left_data = extract_tensor_data!(self.tensor_record, a, self.data);

        // Get the the right side of the add
        let right_data = extract_tensor_data!(self.tensor_record, b, self.data);

        // Preform the operation of the platform vended version of tensor_add 
        let result_data =  tensor_add(left_data, right_data);
        let a_shape = self.tensor_record.get(&a).unwrap().shape;
        let allocate_tensor = self.allocate_tensor(a_shape, result_data);
        return allocate_tensor;
    }

    /// Returns the underlaying data of a tensor, as an array
    /// # Arugments
    /// 'id' - Id for lookup of the tensor
    pub fn get_item(&self, id: TensorID) -> ArrayD<f32>{
        let internal_tensor = self.tensor_record.get(&id).unwrap();
        let shape = internal_tensor.shape;
        let data = &self.data[internal_tensor.data_start_index..(internal_tensor.data_start_index + shape.total_size())];
        let array = ArrayD::from_shape_vec(shape.as_ndarray_shape(), data.to_vec()).unwrap();
        return array;
    }
}