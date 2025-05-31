use std::collections::{HashMap, HashSet};


use ndarray::ArrayD;
use rand_distr::{Distribution, Normal};
use super::{add_op, mul_op, shape::*, sum, InternalTensor, Operation, TensorID};
use crate::utils::*;

#[cfg(target_os = "windows")]
use cant_cpu::prelude::*;

#[cfg(target_os = "macos")]
use cant_metal::prelude::*;

/// A Struct used by the backpropagation functions to help collect common function arugumnets into a single place
pub struct BackproagationPacket<'a> {
    pub incoming_grad: TensorID, // The id of the tensor that we are passing back into the network currently
    pub equation: &'a mut Equation, // a reference to the Equation to use for look ups
    pub operation: Operation
}

// Macro to help make getting values out of the data store
macro_rules! extract_tensor_data {
    ($record:expr, $key:expr, $data:expr) => {{
        let internal_tensor = $record.get(&$key).unwrap();
        let shape = internal_tensor.shape;
        &$data[internal_tensor.data_start_index..(internal_tensor.data_start_index + shape.total_size())]
    }};
}

// Macro to help make getting values out of the grad store
macro_rules! extract_tensor_grad {
    ($record:expr, $key:expr, $grad:expr) => {{
        let internal_tensor = $record.get(&$key).unwrap();
        let shape = internal_tensor.shape;
        &$grad[internal_tensor.grad_start_index..(internal_tensor.grad_start_index + shape.total_size())]
    }};
}

/// Workhorse struct that owns all the data of the equation 
/// also is the interface to go through to take advantage of platform code that speeds
/// common tensor opeartions
pub struct Equation {
    data: Vec<f32>, // All of the data in all of the tensors, kept in one place
    grad: Vec<f32>, // All of the grads for all of the tensors, kept in one place
    tensor_count: usize, // the number of tensors we have allocated, used for tensor ids
    tensor_record: HashMap<TensorID, InternalTensor>, // A lookup table from the tensor id to a internal tensor \
        // (which has the information needed to find tensors in data and grad),
    timing: Timing
}

impl Equation {
    /// Create a new equation, DO NOT USE DIRECTLY
    /// use get_equation()
    pub fn new() -> Equation {
        Equation {
            data: vec![],
            grad: vec![],
            tensor_count: 0,
            tensor_record: HashMap::new(),
            timing: Timing::new()
        }
    }
    
    /// Allocates a tensor, and returns the ID of it, so it can be looked up later
    /// # Arguments
    /// * 'shape' - the shape of the tensor. shape.total_size * 2 memory will be allocated(data and grad)
    /// * 'data' - the data that backs the tensor. It is a flat buffer for simplicty of storagee
    pub fn allocate_tensor(&mut self, shape: Shape, data: Vec<f32>, operation: Operation) -> TensorID {

        let id = self.allocate_tensor_id();
        let total_size = shape.total_size();

        // Allocate our data
        let grad = vec![0.0;total_size];
        let data_start = self.data.len();
        let grad_stat = self.grad.len();

        //Do some record keeping
        let internal_tensor = InternalTensor::new(id, shape, data_start, grad_stat, operation);
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
        return self.allocate_tensor(shape, zero, Operation::Nop);
    }

    /// Allocates a tensor, and returns the ID of it, so it can be looked up later, will be filled with element
    /// # Arguments
    /// * 'shape' - the shape of the tensor. shape.total_size * 2 memory will be allocated(data and grad)
    /// * 'element' - the value that will be filled in all positions
    pub fn allocate_from_element(&mut self, shape: Shape, element: f32) -> TensorID {
        let data = vec![element; shape.total_size()];
        return self.allocate_tensor(shape, data, Operation::Nop);
    }

    /// Allocates a tensor with random values
    /// # Arguments
    /// 'shape' - the shape of desired tensor
    pub fn allocate_random_tesnor(&mut self, shape: Shape) -> TensorID{
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.01).unwrap();
        let data: Vec<f32> = (0..shape.total_size())
            .map(|_| normal.sample(&mut rng))
            .collect();
        return self.allocate_tensor(shape, data, Operation::Nop);
    }

    /// Allocated a tensor id, this is unique id for each tensor, used to look in up later
    fn allocate_tensor_id(&mut self) -> TensorID {
        self.tensor_count += 1;
        return TensorID {
            id: self.tensor_count
        };
    }

    /// util function to start a time with a provided name
    /// # Aruguments
    /// 'name': the name of the timer
    pub fn start_timer(&mut self, name: String) {
        self.timing.start_clock_with_name(name);
    }

    /// util function to end a time with a provided name
    /// # Aruguments
    /// 'name': the name of the timer
    pub fn end_timer(&mut self, name: String) {
        self.timing.end_clock_with_name(name);
    }

    /// Takes two tensors as flat buffers, adds them together at an elementwise level and returns the result
    /// # Arguments
    /// * 'a' : The first tensor
    /// * 'b' : the second tensor
    pub fn add_tensors(&self, a: TensorID, b: TensorID) -> Vec<f32> {
        // Get the left side of the add
        let left_data = extract_tensor_data!(self.tensor_record, a, self.data);
        // Get the the right side of the add
        let right_data = extract_tensor_data!(self.tensor_record, b, self.data);
        // Preform the operation of the platform vended version of tensor_add 
        let result_data =  tensor_add(left_data, right_data);
        return result_data;
    }

    /// Takes two tensors as flat buffers, adds them together at an elementwise level and returns the result
    /// # Arguments
    /// * 'a' : The first tensor
    /// * 'b' : the second tensor
    pub fn add_vector(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        // Preform the operation of the platform vended version of tensor_add 
        let result_data =  tensor_add(a, b);
        return result_data;
    }

    /// Takes two tensors as flat buffers, multiplies them together at an elementwise level and returns the result
    /// # Arguments
    /// * 'a' : The first tensor
    /// * 'b' : the second tensor
    pub fn mul_tensors(&self, a: TensorID, b: TensorID) -> Vec<f32> {
        // Get the left side of the add
        let left_data = extract_tensor_data!(self.tensor_record, a, self.data);
        // Get the the right side of the add
        let right_data = extract_tensor_data!(self.tensor_record, b, self.data);
        // Preform the operation of the platform vended version of tensor_mul
        let result_data =  tensor_mul(left_data, right_data);
        return result_data;
    }

    /// Takes two tensors as flat buffers, multiplies them together at an elementwise level and returns the result
    /// # Arguments
    /// * 'a' : The first tensor
    /// * 'b' : the second tensor
    pub fn mul_vector(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        // Preform the operation of the platform vended version of tensor_mul
        let result_data =  tensor_mul(a, b);
        return result_data;
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

    /// Returns the underlaying grad of a tensor, as an array
    /// # Arugments
    /// 'id' - Id for lookup of the tensor
    pub fn get_grad(&self, id: TensorID) -> ArrayD<f32>{
        let internal_tensor = self.tensor_record.get(&id).unwrap();
        let shape = internal_tensor.shape;
        let grad = &self.grad[internal_tensor.data_start_index..(internal_tensor.data_start_index + shape.total_size())];
        let array = ArrayD::from_shape_vec(shape.as_ndarray_shape(), grad.to_vec()).unwrap();
        return array;
    }

    /// Gets the underlaying backing storage of the tensor, util function for working with grad of a tensor
    /// # Arguments
    /// 'tensor_id' - The tensor we are working with
    pub fn get_grad_flat_buffer(&mut self, tensor_id: TensorID) -> &[f32] {
        return extract_tensor_grad!(self.tensor_record, tensor_id, self.grad);
    }

    /// Gets the underlaying backing storage of the tensor, util function for working with data of a tensor
    /// # Arguments
    /// 'tensor_id' - The tensor we are working with
    pub fn get_data_flat_buffer(&self, tensor_id: TensorID) ->&[f32] {
        return extract_tensor_data!(self.tensor_record, tensor_id, self.data);
    }

    /// Copies data into the grad of tensor_id
    /// # Arugments
    /// 'tensor_id' - Id of for the loopup on the tensor
    /// 'grad' - the grad we are copying in
    pub fn set_tensor_grad(&mut self, tensor_id: TensorID, grad: Vec<f32>) {
        let internal_tensor = &self.tensor_record[&tensor_id];
        assert!(internal_tensor.shape.total_size() == grad.len());
        for i in 0..internal_tensor.shape.total_size() {
            self.grad[internal_tensor.grad_start_index + i] = grad[i];
        }
    }
    /// Adds new grad into the grad of tensor_id
    /// # Arugments
    /// 'tensor_id' - Id of for the loopup on the tensor
    /// 'grad' - the grad we are copying in
    pub fn add_tensor_grad(&mut self, tensor_id: TensorID, grad: Vec<f32>) {
        let internal_tensor = &self.tensor_record[&tensor_id];
        assert!(internal_tensor.shape.total_size() == grad.len());
        for i in 0..internal_tensor.shape.total_size() {
            self.grad[internal_tensor.grad_start_index + i] += grad[i];
        }
    }

    /// Util function that handles passing back a single value in the equation
    /// # Arugments
    /// 'tensor_id' - the tensor that we are passing back
    pub fn backward_for_value(&mut self, tensor_id: TensorID) {
        let internal_tensor = self.tensor_record.get(&tensor_id).unwrap();

        // collects a lot of common aruguments into a single struct
        // helps avoid needed to update a bunch of function signatures
        // in the future
        let operation = internal_tensor.operation.clone();
        let packet = BackproagationPacket {
            incoming_grad: tensor_id,
            equation: self,
            operation
        };

        match operation {
            Operation::Nop => {
                // Intentionally left blank
            },
            Operation::Add(_left_hand_side, _right_hand_side) => {
                add_op::backward_for_add(packet);
            },
            Operation::BroadCast(_from, _to_shape) => {
                panic!("time to implement broadcast");
            },
            Operation::Mul(_left_hand_side,_right_hand_side ) => {
                mul_op::backward_for_mul(packet);
            },
            Operation::Sum(_, _, _) => {
                sum::backward_for_sum(packet);
            }
        }
    }

    /// a util function to preform topological sort on the tensors of the equation for 
    /// the backward pass
    /// # Arugments
    /// 'node' - the id of the tensor that is going to preform the backwards pass
    /// 'visited' - hashmap of notes we have already met
    /// 'stack' - A stack of nodes we need to look at
    fn topological_sort_util(
        &self,
        node: TensorID,
        visited: &mut HashSet<TensorID>,
        stack: &mut Vec<TensorID>,
    ) {
        visited.insert(node);

        // Assuming 'dependencies' method returns all the nodes that the current node depends on
        if let Some(dependencies) = self
            .tensor_record
            .get(&node)
            .map(|n| n.dependencies())
        {
            for dep in dependencies {
                if !visited.contains(&dep) {
                    self.topological_sort_util(dep, visited, stack);
                }
            }
        }

        stack.push(node);
    }

    /// Preforms the backward pass of a value through the network
    /// it only sets the grad, it does not zero it out, or update the values
    /// # Arugments
    /// 'starting_value' - the node we are starting on backward pass on
    pub fn backward(&mut self, starting_value: TensorID) {
        self.start_timer(String::from("backwards_prop"));
        // Get a top sort of all of the nodes of the graph, starting with the node we
        // want to start at
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        self.topological_sort_util(starting_value, &mut visited, &mut stack);
        
        // seed backprop by setting the gradient of the output node(loss) to 1
        // this tells the system, how much does loss change as loss changes(the definition of a gradient), well 1:1
        // this lets backprop apply the chain rule backwards
        let one_grad = vec![1.0];
        self.set_tensor_grad(starting_value, one_grad);

        while let Some(node) = stack.pop() {
           let _ = self.backward_for_value(node);
        }
        
        self.end_timer(String::from("backwards_prop"));
    }
}