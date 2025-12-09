use std::collections::{HashMap, HashSet};

use super::{
    InternalTensor, Operation, TensorID, add_op, chunk_op, cos_op, log, matmul_op, mean_op, mul_op,
    pow_op, reshape, select_op, shape::*, sin_op, std_op, sum_op, tanh_op, transpose_op,
};
use crate::central::index::Indexable;
use crate::central::{
    DebuggingOptions, backward_for_clamp, backwards_for_mask_fill, cat_op, diagonal_op, gather_op,
    max_op, movedim_op, permute, relu_op, sigmoid_op, softmax_op, topk_op, unsqueeze_op,
};
use crate::utils::*;
use ndarray::ArrayD;
use ndarray::Axis;
use rand_distr::{Distribution, Normal};

#[cfg(not(target_os = "macos"))]
use cant_cpu::prelude::*;

#[cfg(target_os = "macos")]
use cant_metal::prelude::*;

/// A Struct used by the backpropagation functions to help collect common function arugumnets into a single place
pub struct BackproagationPacket<'a> {
    pub incoming_grad: TensorID, // The id of the tensor that we are passing back into the network currently
    pub equation: &'a mut Equation, // a reference to the Equation to use for look ups
    pub operation: Operation,
}

// Macro to help make getting values out of the data store
macro_rules! extract_tensor_data {
    ($record:expr, $key:expr, $data:expr) => {{
        let internal_tensor = $record.get(&$key).unwrap();
        let shape = internal_tensor.shape;
        &$data[internal_tensor.data_start_index
            ..(internal_tensor.data_start_index + shape.total_size())]
    }};
}

// Macro to help make getting values out of the grad store
macro_rules! extract_tensor_grad {
    ($record:expr, $key:expr, $grad:expr) => {{
        let internal_tensor = $record.get(&$key).unwrap();
        let shape = internal_tensor.shape;
        &$grad[internal_tensor.grad_start_index
            ..(internal_tensor.grad_start_index + shape.total_size())]
    }};
}

/// Workhorse struct that owns all the data of the equation
/// also is the interface to go through to take advantage of platform code that speeds
/// common tensor opeartions
pub struct Equation {
    data: Vec<f32>,      // All of the data in all of the tensors, kept in one place
    grad: Vec<f32>,      // All of the grads for all of the tensors, kept in one place
    tensor_count: usize, // the number of tensors we have allocated, used for tensor ids
    tensor_record: HashMap<TensorID, InternalTensor>, // A lookup table from the tensor id to a internal tensor \
    // (which has the information needed to find tensors in data and grad),
    timing: Timing,
    _debugging_options: DebuggingOptions,
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
            timing: Timing::new(),
            _debugging_options: DebuggingOptions::default(),
        }
    }

    /// Allocates a tensor, and returns the ID of it, so it can be looked up later
    /// # Arguments
    /// * 'shape' - the shape of the tensor. shape.total_size * 2 memory will be allocated(data and grad)
    /// * 'data' - the data that backs the tensor. It is a flat buffer for simplicty of storagee
    pub fn allocate_tensor(
        &mut self,
        shape: Shape,
        data: Vec<f32>,
        operation: Operation,
    ) -> TensorID {
        let id = self.allocate_tensor_id();
        let total_size = shape.total_size();

        // Allocate our data
        let grad = vec![0.0; total_size];
        let data_start = self.data.len();
        let grad_stat = self.grad.len();

        //Do some record keeping
        let internal_tensor = InternalTensor::new(id, shape, data_start, grad_stat, operation);
        self.tensor_record.insert(id, internal_tensor);

        if self._debugging_options.nan_check {
            for val in &data {
                if val.is_nan() {
                    panic!("Tried to allocate tensor with Nan Value");
                }
            }
        }

        // Extend the vectors by the right length, with 0 init data
        self.data.extend(data);
        self.grad.extend(grad);

        return id;
    }

    /// Allocates a tensor, and returns the ID of it, so it can be looked up later, is all zeroes
    /// # Arguments
    /// * 'shape' - the shape of the tensor. shape.total_size * 2 memory will be allocated(data and grad)
    pub fn allocate_zero_tensor(&mut self, shape: Shape) -> TensorID {
        let zero = vec![0.0; shape.total_size()];
        return self.allocate_tensor(shape, zero, Operation::Nop);
    }

    /// Allocates a tensor, and returns the ID of it, so it can be looked up later, is all ones
    /// # Arguments
    /// * 'shape' - the shape of the tensor. shape.total_size * 2 memory will be allocated(data and grad)
    pub fn allocate_ones_tensor(&mut self, shape: Shape) -> TensorID {
        let zero = vec![1.0; shape.total_size()];
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
    pub fn allocate_random_tesnor(&mut self, shape: Shape) -> TensorID {
        let fan_in = if shape.dimensions().len() >= 2 {
            shape.dimensions()[0]
        } else {
            1
        };
        let fan_out = if shape.dimensions().len() >= 2 {
            shape.dimensions()[1]
        } else {
            shape.total_size()
        };
        let std_dev = (2.0 / (fan_in + fan_out) as f32).sqrt();

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, std_dev as f64).unwrap();
        let data: Vec<f32> = (0..shape.total_size())
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();
        self.allocate_tensor(shape, data, Operation::Nop)
    }

    /// Allocated a tensor id, this is unique id for each tensor, used to look in up later
    fn allocate_tensor_id(&mut self) -> TensorID {
        self.tensor_count += 1;
        return TensorID {
            id: self.tensor_count,
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
        let result_data = tensor_add(left_data, right_data);
        return result_data;
    }

    /// Takes two tensors are flat buffers and preforms matmul on them and returns the result
    /// # Arugments
    /// 'a' - the first tensor
    /// 'b' - The seconf tensor
    pub fn matmul_tensor(&self, a: TensorID, b: TensorID) -> Vec<f32> {
        // Get the left side of the add
        let left_data = extract_tensor_data!(self.tensor_record, a, self.data);
        // Get the the right side of the add
        let right_data = extract_tensor_data!(self.tensor_record, b, self.data);

        // tensor_matmul is written "only" with 4dx4d tensors
        // this works by just assuming that all any tensor less then 4 dimensions
        // is actually 4d, with just a 1 in the missing dimensions
        // so for example: a tensor of dimenion [2, 3] can be see as [1, 1, 2, 3]
        // since that does not actually increase the number of elements in the array
        // and in cant dimensions over 2 are treated as batches, and are not part
        // core matmul action
        let mut a_shape = self.tensor_record[&a].shape.dimensions();
        let mut b_shape = self.tensor_record[&b].shape.dimensions();

        // So we need to get the shapes of our two operands
        let a_shape_missing_dimensions = MAX_DIMS - a_shape.len();
        for _ in 0..a_shape_missing_dimensions {
            a_shape.insert(0, 1);
        }

        // and insert 1s in front of them, to pad out the entire shape
        let b_shape_missing_dimensions = MAX_DIMS - b_shape.len();
        for _ in 0..b_shape_missing_dimensions {
            b_shape.insert(0, 1);
        }

        if b_shape_missing_dimensions == 9 {
            b_shape.swap(8, 9);
        }

        let a_shape = [
            a_shape[0], a_shape[1], a_shape[2], a_shape[3], a_shape[5], a_shape[5], a_shape[6],
            a_shape[7], a_shape[8], a_shape[9],
        ];
        let b_shape = [
            b_shape[0], b_shape[1], b_shape[2], b_shape[3], b_shape[5], b_shape[5], b_shape[6],
            b_shape[7], b_shape[8], b_shape[9],
        ];
        // Before call the actually function that does the matmul
        return tensor_matmul(left_data, a_shape, right_data, b_shape);
    }

    /// takes two matrices as flat vectors and preforms matmul of them
    /// 'a' - the flat array of the first matrix
    /// 'a_shape' - the shape of a
    /// 'b' - the flat array of the second matrix
    /// 'b_shape' - the shape of b
    pub fn matmul_vector(
        &self,
        a: &[f32],
        a_shape: [usize; MAX_DIMS],
        b: &[f32],
        b_shape: [usize; MAX_DIMS],
    ) -> Vec<f32> {
        //TODO: update this to handle MAX_DIM when we get around to ti
        let a_shape = [
            a_shape[0], a_shape[1], a_shape[2], a_shape[3], a_shape[5], a_shape[5], a_shape[6],
            a_shape[7], a_shape[8], a_shape[9],
        ];
        let b_shape = [
            b_shape[0], b_shape[1], b_shape[2], b_shape[3], b_shape[5], b_shape[5], b_shape[6],
            b_shape[7], b_shape[8], b_shape[9],
        ];
        return tensor_matmul(a, a_shape, b, b_shape);
    }

    /// Takes two tensors as flat buffers, adds them together at an elementwise level and returns the result
    /// # Arguments
    /// * 'a' : The first tensor
    /// * 'b' : the second tensor
    pub fn add_vector(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        // Preform the operation of the platform vended version of tensor_add
        let result_data = tensor_add(a, b);
        return result_data;
    }

    /// Takes two tensors as flat buffers, subs them together at an elementwise level and returns the result
    /// # Arguments
    /// * 'a' : The first tensor
    /// * 'b' : the second tensor
    pub fn sub_vector(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        // Preform the operation of the platform vended version of tensor_add
        let result_data = tensor_sub(a, b);
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
        let result_data = tensor_mul(left_data, right_data);
        return result_data;
    }

    /// Takes two tensors as flat buffers, multiplies them together at an elementwise level and returns the result
    /// # Arguments
    /// * 'a' : The first tensor
    /// * 'b' : the second tensor
    pub fn mul_vector(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        // Preform the operation of the platform vended version of tensor_mul
        let result_data = tensor_mul(a, b);
        return result_data;
    }

    /// Returns the underlaying data of a tensor, as an array
    /// # Arugments
    /// 'id' - Id for lookup of the tensor
    pub fn get_item(&self, id: TensorID) -> ArrayD<f32> {
        let internal_tensor = self.tensor_record.get(&id).unwrap();
        let shape = internal_tensor.shape;
        let data = &self.data[internal_tensor.data_start_index
            ..(internal_tensor.data_start_index + shape.total_size())];
        let array = ArrayD::from_shape_vec(shape.as_ndarray_shape(), data.to_vec()).unwrap();
        return array;
    }

    /// Returns the underlaying grad of a tensor, as an array
    /// # Arugments
    /// 'id' - Id for lookup of the tensor
    pub fn get_grad(&self, id: TensorID) -> ArrayD<f32> {
        let internal_tensor = self.tensor_record.get(&id).unwrap();
        let shape = internal_tensor.shape;
        let grad = &self.grad[internal_tensor.data_start_index
            ..(internal_tensor.data_start_index + shape.total_size())];
        let array = ArrayD::from_shape_vec(shape.as_ndarray_shape(), grad.to_vec()).unwrap();
        return array;
    }

    /// Gets the underlaying backing storage of the tensor, util function for working with grad of a tensor
    /// # Arguments
    /// 'tensor_id' - The tensor we are working with
    pub fn get_grad_flat_buffer(&self, tensor_id: TensorID) -> &[f32] {
        return extract_tensor_grad!(self.tensor_record, tensor_id, self.grad);
    }

    /// Gets the underlaying backing storage of the tensor, util function for working with data of a tensor
    /// # Arguments
    /// 'tensor_id' - The tensor we are working with
    pub fn get_data_flat_buffer(&self, tensor_id: TensorID) -> &[f32] {
        return extract_tensor_data!(self.tensor_record, tensor_id, self.data);
    }

    /// Helper function to get the shape of a tensor, without getting the entire tensor
    /// # Arguments
    /// 'tensor_id' - the tensor you are getting the shape for
    pub fn get_tensor_shape(&self, tensor_id: TensorID) -> Shape {
        return self.tensor_record[&tensor_id].shape;
    }

    /// Helper function swap around two axis
    pub fn swap_axes<const MAX_DIMS: usize>(
        &self,
        data: &mut [f32],
        dimensions: [usize; MAX_DIMS],
        axis1: usize,
        axis2: usize,
    ) {
        assert!(axis1 < MAX_DIMS && axis2 < MAX_DIMS, "axis out of bounds");
        if axis1 == axis2 || data.is_empty() {
            return;
        }

        // (Optional) sanity check: product of dims must match data len
        let mut total = 1usize;
        for &d in &dimensions {
            total = total.checked_mul(d).expect("dimension product overflow");
        }
        assert!(
            total == data.len(),
            "data length {} does not match product of dimensions {}",
            data.len(),
            total
        );

        // Scratch buffer
        let mut tmp = vec![0.0f32; data.len()];

        // Original and new dims after swapping the two axes
        let orig_dims = dimensions;
        let mut new_dims = orig_dims;
        new_dims.swap(axis1, axis2);

        // Compute strides (row-major) for original dims
        let mut strides = [0usize; MAX_DIMS];
        strides[MAX_DIMS - 1] = 1;
        for i in (0..MAX_DIMS - 1).rev() {
            strides[i] = strides[i + 1] * orig_dims[i + 1];
        }

        // Compute strides for new dims
        let mut new_strides = [0usize; MAX_DIMS];
        new_strides[MAX_DIMS - 1] = 1;
        for i in (0..MAX_DIMS - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_dims[i + 1];
        }

        // For each flat index in the original layout:
        //  1) decode to multi-index using original strides
        //  2) swap the two axes in that multi-index
        //  3) re-encode with new strides to get destination flat index
        for orig_idx in 0..data.len() {
            // Decode
            let mut rem = orig_idx;
            let mut idx = [0usize; MAX_DIMS];
            for d in 0..MAX_DIMS {
                let s = strides[d];
                // For dimensions of size 0 (shouldn't happen), s could be 0; we asserted sizes multiply to len,
                // so this is safe.
                idx[d] = rem / s;
                rem %= s;
            }

            // Swap the two axes
            idx.swap(axis1, axis2);

            // Encode with new strides
            let mut new_idx = 0usize;
            for d in 0..MAX_DIMS {
                new_idx += idx[d] * new_strides[d];
            }

            tmp[new_idx] = data[orig_idx];
        }

        data.copy_from_slice(&tmp);
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
        if !self.tensor_record.contains_key(&tensor_id) {
            panic!("{:?} is missing from tensor record", tensor_id);
        }

        if self._debugging_options.nan_check {
            for val in &grad {
                if val.is_nan() {
                    panic!("NaN grad found");
                }

                if val.is_infinite() {
                    panic!("Infnite grad found");
                }
            }
        }

        let internal_tensor = &self.tensor_record[&tensor_id];

        assert!(internal_tensor.shape.total_size() == grad.len());

        for i in 0..internal_tensor.shape.total_size() {
            self.grad[internal_tensor.grad_start_index + i] += grad[i];
        }
    }

    pub fn set_single_value(&mut self, tensor_id: TensorID, index: Indexable, value: f32) {
        let internal_tensor = &self.tensor_record[&tensor_id];
        let mut final_index;
        match index {
            Indexable::Quadruable(a, b, c, d) => {
                let mut current_shape = internal_tensor.shape.dimensions();
                let missing_indices = 4 - current_shape.len();
                for _ in 0..missing_indices {
                    current_shape.insert(0, 1);
                }

                // Calculate strides for row-major order
                let stride_0 = current_shape[1] * current_shape[2] * current_shape[3];
                let stride_1 = current_shape[2] * current_shape[3];
                let stride_2 = current_shape[3];
                let stride_3 = 1;

                final_index = a * stride_0 + b * stride_1 + c * stride_2 + d * stride_3;

                final_index += internal_tensor.data_start_index;
                self.data[final_index] = value;
            }
            _ => {
                panic!(
                    "Should not be calling set_single_value with anything other then a quadruable indexable"
                );
            }
        }
    }

    /// Util function that handles passing back a single value in the equation
    /// # Arugments
    /// 'incoming_grad' - the tensor that we are passing back
    pub fn backward_for_value(&mut self, incoming_grad: TensorID) {
        let internal_tensor = self.tensor_record.get(&incoming_grad).unwrap();

        // collects a lot of common aruguments into a single struct
        // helps avoid needed to update a bunch of function signatures
        // in the future
        let operation = internal_tensor.operation.clone();
        let packet: BackproagationPacket<'_> = BackproagationPacket {
            incoming_grad: incoming_grad,
            equation: self,
            operation,
        };

        match operation {
            Operation::Nop => {
                // Intentionally left blank
            }
            Operation::Add(_left_hand_side, _right_hand_side) => {
                add_op::backward_for_add(packet);
            }
            Operation::BroadCast(from, to_shape) => {
                let from_grad = self.get_grad(from);
                let from_shape = from_grad.shape();
                let from_shape = padding_dimenions_to_max(from_shape.to_vec());
                let dimensions = to_shape.dimensions();
                let dimensions = padding_dimenions_to_max(dimensions);
                let mut result = self.get_grad(incoming_grad);

                // Sum dimensions from right to left, accounting for changing tensor dimensions
                // We need to track how many dimensions we've already summed
                let mut dims_summed = 0;
                for index in 0..from_shape.len() {
                    let input_dim = dimensions[dimensions.len() - 1 - index];
                    let original_dim = if index < from_shape.len() {
                        from_shape[from_shape.len() - 1 - index]
                    } else {
                        1
                    };

                    if original_dim == 1 && input_dim != 1 {
                        // Calculate the axis to sum, accounting for previously summed dimensions
                        let axis_to_sum = if result.ndim() > index - dims_summed {
                            result.ndim() - 1 - (index - dims_summed)
                        } else {
                            // Skip this if we've already reduced too many dimensions
                            continue;
                        };
                        result = result.sum_axis(Axis(axis_to_sum));
                        dims_summed += 1;
                    }
                }
                self.add_tensor_grad(from, result.to_owned().into_raw_vec());
            }
            Operation::Mul(_left_hand_side, _right_hand_side) => {
                mul_op::backward_for_mul(packet);
            }
            Operation::Sum(_, _, _, _) => {
                sum_op::backward_for_sum(packet);
            }
            Operation::Pow(_base, _power) => {
                pow_op::backward_for_pow(packet);
            }
            Operation::Matmul(_left, _right) => {
                matmul_op::backward_for_matmul(packet);
            }
            Operation::Reshape(_from, _shape) => {
                reshape::backward_for_reshape(packet);
            }
            Operation::Exp(from) => {
                let data = self.get_data_flat_buffer(incoming_grad);
                let grad = self.get_grad_flat_buffer(incoming_grad);
                let result = self.mul_vector(data, grad);
                self.add_tensor_grad(from, result);
            }
            Operation::Select(_source, _indices) => {
                select_op::backward_for_select(packet);
            }
            Operation::Tanh(_source) => {
                tanh_op::backward_for_tanh(packet);
            }
            Operation::Mean(_, _, _) => {
                mean_op::backward_for_mean(packet);
            }
            Operation::Std(_, _, _) => {
                std_op::backward_for_std(packet);
            }
            Operation::Softmax(_, _) => {
                softmax_op::backward_for_softmax(packet);
            }
            Operation::Log(_) => {
                log::backwards_for_log(packet);
            }
            Operation::Transpose(_, _, _) => {
                transpose_op::backwards_for_transpose(packet);
            }
            Operation::RELU(_) => {
                relu_op::backward_for_relu(packet);
            }
            Operation::MaskFill(_, _, _) => {
                backwards_for_mask_fill(packet);
            }
            Operation::Clamp(_, _, _) => {
                backward_for_clamp(packet);
            }
            Operation::Cos(_) => {
                cos_op::backward_for_cos(packet);
            }
            Operation::Sin(_) => {
                sin_op::backward_for_sin(packet);
            }
            Operation::Unsqueeze(_, _) => {
                unsqueeze_op::backward_for_unsqueeze(packet);
            }
            Operation::Chunk(_, _, _) => {
                chunk_op::backward_for_chunk(packet);
            }
            Operation::Cat(_, _, _) => {
                cat_op::backwards_for_cat(packet);
            }
            Operation::Diagonal(_, _, _, _) => {
                diagonal_op::backwards_for_diagonal(packet);
            }
            Operation::MoveDim(_, _, _) => {
                movedim_op::backward_for_movedim(packet);
            }
            Operation::Permute(_, _, _) => {
                permute::backward_for_permute(packet);
            }
            Operation::Topk(_, _, _, _, _, _) => {
                topk_op::backward_for_topk(packet);
            }
            Operation::Gather(_, _, _) => {
                gather_op::backward_for_gather(packet);
            }
            Operation::Max(_, _, _, _) => {
                max_op::backwards_for_max(packet);
            }
            Operation::Sigmoid(_) => {
                sigmoid_op::backward_for_sigmoid(packet);
            }
            Operation::SmoothL1Loss(_, _) => {
                panic!()
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
        if let Some(dependencies) = self.tensor_record.get(&node).map(|n| n.dependencies()) {
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

    pub fn trace_backwrads_path(&self, starting_value: TensorID) {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        self.topological_sort_util(starting_value, &mut visited, &mut stack);
        let mut output_string = String::from("");

        while let Some(node) = stack.pop() {
            let internal_tensor = self.tensor_record.get(&node).unwrap();
            output_string += &internal_tensor.id.id.to_string();
        }
    }

    // Zeroes out the grad, important to call before calling backwards on a value
    pub fn zero_grad(&mut self) {
        for g in &mut self.grad {
            *g = 0.0;
        }
        //        self.grad = vec![0.0;self.grad.len()];
    }

    pub fn clip_grad(&mut self, max_norm: f32) {
        for g in self.grad.iter_mut() {
            *g = g.clamp(-max_norm, max_norm);
        }
    }

    /// Updates all parameters for all tensor that are marked for needed gradients(set_requires_grad)
    /// Arguments
    /// 'learning_rate': a singe learning rate applied to all parameters
    pub fn update_parameters(&mut self, learning_rate: f32) {
        for (_k, v) in &self.tensor_record {
            if v.requires_grad {
                for i in 0..v.shape.total_size() {
                    let grad_anchor_point = v.grad_start_index;
                    let data_anchor_point = v.data_start_index;
                    let _update = learning_rate * self.grad[grad_anchor_point + i];
                    if self._debugging_options.nan_check {
                        if _update.is_nan() {
                            panic!("Tried to update parameter with NaN value");
                        }
                    }
                    self.data[data_anchor_point + i] +=
                        learning_rate * self.grad[grad_anchor_point + i];
                }
            }
        }
    }

    /// Updates the paramaters of a single tensor
    /// Arugments
    /// 'parameter': the tensor that will have its weights updated 
    /// 'learning_rate': the learning rate applied to each gradient
    pub fn update_single_parameter(&mut self, parameter: TensorID, learning_rate: f32, weight_decay: f32) {
        let v = &self.tensor_record[&parameter];
        if v.requires_grad {
            for i in 0..v.shape.total_size() {
                let grad_anchor_point = v.grad_start_index;
                let data_anchor_point = v.data_start_index;
                let _update = learning_rate * self.grad[grad_anchor_point + i];
                if self._debugging_options.nan_check {
                    if _update.is_nan() {
                        panic!("Tried to update parameter with NaN value");
                    }
                }
                self.data[data_anchor_point + i] = self.data[data_anchor_point + i] +
                // Standard applying the learning rate
                learning_rate * (self.grad[grad_anchor_point + i] + 
                        // This is applying weight decay
                        (self.data[data_anchor_point + i] * weight_decay));
            }
        }
        else {
            panic!("Update single parameter called on a tensor that does not have grad required: Tensor ID {:?}", parameter);
        }
    }

    /// Helper function for setting the internal tensor to know if it needs gradient or not
    /// # Arguments
    /// 'tensor_id' : which Tensor we are setting
    /// 'requires_grad' : what we are setting the bool too
    pub fn set_is_grequires_grad(&mut self, tensor_id: TensorID, requires_grad: bool) {
        self.tensor_record
            .get_mut(&tensor_id)
            .unwrap()
            .requires_grad = requires_grad;
    }

    /// Helper function for setting the internal tensor to know if it needs gradient or not
    /// # Arguments
    /// 'tensor_id' : which Tensor we are setting
    /// 'requires_grad' : what we are setting the bool too
    pub fn set_keep_alive(&mut self, tensor_id: TensorID, keep_alive: bool) {
        self.tensor_record.get_mut(&tensor_id).unwrap().keep_alive = keep_alive;
    }

    pub fn validate_tensor_store(&self) {
        for d in &self.data {
            if d.is_nan() || d.is_infinite() {
                if d.is_nan() {
                    panic!("Bad Data in tensor data, NaN value found");
                }

                if d.is_infinite() {
                    panic!("Bad Data is tensor data, infinite value found");
                }
            }
        }

        for d in &self.grad {
            if d.is_nan() || d.is_infinite() {
                if d.is_nan() {
                    panic!("Bad Data in tensor data, NaN value found");
                }

                if d.is_infinite() {
                    panic!("Bad Data is tensor data, infinite value found");
                }
            }
        }
    }

    pub fn compact_tensor_store(&mut self) {
        let mut new_data_store = Vec::with_capacity(self.data.len());
        let mut new_grad_store = Vec::with_capacity(self.grad.len());
        let mut offset_map = HashMap::new();

        for tensor in self.tensor_record.values_mut().filter(|t| t.keep_alive) {
            let size = tensor.shape.total_size();

            // Copy data
            let data_slice = &self.data[tensor.data_start_index..tensor.data_start_index + size];
            let grad_slice = &&self.grad[tensor.grad_start_index..tensor.grad_start_index + size];

            let new_data_start = new_data_store.len();
            new_data_store.extend_from_slice(data_slice);

            let new_grad_start = new_grad_store.len();
            new_grad_store.extend_from_slice(grad_slice);

            // Update tensor indices
            tensor.data_start_index = new_data_start;
            tensor.grad_start_index = new_grad_start;
            tensor.keep_alive = true;

            offset_map.insert(tensor.id, tensor); // if needed
        }

        // Replace data store
        self.data = new_data_store;
        self.grad = new_grad_store;

        // Remove tensors that were not copied
        self.tensor_record.retain(|_, t| t.keep_alive);
    }

    pub fn garbage_collect(&mut self) {
        self.compact_tensor_store();
    }
}
