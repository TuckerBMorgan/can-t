use crate::central::*;
use std::ops::{Add, Sub};

use super::get_equation;
use crate::utils::handle_broadcasting;

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        // We want to make sure that the two operands can be added on a elementwise way
        // so we try to broadcast them together if they do not equal each other
        let (working_lfs, working_rhs) = handle_broadcasting(self, rhs);

        // We vend out the actual work to the equation, which in turn uses libs that take advantage of platform libs to speed it up
        let data = get_equation().add_tensors(working_lfs.id, working_rhs.id);
        let return_tensor = Tensor::create_tensor_data_and_shape_and_operation(
            working_lfs.shape,
            data,
            Operation::Add(working_lfs.id, working_rhs.id),
        );
        return return_tensor;
    }
}

/// overload the Add with f32 operator, it creates a little tensor to hold it
impl Add<f32> for Tensor {
    type Output = Self;
    fn add(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self + right_hand_as_tesnor
    }
}

/// Overload the sub operator for the Tensor struct
/// This will allow us to subtract two tensors together
/// it does it by negating the right hand side tensor and then adding them together
impl Sub for Tensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

// Overload the sub operator for the Tensor struct
// This will allow us to subtract a tensor and a f32 together
// it will turn the f32 into a tensor and then subtract them together
impl Sub<Tensor> for f32 {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(rhs.shape.clone(), self);
        right_hand_as_tesnor - rhs
    }
}

// Overload the sub operator for the Tensor struct
// This will allow us to subtract a tensor and a f32 together
// it will turn the f32 into a tensor and then subtract them together
impl Sub<f32> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self - right_hand_as_tesnor
    }
}

/// Overload F32 + tensor, flips it around to use existing add op
impl Add<Tensor> for f32 {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        rhs + self
    }
}

/// Handles calculating and passing back the gradient of an add operation
pub fn backward_for_add(backprop_backet: BackproagationPacket) {
    // The stored operation must be an Add; otherwise this backward is being called incorrectly.
    if let Operation::Add(left_hand_side, right_hand_side) = backprop_backet.operation {
        // In an addition operation:  z = x + y
        //
        // The derivative of z with respect to each input is 1:
        //   ∂z/∂x = 1
        //   ∂z/∂y = 1
        //
        // Therefore, the incoming gradient dL/dz is passed unchanged to both inputs:
        //   dL/dx = dL/dz
        //   dL/dy = dL/dz
        //
        // Here we fetch the incoming gradient buffer (dL/dz) for this node.
        let grad = backprop_backet
            .equation
            .get_grad_flat_buffer(backprop_backet.incoming_grad)
            .to_owned();

        // Add the incoming gradient to the right-hand side tensor's accumulated gradient.
        backprop_backet
            .equation
            .add_tensor_grad(right_hand_side, grad.to_vec());

        // Add the same incoming gradient to the left-hand side tensor's accumulated gradient.
        backprop_backet
            .equation
            .add_tensor_grad(left_hand_side, grad.to_vec());
    } else {
        panic!("Wrong operation for backwards add");
    }
}

#[cfg(test)]
mod tests {
    use crate::central::Shape;
    use crate::central::Tensor;
    use crate::utils::GGUFFile;
    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    pub fn basic_add_test() {
        let a = Tensor::element(Shape::new(vec![1]), 1.0);
        let b = Tensor::element(Shape::new(vec![1]), 1.0);
        let c = a + b;
        let c_item = c.item();
        assert!(c_item[0] == 2.0);
    }

    #[test]
    pub fn basic_2d_test() {
        let a = Tensor::element(Shape::new(vec![5, 5]), 5.0);
        let b = Tensor::element(Shape::new(vec![5, 5]), 1.0);
        let c = a + b;
        let c_item = c.item();
        for i in c_item {
            assert!(i == 6.0);
        }
    }

    #[test]
    pub fn basic_3d_test() {
        let a = Tensor::element(Shape::new(vec![5, 5, 5]), 5.0);
        let b = Tensor::element(Shape::new(vec![5, 5, 5]), 7.0);
        let c = a + b;
        let c_item = c.item();
        for i in c_item {
            assert!(i == 12.0);
        }
    }

    #[test]
    pub fn basic_3d_brodcast() {
        let a = Tensor::element(Shape::new(vec![5, 5, 5]), 5.0);
        let b = Tensor::element(Shape::new(vec![5, 5, 5]), 7.0);
        let c = a + b;
        let c_item = c.item();
        for i in c_item {
            assert!(i == 12.0);
        }
    }

    #[test]
    pub fn basic_4d_test() {
        let a = Tensor::element(Shape::new(vec![5, 6, 3, 10]), 5.0);
        let b = Tensor::element(Shape::new(vec![5, 6, 3, 10]), 5.0);
        let c = a + b;
        let c_item = c.item();
        for i in c_item {
            assert!(i == 10.0);
        }
    }

    #[test]
    pub fn basic_2d_broadcast_test() {
        let a = Tensor::element(Shape::new(vec![1, 10]), 5.0);
        let b = Tensor::element(Shape::new(vec![3, 10]), 5.0);
        let c = a + b;
        let c_item = c.item();
        assert!(
            c_item.len() == 30,
            "length of final array {:?}",
            c_item.len()
        );
        for i in c_item {
            assert!(i == 10.0);
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    pub fn do_not_add_tensors_of_different_size_but_same_length() {
        // The two tensors have different shapes, but whos length will be the same by the time
        // we get to the bare metal add, so we need to make sure other things catch this case
        let a = Tensor::element(Shape::new(vec![5, 6, 3, 10]), 5.0);
        let b = Tensor::element(Shape::new(vec![5, 3, 6, 10]), 5.0);
        let _ = a + b;
    }

    #[test]
    pub fn backprop_add_test() {
        let epsilon = 1e-5;
        let mut gguf_file =
            GGUFFile::new(String::from("./models/tests/add/add_broadcast_test.gguf"));

        let tensor_a =
            Tensor::from_gguf_file("add_broadcast_test_tensor_a".to_string(), &mut gguf_file);
        let tensor_b =
            Tensor::from_gguf_file("add_broadcast_test_tensor_b".to_string(), &mut gguf_file);
        let tensor_c_real =
            Tensor::from_gguf_file("add_broadcast_test_tensor_c".to_string(), &mut gguf_file);

        let tensor_c = tensor_a + tensor_b;
        let tensor_c_item = tensor_c.item();

        let together = tensor_c_item.iter().zip(tensor_c_real.item());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon));
        }
    }

    #[test]
    pub fn multiple_add_backprop_test() {
        let a = Tensor::element(Shape::new(vec![1]), 5.0);
        let b = Tensor::element(Shape::new(vec![1]), 10.0);
        let c = b + a;
        let d = Tensor::element(Shape::new(vec![1]), 20.9);
        let e = c + d;
        e.backward();
        assert!(a.grad()[0] == 1.0);
        assert!(b.grad()[0] == 1.0);
        assert!(c.grad()[0] == 1.0);
        assert!(d.grad()[0] == 1.0);
        assert!(e.grad()[0] == 1.0);
    }

    //TODO: undo this comment once we have operations for reducing multidimension arrays to
    // single values
    #[test]
    pub fn multiple_2d_add_backprop_test() {
        let a = Tensor::element(Shape::new(vec![10, 5]), 5.0);
        let b = Tensor::element(Shape::new(vec![10, 5]), 10.0);
        let c = b + a;
        let c = c.sum(vec![1], false);
        let c = c.sum(vec![0], true);
        c.backward();

        assert!(c.shape.total_size() == 1);
        let data = c.grad();
        for datum in data {
            assert!(datum == 1.0, "Grad should be 1.0, was {}", datum);
        }
    }

    #[test]
    pub fn multiple_3d_add_backprop_test() {
        let a = Tensor::element(Shape::new(vec![5, 5, 5]), 5.0);
        let b = Tensor::element(Shape::new(vec![5, 5, 5]), 10.0);
        let c = b + a;
        let c = c.sum(vec![2], false);
        let c = c.sum(vec![1], false);
        let c = c.sum(vec![0], true);
        c.backward();

        assert!(c.shape.total_size() == 1);
        let data = c.grad();
        for datum in data {
            assert!(datum == 1.0);
        }
    }
}
