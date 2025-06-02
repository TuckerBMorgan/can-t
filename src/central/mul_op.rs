use super::get_equation;
use crate::central::*;
use std::ops::{Mul, Neg, Div};
use crate::utils::handle_broadcasting;

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        // We want to make sure that the two operands can be muled on a elementwise way
        let (working_lfs, working_rhs) = handle_broadcasting(self, rhs);

        // We vend out the actual work to the equation, which in turn uses libs that take advantage of platform libs to speed it up
        let data = get_equation().mul_tensors(working_lfs.id, working_rhs.id);
        let return_tensor = Tensor::create_tensor_data_and_shape_and_operation(
            working_lfs.shape,
            data,
            Operation::Mul(working_lfs.id, working_rhs.id),
        );
        return return_tensor;
    }
}

/// Overload the multiplication operator for the Tensor struct
/// This will allow us to multiply a tensor by a scalar
/// we will convert the scalar into a tensor and then multiply the two tensors together
impl Mul<f32> for Tensor {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self * right_hand_as_tesnor
    }
}

/// Convinence function to allow us to handle the element wise negation of a tensor
impl Neg for Tensor {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

/// Overload the Division operator for the Tensor struct
/// we are going to use the fact that a/b = a * b^-1
impl Div for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        // we take advantage of the fact that a/b = a * b^-1
        // to let us keep the code simplier
        let intermidiate = rhs.pow(-1.0);
        self * intermidiate
    }
}

/// Overload the Division operator for the Tensor struct
/// This will allow us to divide a tensor by a scalar
/// we will convert the scalar into a tensor and then divide the two tensors together
impl Div<f32> for Tensor {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self / right_hand_as_tesnor
    }
}

/// Overload the Multiplication operator for the f32 struct
/// This will allow us to multiply a scalar by a tensor
/// we will convert the scalar into a tensor and then multiply the two tensors together
impl Mul<Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        rhs * self
    }
}


pub fn backward_for_mul(backprop_backet: BackproagationPacket) {
    if let Operation::Mul(left_hand_side, right_hand_side) = backprop_backet.operation {
        // for the mul operation, the gradient is simply the incoming gradient for both the left and right right operand
        let grad = backprop_backet
            .equation
            .get_grad_flat_buffer(backprop_backet.incoming_grad)
            .to_owned();

        //// The derivative of a * b is a' * b for a and  a * b for b
        let left_hand_data = backprop_backet
            .equation
            .get_data_flat_buffer(left_hand_side);
        let left_hand_additional_grad = backprop_backet.equation.mul_vector(left_hand_data, &grad);

        // we are using the equation provided functions here, as they can take advantage of hard acceleration
        // and hide it for us
        let right_hand_data = backprop_backet
            .equation
            .get_data_flat_buffer(right_hand_side);
        let right_hand_additional_grad =
            backprop_backet.equation.mul_vector(right_hand_data, &grad);

        // This is intentionally fliped. the derivative of an operand in a multiplication is the incoming grad multiplied by
        // the value of the other operand. As how much the output of itself is changed, is directly dependent on the other operand
        backprop_backet
            .equation
            .add_tensor_grad(right_hand_side, left_hand_additional_grad.to_vec());
        backprop_backet
            .equation
            .add_tensor_grad(left_hand_side, right_hand_additional_grad.to_vec());
    } else {
        panic!("backward_for_mul called with the wrong operand");
    }
}

#[cfg(test)]
mod test {
    use crate::central::Shape;
    use crate::central::Tensor;

    #[test]
    pub fn basic_mul_test() {
        let a = Tensor::element(Shape::new(vec![1]), 1.0);
        let b = Tensor::element(Shape::new(vec![1]), 1.0);
        let c = a * b;
        let c_item = c.item();
        assert!(c_item[0] == 1.0);
    }

    #[test]
    pub fn basic_neg_test() {
        let a = Tensor::element(Shape::new(vec![1]), 1.0);

        let c = -a;
        let c_item = c.item();
        assert!(c_item[0] == -1.0);
    }

    #[test]
    pub fn basic_div_test() {
        let a = Tensor::element(Shape::new(vec![1]), 10.0);
        let b = Tensor::element(Shape::new(vec![1]), 2.0);
        let c = a / b;
        let c_item = c.item();
        assert!(c_item[0] == 5.0);
    }

    #[test]
    pub fn basic_2d_test() {
        let a = Tensor::element(Shape::new(vec![5, 5]), 5.0);
        let b = Tensor::element(Shape::new(vec![5, 5]), 1.0);
        let c = a * b;
        let c_item = c.item();
        for i in c_item {
            assert!(i == 5.0);
        }
    }

    #[test]
    pub fn basic_3d_test() {
        let a = Tensor::element(Shape::new(vec![5, 5, 5]), 5.0);
        let b = Tensor::element(Shape::new(vec![5, 5, 5]), 7.0);
        let c = a * b;
        let c_item = c.item();
        for i in c_item {
            assert!(i == 35.0);
        }
    }

    #[test]
    pub fn basic_4d_test() {
        let a = Tensor::element(Shape::new(vec![5, 6, 3, 10]), 5.0);
        let b = Tensor::element(Shape::new(vec![5, 6, 3, 10]), 5.0);
        let c = a * b;
        let c_item = c.item();
        for i in c_item {
            assert!(i == 25.0);
        }
    }

    #[test]
    pub fn basic_2d_broadcast_test() {
        let a = Tensor::element(Shape::new(vec![1, 10]), 5.0);
        let b = Tensor::element(Shape::new(vec![3, 10]), 5.0);
        let c = a * b;
        let c_item = c.item();
        assert!(
            c_item.len() == 30,
            "length of final array {:?}",
            c_item.len()
        );
        for i in c_item {
            assert!(i == 25.0);
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    pub fn do_not_mul_tensors_of_different_size_but_same_length() {
        // The two tensors have different shapes, but whos length will be the same by the time
        // we get to the bare metal mul, so we need to make sure other things catch this case
        let a = Tensor::element(Shape::new(vec![5, 6, 3, 10]), 5.0);
        let b = Tensor::element(Shape::new(vec![5, 3, 6, 10]), 5.0);
        let _ = a * b;
    }

    #[test]
    pub fn backprop_mul_test() {
        let a = Tensor::element(Shape::new(vec![1]), 5.0);
        let b = Tensor::element(Shape::new(vec![1]), 10.0);
        let c = b * a;
        c.backward();
        assert!(
            a.grad()[0] == 10.0,
            "grad was supposed to be 10 was {}",
            a.grad()[0]
        );
    }

    #[test]
    pub fn multiple_mul_backprop_test() {
        let a = Tensor::element(Shape::new(vec![1]), 5.0);
        let b = Tensor::element(Shape::new(vec![1]), 10.0);
        let c = b * a;
        let d = Tensor::element(Shape::new(vec![1]), 20.0);
        let e = c * d;
        e.backward();
        assert!(a.grad()[0] == 200.0,);
        assert!(b.grad()[0] == 100.0);
        assert!(c.grad()[0] == 20.0);
        assert!(d.grad()[0] == 50.0);
        assert!(e.grad()[0] == 1.0);
    }
}
