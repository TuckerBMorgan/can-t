use std::ops::Mul;
use  crate::central::*;
use super::get_equation;

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {

        // We want to make sure that the two operands can be muled on a elementwise way
        // so we try to broadcast them together if they do not equal each other
        /// TODO: THIS WILL LIKELY BE A COMMON OPERATION IS THERE A WAY TO MAKE THIS SIMPLIER
        let mut working_rhs = rhs;
        if working_rhs.shape != self.shape {
            if self.shape.can_broadcast(working_rhs.shape) {
                working_rhs = self.broadcast(working_rhs.shape);
            }
        }
        let mut working_lfs = self;
        if working_lfs.shape != working_rhs.shape {
            if working_lfs.shape.can_broadcast(working_rhs.shape) {
                working_lfs = working_rhs.broadcast(working_lfs.shape);
            }
        }
        
        assert!(working_lfs.shape == working_rhs.shape);

        // We vend out the actual work to the equation, which in turn uses libs that take advantage of platform libs to speed it up
        let data = get_equation().mul_tensors(working_lfs.id, working_rhs.id);
        let return_tensor = Tensor::create_tensor_data_and_shape_and_operation(working_lfs.shape, data, Operation::Mul(working_lfs.id, working_rhs.id));
        return return_tensor;
    }
}

pub fn backward_for_mul(backprop_backet: BackproagationPacket) {
    if let Operation::Mul(left_hand_side, right_hand_side) = backprop_backet.operation {

        // for the mul operation, the gradient is simply the incoming gradient for both the left and right right operand
        let grad = backprop_backet.equation.get_grad_flat_buffer(backprop_backet.incoming_grad).to_owned();

        //// The derivative of a * b is a' * b for a and  a * b for b 
        let left_hand_data = backprop_backet.equation.get_data_flat_buffer(left_hand_side);
        let left_hand_additional_grad = backprop_backet.equation.mul_vector(left_hand_data, &grad);

        // we are using the equation provided functions here, as they can take advantage of hard acceleration
        // and hide it for us
        let right_hand_data = backprop_backet.equation.get_data_flat_buffer(right_hand_side);
        let right_hand_additional_grad = backprop_backet.equation.mul_vector(right_hand_data, &grad);


        // This is intentionally fliped. the derivative of an operand in a multiplication is the incoming grad multiplied by 
        // the value of the other operand. As how much the output of itself is changed, is directly dependent on the other operand
        backprop_backet.equation.add_tensor_grad(right_hand_side, left_hand_additional_grad.to_vec());
        backprop_backet.equation.add_tensor_grad(left_hand_side, right_hand_additional_grad.to_vec());
    }
    else {
        panic!("backward_for_mul called with the wrong operand");
    }
}

#[cfg(test)]
mod test {
    use crate::central::Tensor;
    use crate::central::Shape;

    #[test]
    pub fn basic_mul_test() {
        let a = Tensor::element(Shape::new(vec![1]), 1.0);
        let b = Tensor::element(Shape::new(vec![1]), 1.0);
        let c = a * b;
        let c_item = c.item();
        assert!(c_item[0] == 1.0);
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
        assert!(c_item.len() == 30, "length of final array {:?}", c_item.len());
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
        assert!(a.grad()[0] == 10.0, "grad was supposed to be 10 was {}", a.grad()[0]);
    }

    #[test]
    pub fn multiple_mul_backprop_test() {
        let a = Tensor::element(Shape::new(vec![1]), 5.0);
        let b = Tensor::element(Shape::new(vec![1]), 10.0);
        let c = b * a;
        let d= Tensor::element(Shape::new(vec![1]), 20.0);
        let e = c * d;
        e.backward();
        assert!(a.grad()[0] == 200.0,);
        assert!(b.grad()[0] == 100.0);
        assert!(c.grad()[0] == 20.0);
        assert!(d.grad()[0] == 50.0);
        assert!(e.grad()[0] == 1.0);
    }
}
