
use std::ops::{Add, Sub};
use  crate::central::*;

use super::get_equation;

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {

        // We want to make sure that the two operands can be added on a elementwise way
        // so we try to broadcast them together if they do not equal each other
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
        let result_id = get_equation().add_tensors(working_lfs.id, working_rhs.id);


        return Tensor::create_tensor_from_id_and_shape_and_operation(working_rhs.shape,result_id, Operation::Add(working_lfs.id, working_rhs.id));
   }
}

#[cfg(test)]
mod test {
    use crate::central::Tensor;
    use crate::central::Shape;

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
        assert!(c_item.len() == 30, "length of final array {:?}", c_item.len());
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

}
