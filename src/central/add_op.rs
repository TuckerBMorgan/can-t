
use std::ops::{Add, Sub};
use  crate::central::*;

use super::get_equation;

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.shape == rhs.shape);
        let result_id = get_equation().add_tensors(self.id, rhs.id);
        return Tensor::create_tensor_from_id_and_shape_and_operation(self.shape,result_id, Operation::Add(self.id, rhs.id));
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
    #[should_panic(expected = "assertion failed")]
    pub fn do_not_add_tensors_of_different_size_but_same_length() {
        // The two tensors have different shapes, but whos length will be the same by the time 
        // we get to the bare metal add, so we need to make sure other things catch this case
        let a = Tensor::element(Shape::new(vec![5, 6, 3, 10]), 5.0);
        let b = Tensor::element(Shape::new(vec![5, 3, 6, 10]), 5.0);
        let _ = a + b;
    }

}
