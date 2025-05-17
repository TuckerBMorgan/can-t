
use std::ops::{Add, Sub};
use  crate::central::*;

use super::get_equation;

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
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
}
