
use crate::central::*;
use std::ops::Shl;
use super::get_equation;
use crate::utils::handle_broadcasting;

 
impl Shl for Tensor {
    type Output = Tensor;
    fn shl(self, rhs: Self) -> Self::Output {
     //   assert!(self.shape.can_matmul(rhs.shape), "Invalid operands for matmul left hand {:?} right hand {:?}", self.shape, rhs.shape);
        // We want to make sure that the two operands can be matmuled together
        // so we try to broadcast them together if they do not equal each other

        // we vend the actual work of doing the matmul to the equation, which can take advantage of platform libs
        let data = get_equation().matmul_tensor(self.id,rhs.id);

        // Shape has helper funciton to figure out the shape for us
        let matmul_shape = self.shape.matmul_shape(rhs.shape);

        let return_tensor = Tensor::create_tensor_data_and_shape_and_operation(matmul_shape, data, Operation::Matmul(self.id, rhs.id ));
        return return_tensor;
    }
}
#[cfg(test)]
mod tests {
    use crate::central::{Shape, Tensor};
    use crate::utils::GGUFFile;
    
    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    fn compare_tensors(a: Tensor, b: Tensor) {
        let epsilon = 1e-5;
        let a_item = a.item();
        let together = a_item.iter().zip(b.item());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", *a, b);
        }
    }

    #[test]
    pub fn basic_matmul_test() {
        let a = Tensor::randn(Shape::new(vec![2]));
        let b = Tensor::randn(Shape::new(vec![2, 2]));
        let _c = a << b;
    }

    #[test]
    pub fn matmul_4x4() {
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/matmul/matmul_4x4.gguf",
        ));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_4x4_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_4x4_tensor_b"), &mut gguf_file);
        let tensor_x_real = Tensor::from_gguf_file(String::from("matmul_4x4_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }
}