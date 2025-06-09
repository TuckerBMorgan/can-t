
use crate::central::*;
use std::ops::Shl;
use super::get_equation;
use crate::utils::handle_broadcasting;

 
impl Shl for Tensor {
    type Output = Tensor;
    fn shl(self, rhs: Self) -> Self::Output {
        
        // Broadcasting and checking if you can matmul at all are covered by this function 
        let (left_hand_broadcast_shape, right_hand_broadcast_shape) = Shape::matmul_broadcast(self.shape, rhs.shape);

        // Do the final shape based of of the PRE broadcast shapes       
        let matmul_shape = self.shape.matmul_shape(rhs.shape);

        // Broadcast the tensors over to their right shape
        let left_hand = self.broadcast(left_hand_broadcast_shape);
        let right_hand = rhs.broadcast(right_hand_broadcast_shape);

        // we vend the actual work of doing the matmul to the equation, which can take advantage of platform libs
        let data = get_equation().matmul_tensor(left_hand.id,right_hand.id);
        let return_tensor = Tensor::create_tensor_data_and_shape_and_operation(matmul_shape, data, Operation::Matmul(left_hand.id, right_hand.id ));
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


    #[test]
    pub fn matmul_4x3() {
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/matmul/matmul_4x3.gguf",
        ));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_4x3_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_4x3_tensor_b"), &mut gguf_file);
        let tensor_x_real = Tensor::from_gguf_file(String::from("matmul_4x3_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }
}