use super::get_equation;
use crate::{central::*, utils::padding_dimenions_to_max};
use std::ops::Shl;

impl Shl for Tensor {
    type Output = Tensor;
    fn shl(self, rhs: Self) -> Self::Output {
        // Matmul broadcast rules, if the right hand has only one dimension, we want to add a 1 length dimenion
        // so it does make a mistake
        let mut working_rhs = rhs;
        if working_rhs.shape.dimensions().len() == 1 {
            working_rhs =
                working_rhs.reshape(Shape::new(vec![working_rhs.shape.dimensions()[0], 1]));
        }

        // Broadcasting and checking if you can matmul at all are covered by this function
        let (left_hand_broadcast_shape, right_hand_broadcast_shape) =
            Shape::matmul_broadcast(self.shape, working_rhs.shape);

        // Do the final shape based of of the PRE broadcast shapes
        let matmul_shape = self.shape.matmul_shape(working_rhs.shape);

        // Broadcast the tensors over to their right shape
        let left_hand = self.broadcast(left_hand_broadcast_shape);
        let right_hand = working_rhs.broadcast(right_hand_broadcast_shape);

        // we vend the actual work of doing the matmul to the equation, which can take advantage of platform libs
        let data = get_equation().matmul_tensor(left_hand.id, right_hand.id);
        let return_tensor = Tensor::create_tensor_data_and_shape_and_operation(
            matmul_shape,
            data,
            Operation::Matmul(left_hand.id, right_hand.id),
        );
        return return_tensor;
    }
}

pub fn backward_for_matmul(backprop_backet: BackproagationPacket) {
    if let Operation::Matmul(left_hand_side, right_hand_side) = backprop_backet.operation {
        // Get the grad, it stays the same
        let grad = backprop_backet
            .equation
            .get_grad(backprop_backet.incoming_grad)
            .to_owned();
        let grad_dimensions_length = grad.shape().len();
        let mut grad_dimensions_reformed = grad.shape().to_vec();

        // Pad out the shape so it 4 long, adding a one does not change the matrix
        let added_dimensions = MAX_DIMS - grad_dimensions_length;
        for _ in 0..added_dimensions {
            grad_dimensions_reformed.insert(0, 1);
        }

        // Then get the left and right hand side
        // reshape it to 4ds
        // These are both COPY opeartions, which will cause allocations hitches
        let mut left_hand_weights = backprop_backet
            .equation
            .get_data_flat_buffer(left_hand_side)
            .to_vec();
        let left_hand_shape = backprop_backet.equation.get_tensor_shape(left_hand_side);
        let mut padded_left_hand_shape = padding_dimenions_to_max(left_hand_shape.dimensions());

        let mut right_hand_weights = backprop_backet
            .equation
            .get_data_flat_buffer(right_hand_side)
            .to_vec();
        let right_hand_shape = backprop_backet.equation.get_tensor_shape(right_hand_side);
        let mut padded_right_hand_shape = padding_dimenions_to_max(right_hand_shape.dimensions());

        // TODO: update this comment
        // Treat them all as if they where 4d matrices, makes everything that comes after simpler
        // and because we are not actually adding or removing elements, it results in the same opeartions
        // ex: a matrixes of size [4] is the equivilent to one of size [4, 1] or [1, 4]
        backprop_backet
            .equation
            .swap_axes(&mut left_hand_weights, padded_left_hand_shape, MAX_DIMS - 2, MAX_DIMS - 1);

        backprop_backet
            .equation
            .swap_axes(&mut right_hand_weights, padded_right_hand_shape, MAX_DIMS - 2, MAX_DIMS - 1);

        // we also need to swap the indices themselves
        let hold = padded_left_hand_shape[MAX_DIMS - 2];
        padded_left_hand_shape[MAX_DIMS - 2] = padded_left_hand_shape[MAX_DIMS - 1];
        padded_left_hand_shape[MAX_DIMS - 1] = hold;

        let hold = padded_right_hand_shape[MAX_DIMS - 2];
        padded_right_hand_shape[MAX_DIMS - 2] = padded_right_hand_shape[MAX_DIMS - 1];
        padded_right_hand_shape[MAX_DIMS - 1] = hold;

        // everything wants know sized arrays, so just making it quick here
        let grad_dimensions_reformed: [usize; 10] = [
            grad_dimensions_reformed[0],
            grad_dimensions_reformed[1],
            grad_dimensions_reformed[2],
            grad_dimensions_reformed[3],
            grad_dimensions_reformed[4],
            grad_dimensions_reformed[5],
            grad_dimensions_reformed[6],
            grad_dimensions_reformed[7],
            grad_dimensions_reformed[8],
            grad_dimensions_reformed[9],
        ];
        let grad_as_vec = grad.into_raw_vec();

        // we hand off actually doing the math to matmul vector as then it can vend out to platform code, and we don't need to pollute this code with that
        let left_hand_result = backprop_backet.equation.matmul_vector(
            &grad_as_vec,
            grad_dimensions_reformed,
            &right_hand_weights,
            padded_right_hand_shape,
        );

        let right_hand_result = backprop_backet.equation.matmul_vector(
            &left_hand_weights,
            padded_left_hand_shape,
            &grad_as_vec,
            grad_dimensions_reformed,
        );

        // Finally add the grad to the tensors
        backprop_backet
            .equation
            .add_tensor_grad(left_hand_side, left_hand_result);
        backprop_backet
            .equation
            .add_tensor_grad(right_hand_side, right_hand_result);
    } else {
        panic!("Wrong opeartions for matmul backward");
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
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/matmul/matmul_4x4.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_4x4_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_4x4_tensor_b"), &mut gguf_file);
        let tensor_x_real =
            Tensor::from_gguf_file(String::from("matmul_4x4_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }

    #[test]
    pub fn matmul_4x3() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/matmul/matmul_4x3.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_4x3_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_4x3_tensor_b"), &mut gguf_file);
        let tensor_x_real =
            Tensor::from_gguf_file(String::from("matmul_4x3_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }

    #[test]
    pub fn matmul_4x2() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/matmul/matmul_4x2.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_4x2_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_4x2_tensor_b"), &mut gguf_file);
        let tensor_x_real =
            Tensor::from_gguf_file(String::from("matmul_4x2_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }

    #[test]
    pub fn matmul_4x1() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/matmul/matmul_4x1.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_4x1_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_4x1_tensor_b"), &mut gguf_file);
        let tensor_x_real =
            Tensor::from_gguf_file(String::from("matmul_4x1_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }

    #[test]
    pub fn matmul_1x4() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/matmul/matmul_1x4.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_1x4_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_1x4_tensor_b"), &mut gguf_file);
        let tensor_x_real =
            Tensor::from_gguf_file(String::from("matmul_1x4_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }

    #[test]
    pub fn matmul_2x4() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/matmul/matmul_2x4.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_2x4_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_2x4_tensor_b"), &mut gguf_file);
        let tensor_x_real =
            Tensor::from_gguf_file(String::from("matmul_2x4_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }

    #[test]
    pub fn matmul_3x4() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/matmul/matmul_3x4.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_3x4_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_3x4_tensor_b"), &mut gguf_file);
        let tensor_x_real =
            Tensor::from_gguf_file(String::from("matmul_3x4_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }

    #[test]
    pub fn matmul_1x2() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/matmul/matmul_1x2.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_1x2_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_1x2_tensor_b"), &mut gguf_file);
        let tensor_x_real =
            Tensor::from_gguf_file(String::from("matmul_1x2_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }

    #[test]
    pub fn matmul_2x2() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/matmul/matmul_2x2.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_2x2_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_2x2_tensor_b"), &mut gguf_file);
        let tensor_x_real =
            Tensor::from_gguf_file(String::from("matmul_2x2_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }

    #[test]
    pub fn matmul_3x2() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/matmul/matmul_3x2.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_3x2_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_3x2_tensor_b"), &mut gguf_file);
        let tensor_x_real =
            Tensor::from_gguf_file(String::from("matmul_3x2_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }

    #[test]
    pub fn matmul_3x1() {
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/matmul/matmul_3x1.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("matmul_3x1_tensor_a"), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file(String::from("matmul_3x1_tensor_b"), &mut gguf_file);
        let tensor_x_real =
            Tensor::from_gguf_file(String::from("matmul_3x1_tensor_x"), &mut gguf_file);
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
    }

    #[test]
    pub fn matmul_backward_test_basic() {
        let epsilon = 1e-5;
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/matmul/matmul_backward_test_basic.gguf",
        ));

        let tensor_a = Tensor::from_gguf_file(
            String::from("matmul_backward_test_basic_tensor_a"),
            &mut gguf_file,
        );
        let tensor_a_grad_real = Tensor::from_gguf_file(
            String::from("matmul_backward_test_basic_tensor_a_grad"),
            &mut gguf_file,
        );
        let tensor_b = Tensor::from_gguf_file(
            String::from("matmul_backward_test_basic_tensor_b"),
            &mut gguf_file,
        );
        let tensor_x_real = Tensor::from_gguf_file(
            String::from("matmul_backward_test_basic_tensor_c"),
            &mut gguf_file,
        );
        let tensor_x = tensor_a << tensor_b;
        compare_tensors(tensor_x, tensor_x_real);
        tensor_x.backward();

        let tensor_a_real_grad_item = tensor_a_grad_real.item();
        let together = tensor_a_real_grad_item.iter().zip(tensor_a.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon));
        }
    }

    #[test]
    pub fn matmul_backward_test_4x4() {
        let epsilon = 1e-5;
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/matmul/matmul_backward_test_4x4.gguf",
        ));

        let tensor_a = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x4_tensor_a"),
            &mut gguf_file,
        );
        let tensor_a_grad_real = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x4_tensor_a_grad"),
            &mut gguf_file,
        );
        let tensor_b = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x4_tensor_b"),
            &mut gguf_file,
        );
        let middle_sum_grad = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x4_middle_sum_grad"),
            &mut gguf_file,
        );
        let first_sum_grad = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x4_first_sum_grad"),
            &mut gguf_file,
        );
        let tensor_c_real = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x4_tensor_c"),
            &mut gguf_file,
        );

        let tensor_c = tensor_a << tensor_b;
        let first_sum = tensor_c.sum(vec![0], false);
        let tensor_c = first_sum.sum(vec![0], false);
        let middle_sum = tensor_c.sum(vec![0], false);
        let tensor_c = middle_sum.sum(vec![0], true);
        compare_tensors(tensor_c, tensor_c_real);
        tensor_c.backward();

        let first_sum_real_grad_item = first_sum_grad.item();
        let together = first_sum_real_grad_item.iter().zip(first_sum.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }

        let middle_sum_real_grad_item = middle_sum_grad.item();
        let together = middle_sum_real_grad_item.iter().zip(middle_sum.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }

        let tensor_a_real_grad_item = tensor_a_grad_real.item();
        let together = tensor_a_real_grad_item.iter().zip(tensor_a.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }
    }

    #[test]
    pub fn matmul_backward_test_4x3() {
        let epsilon = 1e-5;
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/matmul/matmul_backward_test_4x3.gguf",
        ));

        let tensor_a = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x3_tensor_a"),
            &mut gguf_file,
        );
        let tensor_a_grad_real = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x3_tensor_a_grad"),
            &mut gguf_file,
        );
        let tensor_b = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x3_tensor_b"),
            &mut gguf_file,
        );
        let middle_sum_grad = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x3_middle_sum_grad"),
            &mut gguf_file,
        );
        let first_sum_grad = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x3_first_sum_grad"),
            &mut gguf_file,
        );
        let tensor_c_real = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x3_tensor_c"),
            &mut gguf_file,
        );

        let tensor_c = tensor_a << tensor_b;
        let first_sum = tensor_c.sum(vec![0], false);
        let tensor_c = first_sum.sum(vec![0], false);
        let middle_sum = tensor_c.sum(vec![0], false);
        let tensor_c = middle_sum.sum(vec![0], true);
        compare_tensors(tensor_c, tensor_c_real);
        tensor_c.backward();

        let first_sum_real_grad_item = first_sum_grad.item();
        let together = first_sum_real_grad_item.iter().zip(first_sum.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }

        let middle_sum_real_grad_item = middle_sum_grad.item();
        let together = middle_sum_real_grad_item.iter().zip(middle_sum.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }

        let tensor_a_real_grad_item = tensor_a_grad_real.item();
        let together = tensor_a_real_grad_item.iter().zip(tensor_a.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }
    }

    #[test]
    pub fn matmul_backward_test_4x2() {
        let epsilon = 1e-5;
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/matmul/matmul_backward_test_4x2.gguf",
        ));

        let tensor_a = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x2_tensor_a"),
            &mut gguf_file,
        );
        let tensor_a_grad_real = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x2_tensor_a_grad"),
            &mut gguf_file,
        );
        let tensor_b = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x2_tensor_b"),
            &mut gguf_file,
        );
        let middle_sum_grad = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x2_middle_sum_grad"),
            &mut gguf_file,
        );
        let first_sum_grad = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x2_first_sum_grad"),
            &mut gguf_file,
        );
        let tensor_c_real = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x2_tensor_c"),
            &mut gguf_file,
        );

        let tensor_c = tensor_a << tensor_b;
        let first_sum = tensor_c.sum(vec![0], false);
        let tensor_c = first_sum.sum(vec![0], false);
        let middle_sum = tensor_c.sum(vec![0], false);
        let tensor_c = middle_sum.sum(vec![0], true);
        compare_tensors(tensor_c, tensor_c_real);
        tensor_c.backward();

        let first_sum_real_grad_item = first_sum_grad.item();
        let together = first_sum_real_grad_item.iter().zip(first_sum.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }

        let middle_sum_real_grad_item = middle_sum_grad.item();
        let together = middle_sum_real_grad_item.iter().zip(middle_sum.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }

        let tensor_a_real_grad_item = tensor_a_grad_real.item();
        let together = tensor_a_real_grad_item.iter().zip(tensor_a.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }
    }

    #[test]
    pub fn matmul_backward_test_4x1() {
        let epsilon = 1e-5;
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/matmul/matmul_backward_test_4x1.gguf",
        ));

        let tensor_a = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x1_tensor_a"),
            &mut gguf_file,
        );
        let tensor_a_grad_real = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x1_tensor_a_grad"),
            &mut gguf_file,
        );
        let tensor_b = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x1_tensor_b"),
            &mut gguf_file,
        );
        let middle_sum_grad = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x1_middle_sum_grad"),
            &mut gguf_file,
        );
        let first_sum_grad = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x1_first_sum_grad"),
            &mut gguf_file,
        );
        let tensor_c_real = Tensor::from_gguf_file(
            String::from("matmul_backward_test_4x1_tensor_c"),
            &mut gguf_file,
        );

        let tensor_c = tensor_a << tensor_b;
        let first_sum = tensor_c.sum(vec![0], false);
        let tensor_c = first_sum.sum(vec![0], false);
        let middle_sum = tensor_c.sum(vec![0], false);
        let tensor_c = middle_sum.sum(vec![0], true);
        compare_tensors(tensor_c, tensor_c_real);
        tensor_c.backward();

        let first_sum_real_grad_item = first_sum_grad.item();
        let together = first_sum_real_grad_item.iter().zip(first_sum.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }

        let middle_sum_real_grad_item = middle_sum_grad.item();
        let together = middle_sum_real_grad_item.iter().zip(middle_sum.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }

        let tensor_a_real_grad_item = tensor_a_grad_real.item();
        let together = tensor_a_real_grad_item.iter().zip(tensor_a.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", a, b);
        }
    }
}
