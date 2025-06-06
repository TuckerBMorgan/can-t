use std::collections::HashSet;

use crate::central::Shape;
use crate::central::*;
use ndarray::prelude::*;

use super::get_equation;

impl Tensor {
    pub fn sum(&self, mut axes: Vec<usize>, keep_dimensions: bool) -> Tensor {
        // Sanity Check that there is no double sums provided
        let mut seen_dimens: HashSet<usize> = HashSet::new();
        for dim in axes.clone() {
            if seen_dimens.contains(&dim) {
                panic!("dimensions summed must be unique");
            } else {
                seen_dimens.insert(dim);
            }
        }

        let mut new_shape = self.shape.clone();
        // If we are not keep the axes just remove them from the shape at the start
        if !keep_dimensions {
            let mut number_of_removed_axis = 0;
            for dim in axes.clone() {
                new_shape = new_shape.remove_index(dim - number_of_removed_axis);
                number_of_removed_axis += 1;
            }
        }
        // otherwise swap a 1 into the removed dimension
        else {
            for dim in axes.clone() {
                new_shape = new_shape.swap_index(dim, 1);
            }
        }

        // Sort the axes smallest to largest to make it easier to track how much we should subtracts
        axes.sort();
        let mut number_of_summed_axes = 0;
        let mut item = self.item();

        for dimension in axes.clone() {
            item = item.sum_axis(Axis(dimension - number_of_summed_axes));
            // As we remove axis(ndarray does not preserve them) we need to substract from the dimension we want to remove
            // so help offset stuff
            number_of_summed_axes += 1;
        }

        // We need to copy the dimensions we are summing into an array, so it can fit into the enum Operation
        let mut dimensions_as_array = [0_usize; 4];
        for (index, dim) in axes.iter().enumerate() {
            dimensions_as_array[index] = *dim;
        }

        let sum_opeartion = Operation::Sum(self.id, dimensions_as_array, axes.len(), keep_dimensions);
        let new_tensor = Tensor::create_tensor_data_and_shape_and_operation(
            new_shape,
            item.into_raw_vec(),
            sum_opeartion,
        );
        return new_tensor;
    }
}
pub fn backward_for_sum(backprop_backet: BackproagationPacket) {
    if let Operation::Sum(from, dimensions, dimensions_count, keep_dimensions) = backprop_backet.operation {
        // Get the incoming grad and broadcast it back to the shape we want
        let grad = backprop_backet
            .equation
            .get_grad(backprop_backet.incoming_grad);
        let from_array = backprop_backet.equation.get_grad(from);
        let mut working_shape = grad.shape().to_vec();


        // if we removed the dimensions, we need to add back the space that they where
        if !keep_dimensions {
            // we loop over the from shape, and 1 out all of the dimensions we removed, so that
            // we can then broadcast along those dimensions the grad
            let mut brodcastable_shape = from_array.shape().to_vec();
            for index in 0..dimensions_count {
                brodcastable_shape[dimensions[index]] = 1;
            } 
            working_shape = brodcastable_shape;
        }

        // Reshape the grad, just in case we had to add in dimensions
        let grad = grad.into_shape(working_shape).unwrap();

        let grad_broadcasted = grad.broadcast(from_array.shape()).unwrap();
        backprop_backet
            .equation
            .add_tensor_grad(from, grad_broadcasted.into_owned().into_raw_vec());
    } else {
        panic!("backward_for_sum called with the wrong operation");
    }
}
#[cfg(test)]
mod tests {
    use crate::{central::*, utils::GGUFFile};
    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    pub fn basic_sum_test() {
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/sum/sum_test_file_container.gguf",
        ));
        let presum_tensor = Tensor::from_gguf_file("sum_test_pre_sum_model".to_string(), &mut gguf_file);
        let postsum_tensor = Tensor::from_gguf_file("sum_test_post_sum_model".to_string(), &mut gguf_file);
        let result = presum_tensor.sum(vec![1], true);
        let cant_result = result.item();
        let pytorch_result = postsum_tensor.item();

        let binded = cant_result.iter().zip(pytorch_result);
        for (a, b) in binded {
            assert!(*a == b);
        }
    }

    #[test]
    pub fn basic_double_index_sum_test() {
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/sum/double_index_sum_test_file_container.gguf",
        ));
        let presum_tensor = Tensor::from_gguf_file("sum_test_pre_double_index_sum_model".to_string(), &mut gguf_file);
        let postsum_tensor = Tensor::from_gguf_file("sum_test_post_double_index_sum_model".to_string(), &mut gguf_file);
        let result = presum_tensor.sum(vec![1, 2], true);
        let cant_result = result.item();
        let pytorch_result = postsum_tensor.item();

        let binded = cant_result.iter().zip(pytorch_result);
        for (a, b) in binded {
            assert!(approx_equal(*a, b, 1e-6));
        }
    }

    #[test]
    pub fn basic_double_index_with_skip_sum_test() {
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/sum/double_index_with_skip_sum__test_file_container.gguf",
        ));
        let presum_tensor = Tensor::from_gguf_file("sum_test_pre_double_index_with_skip_sum_model".to_string(), &mut gguf_file);

        let postsum_tensor = Tensor::from_gguf_file("sum_test_post_double_index_with_skip_sum__model".to_string(), &mut gguf_file);
        let result = presum_tensor.sum(vec![1, 3], true);
        let cant_result = result.item();
        let pytorch_result = postsum_tensor.item();
        let binded = cant_result.iter().zip(pytorch_result);
        for (a, b) in binded {
            assert!(approx_equal(*a, b, 1e-6));
        }
    }

    #[test]
    pub fn sum_backward_test_container_test() {
        let epsilon = 1e-6;
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/sum/sum_backward_test_container.gguf",
        ));
        let tensor_a = Tensor::from_gguf_file("sum_backward_test_container_tensor_a".to_string(), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file("sum_backward_test_container_tensor_b".to_string(), &mut gguf_file);
        let tensor_a_grad_real = Tensor::from_gguf_file("sum_backward_test_container_tensor_a_grad".to_string(), &mut gguf_file);
        let tensor_b_grad_real = Tensor::from_gguf_file("sum_backward_test_container_tensor_b_grad".to_string(), &mut gguf_file);
        let multiplied_real = Tensor::from_gguf_file("sum_backward_test_container_tensor_multiplied".to_string(), &mut gguf_file);
        let multiplied_real_grad = Tensor::from_gguf_file("sum_backward_test_container_tensor_multiplied_grad".to_string(), &mut gguf_file);
        let summed_real = Tensor::from_gguf_file("sum_backward_test_container_tensor_summed".to_string(), &mut gguf_file);

        let multiplied = tensor_a * tensor_b;
        let multiplied_item = multiplied.item();
        let together = multiplied_item.iter().zip(multiplied_real.item());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon));
        }

        let summed = multiplied.sum(vec![1, 2, 3], false);
        let summed_item = summed.item();
        let together = summed_item.iter().zip(summed_real.item());
        for (a, b) in together {
            assert!(approx_equal(*a,b, epsilon));
        }

        summed.backward();

        let multiplied_grad = multiplied.grad();
        let together = multiplied_grad.iter().zip(multiplied_real_grad.item());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon))
        }

        
        let tensor_a_grad = tensor_a.grad();
        let together = tensor_a_grad.iter().zip(tensor_a_grad_real.item());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon));
        }

        let tensor_b_grad = tensor_b.grad();
        let together = tensor_b_grad.iter().zip(tensor_b_grad_real.item());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon));
        }
    }

    #[test]
    pub fn sum_and_broadcast_backward_test() {
        let epsilon = 1e-5;
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/sum/sum_and_broadcast_backward_test_container.gguf",
        ));
        let tensor_a = Tensor::from_gguf_file("sum_and_broadcast_backward_test_container_tensor_a".to_string(), &mut gguf_file);
        let tensor_b = Tensor::from_gguf_file("sum_and_broadcast_backward_test_container_tensor_b".to_string(), &mut gguf_file);
        let tensor_a_grad_real = Tensor::from_gguf_file("sum_and_broadcast_backward_test_container_tensor_a_grad".to_string(), &mut gguf_file);
        let tensor_b_grad_real = Tensor::from_gguf_file("sum_and_broadcast_backward_test_container_tensor_b_grad".to_string(), &mut gguf_file);
        let multiplied_real = Tensor::from_gguf_file("sum_and_broadcast_backward_test_container_tensor_multiplied".to_string(), &mut gguf_file);
        let multiplied_real_grad = Tensor::from_gguf_file("sum_and_broadcast_backward_test_container_tensor_multiplied_grad".to_string(), &mut gguf_file);
        let summed_real = Tensor::from_gguf_file("sum_and_broadcast_backward_test_container_tensor_summed".to_string(), &mut gguf_file);

        let multiplied = tensor_a + tensor_b;
        let multiplied_item = multiplied.item();
        let together = multiplied_item.iter().zip(multiplied_real.item());

        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon));
        }

        let summed = multiplied.sum(vec![1, 2], false);
        let summed_item = summed.item();
        let together = summed_item.iter().zip(summed_real.item());

        for (a, b) in together {
            assert!(approx_equal(*a,b, epsilon));
        }

        summed.backward();

        let multiplied_grad = multiplied.grad();
        let together = multiplied_grad.iter().zip(multiplied_real_grad.item());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon))
        }

        
        let tensor_a_grad = tensor_a.grad();
        let together = tensor_a_grad.iter().zip(tensor_a_grad_real.item());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon));
        }

        let tensor_b_grad = tensor_b.grad();
        let together = tensor_b_grad.iter().zip(tensor_b_grad_real.item());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon));
        }
    }
}
