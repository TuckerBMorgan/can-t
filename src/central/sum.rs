use std::collections::HashSet;

use crate::central::*;
use crate::central::Shape;
use ndarray::prelude::*;

use super::get_equation;

impl Tensor {
    pub fn sum(&self,mut axes: Vec<usize>, keep_dimensions: bool) -> Tensor {

        // Sanity Check that there is no double sums provided
        let mut seen_dimens : HashSet<usize> = HashSet::new();
        for dim in axes.clone() {
            if seen_dimens.contains(&dim) {
                panic!("dimensions summed must be unique");
            }
            else {
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
        let mut dimensions_as_array = [0_usize;4];
        for (index, dim) in axes.iter().enumerate() {
            dimensions_as_array[index] = *dim;
        }

        let sum_opeartion = Operation::Sum(self.id, dimensions_as_array, axes.len());
        let new_tensor = Tensor::create_tensor_data_and_shape_and_operation(new_shape, item.into_raw_vec(), sum_opeartion);
        return new_tensor;

    }
}
pub fn backward_for_sum(backprop_backet: BackproagationPacket) {
    if let Operation::Sum(from, _dimensions, _dimensions_count) = backprop_backet.operation {
        
        // Get the incoming grad and broadcast it back to the shape we want
        let grad = backprop_backet.equation.get_grad(backprop_backet.incoming_grad);
        let from_array = backprop_backet.equation.get_grad(from);

        let grad_broadcasted = grad.broadcast(from_array.shape()).unwrap();
        backprop_backet.equation.add_tensor_grad(from, grad_broadcasted.into_owned().into_raw_vec());
    }
    else {
        panic!("backward_for_sum called with the wrong operation");
    }
}
#[cfg(test)]
mod tests {
    use crate::central::*;

    #[test]
    pub fn basic_sum_test() {
        let tensor = Tensor::randn(Shape::new(vec![5, 4]));
        let result = tensor.sum(vec![0, 1], true);
    }

}