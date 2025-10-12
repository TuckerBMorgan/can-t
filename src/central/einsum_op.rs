use std::collections::HashMap;

use crate::central::*;

impl Tensor {
    /*
    pub fn einsum(&self, einsum_def: String, other: Tensor) -> Tensor {

        // Split the definition into two parts
        // left side = what we are operating on
        // right side = what we are turning it into
        let parts : Vec<&str> = einsum_def.split("->").collect();
        assert!(parts.len() == 2, "For the moment we only support a very basic version of einsum, must have two parts seperate by ->");

        // Next we need to split up our operands, so we can assign a letter to each of their axis
        let operands : Vec<&str> = parts[0].split(",").collect();
        assert!(operands.len() == 1, "We only support two operands at the moment");

        let left_axis_count = self.shape.number_of_dimension();
        assert!(left_axis_count == operands[0].len(), "The number of provided axis must match the number of dimensions in the matrix");

        let right_axis_count = other.shape.number_of_dimension();
        assert!(right_axis_count == operands[1].len(), "The number of provided axis must match the number of dimensions in the matrix");

        let mut axis_to_dimension_size = HashMap::new();

        let mut i = 0;
        for c in operands[0].as_ {

            i+= 1;
        }


        Tensor::from_vec(vec![0.0], vec![1])

    }
    */
}
