use super::{BackproagationPacket, Operation, Tensor};
use crate::central::*;
use regex::Regex;
use std::{collections::HashMap, hash::Hash, mem::swap};

impl Tensor {

    fn dim_list_to_bitset(summed_dimensions: &Vec<usize>, number_of_dimensions: usize) -> Vec<bool> {
        let mut bitset = vec![false;64];
        for index in 0..summed_dimensions.len() {
            let dimension = summed_dimensions[index];
            assert!(bitset[dimension] == false);
            bitset[dimension] = true;
        }
        return bitset;
    }

    fn sumproduct_pair(left: Tensor, right: Tensor, sum_dimensions: Vec<usize>, keep_dimensions: bool) -> Tensor {
        if sum_dimensions.is_empty() {
            return left * right;
        }

        let mut left = left;
        let mut right = right;

        let number_left_dimensions = left.shape.number_of_dimension();
        let summed_dimensions = Tensor::dim_list_to_bitset(&sum_dimensions, number_left_dimensions);
        let mut lro = vec![];
        let mut lro_size = 1;
        let mut lo = vec![];
        let mut lo_size = 1;
        let mut ro = vec![];
        let mut ro_size = 1;
        let mut sum_size = 1;
        

        for i in 0..number_left_dimensions {
            let sl = left.shape.dimensions()[i] != 1;
            let sr = right.shape.dimensions()[i] != 1;

            if summed_dimensions[i] == true {
                if sl == true && sr == true {
                    sum_size *= left.shape.dimensions()[i];                    
                }
                else if sl == true {
                    left = left.sum(vec![i], true);
                }
                else if sr == true {
                    right = right.sum(vec![i], true);
                }
            } else if sl == true && sr == true {
                lro.push(i);
                lro_size *= left.shape.dimensions()[i];
            }
            else if sl == true {
                lo.push(i);
                lo_size *= left.shape.dimensions()[i];
            }
            else {
                ro.push(i);
                ro_size *= right.shape.dimensions()[i];
            }
        }

        let swap_lo_ro = lo.is_empty() == false && ro.is_empty() == false && ro.last() < lo.first();

        if swap_lo_ro {
            let hold_tensor = left;
            left = right;
            right = hold_tensor;

            let hold_ro = ro.clone();
            ro = lo;
            lo = hold_ro;

            let hold_ro_size = ro_size.clone();
            ro_size = lo_size;
            lo_size = hold_ro_size;
        }


        let number_of_output_dimension = lro.len() + lo.len() + sum_dimensions.len() + ro.len();

        let mut output_size = vec![];

        output_size.reserve(number_of_output_dimension);

        for d in &lro {
            output_size.push(left.shape.dimensions()[*d]);
        }

        for d in &lo {
            output_size.push(left.shape.dimensions()[*d]);
        }

        for d in &sum_dimensions {
            // This is an odd line in the original code
            // it does emblack back
            // and has a (vodi)(d)
            // Need more context on this
            output_size.push(1);
        }

        for d in &ro {
            output_size.push(right.shape.dimensions()[*d]);
        }

        let sum_dimensions = sum_dimensions;
        let mut lpermutations = lro.clone();
        lpermutations.append(&mut lo.clone());
        lpermutations.append(&mut sum_dimensions.clone());
        lpermutations.append(&mut ro.clone());

        let mut rpermutations = lro.clone();
        rpermutations.append(&mut sum_dimensions.clone());
        rpermutations.append(&mut ro.clone());
        rpermutations.append(&mut lo.clone());


        let mut out_permutations = vec![0;number_of_output_dimension];
        {
            let mut i = 0;

            for it in &lro { 
                out_permutations[*it] = i;
                i += 1;
            }

            for it in &lo {
                out_permutations[*it] = i as usize;
                i+=1;
            }

            for it in &sum_dimensions {
                out_permutations[*it] = i;
                i += 1;
            }

            for it in &ro {
                out_permutations[*it] = i;
                i += 1;
            }
            
        }


        let left = left.permute(lpermutations).reshape(Shape::new(vec![lro_size, lo_size, sum_size]));

        let right = right.permute(rpermutations).reshape(Shape::new(vec![lro_size, sum_size, ro_size]));

        // In the original c++ this is a call to bmm, (batched matrix multlpy)
        // Cant does this by default as all Matrix Multiply
        // it is likely worth some time to understand why pytorch does not have them as the same function
        let result = left << right;
        println!("Left {:?}", left.shape);
        println!("Right {:?}", right.shape);
        println!("Result {:?}", result.shape);
        println!("Output Size {:?}", output_size);
        println!("Out Permuitation {:?}", out_permutations);
        let result = result.reshape(Shape::new(output_size)).permute(out_permutations);

        return result;
    }

    // Adapated from https://github.com/pytorch/pytorch/blob/0e083942ccd98c5a2d9386f0b2c4bbc901be40ef/aten/src/ATen/native/Linear.cpp einsum function
    pub fn einsum(einsum_notation: &str, operands: Vec<Tensor>) -> Tensor {
        // Labels must be in range [A-Za-z]
        const NUM_OF_LETTERS: u8 = b'z' - b'a' + 1;
        const TOTAL_LABELS: u8 = NUM_OF_LETTERS * 2;
        const ELLIPSIS: u8 = TOTAL_LABELS;

        fn label_to_subscript(label: u8) -> u8 {
            if label.is_ascii_uppercase() {
                return label - b'A';
            } else {
                return label - b'a' + NUM_OF_LETTERS;
            }
        }

        let number_of_operands = operands.len();
        let arrow_position = einsum_notation.find("->");
        if arrow_position.is_none() {
            panic!("Only Einsum functions with -> in them are supported");
        }
        let lhs = &einsum_notation[0..arrow_position.unwrap()];

        let mut operand_labels = vec![vec![]; number_of_operands];
        let mut lhs_chars = lhs.chars();
        let mut ellipsis_in_input = false;
        let mut current_operand = 0;

        loop {
            let next = lhs_chars.next();
            match next {
                Some(lhs_char) => {
                    match lhs_char {
                        ' ' => {
                            // Ignore white space
                        }
                        '.' => {
                            if ellipsis_in_input {
                                panic!("Double ellipse in operand");
                            }
                            // Are there enough characters left in the string
                            let next_next = lhs_chars.next();
                            let next_next_next = lhs_chars.next();

                            if next_next.is_none() || next_next_next.is_none() {
                                println!("improper number of '.' characters of ellipis");
                            }

                            // And are they both ellipsis
                            let next_next = next_next.unwrap();
                            let next_next_next = next_next_next.unwrap();

                            if next_next != '.' || next_next_next != '.' {
                                println!("improper number of '.' characters of ellipis");
                            }

                            operand_labels[current_operand].push(ELLIPSIS);
                            ellipsis_in_input = true;
                        }
                        ',' => {
                            // Move us onto the next opearand
                            current_operand += 1;
                            if current_operand > number_of_operands {
                                panic!(
                                    "fewer opearands where provided that exists in the input string"
                                );
                            }

                            // And reset our control variable
                            ellipsis_in_input = false;
                        }
                        _ => {
                            // Default case
                            if lhs_char.is_alphabetic() == false {
                                panic!("invalid character in input string");
                            }

                            operand_labels[current_operand]
                                .push(label_to_subscript(lhs_char as u8));
                        }
                    }
                }
                _ => {
                    break;
                }
            }
        }

        assert!(
            current_operand == number_of_operands - 1,
            "Improper number of operands passed to einsum"
        );

        let mut label_count = vec![0; TOTAL_LABELS as usize];

        let mut number_of_ellipsis_dimensions = 0;

        for i in 0..number_of_operands {
            let operands = operands[i];
            let labels = &operand_labels[i];
            let number_of_dims = operands.shape;

            let mut number_labels = labels.len();

            //          let mut has_ellipsis = false;

            for label in labels {
                if *label == ELLIPSIS {
                    number_labels -= 1;
                    //                    has_ellipsis = true;
                    number_of_ellipsis_dimensions = number_of_ellipsis_dimensions
                        .max(number_of_dims.number_of_dimension() - number_labels);
                } else {
                    label_count[*label as usize] += 1;
                }
            }
        }

        let mut permanent_label_index = vec![-1; TOTAL_LABELS as usize];

        let mut permanent_index = 0;

        let mut index_of_ellipsis = 0;
        let mut ellipsis_in_output = false;

        if arrow_position.is_none() {
            panic!("Unsupported case of no right hand side in einsum");
        } else {
            let rhs = &einsum_notation[arrow_position.unwrap() + 2..];
            let mut rhs_chars = rhs.chars();
            loop {
                let next = rhs_chars.next();

                match next {
                    Some(rhs_char) => {
                        match rhs_char {
                            ' ' => {}
                            '.' => {
                                assert!(
                                    ellipsis_in_output == false,
                                    "Double Ellipsis found in output"
                                );
                                // Are there enough characters left in the string
                                let next_next = rhs_chars.next();
                                let next_next_next = rhs_chars.next();

                                if next_next.is_none() || next_next_next.is_none() {
                                    println!("improper number of '.' characters of ellipis");
                                }

                                // And are they both ellipsis
                                let next_next = next_next.unwrap();
                                let next_next_next = next_next_next.unwrap();

                                if next_next != '.' || next_next_next != '.' {
                                    println!("improper number of '.' characters of ellipis");
                                }
                                index_of_ellipsis = permanent_index;
                                permanent_index += number_of_ellipsis_dimensions;
                                ellipsis_in_output = true;
                            }
                            _ => {
                                if rhs_char.is_alphabetic() == false {
                                    panic!("Invalid index character");
                                }

                                let index = label_to_subscript(rhs_char as u8);

                                assert!(
                                    label_count[index as usize] > 0
                                        && permanent_label_index[index as usize] == -1,
                                    "index appears more then once in the output index"
                                );

                                permanent_label_index[index as usize] = permanent_index as isize;
                                permanent_index += 1;
                            }
                        }
                    }
                    _ => {
                        break;
                    }
                }
            }
        }

        let number_of_output_dimensions = permanent_index;

        if ellipsis_in_output == false {
            index_of_ellipsis = permanent_index;
            permanent_index += number_of_ellipsis_dimensions;
        }

        for label in 0..TOTAL_LABELS {
            if label_count[label as usize] > 0 && permanent_label_index[label as usize] == -1 {
                permanent_label_index[label as usize] = permanent_index as isize;
                permanent_index += 1;
            }
        }

        let mut label_size = vec![1; TOTAL_LABELS as usize];
        let mut size_of_ellipsis = vec![1; number_of_ellipsis_dimensions];
        let mut dimenion_count = vec![0; permanent_index];

        let mut operands_stack: Vec<Tensor> = vec![];
        for i in 0..number_of_operands {
            let mut operand = operands[i];

            let mut permutation = vec![-1; permanent_index];
            let mut dimension = 0;

            for s in &operand_labels[i] {
                if *s == ELLIPSIS {
                    let number_of_dimensions =
                        operands[i].shape.number_of_dimension() - operand_labels[i].len() - 1;
                    for ii in (number_of_ellipsis_dimensions - number_of_dimensions)
                        ..number_of_ellipsis_dimensions
                    {
                        if operand.shape.dimensions()[dimension] != 1 {
                            assert!(
                                size_of_ellipsis[ii] == 1
                                    || size_of_ellipsis[ii]
                                        == operand.shape.dimensions()[dimension]
                            );

                            size_of_ellipsis[ii] = operand.shape.dimensions()[dimension];
                            dimenion_count[index_of_ellipsis + ii] =
                                dimenion_count[index_of_ellipsis + ii] + 1;
                        }

                        permutation[index_of_ellipsis + ii] = dimension as isize;
                        dimension += 1;
                    }
                } else if permutation[permanent_label_index[*s as usize] as usize] == -1 {
                    if operand.shape.dimensions()[dimension] != 1 {
                        assert!(
                            label_size[*s as usize] == 1
                                || label_size[*s as usize] == operand.shape.dimensions()[dimension]
                        );
                        label_size[*s as usize] = operand.shape.dimensions()[dimension];
                        dimenion_count[permanent_label_index[*s as usize] as usize] =
                            dimenion_count[permanent_label_index[*s as usize] as usize] + 1;
                    }
                    permutation[permanent_label_index[*s as usize] as usize] = dimension as isize;
                    dimension += 1;
                } else {
                    let previous_dimension =
                        permutation[permanent_label_index[*s as usize] as usize];

                    let diag = operand.diagonal(0, previous_dimension as usize, dimension);
                    let last_dimension = diag.shape.dimensions()[diag.shape.number_of_dimension() - 1];
                    operand = diag.movedim(last_dimension,previous_dimension as usize );
                }
            }
            for val in &mut permutation {
                if *val == -1 {
                    operand = operand.unsqueeze(dimension as isize);
                    *val = dimension as isize;
                    dimension += 1;
                }
            }
            operands_stack.push(operand.permute(permutation.iter().map(|x|*x as usize).collect()));
        }

        while operands_stack.len() > 1 {
            let mut i = 0;
            let mut j = 1;

            // In the original code there is a branch here
            // to handle a contraction path
            // as we don't do that it is not present
            // to reduce complexity

            let mut operand_a = operands_stack[i];
            let mut operand_b = operands_stack[j];
            
            operands_stack.remove(j);
            operands_stack.remove(i);
            
            let mut summed_dimensions = vec![];
            
            let mut a_dimensions_to_sum = vec![];
            let mut b_dimensions_to_sum = vec![];

            for dim in number_of_output_dimensions..permanent_index {
                
                let sa = operand_a.shape.dimensions()[dim] != 1;
                let sb = operand_b.shape.dimensions()[dim] != 1;

                if sa && sb {
                    assert!(operand_a.shape.dimensions()[dim] == operand_b.shape.dimensions()[dim]);
                    dimenion_count[dim] -= 1;
                    if dimenion_count[dim] == 1 {
                        summed_dimensions.push(dim);
                        dimenion_count[dim] = 0;
                    }
                }
                else if dimenion_count[dim] == 1 {
                    if sa == true {
                        a_dimensions_to_sum.push(dim);
                        dimenion_count[dim] = 0;
                    } else if sb == true {
                        b_dimensions_to_sum.push(dim);
                        dimenion_count[dim] = 0;
                    }
                }
            }

            if a_dimensions_to_sum.is_empty() == false {
                operand_a = operand_a.sum(a_dimensions_to_sum, true);
            }
            if b_dimensions_to_sum.is_empty() == false {
                operand_b = operand_b.sum(b_dimensions_to_sum, true);
            }

            operands_stack.insert(0, Tensor::sumproduct_pair(operand_a, operand_b, summed_dimensions, true));
        }

        if permanent_index - number_of_output_dimensions > 0 {
            if number_of_operands > 1 {
                let mut sizes = operands_stack[0].shape.dimensions();
                for dim in (number_of_output_dimensions..(permanent_index - 1)).rev() {
                    sizes.remove(dim);
                }
                return operands_stack[0].reshape(Shape::new(sizes));
            }
            else {
                let mut sum_dims = vec![0;permanent_index - number_of_output_dimensions];
                for i in 0..number_of_output_dimensions {
                    sum_dims.push(i);
                }

                return operands_stack[0].sum(sum_dims, false);
            }
        }

        return operands_stack[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Need to support cases 
    // i,j -> ij
    // "qhmd,khmd->hmqk"
    // beck,bk->bec
    // bec,be->bc
    #[test]
    fn basic_test() {
        let a = Tensor::arange(0, 4,1).reshape(Shape::new(vec![2, 2]));
        let b = Tensor::arange(5, 4,1).reshape(Shape::new(vec![2, 2]));
        println!("{:?}", a.item());
        println!("{:?}", b.item());
        let c =Tensor::einsum("ij,jk->ik", vec![a, b]);
        println!("{:?}", c.item())
    }

    #[test]
    fn basic_1d_1d_test() {
        let a = Tensor::arange(0, 4,1).reshape(Shape::new(vec![4]));
        let b = Tensor::arange(5, 4,1).reshape(Shape::new(vec![4]));
        println!("{:?}", a.item());
        println!("{:?}", b.item());
        let c =Tensor::einsum("i,k->ik", vec![a, b]);
        println!("{:?}", c.item())
    }

    #[test]
    // bec,be->bc
    fn basic_batch_test() {
        let a = Tensor::arange(0, 12,1).reshape(Shape::new(vec![3, 2, 2]));
        let b = Tensor::arange(0, 6,1).reshape(Shape::new(vec![3, 2]));
        println!("{:?}", a.item());
        println!("{:?}", b.item());
        let c = Tensor::einsum("bec,be->bc", vec![a, b]);
        println!("{:?}", c.item())
    }
}
