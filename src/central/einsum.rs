use super::{BackproagationPacket, Operation, Tensor};
use crate::central::*;
use regex::Regex;
use std::{collections::HashMap, hash::Hash};

impl Tensor {
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
                            if current_operand < number_of_operands {
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
        let mut size_of_ellipsis = vec![1, number_of_ellipsis_dimensions];
        let mut dimenion_count = vec![0, permanent_index];

        let mut ops: Vec<Tensor> = vec![];
        for i in 0..number_of_operands {
            let operand = operands[i];

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
                    permutation[label_size[*s as usize]] = dimension as isize;
                    dimension += 1;
                } else {
                    let previous_dimension =
                        permutation[permanent_label_index[*s as usize] as usize];
                }
            }
        }

        Tensor::randn(Shape::new(vec![1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
