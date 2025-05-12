use core::panic;

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    dimension: [usize;4], // Right to left, the length of up to 4 dimensions, Capping at 4 since that is the most we will encounter
    in_use_dimension: usize // the number of the 4 dimensions that we are using
}

impl Shape {
    /// Creates a new `Shape` instance with the given dimensions
    ///
    /// # Arguments
    ///
    /// * `dimensions` - A vector of indices representing the shape.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of indices exceeds the maximum number of indices.
    /// This function will panic if there are 0 supplied dimensions
    pub fn new(dimensions: Vec<usize>) -> Shape {
        assert!(dimensions.len() <= 4);
        assert!(dimensions.len() != 0);
        //TODO: do an assert for cases such as [1, 0, 1], which is invalid
        let in_use_dimension = dimensions.len();

        let mut final_dimensions = [0, 0, 0, 0];
        for i in 0..in_use_dimension {
            assert!(dimensions[i] != 0, "tried to create a shape with a 0 dimension");
            final_dimensions[i] = dimensions[i];
        }

        Shape { dimension: final_dimensions, in_use_dimension }
    }

    /// Returns the number of dimenions of the shape
    pub fn number_of_dimension(&self) -> usize {
        return self.in_use_dimension;
    }

    /// Returns the total number of elements in this shape
    pub fn total_size(&self) -> usize {
        let mut total = self.dimension[0];
        // we start with total as the frist dimension to not have a 0 there and then do a mul by 0
        for i in 1..self.in_use_dimension {
            total *= self.dimension[i];
        }

        return total;
    }

    /// Returns the dimensions of the shape
    pub fn dimensions(&self) -> Vec<usize> {
        let mut dimensions = vec![];
        for i in 0..self.in_use_dimension {
            dimensions.push(self.dimension[i]);
        }
        return dimensions;
    }

    /// Returns a new shape with the dimension at the provided index removed
    /// 
    /// # Arguments
    /// * 'index' - The 0base index to be removed from the shape
    pub fn remove_index(&self, index: usize) -> Shape {
        assert!(index < self.in_use_dimension);

        let mut new_dimension = vec![];
        for i in 0..self.in_use_dimension {
            // Skip the dimension provided
            if i != index {
                new_dimension.push(self.dimension[i]);
            }
        }

        return Shape::new(new_dimension);
    }

    /// Returns a new shape with the provided dimension added at the provided index
    /// # Arguments
    /// * 'dimension' - the dimension to be inserted
    /// * 'index' - the 0base index that it will be added at 
    pub fn add_dimension_at_index(&self, dimension: usize, index: usize) -> Shape {
        assert!(self.number_of_dimension() <= 3, "cannot add more dimensions to this shape");

        //TODO: Add dynamic index assert
        assert!(index <= 3, "Index high then possible dimenions");
        assert!(dimension != 0, "Cannot inset zero dimension");
        let mut current_dimension = self.dimensions();
        current_dimension.insert(index, dimension);
        return Shape::new(current_dimension);
    }

    /// Returns if two shapes can be matmuled together
    /// This only covers 2x2 cases or lower
    /// higher dimenions are considered "batched" matmul
    /// # Arguments 
    /// * 'other' - the other shape provided to check against
    pub fn can_matmul(&self, other: Shape) -> bool {
        if self.number_of_dimension() > 2 || other.number_of_dimension() > 2{
            assert!(false == true, "assertion failed")
        }

        let last_dimension = self.dimensions()[self.dimensions().len() - 1];
        let first_dimension = other.dimensions()[0];
        return last_dimension == first_dimension;
    }

    /// Returns a new shape that would be the result of two matrices of the provided shapes being matmul together
    /// # Arguments
    /// * 'other' - the shape of ther right side operand
    pub fn matmul_shape(&self, other: Shape) -> Shape {
        let mut final_dimensions = vec![];

        // Following rules roughly laid out in https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
        // If they are both vectors(1d matrix) then the result will be a single scalar, so the shape will be 1 d
        if self.number_of_dimension() == 1 && other.number_of_dimension() == 1 {
            assert!(self.dimensions()[0] == other.dimensions()[0], "Mis matching dimensions for 1d@1d matrix multiply a{:?} b{:?}", self, other);
            return Shape::new(vec![1]);
        }

        // If they are both 2d, follow the standard forumla, MxN @ NxB = MxB
        if self.number_of_dimension() == 2 && other.number_of_dimension() == 2 {
            assert!(self.dimensions()[1] == other.dimensions()[0], "Mis matching dimensions for 2d@2d matrix multiply a{:?} b{:?}", self, other);
            return Shape::new(vec![self.dimensions()[0], other.dimensions()[1]]);
        }

        // if one is 2d and the other is 1d, take the leading dimension
        if self.number_of_dimension() == 2 && other.number_of_dimension() == 1 {
            assert!(self.dimensions()[1] == other.dimensions()[0], "Mis matching dimensions for 2d@1d matrix multiply a{:?} b{:?}", self, other);
            return Shape::new(vec![self.dimensions()[0]]);
        }
        // if one is 1d and the other is 2d, take the trailing dimension
        if self.number_of_dimension() == 1 && other.number_of_dimension() == 2 {
            assert!(self.dimensions()[0] == other.dimensions()[0], "Mis matching dimensions for 1d@2d matrix multiply a{:?} b{:?}", self, other);
            return Shape::new(vec![other.dimensions()[1]]);
        }

        let mut one_dimension_added_to_left = false;
        let mut one_dimension_add_to_right = false;
        // Then get left and right operand for the batch test
        let left_hand_working_shape = {

            if self.number_of_dimension() == 4 || self.number_of_dimension() == 3 {
                let return_shape = self.clone();
                if self.number_of_dimension() == 3 {
                    return_shape.remove_index(0)
                }
                else {
                    return_shape.remove_index(0).remove_index(0)
                }
            }

            else {
                if self.number_of_dimension() == 2 {
                    self.clone()
                }
                else {
                    //NOTE: this is different then the right side, we are prepending, below does a append
                    one_dimension_added_to_left = true;
                    self.add_dimension_at_index(1, 0)
                }
            }
        };

        let right_hand_working_shape = {
            if other.number_of_dimension() == 4 || other.number_of_dimension() == 3 {
                let return_shape = other.clone();
                if self.number_of_dimension() == 3 {
                    return_shape.remove_index(0)
                }
                else {
                    return_shape.remove_index(0).remove_index(0)
                }
            }

            else {
                if other.number_of_dimension() == 2 {
                    other.clone()
                }
                else {
                    //NOTE: this is different then the left side, we are appending, above does a prepend
                    one_dimension_add_to_right = true;
                    other.add_dimension_at_index(1, 1)
                }
            }
        };

        let mut dimensions = vec![left_hand_working_shape.dimensions()[0], right_hand_working_shape.dimensions()[1]];

        // We need to remove the added dimensions that made it easier to and simpler to make the new shape
        if one_dimension_added_to_left {
            dimensions.remove(0);
        }
        else if one_dimension_add_to_right {
            dimensions.remove(1);
        }


        // We need to the batch dimensions (the lead dimensions of any vector greater then 2 in length)
        let mut left_side_batch_dimension = vec![];
        let mut right_side_batch_dimension = vec![];

        let number_of_left_side_batch_dimension = self.number_of_dimension() - 2;
        let number_of_right_side_batch_dimension = other.number_of_dimension() - 2;

        if number_of_left_side_batch_dimension == 0 {
            for i in 0..number_of_right_side_batch_dimension {
                final_dimensions.push(other.dimensions()[i]);
            }
            
            for i in 0..dimensions.len() {
                final_dimensions.push(dimensions[i]);
            }
            return Shape::new(final_dimensions);
        }

        if number_of_right_side_batch_dimension == 0 {
            for i in 0..number_of_left_side_batch_dimension {
                final_dimensions.push(self.dimensions()[i]);
            }
            
            for i in 0..dimensions.len() {
                final_dimensions.push(dimensions[i]);
            }
            return Shape::new(final_dimensions);
        }

        for i in 0..number_of_left_side_batch_dimension {
            left_side_batch_dimension.push(self.dimensions()[i]);
        }

        for i in 0..number_of_right_side_batch_dimension {
            right_side_batch_dimension.push(other.dimensions()[i]);
        }

        let left_side_batch_dimension_shape = Shape::new(left_side_batch_dimension);
        let right_side_batch_dimension_shape = Shape::new(right_side_batch_dimension);

        let broadcast_shape = left_side_batch_dimension_shape.broadcast_shape(right_side_batch_dimension_shape);

        let broadcast_dimension = broadcast_shape.dimensions();
        
        for i in 0..broadcast_dimension.len() {
            final_dimensions.push(broadcast_dimension[i]);
        }
        
        for i in 0..dimensions.len() {
            final_dimensions.push(dimensions[i]);
        }

        return Shape::new(final_dimensions);
    }

    /// Checks if the two shapes are broadcastable
    /// # Arguments
    /// * 'other' - The shape we are testing against
    pub fn can_broadcast(&self, other: Shape) -> bool {
        // One common requirement of shapes for broadcastsing is that they both must have at least 1 dimensions
        // we can skip that check as it creating a shape with less then one dimenion will cause an assert failure
        let self_dimensions = self.dimensions();
        let other_dimensions = other.dimensions();

        let lowest_length = usize::min(self_dimensions.len(), other_dimensions.len());

        for i in 0..lowest_length {
            let self_index = self_dimensions[self_dimensions.len() - 1 - i];
            let other_index = other_dimensions[other_dimensions.len() - 1 - i];
            if self_index != other_index && (self_index != 1 && other_index != 1) {
                return false;
            }
        }


        return true;
    }

    /// Returns the shape of that results from two shapes being brodcasted
    /// # Arguments
    /// * 'other' - The shape we are testing against
    pub fn broadcast_shape(&self, other: Shape) -> Shape {
        assert!(self.can_broadcast(other), "Cannot broadcast {:?} {:?}", self, other);
        let self_dimensions = self.dimensions();
        let other_dimensions = other.dimensions();


        let lowest_length = usize::min(self_dimensions.len(), other_dimensions.len());
        let mut dimension_in_reverse = vec![];

        for i in 0..lowest_length {
            let self_index = self_dimensions[self_dimensions.len() - 1 - i];
            let other_index = other_dimensions[other_dimensions.len() - 1 - i];
            if self_index == other_index || (self_index == 1 || other_index == 1) {
                // if the two dimensions match, it just needs to pick one(max will do this)
                // other wise we want to the dimension that is not 1
                dimension_in_reverse.push(usize::max(self_index, other_index));
            }
        }

        // If they are the same length we are just early return and be done with this
        if self_dimensions.len() == other_dimensions.len() { 
            return Shape::new(dimension_in_reverse.iter().rev().map(|x|*x).collect());
        }

        if self_dimensions.len() > other_dimensions.len() {
            let diff = self_dimensions.len() - other_dimensions.len();
            let extra = &self_dimensions[0..diff];
            dimension_in_reverse.extend(extra.iter().rev().map(|x|*x));
        }

        dimension_in_reverse.reverse();
        return Shape::new(dimension_in_reverse);
    }

    /// Returns the cant lib shape type in the right format to passed to an ndarray ArrayD
    pub fn as_ndarray_shape(&self) -> Vec<usize> {
        let mut shape = Vec::new();
        for i in 0..self.in_use_dimension {
            shape.push(self.dimension[i]);
        }
        shape
    }

}

#[cfg(test)]   
mod tests {
    use super::Shape;

    #[test]
    pub fn basic_allocation_test() {
        let shape = Shape::new(vec![1]);
        println!("shape {}", shape.total_size());
        assert!(shape.total_size() == 1);

        let dimensions = shape.dimensions();

        assert!(dimensions[0] == 1);
    }

    #[test]
    pub fn basic_allocation_test_2() {
        let shape = Shape::new(vec![1, 2, 3, 4]);
        assert!(shape.total_size() == (1 * 2 * 3 * 4));

        let dimensions = shape.dimensions();

        for i in 0..4 {
            assert!(dimensions[i] == i + 1);
        }
    }

    #[test]
    pub fn basic_remove_test() {

        // Set up a basic shape
        let shape = Shape::new(vec![1, 2, 3, 4]);
        assert!(shape.total_size() == (1 * 2 * 3 * 4));
        let dimensions = shape.dimensions();
        for i in 0..4 {
            assert!(dimensions[i] == i + 1, "dimensions {:?}", dimensions);
        }

        // Remove just the last dimension
        let new_shape = shape.remove_index(3);
        assert!(new_shape.total_size() == (1 * 2 * 3 ), "dimensions {:?}", dimensions);

        let dimensions = new_shape.dimensions();

        for i in 0..3 {
            assert!(dimensions[i] == i + 1, "dimensions {:?}", dimensions);
        }

        // Remove the middle index
        let second_new_shape = new_shape.remove_index(1);

        assert!(second_new_shape.total_size() == (1 * 3 ));
        let dimensions = second_new_shape.dimensions();
        assert!(dimensions[0] == 1, "dimensions {:?}", dimensions);
        assert!(dimensions[1] == 3, "dimensions {:?}", dimensions);
    }


    #[test]
    pub fn basic_add_test() {
        let shape = Shape::new(vec![1, 2]);
        assert!(shape.total_size() == 1 * 2, "shape {:?} is wrong", shape);
        let dimensions = shape.dimensions();
        assert!(dimensions.len() == 2);
        assert!(dimensions[0] == 1, "dimensions {:?}", dimensions);
        assert!(dimensions[1] == 2,  "dimensions {:?}", dimensions);

        let new_shape = shape.add_dimension_at_index(3, 2);
        let new_dimensions = new_shape.dimensions();
        assert!(new_dimensions.len() == 3);
        assert!(new_dimensions[0] == 1);
        assert!(new_dimensions[1] == 2);
        assert!(new_dimensions[2] == 3);

        let new_shape = new_shape.add_dimension_at_index(4, 1);
        let new_dimensions = new_shape.dimensions();
        assert!(new_dimensions.len() == 4); 
        assert!(new_dimensions[0] == 1);
        assert!(new_dimensions[1] == 4);
        assert!(new_dimensions[2] == 2);
        assert!(new_dimensions[3] == 3);
    }

    #[test]
    pub fn add_remove_test() {
        let start = Shape::new(vec![10, 1]);
        let next = start.add_dimension_at_index(1, 0);
        let dimensions = next.dimensions();
        assert!(dimensions[0] == 1);
        assert!(dimensions[1] == 10);
        assert!(dimensions[2] == 1);

        let next = next.remove_index(1);
        let dimensions = next.dimensions();
        assert!(dimensions[0] == 1);
        assert!(dimensions[1] == 1);
    }

    #[test]
    pub fn complicated_broadcast_test() {
        let a = Shape::new(vec![10, 1]);
        let b = Shape::new(vec![1, 15]);
        let new_shape = a.broadcast_shape(b);
        let dimensions = new_shape.dimensions();
        assert!(dimensions[0] == 10);
        assert!(dimensions[1] == 15);
    }

    #[test]
    pub fn complicated_matmul_test_shape() {
        let a = Shape::new(vec![10, 1, 5, 6]);
        let b = Shape::new(vec![1, 15, 6, 7]);
        let new_shape = a.matmul_shape(b);
        let dimensions = new_shape.dimensions();
        assert!(dimensions[0] == 10);
        assert!(dimensions[1] == 15);
        assert!(dimensions[2] == 5);
        assert!(dimensions[3] == 7);

        let a = Shape::new(vec![1, 5, 6]);
        let b = Shape::new(vec![15, 6, 7]);
        let new_shape = a.matmul_shape(b);
        let dimensions = new_shape.dimensions();

        assert!(dimensions[0] == 15);
        assert!(dimensions[1] == 5);
        assert!(dimensions[2] == 7);
    }

    #[test]
    pub fn basic_matmul_shape_tests() {
        let shape_a = Shape::new(vec![10, 10]);
        let shape_b = Shape::new(vec![10, 10]);
        let new_shape = shape_a.matmul_shape(shape_b);
        let dimensions = new_shape.dimensions();
        assert!(dimensions[0] == 10);
        assert!(dimensions[1] == 10);
        
        let shape_a = Shape::new(vec![1, 10]);
        let shape_b = Shape::new(vec![10, 1]);
        let new_shape = shape_a.matmul_shape(shape_b);
        let dimensions = new_shape.dimensions();
        assert!(dimensions[0] == 1);
        assert!(dimensions[1] == 1);
        
        let shape_a = Shape::new(vec![2, 1, 10]);
        let shape_b = Shape::new(vec![10, 1]);
        let new_shape = shape_a.matmul_shape(shape_b);
        let dimensions = new_shape.dimensions();
        assert!(dimensions[0] == 2);
        assert!(dimensions[1] == 1);
        assert!(dimensions[2] == 1);
    }

    #[test]
    pub fn can_matmul_test() {
        let shape_a = Shape::new(vec![10, 10]);
        let shape_b = Shape::new(vec![10, 10]);
        assert!(shape_a.can_matmul(shape_b));

        let shape_a = Shape::new(vec![10, 10]);
        let shape_b = Shape::new(vec![1, 10]);
        assert!(shape_a.can_matmul(shape_b) == false);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    pub fn bad_shape_test() {
        let _shape = Shape::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    pub fn bad_shape_test_remove_at_zero() {
        let shape = Shape::new(vec![1]);
        let _new_shape = shape.remove_index(0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    pub fn bad_can_matmul() {
        let a = Shape::new(vec![10, 1, 5, 6]);
        let b = Shape::new(vec![1, 15, 6, 7]);
        a.can_matmul(b);
    }

    #[test]
    pub fn simple_matmul_test_1x1() {
        let a = Shape::new(vec![6]);
        let b = Shape::new(vec![6]);
        let matmul_shape = a.matmul_shape(b);
        assert!(matmul_shape.dimensions()[0] == 1);
    }

    #[test]
    pub fn simple_matmul_test_1x2() {
        let a = Shape::new(vec![6]);
        let b = Shape::new(vec![6, 1]);
        let matmul_shape = a.matmul_shape(b);
        assert!(matmul_shape.dimensions()[0] == 1);
    }

    #[test]
    pub fn simple_matmul_test_2x1() {
        let a = Shape::new(vec![1, 6]);
        let b = Shape::new(vec![6]);
        let matmul_shape = a.matmul_shape(b);
        assert!(matmul_shape.dimensions()[0] == 1);
    }

    #[test]
    pub fn bad_broadacst_test() {
        let a = Shape::new(vec![4, 6]);
        let b = Shape::new(vec![5, 6]);
        assert!(a.can_broadcast(b) == false);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast Shape")]
    pub fn bad_broadacst_shape_test() {
        let a = Shape::new(vec![4, 6]);
        let b = Shape::new(vec![5, 6]);
        a.broadcast_shape(b);
    }


    /* 
    #[test]
    pub fn simple_matmul_test_3x1() {
        let a = Shape::new(vec![1, 2, 6]);
        let b = Shape::new(vec![6]);
        let matmul_shape = a.matmul_shape(b);
        assert!(matmul_shape.dimensions()[0] == 1);
    }
    */
}