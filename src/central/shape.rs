// To keep the library simple we have a forced max of the number of dimensions we work with
pub const MAX_DIMS: usize = 10;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dimension: [usize; MAX_DIMS], // Right to left, the length of up to 4 dimensions, Capping at 4 since that is the most we will encounter
    in_use_dimension: usize,      // the number of the 4 dimensions that we are using
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
        assert!(
            dimensions.len() <= MAX_DIMS,
            "To many dimensions provided {:?}",
            dimensions
        );
        assert!(
            dimensions.len() != 0,
            "Must provide at least one dimension {:?}",
            dimensions
        );
        //TODO: do an assert for cases such as [1, 0, 1], which is invalid
        let in_use_dimension = dimensions.len();

        let mut final_dimensions: [usize; MAX_DIMS] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        for i in 0..in_use_dimension {
            assert!(
                dimensions[i] != 0,
                "tried to create a shape with a 0 dimension"
            );
            final_dimensions[i] = dimensions[i];
        }

        Shape {
            dimension: final_dimensions,
            in_use_dimension,
        }
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

    /// Returns a new shape where the dimenion at origin has been moved to destination
    ///
    /// # Arguments
    /// * 'origin' - The source location we are pulling the dimension from
    /// * 'destination' - The location we are moving the dimenion to
    pub fn move_index(&self, origin: usize, destination: usize) -> Shape {
        let mut current_dimension = self.dimensions();
        let max = origin.max(destination);
        assert!(
            max > current_dimension.len(),
            "too high dimension provided to move_index"
        );
        let element = current_dimension.remove(origin);
        current_dimension.insert(destination, element);
        return Shape::new(current_dimension);
    }

    /// Returns a new shape where the existing dimensions are rearranged according to the order set in permutation
    ///
    /// # Arguments
    /// * 'premutation' - The order the dimensions should be rearranged into
    pub fn permute(&self, permutation: &[usize]) -> Shape {
        let dims = self.dimensions();
        let rank = dims.len();
        assert!(
            permutation.len() == rank,
            "permute: expected {} indices, got {}",
            rank,
            permutation.len()
        );

        // O(n) validation: in-range + no duplicates
        let mut seen = vec![false; rank];
        for &p in permutation {
            assert!(p < rank, "permute: index {} out of range 0..{}", p, rank);
            assert!(!seen[p], "permute: duplicate index {}", p);
            seen[p] = true;
        }

        let new_dims = permutation.iter().map(|&i| dims[i]).collect();
        Shape::new(new_dims)
    }

    /// Returns a new shape with the dimensions changed to a new ones
    ///
    /// # Arguments
    /// *'index' - the 0base index we will be swapping
    /// *'new_dimension' the new dimension will be putting in there
    pub fn swap_index(&self, index: usize, dimension: usize) -> Shape {
        assert!(index < self.in_use_dimension);

        let mut new_dimension = vec![];
        for i in 0..self.in_use_dimension {
            // Skip the dimension provided
            if i != index {
                new_dimension.push(self.dimension[i]);
            } else if i == index {
                new_dimension.push(dimension);
            }
        }

        return Shape::new(new_dimension);
    }

    /// Returns a new shape with the provided dimension added at the provided index
    /// # Arguments
    /// * 'dimension' - the dimension to be inserted
    /// * 'index' - the 0base index that it will be added at
    pub fn add_dimension_at_index(&self, dimension: usize, index: usize) -> Shape {
        assert!(
            self.number_of_dimension() <= MAX_DIMS,
            "cannot add more dimensions to this shape"
        );

        //TODO: Add dynamic index assert
        assert!(index <= MAX_DIMS, "Index high then possible dimenions");
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
        if self.number_of_dimension() > 2 || other.number_of_dimension() > 2 {
            assert!(false == true, "assertion failed")
        }

        let last_dimension = self.dimensions()[self.dimensions().len() - 1];
        let first_dimension = other.dimensions()[0];
        return last_dimension == first_dimension;
    }

    /// Returns the shape that readies two matrices ready for matmul
    /// Broadcasting matrices is slightly different then normal
    /// as we only want to broadcast the batch dimensions [batch, batch, outer, inner]
    /// Which are those dimensions over 2
    /// for simplicity of matmul, we treat all matrices as if they where MAX_DIMSxMAX_DIMS
    /// # Arguments
    /// * 'a' - Left hand shape
    /// * 'b' - Right hand shape
    pub fn matmul_broadcast(a: Shape, b: Shape) -> (Shape, Shape) {
        let mut a_new_shape = vec![1; MAX_DIMS];
        let mut b_new_shape = vec![1; MAX_DIMS];

        // Right-align A into the last MAX_DIMS slots
        for (index, &dimension) in a.dimensions().iter().rev().enumerate() {
            a_new_shape[MAX_DIMS - 1 - index] = dimension;
        }

        // Right-align B; if B is 1-D, treat it as (k, 1) so it can act like a column vector.
        if b.number_of_dimension() == 1 {
            // put k in the penultimate dim; final dim (n) stays 1
            b_new_shape[MAX_DIMS - 2] = b.dimensions()[0];
        } else {
            for (index, &dimension) in b.dimensions().iter().rev().enumerate() {
                b_new_shape[MAX_DIMS - 1 - index] = dimension;
            }
        }

        // Broadcast all "batch" dimensions (everything except the last two)
        for i in 0..(MAX_DIMS - 2) {
            let ad = a_new_shape[i];
            let bd = b_new_shape[i];

            // Check broadcastability for this batch dim
            if ad != 1 && bd != 1 && ad != bd {
                panic!("None broadcastable shapes {:?} {:?}", a, b);
            }

            // Apply broadcasting if needed
            if ad == 1 || bd == 1 {
                let final_dimension = usize::max(ad, bd);
                a_new_shape[i] = final_dimension;
                b_new_shape[i] = final_dimension;
            }
        }

        // Note: as before, we don't check matmul core dims here:
        // A[..., m, k] x B[..., k, n] -> result checks happen elsewhere.

        (Shape::new(a_new_shape), Shape::new(b_new_shape))
    }

    /// Returns a new shape that would be the result of two matrices of the provided shapes being matmul together
    /// # Arguments
    /// * 'other' - the shape of ther right side operand
    pub fn matmul_shape(&self, other: Shape) -> Shape {
        let a_dims = self.dimensions().to_vec();
        let b_dims = other.dimensions().to_vec();

        // 1) 1D @ 1D -> scalar (keep your convention: shape [1])
        if a_dims.len() == 1 && b_dims.len() == 1 {
            assert!(
                a_dims[0] == b_dims[0],
                "Mis matching dimensions for 1d@1d matrix multiply a{:?} b{:?}",
                self,
                other
            );
            return Shape::new(vec![1]);
        }

        // Build working 2D-or-higher shapes by "unsqueezing" vectors:
        // If A is 1D [k] -> [1, k]
        // If B is 1D [k] -> [k, 1]
        let a_was_vec = a_dims.len() == 1;
        let b_was_vec = b_dims.len() == 1;

        let a_work = if a_was_vec {
            vec![1, a_dims[0]]
        } else {
            a_dims.clone()
        };
        let b_work = if b_was_vec {
            vec![b_dims[0], 1]
        } else {
            b_dims.clone()
        };

        // 2) Validate matrix core dims: A[..., m, k] @ B[..., k, n]
        assert!(
            a_work.len() >= 2 && b_work.len() >= 2,
            "Both operands must be at least 1D (after normalization)."
        );
        let a_m = a_work[a_work.len() - 2];
        let a_k = a_work[a_work.len() - 1];
        let b_k = b_work[b_work.len() - 2];
        let b_n = b_work[b_work.len() - 1];

        assert!(
            a_k == b_k,
            "Mis matching core dims a_k={} vs b_k={} for matmul a{:?} b{:?}",
            a_k,
            b_k,
            self,
            other
        );

        // 3) Broadcast batch dims (everything except the last two), right-aligned
        let a_batch = &a_work[..a_work.len() - 2];
        let b_batch = &b_work[..b_work.len() - 2];

        // Right-align and broadcast
        let mut result_batch = vec![];
        let max_batch_len = a_batch.len().max(b_batch.len());
        for i in 0..max_batch_len {
            // pick from the right
            let a_dim = if i < a_batch.len() {
                a_batch[a_batch.len() - 1 - i]
            } else {
                1
            };
            let b_dim = if i < b_batch.len() {
                b_batch[b_batch.len() - 1 - i]
            } else {
                1
            };

            if a_dim != 1 && b_dim != 1 && a_dim != b_dim {
                panic!(
                    "None broadcastable batch dims at offset {}: a_dim={}, b_dim={} for a{:?} b{:?}",
                    i, a_dim, b_dim, self, other
                );
            }
            result_batch.push(usize::max(a_dim, b_dim));
        }
        result_batch.reverse();

        // 4) Compose result matrix dims, then squeeze back if a/b were vectors
        let mut result = result_batch;
        result.push(a_m);
        result.push(b_n);

        if a_was_vec {
            // drop the 'm' dimension -> batch + [n]
            // (remove the element just before the last: index len-2)
            let drop_idx = result.len() - 2;
            result.remove(drop_idx);
        }
        if b_was_vec {
            // drop the 'n' dimension -> batch + [m]
            // (remove the last element)
            result.pop();
        }

        // Special-cases naturally fall out:
        // - 2D@2D -> [m, n]
        // - 2D@1D -> [m]
        // - 1D@2D -> [n]
        // - ND@MD -> batch-broadcasted + [m, n]

        Shape::new(result)
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

    pub fn should_broadcast(&self, other: Shape) -> bool {
        let self_dimensions = self.dimensions();
        let other_dimensions = other.dimensions();
        return self_dimensions.len() <= other_dimensions.len();
    }

    /// Returns the shape of that results from two shapes being brodcasted
    /// # Arguments
    /// * 'other' - The shape we are testing against
    pub fn broadcast_shape(&self, other: Shape) -> Shape {
        assert!(
            self.can_broadcast(other),
            "Cannot broadcast {:?} {:?}",
            self,
            other
        );
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
            return Shape::new(dimension_in_reverse.iter().rev().map(|x| *x).collect());
        }

        if self_dimensions.len() > other_dimensions.len() {
            let diff = self_dimensions.len() - other_dimensions.len();
            let extra = &self_dimensions[0..diff];
            dimension_in_reverse.extend(extra.iter().rev().map(|x| *x));
        }

        dimension_in_reverse.reverse();
        // Broadcasted shapes should always be the same time, so this is a cheap final check to
        // help catch cases where that is not true and should be fixed right away

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

    /// checks if you can reshape one shape into another
    pub fn can_reshape_to(&self, other_shape: Shape) -> bool {
        // with reshapeing you can't add or remove elements, so they total size must be equal
        if self.total_size() != other_shape.total_size() {
            return false;
        }

        return true;
    }

    /// adds a dimension of 1 at the provided dimensions
    pub fn unsqueeze(&self, dimension: isize) -> Shape {
        let rank = self.number_of_dimension();
        // Valid dimensions in PyTorch semantics are [-rank - 1, rank]
        if dimension < -(rank as isize) - 1 || dimension > rank as isize {
            panic!(
                "Dimension provided to unsequeeze larger then number of dimensions the tensor has"
            );
        }

        let resolved_index = if dimension >= 0 {
            dimension as usize
        } else {
            // Negative dims insert counting from the end, so `-1` appends.
            (rank as isize + dimension + 1) as usize
        };

        let mut current_dimensions = self.dimensions();
        current_dimensions.insert(resolved_index, 1);

        Shape::new(current_dimensions)
    }

    pub fn generate_all_positions(&self) -> Vec<Vec<usize>> {
        let shape = self.dimensions();

        if shape.is_empty() {
            return vec![Vec::new()];
        }
        if shape.iter().any(|&d| d == 0) {
            return Vec::new();
        }

        let total: usize = shape.iter().product();
        let k = shape.len();
        let mut result = Vec::with_capacity(total);
        let mut idx = vec![0usize; k];

        loop {
            result.push(idx.clone());

            // increment like an odometer (last dim fastest)
            for dim in (0..k).rev() {
                idx[dim] += 1;
                if idx[dim] < shape[dim] {
                    break; // normal carry resolved
                } else {
                    idx[dim] = 0; // carry to the next more-significant dim
                    if dim == 0 {
                        return result; // overflowed the most-significant dim
                    }
                }
            }
        }
    }

    pub fn compute_strides_row_major(&self) -> Vec<usize> {
        let shape = self.dimensions();
        let n = shape.len();
        if n == 0 {
            return Vec::new(); // scalar: no strides
        }
        let mut strides = vec![0usize; n];
        let mut acc = 1usize;
        // last dim stride = 1; proceed backward
        for (i, &dim) in shape.iter().enumerate().rev() {
            strides[i] = acc;
            acc = acc
                .checked_mul(dim)
                .expect("stride multiplication overflowed usize");
        }
        strides
    }
}

#[cfg(test)]
mod tests {
    use super::Shape;

    #[test]
    pub fn basic_allocation_test() {
        let shape = Shape::new(vec![1]);
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
        assert!(
            new_shape.total_size() == (1 * 2 * 3),
            "dimensions {:?}",
            dimensions
        );

        let dimensions = new_shape.dimensions();

        for i in 0..3 {
            assert!(dimensions[i] == i + 1, "dimensions {:?}", dimensions);
        }

        // Remove the middle index
        let second_new_shape = new_shape.remove_index(1);

        assert!(second_new_shape.total_size() == (1 * 3));
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
        assert!(dimensions[1] == 2, "dimensions {:?}", dimensions);

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
    #[should_panic(expected = "Must provide at least one dimension []")]
    pub fn bad_shape_test() {
        let _shape = Shape::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "Must provide at least one dimension []")]
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

    #[test]
    pub fn simple_matmul_test_3x1() {
        let a = Shape::new(vec![1, 2, 6]);
        let b = Shape::new(vec![6]);
        let matmul_shape = a.matmul_shape(b);
        assert!(matmul_shape.dimensions()[0] == 1);
        assert!(matmul_shape.dimensions()[1] == 2);
    }

    #[test]
    pub fn simple_matmul_test_1x3() {
        let a = Shape::new(vec![2]);
        let b = Shape::new(vec![1, 2, 6]);
        let matmul_shape = a.matmul_shape(b);
        assert!(matmul_shape.dimensions()[0] == 1);
        assert!(matmul_shape.dimensions()[1] == 6);
    }

    #[test]
    pub fn simple_matmul_test_2x3() {
        let a = Shape::new(vec![6, 2]);
        let b = Shape::new(vec![1, 2, 6]);
        let matmul_shape = a.matmul_shape(b);
        assert!(matmul_shape.dimensions()[0] == 1);
        assert!(matmul_shape.dimensions()[1] == 6);
        assert!(matmul_shape.dimensions()[2] == 6);
    }

    #[test]
    pub fn simple_matmul_test_4x3() {
        let a = Shape::new(vec![15, 15, 6, 2]);
        let b = Shape::new(vec![1, 2, 6]);
        let matmul_shape = a.matmul_shape(b);
        assert!(matmul_shape.dimensions()[0] == 15);
        assert!(matmul_shape.dimensions()[1] == 15);
        assert!(matmul_shape.dimensions()[2] == 6);
        assert!(matmul_shape.dimensions()[3] == 6);
    }

    #[test]
    fn unsqueeze_inserts_dimension_at_positive_index() {
        let shape = Shape::new(vec![2, 3, 4]);
        let unsqueezed = shape.unsqueeze(1);
        assert_eq!(unsqueezed.dimensions(), vec![2, 1, 3, 4]);
        let append = shape.unsqueeze(3);
        assert_eq!(append.dimensions(), vec![2, 3, 4, 1]);
    }

    #[test]
    fn unsqueeze_supports_negative_index_from_end() {
        let shape = Shape::new(vec![5, 7]);
        let prepend = shape.unsqueeze(-3);
        assert_eq!(prepend.dimensions(), vec![1, 5, 7]);
        let append = shape.unsqueeze(-1);
        assert_eq!(append.dimensions(), vec![5, 7, 1]);
    }

    #[test]
    #[should_panic(
        expected = "Dimension provided to unsequeeze larger then number of dimensions the tensor has"
    )]
    fn unsqueeze_panics_when_index_out_of_range() {
        let shape = Shape::new(vec![3, 3]);
        let _ = shape.unsqueeze(3);
    }
}
