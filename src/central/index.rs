use crate::central::*;

/// An abstraction I have found useful in the past
/// Since we have a capped length the tensor can be at, it helps
#[derive(Debug)]
pub enum Indexable {
    Single(usize),
    Double(usize, usize),
    Triple(usize, usize, usize),
    Quadruable(usize, usize, usize, usize),
    FromTensor(TensorID),
}

impl From<[usize; 1]> for Indexable {
    fn from(value: [usize; 1]) -> Self {
        return Indexable::Single(value[0]);
    }
}

impl From<[usize; 2]> for Indexable {
    fn from(value: [usize; 2]) -> Self {
        return Indexable::Double(value[0], value[1]);
    }
}

impl Tensor {
    pub fn set_index(&self, index: Indexable, new_data: f32) {
        // Convert all of the indices to a Quadruable index, makes things simplier going forward
        let using_indexable = match index {
            Indexable::Single(index) => Indexable::Quadruable(0, 0, 0, index),
            Indexable::Double(a, b) => Indexable::Quadruable(0, 0, a, b),
            Indexable::Triple(a, b, c) => Indexable::Quadruable(0, a, b, c),
            Indexable::Quadruable(a, b, c, d) => Indexable::Quadruable(a, b, c, d),
            _ => {
                panic!("Not implemented");
            }
        };

        //
        get_equation().set_single_value(self.id, using_indexable, new_data);
    }
}

#[cfg(test)]
pub mod tests {
    #[test]
    fn basic_set_test() {}
}
