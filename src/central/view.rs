use crate::central::index::Indexable;
use crate::central::Tensor;


impl Tensor {
    /// Creates a view of the tensor using the provided indexable
    /// # Arguments
    /// * 'indexable' - The indexing method to use
    pub fn view(&self, indexable: Indexable) -> Tensor {
        match indexable {
            Indexable::FromTensor(index_tensor_id) => {
                // Use the select operation for tensor-based indexing (embedding lookup)
                self.select(index_tensor_id)
            },
            _ => {
                panic!("Only FromTensor indexing is currently supported for view operation");
            }
        }
    }
}