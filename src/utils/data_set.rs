use crate::central::Tensor;
use mnist::{Mnist, MnistBuilder};

pub trait DataSet {
    fn length(&self) -> usize;
    fn get_item(&self, index: usize) -> (Tensor, Tensor);
}

pub struct MnistDataSet {
    training_images: Vec<u8>,
    training_labels: Vec<u8>,
    _test_images: Vec<u8>,
    _test_labels: Vec<u8>,
}

impl MnistDataSet {
    // each image is a gray scale image of the digit, 28x28 in size, 28 * 28 = 784
    const MNIST_DATA_LENGTH: usize = 784;

    const MNIST_LABEL_LENGTH: usize = 1;

    pub fn new() -> MnistDataSet {
        let (training_images, training_labels, _test_images, _test_labels) =
            MnistDataSet::load_data_basic();
        MnistDataSet {
            training_images,
            training_labels,
            _test_images,
            _test_labels,
        }
    }

    fn load_data_basic() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new().label_format_digit().finalize();

        (trn_img, trn_lbl, tst_img, tst_lbl)
    }
}

impl DataSet for MnistDataSet {
    fn get_item(&self, index: usize) -> (Tensor, Tensor) {
        assert!(
            index * MnistDataSet::MNIST_DATA_LENGTH
                <= self.training_images.len() - MnistDataSet::MNIST_DATA_LENGTH
        );
        let data_start_offset = index * MnistDataSet::MNIST_DATA_LENGTH;
        let training_data = &self.training_images
            [data_start_offset..(data_start_offset + MnistDataSet::MNIST_DATA_LENGTH)];
        // The data is generally stored as u8 bytes of 0..255, this all works a little better if we move it to be between 0.0 and 1.0
        let data_mapped: Vec<f32> = training_data.iter().map(|x| *x as f32 / 255.0).collect();
        let label_start_offset = index * MnistDataSet::MNIST_LABEL_LENGTH;

        let label_data = &self.training_labels[label_start_offset];

        let mut one_hot = vec![0.0f32; 10];
        one_hot[*label_data as usize] = 1.0;

        let data_tensor = Tensor::from_vec(data_mapped, vec![MnistDataSet::MNIST_DATA_LENGTH]);
        let label_tensor = Tensor::from_vec(one_hot, vec![MnistDataSet::MNIST_LABEL_LENGTH * 10]);
        (data_tensor, label_tensor)
    }

    fn length(&self) -> usize {
        return self.training_images.len() / MnistDataSet::MNIST_DATA_LENGTH;
    }
}
