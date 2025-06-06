use crate::central::*;

impl Tensor  {
    /// returns a tensors with each element raised to the provided power
    /// arguments 
    /// 'power' - the power we are will raise each element to 
    pub fn pow(&self, power: f32) -> Tensor {
        let power_as_tensor = Tensor::element(Shape::new(vec![1]), power);
        let data : Vec<f32> = self
            .item()
            .into_iter()
            .map(|x| x.powf(power))
            .collect();
        return Tensor::create_tensor_data_and_shape_and_operation( self.shape, data, Operation::Pow(self.id, power_as_tensor.id));
    }
}

pub fn backward_for_pow(backprop_backet: BackproagationPacket) {
    if let Operation::Pow(base, power) = backprop_backet.operation {
        const EPS: f32 = 1.0e-12;
        let base_data = backprop_backet.equation.get_item(base);
        let power_data = backprop_backet.equation.get_item(power);

        let power = power_data[0];
        let p_minus_1 = power - 1.0;


        // the derivative of x^n = n * x ^ n-1
        // and we add in a little EPS to avoid zeros ^ X, since that is INF
        let grad_update = base_data.mapv(|x|{
            power * (x + EPS).powf(p_minus_1)
        }) *  backprop_backet.equation.get_grad(backprop_backet.incoming_grad);
        // Then we need to multiply it by the original grad

        // and then we are done
        backprop_backet.equation.add_tensor_grad(base, grad_update.to_owned().into_raw_vec());
    }
}

#[cfg(test)]
mod tests {
    use crate::{central::*, utils::GGUFFile};
    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    fn basic_pow_test() {
        let tensor = Tensor::element(Shape::new(vec![4]), 5.0);
        let tensor_pow = tensor.pow(2.0);
        for d in tensor_pow.item() {
            assert!(d == 25.0);
        }
    }

    #[test]
    fn advance_pow_test() {
        let epsilon = 1e-5;
        let mut gguf_file = GGUFFile::new(String::from(
            "./models/tests/pow/advance_pow_test.gguf",
        ));

        let tensor_a = Tensor::from_gguf_file("add_broadcast_test_tensor_a".to_string(), &mut gguf_file);
        let tensor_pow_real = Tensor::from_gguf_file("add_broadcast_test_tensor_pow".to_string(), &mut gguf_file);
        let tensor_pow = tensor_a.pow(3.0);

        let tensor_pow_item = tensor_pow.item();
        let together = tensor_pow_item.iter().zip(tensor_pow_real.item());

        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon));
        }
    }
}