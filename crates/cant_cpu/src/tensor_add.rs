/// Takes two tensors as flat buffers, adds them together at an elementwise level and returns the result
/// # Arguments
/// * 'a' : The first vector
/// * 'b' : the second vector
pub fn tensor_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert!(
        a.len() == b.len(),
        "tensor add requires that both vectors be the same length"
    );
    let mut result = vec![0.0; b.len()];
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
    return result;
}
