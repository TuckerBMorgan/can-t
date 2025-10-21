// A minimal, self-contained einsum implementation in Rust (no external crates).
// Scope:
// - Supports lowercase letter labels a-z
// - No ellipsis; each input term is a sequence of labels like "ij" or "bnm"
// - Broadcasting only for size-1 dimensions when the same label appears across tensors
// - Tensors are provided as flat slices (row-major), with a provided shape array of length 4.
//   The effective rank is the number of labels in that input term; we read the first R dims.
// - Multiple inputs supported. Output is a flat Vec<f32> with its shape returned alongside.
//
// Example usage at bottom (see `main`).

use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: Vec<usize>,   // effective shape, trimmed to the rank implied by labels
    strides: Vec<usize>, // row-major strides
    labels: Vec<char>,
}

#[derive(Debug)]
pub struct EinsumResult {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub labels: Vec<char>, // output labels, order as in notation
}

pub fn einsum(
    einsum_notation: &str,
    tensors_and_shape: Vec<(&[f32], &[usize; 4])>,
) -> Result<EinsumResult, String> {
    // 1) Parse notation: "lhs1,lhs2,...->out"
    let parts: Vec<&str> = einsum_notation.split("->").collect();
    if parts.len() != 2 {
        return Err("Notation must contain a single '->'".into());
    }
    let inputs_str = parts[0].trim();
    let output_str = parts[1].trim();

    let input_terms: Vec<&str> = inputs_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if input_terms.is_empty() {
        return Err("No input terms found".into());
    }
    if input_terms.len() != tensors_and_shape.len() {
        return Err(format!(
            "Number of input terms ({}) != number of tensors ({})",
            input_terms.len(),
            tensors_and_shape.len()
        ));
    }

    let out_labels: Vec<char> = output_str.chars().collect();
    validate_labels(&out_labels)?;

    // 2) Build TensorViews for each input term
    let mut inputs: Vec<TensorView> = Vec::with_capacity(input_terms.len());
    for (term, (data, shape4)) in input_terms.iter().zip(tensors_and_shape.into_iter()) {
        let labels: Vec<char> = term.chars().collect();
        validate_labels(&labels)?;
        let rank = labels.len();
        if rank == 0 {
            return Err("Empty label term not allowed".into());
        }
        // Effective shape: take the first `rank` dims from provided shape
        let mut shape = shape4[..rank].to_vec();
        // Basic sanity: non-zero dims
        if shape.iter().any(|&d| d == 0) {
            return Err("Zero dimension encountered".into());
        }
        // Compute row-major strides
        let strides = make_strides(&shape);
        // Size check vs data length (allow extra tail dims in shape4 beyond rank)
        let expected = shape.iter().product::<usize>();
        if expected != data.len() {
            return Err(format!(
                "Data length {} does not match product of effective shape {} for term '{}'",
                data.len(), expected, term
            ));
        }
        inputs.push(TensorView { data, shape, strides, labels });
    }

    // 3) Collect all labels and classify into kept vs reduced
    let mut all_labels: HashSet<char> = HashSet::new();
    for t in &inputs {
        for &c in &t.labels { all_labels.insert(c); }
    }
    let out_set: HashSet<char> = out_labels.iter().copied().collect();
    for &c in &out_labels { if !all_labels.contains(&c) { return Err(format!("Output label '{}' not present in inputs", c)); } }

    let mut red_labels: Vec<char> = Vec::new();
    for &c in &all_labels {
        if !out_set.contains(&c) { red_labels.push(c); }
    }
    red_labels.sort(); // deterministic order for loops

    // 4) Infer output dims from the first occurrence of each output label, check consistency
    let mut out_dims: Vec<usize> = Vec::with_capacity(out_labels.len());
    for &c in &out_labels {
        let mut dim: Option<usize> = None;
        for t in &inputs {
            if let Some(ax) = t.labels.iter().position(|&x| x == c) {
                let d = t.shape[ax];
                match dim {
                    None => dim = Some(d),
                    Some(prev) => {
                        if prev != d && prev != 1 && d != 1 {
                            return Err(format!(
                                "Dimension mismatch for label '{}': {} vs {}",
                                c, prev, d
                            ));
                        }
                        dim = Some(prev.max(d)); // allow size-1 broadcasting
                    }
                }
            }
        }
        out_dims.push(dim.ok_or_else(|| format!("Could not infer dimension for output label '{}'", c))?);
    }

    // 5) Infer reduction dims similarly
    let mut red_dims: Vec<usize> = Vec::with_capacity(red_labels.len());
    for &c in &red_labels {
        let mut dim: Option<usize> = None;
        for t in &inputs {
            if let Some(ax) = t.labels.iter().position(|&x| x == c) {
                let d = t.shape[ax];
                match dim {
                    None => dim = Some(d),
                    Some(prev) => {
                        if prev != d && prev != 1 && d != 1 {
                            return Err(format!(
                                "Dimension mismatch for reduction label '{}': {} vs {}",
                                c, prev, d
                            ));
                        }
                        dim = Some(prev.max(d));
                    }
                }
            }
        }
        red_dims.push(dim.ok_or_else(|| format!("Could not infer dimension for reduction label '{}'", c))?);
    }

    // 6) Build mapping from label -> (is_output, index_in_out_or_red, dim)
    let mut label_to_out_idx: HashMap<char, usize> = HashMap::new();
    for (i, &c) in out_labels.iter().enumerate() { label_to_out_idx.insert(c, i); }
    let mut label_to_red_idx: HashMap<char, usize> = HashMap::new();
    for (i, &c) in red_labels.iter().enumerate() { label_to_red_idx.insert(c, i); }

    // 7) Allocate output
    let out_size = out_dims.iter().product();
    let mut out = vec![0.0f32; out_size];
    let out_strides = make_strides(&out_dims);

    // 8) Iteration helpers
    let mut out_idx = vec![0usize; out_dims.len()];
    loop {
        // Compute flat index for output
        let mut out_flat = 0usize;
        for (i, &v) in out_idx.iter().enumerate() { 
            out_flat += v * out_strides[i]; 
        }

        let mut acc = 0.0f32;
        // reduction loop
        let mut red_idx = vec![0usize; red_dims.len()];
        loop {
            // For each tensor, fetch the element corresponding to current (out_idx, red_idx)
            let mut prod = 1.0f32;
            for t in &inputs {
                let mut tensor_multi: Vec<usize> = vec![0; t.labels.len()];
                for (ax, &lab) in t.labels.iter().enumerate() {
                    if let Some(&oi) = label_to_out_idx.get(&lab) {
                        tensor_multi[ax] = out_idx[oi].min(t.shape[ax]-1); // broadcast if needed
                    } else if let Some(&ri) = label_to_red_idx.get(&lab) {
                        tensor_multi[ax] = red_idx[ri].min(t.shape[ax]-1);
                    } else {
                        return Err(format!("Label '{}' not found in out or red sets", lab));
                    }
                }
                let mut flat = 0usize;
                
                for (i, &v) in tensor_multi.iter().enumerate() { 
                    flat += v * t.strides[i]; 
                }

                prod *= t.data[flat];
            }
            acc += prod;

            if !bump_index(&mut red_idx, &red_dims) { break; }
        }
        out[out_flat] = acc;

        if !bump_index(&mut out_idx, &out_dims) { break; }
    }

    Ok(EinsumResult { data: out, shape: out_dims, labels: out_labels })
}

fn validate_labels(labels: &Vec<char>) -> Result<(), String> {
    if labels.is_empty() { return Err("Empty label list".into()); }
    let mut seen = HashSet::new();
    for &c in labels {
        if !('a'..='z').contains(&c) {
            return Err(format!("Invalid label '{}': only a-z supported", c));
        }
        if !seen.insert(c) {
            return Err(format!("Duplicate label '{}' within a single term not allowed", c));
        }
    }
    Ok(())
}

fn make_strides(shape: &Vec<usize>) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn bump_index(idx: &mut Vec<usize>, dims: &Vec<usize>) -> bool {
    // returns false when it overflows (done)
    for i in (0..idx.len()).rev() {
        idx[i] += 1;
        if idx[i] < dims[i] { return true; }
        idx[i] = 0;
    }
    false
}

// ------------------------- Demo -------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_equivalence() {
        // 'ij,jk->ik' with A: (2,3), B: (3,4)
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let b = vec![
            1.0, 0.0, 2.0, 1.0,
            0.0, 1.0, 0.0, 2.0,
            3.0, 1.0, 1.0, 0.0,
        ];
        let res = einsum("ij,jk->ik", vec![(&a, &[2,3,1,1]), (&b, &[3,4,1,1])]).unwrap();
        assert_eq!(res.shape, vec![2,4]);
        // naive matmul reference
        let mut ref_out = vec![0.0f32; 2*4];
        for i in 0..2 { for k in 0..4 { let mut s=0.0; for j in 0..3 { s += a[i*3+j]*b[j*4+k]; } ref_out[i*4+k]=s; }}
        for (x,y) in res.data.iter().zip(ref_out.iter()) { assert!((x-y).abs()<1e-5); }
    }

    #[test]
    fn bilinear_demo() {
        // 'bn,anm,bm->ba' like your example; small sizes
        let l = vec![
            0.1, -0.2, 0.3, 0.4, 0.5,  // b=0, n=5
            -0.3, 0.2, -0.1, 0.0, 0.7, // b=1, n=5
        ];
        let a = vec![
            // a=0 (5x4)
            0.1, 0.2, 0.3, 0.4,
            0.0, 0.1, 0.0, 0.2,
            -0.3, 0.2, 0.1, -0.1,
            0.1, -0.2, 0.2, 0.3,
            0.2, 0.0, 0.1, -0.2,
            // a=1 (5x4)
            -0.1, 0.2, -0.2, 0.1,
            0.1, 0.0, 0.1, 0.2,
            0.2, -0.2, 0.3, 0.0,
            -0.1, 0.2, -0.3, 0.1,
            0.0, 0.1, 0.0, 0.1,
        ];
        let r = vec![
            0.5, -0.1, 0.0, 0.2,   // b=0, m=4
            -0.3, 0.4, 0.1, 0.0,   // b=1, m=4
        ];
        let res = einsum("bn,anm,bm->ba",
            vec![(&l, &[2,5,1,1]), (&a, &[2,5,4,1]), (&r, &[2,4,1,1])]
        ).unwrap();
        assert_eq!(res.shape, vec![2,2]);
    }
}

// Simple manual demo
fn main() {
    let a = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]; // (2,3)
    let b = vec![
        1.0, 0.0, 2.0, 1.0,
        0.0, 1.0, 0.0, 2.0,
        3.0, 1.0, 1.0, 0.0,
    ]; // (3,4)
    let res = einsum("ij,jk->ik", vec![(&a, &[2,3,1,1]), (&b, &[3,4,1,1])]).unwrap();
    println!("out shape = {:?}", res.shape);
    println!("out data  = {:?}", res.data);
}
