import torch
import torch.nn as nn
from gguf import GGUFWriter
import numpy as np

def add_tensor_safe(writer: GGUFWriter, name: str, t: torch.Tensor) -> None:
    """
    Add a tensor to the GGUF file guaranteeing it always has at least one dim.
    Scalars (ndim == 0) are reshaped to shape (1,).
    """
    # First ensure tensor has at least 1 dimension
    if t.ndim == 0:
        t = t.unsqueeze(0)  # Convert 0-d scalar to 1-d tensor with shape [1]
    
    arr = t.detach().cpu().numpy()
    writer.add_tensor(name, arr)

def test_layer_norm_forward_single_sample():
    """
    PyTorch equivalent of the Rust test_layer_norm_forward_single_sample test.
    Creates the same LayerNorm test and saves tensors to GGUF format.
    """
    print("=== PyTorch LayerNorm Single Sample Test ===")
    
    # Create LayerNorm layer with 4 features (same as Rust test)
    layer_norm = nn.LayerNorm(4, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Test with single sample: [1, 2, 3, 4] (same as Rust test)
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # shape [1, 4]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    
    # Calculate expected values manually to verify
    input_flat = input_tensor.flatten()
    mean = input_flat.mean()
    var = input_flat.var(unbiased=False)  # Use population variance like LayerNorm
    eps = layer_norm.eps
    
    print(f"Manual calculation:")
    print(f"  Mean: {mean}")
    print(f"  Variance: {var}")
    print(f"  Epsilon: {eps}")
    print(f"  Std: {torch.sqrt(var + eps)}")
    
    # Verify mean is approximately 0 for normalized output
    output_mean = output.mean()
    print(f"Output mean: {output_mean} (should be ~0)")
    
    # Check all values are finite
    all_finite = torch.all(torch.isfinite(output))
    print(f"All output values finite: {all_finite}")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_single_sample.gguf",
        "layer_norm_single_sample"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_single_sample_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_single_sample_output", output)
    add_tensor_safe(writer, "layer_norm_single_sample_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_single_sample_bias", layer_norm.bias)
    
    # Save intermediate values for verification
    add_tensor_safe(writer, "layer_norm_single_sample_mean", mean)
    add_tensor_safe(writer, "layer_norm_single_sample_variance", var)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_single_sample.gguf")
    
    return input_tensor, output, layer_norm.weight, layer_norm.bias

def test_layer_norm_forward_zero_variance():
    """
    PyTorch equivalent of the Rust test_layer_norm_forward_zero_variance test.
    Tests LayerNorm with uniform input that has zero variance.
    """
    print("\n=== PyTorch LayerNorm Zero Variance Test ===")
    
    # Create LayerNorm layer with 4 features (same as Rust test)
    layer_norm = nn.LayerNorm(4, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Test with constant values (zero variance case) - same as Rust test
    input_tensor = torch.tensor([[2.0, 2.0, 2.0, 2.0]])  # shape [1, 4]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    print(f"LayerNorm epsilon: {layer_norm.eps}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    
    # Manual calculation to understand what happens
    input_flat = input_tensor.flatten()
    mean = input_flat.mean()  # Should be 2.0
    var = input_flat.var(unbiased=False)  # Should be 0.0 (all values same)
    eps = layer_norm.eps  # Default 1e-5
    std = torch.sqrt(var + eps)  # sqrt(0 + 1e-5) = sqrt(1e-5)
    
    print(f"Manual calculation:")
    print(f"  Mean: {mean}")
    print(f"  Variance: {var}")
    print(f"  Epsilon: {eps}")
    print(f"  Std: {std}")
    print(f"  Centered: {input_flat - mean}")  # Should be [0, 0, 0, 0]
    print(f"  Normalized: {(input_flat - mean) / std}")  # Should be [0, 0, 0, 0]
    
    # Check output properties
    all_finite = torch.all(torch.isfinite(output))
    max_abs_value = torch.max(torch.abs(output))
    
    print(f"All output values finite: {all_finite}")
    print(f"Max absolute output value: {max_abs_value}")
    print(f"Output values very small (< 0.1): {torch.all(torch.abs(output) < 0.1)}")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_zero_variance.gguf",
        "layer_norm_zero_variance"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_zero_variance_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_zero_variance_output", output)
    add_tensor_safe(writer, "layer_norm_zero_variance_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_zero_variance_bias", layer_norm.bias)
    
    # Save intermediate values for verification
    add_tensor_safe(writer, "layer_norm_zero_variance_mean", mean)
    add_tensor_safe(writer, "layer_norm_zero_variance_variance", var)
    add_tensor_safe(writer, "layer_norm_zero_variance_std", std)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_zero_variance.gguf")
    
    return input_tensor, output, layer_norm.weight, layer_norm.bias

def test_layer_norm_backward_zero_variance():
    """
    PyTorch equivalent of the Rust test_layer_norm_backward_zero_variance test.
    Tests gradient behavior when LayerNorm receives uniform input (zero variance).
    """
    print("\n=== PyTorch LayerNorm Backward Zero Variance Test ===")
    
    # Create LayerNorm layer with 3 features (same as Rust test)
    layer_norm = nn.LayerNorm(3, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Input with zero variance (all same values) - same as Rust test [5,5,5]
    input_tensor = torch.tensor([[5.0, 5.0, 5.0]], requires_grad=True)  # shape [1, 3]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input requires_grad: {input_tensor.requires_grad}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    print(f"LayerNorm epsilon: {layer_norm.eps}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    print(f"Output: {output}")
    print(f"Output requires_grad: {output.requires_grad}")
    
    # Create loss (sum like in Rust test)
    loss = output.sum()
    print(f"Loss: {loss}")
    
    # Backward pass
    loss.backward()
    
    # Get gradients
    input_grad = input_tensor.grad
    weight_grad = layer_norm.weight.grad
    bias_grad = layer_norm.bias.grad
    
    print(f"Input gradients: {input_grad}")
    print(f"Weight gradients: {weight_grad}")
    print(f"Bias gradients: {bias_grad}")
    
    # Analyze gradient properties
    input_grad_finite = torch.all(torch.isfinite(input_grad))
    weight_grad_finite = torch.all(torch.isfinite(weight_grad))
    bias_grad_finite = torch.all(torch.isfinite(bias_grad))
    
    input_grad_small = torch.all(torch.abs(input_grad) < 1e-3)
    max_input_grad = torch.max(torch.abs(input_grad))
    max_weight_grad = torch.max(torch.abs(weight_grad))
    max_bias_grad = torch.max(torch.abs(bias_grad))
    
    print(f"\nGradient Analysis:")
    print(f"  Input gradients finite: {input_grad_finite}")
    print(f"  Weight gradients finite: {weight_grad_finite}")
    print(f"  Bias gradients finite: {bias_grad_finite}")
    print(f"  Input gradients small (< 1e-3): {input_grad_small}")
    print(f"  Max |input_grad|: {max_input_grad}")
    print(f"  Max |weight_grad|: {max_weight_grad}")
    print(f"  Max |bias_grad|: {max_bias_grad}")
    
    # Manual gradient calculation to understand what's happening
    print(f"\nManual Analysis:")
    with torch.no_grad():
        # For uniform input, centered = [0, 0, 0]
        # normalized = [0, 0, 0]
        # output = normalized * weight + bias = [0, 0, 0]
        # loss = sum(output) = 0
        # d_loss/d_output = [1, 1, 1]
        # d_output/d_bias = [1, 1, 1] -> bias_grad = [1, 1, 1]
        # d_output/d_weight = normalized = [0, 0, 0] -> weight_grad = [0, 0, 0]
        # d_output/d_input is complex but should be very small due to normalization
        print(f"  Expected bias gradients: [1, 1, 1] (d_output/d_bias)")
        print(f"  Expected weight gradients: [0, 0, 0] (d_output/d_weight = normalized)")
        print(f"  Expected input gradients: very small (normalization effect)")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_backward_zero_variance.gguf",
        "layer_norm_backward_zero_variance"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_backward_zero_variance_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_backward_zero_variance_output", output)
    add_tensor_safe(writer, "layer_norm_backward_zero_variance_loss", loss)
    add_tensor_safe(writer, "layer_norm_backward_zero_variance_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_backward_zero_variance_bias", layer_norm.bias)
    
    # Save gradients
    add_tensor_safe(writer, "layer_norm_backward_zero_variance_input_grad", input_grad)
    add_tensor_safe(writer, "layer_norm_backward_zero_variance_weight_grad", weight_grad)
    add_tensor_safe(writer, "layer_norm_backward_zero_variance_bias_grad", bias_grad)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_backward_zero_variance.gguf")
    
    return input_tensor, output, input_grad, weight_grad, bias_grad

def test_layer_norm_forward_batch():
    """
    PyTorch equivalent of the Rust test_layer_norm_forward_batch test.
    Tests LayerNorm with batch processing (2 samples of 3 features each).
    """
    print("\n=== PyTorch LayerNorm Forward Batch Test ===")
    
    # Create LayerNorm layer with 3 features (same as Rust test)
    layer_norm = nn.LayerNorm(3, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Test with batch of 2 samples - same as Rust test
    # Sample 1: [1, 2, 3], Sample 2: [4, 5, 6]
    input_tensor = torch.tensor([
        [1.0, 2.0, 3.0],  # Sample 1
        [4.0, 5.0, 6.0],  # Sample 2
    ])  # shape [2, 3]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    print(f"LayerNorm epsilon: {layer_norm.eps}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    
    # Manual calculation for each sample to verify
    print(f"\nManual calculation:")
    
    for i, sample in enumerate(input_tensor):
        sample_mean = sample.mean()
        sample_var = sample.var(unbiased=False)  # Population variance like LayerNorm
        eps = layer_norm.eps
        sample_std = torch.sqrt(sample_var + eps)
        
        print(f"  Sample {i+1}: {sample}")
        print(f"    Mean: {sample_mean}")
        print(f"    Variance: {sample_var}")
        print(f"    Std: {sample_std}")
        print(f"    Centered: {sample - sample_mean}")
        print(f"    Normalized: {(sample - sample_mean) / sample_std}")
    
    # Check each sample is normalized independently
    print(f"\nBatch Independence Check:")
    for batch_idx in range(2):
        sample_output = output[batch_idx]
        sample_mean = sample_output.mean()
        print(f"  Sample {batch_idx+1} output mean: {sample_mean} (should be ~0)")
        
        # Check all values are finite
        all_finite = torch.all(torch.isfinite(sample_output))
        print(f"  Sample {batch_idx+1} all finite: {all_finite}")
    
    # Overall checks
    all_finite = torch.all(torch.isfinite(output))
    max_abs_value = torch.max(torch.abs(output))
    
    print(f"\nOverall checks:")
    print(f"  All output values finite: {all_finite}")
    print(f"  Max absolute output value: {max_abs_value}")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_batch.gguf",
        "layer_norm_batch"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_batch_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_batch_output", output)
    add_tensor_safe(writer, "layer_norm_batch_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_batch_bias", layer_norm.bias)
    
    # Save per-sample statistics for verification
    sample1_mean = input_tensor[0].mean()
    sample1_var = input_tensor[0].var(unbiased=False)
    sample2_mean = input_tensor[1].mean()
    sample2_var = input_tensor[1].var(unbiased=False)
    
    add_tensor_safe(writer, "layer_norm_batch_sample1_mean", sample1_mean)
    add_tensor_safe(writer, "layer_norm_batch_sample1_var", sample1_var)
    add_tensor_safe(writer, "layer_norm_batch_sample2_mean", sample2_mean)
    add_tensor_safe(writer, "layer_norm_batch_sample2_var", sample2_var)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_batch.gguf")
    
    return input_tensor, output, layer_norm.weight, layer_norm.bias

def test_layer_norm_backward_simple():
    """
    PyTorch equivalent of the Rust test_layer_norm_backward_simple test.
    Tests basic backward pass functionality with simple input.
    """
    print("\n=== PyTorch LayerNorm Backward Simple Test ===")
    
    # Create LayerNorm layer with 2 features (same as Rust test)
    layer_norm = nn.LayerNorm(2, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Simple input [1, 2] - same as Rust test
    input_tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)  # shape [1, 2]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input requires_grad: {input_tensor.requires_grad}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    print(f"LayerNorm epsilon: {layer_norm.eps}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    print(f"Output: {output}")
    print(f"Output requires_grad: {output.requires_grad}")
    
    # Create loss (sum like in Rust test: sum(vec![0, 1], true))
    loss = output.sum()
    print(f"Loss: {loss}")
    
    # Backward pass
    loss.backward()
    
    # Get gradients
    input_grad = input_tensor.grad
    weight_grad = layer_norm.weight.grad
    bias_grad = layer_norm.bias.grad
    
    print(f"Input gradients: {input_grad}")
    print(f"Weight gradients: {weight_grad}")
    print(f"Bias gradients: {bias_grad}")
    
    # Check that gradients exist and are finite
    input_grad_finite = torch.all(torch.isfinite(input_grad))
    weight_grad_finite = torch.all(torch.isfinite(weight_grad))
    bias_grad_finite = torch.all(torch.isfinite(bias_grad))
    
    print(f"\nGradient Analysis:")
    print(f"  Input gradients finite: {input_grad_finite}")
    print(f"  Weight gradients finite: {weight_grad_finite}")
    print(f"  Bias gradients finite: {bias_grad_finite}")
    
    # Calculate expected gradients manually
    print(f"\nManual Analysis:")
    with torch.no_grad():
        # Input: [1, 2], Mean: 1.5, Centered: [-0.5, 0.5]
        # Variance: 0.25, Std: 0.5, Normalized: [-1, 1]
        # Output: normalized * weight + bias = [-1, 1] * [1, 1] + [0, 0] = [-1, 1]
        # Loss: sum(output) = 0
        # d_loss/d_output = [1, 1]
        # d_output/d_bias = [1, 1] -> bias_grad = [1, 1]
        # d_output/d_weight = normalized = [-1, 1] -> weight_grad = [-1, 1]
        # d_output/d_input is complex but should be non-zero
        print(f"  Expected bias gradients: [1, 1] (d_output/d_bias)")
        print(f"  Expected weight gradients: [-1, 1] (d_output/d_weight = normalized)")
        print(f"  Expected input gradients: non-zero (complex LayerNorm derivative)")
    
    # Check gradient magnitudes
    max_input_grad = torch.max(torch.abs(input_grad))
    max_weight_grad = torch.max(torch.abs(weight_grad))
    max_bias_grad = torch.max(torch.abs(bias_grad))
    
    print(f"  Max |input_grad|: {max_input_grad}")
    print(f"  Max |weight_grad|: {max_weight_grad}")
    print(f"  Max |bias_grad|: {max_bias_grad}")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_backward_simple.gguf",
        "layer_norm_backward_simple"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_backward_simple_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_backward_simple_output", output)
    add_tensor_safe(writer, "layer_norm_backward_simple_loss", loss)
    add_tensor_safe(writer, "layer_norm_backward_simple_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_backward_simple_bias", layer_norm.bias)
    
    # Save gradients
    add_tensor_safe(writer, "layer_norm_backward_simple_input_grad", input_grad)
    add_tensor_safe(writer, "layer_norm_backward_simple_weight_grad", weight_grad)
    add_tensor_safe(writer, "layer_norm_backward_simple_bias_grad", bias_grad)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_backward_simple.gguf")
    
    return input_tensor, output, input_grad, weight_grad, bias_grad

def test_layer_norm_forward_known_values():
    """
    PyTorch equivalent of the Rust test_layer_norm_forward_known_values test.
    Tests simple case [0,1] with manual calculation verification.
    """
    print("\n=== PyTorch LayerNorm Forward Known Values Test ===")
    
    # Create LayerNorm layer with 2 features (same as Rust test)
    layer_norm = nn.LayerNorm(2, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Test with known simple case: [0, 1] - same as Rust test
    input_tensor = torch.tensor([[0.0, 1.0]])  # shape [1, 2]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    print(f"LayerNorm epsilon: {layer_norm.eps}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    
    # Manual calculation verification (same as Rust test)
    print(f"\nManual calculation verification:")
    print(f"  Mean of [0, 1] is 0.5")
    print(f"  Centered: [-0.5, 0.5]")
    print(f"  Variance: (0.25 + 0.25) / 2 = 0.25")
    print(f"  Std: sqrt(0.25 + 1e-5) ≈ 0.5")
    print(f"  Normalized: [-1, 1]")
    
    # Verify the expected output values
    expected_output = torch.tensor([[-1.0, 1.0]])
    print(f"Expected output: {expected_output}")
    print(f"Actual output: {output}")
    print(f"Match: {torch.allclose(output, expected_output, atol=1e-4)}")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_forward_known_values.gguf",
        "layer_norm_forward_known_values"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_forward_known_values_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_forward_known_values_output", output)
    add_tensor_safe(writer, "layer_norm_forward_known_values_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_forward_known_values_bias", layer_norm.bias)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_forward_known_values.gguf")
    
    return input_tensor, output, layer_norm.weight, layer_norm.bias

def test_layer_norm_backward_batch():
    """
    PyTorch equivalent of the Rust test_layer_norm_backward_batch test.
    Tests batch backward pass functionality with gradient computation.
    """
    print("\n=== PyTorch LayerNorm Backward Batch Test ===")
    
    # Create LayerNorm layer with 3 features (same as Rust test)
    layer_norm = nn.LayerNorm(3, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Batch input with 2 samples of 3 features each - same as Rust test
    input_tensor = torch.tensor([
        [1.0, 2.0, 3.0],  # Sample 1
        [4.0, 5.0, 6.0],  # Sample 2
    ], requires_grad=True)  # shape [2, 3]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input requires_grad: {input_tensor.requires_grad}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    print(f"LayerNorm epsilon: {layer_norm.eps}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    print(f"Output requires_grad: {output.requires_grad}")
    
    # Create loss (sum all dimensions like Rust test: sum(vec![0, 1], true))
    loss = output.sum()
    print(f"Loss: {loss}")
    
    # Backward pass
    loss.backward()
    
    # Get gradients
    input_grad = input_tensor.grad
    weight_grad = layer_norm.weight.grad
    bias_grad = layer_norm.bias.grad
    
    print(f"Input gradients: {input_grad}")
    print(f"Weight gradients: {weight_grad}")
    print(f"Bias gradients: {bias_grad}")
    
    # Check that gradients exist and are finite (same as Rust test)
    print(f"\nGradient Analysis:")
    
    # Input gradients - check all batch/feature combinations
    input_grad_finite = torch.all(torch.isfinite(input_grad))
    print(f"  Input gradients finite: {input_grad_finite}")
    
    # Weight and bias gradients - check all features
    weight_grad_finite = torch.all(torch.isfinite(weight_grad))
    bias_grad_finite = torch.all(torch.isfinite(bias_grad))
    print(f"  Weight gradients finite: {weight_grad_finite}")
    print(f"  Bias gradients finite: {bias_grad_finite}")
    
    # Additional analysis
    max_input_grad = torch.max(torch.abs(input_grad))
    max_weight_grad = torch.max(torch.abs(weight_grad))
    max_bias_grad = torch.max(torch.abs(bias_grad))
    
    print(f"  Max |input_grad|: {max_input_grad}")
    print(f"  Max |weight_grad|: {max_weight_grad}")
    print(f"  Max |bias_grad|: {max_bias_grad}")
    
    # Manual analysis of expected gradients
    print(f"\nManual Analysis:")
    print(f"  Expected bias gradients: [2, 2, 2] (one per feature, sum over batch)")
    print(f"  Expected weight gradients: normalized outputs summed over batch")
    print(f"  Expected input gradients: complex LayerNorm derivative per sample")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_backward_batch.gguf",
        "layer_norm_backward_batch"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_backward_batch_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_backward_batch_output", output)
    add_tensor_safe(writer, "layer_norm_backward_batch_loss", loss)
    add_tensor_safe(writer, "layer_norm_backward_batch_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_backward_batch_bias", layer_norm.bias)
    
    # Save gradients
    add_tensor_safe(writer, "layer_norm_backward_batch_input_grad", input_grad)
    add_tensor_safe(writer, "layer_norm_backward_batch_weight_grad", weight_grad)
    add_tensor_safe(writer, "layer_norm_backward_batch_bias_grad", bias_grad)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_backward_batch.gguf")
    
    return input_tensor, output, input_grad, weight_grad, bias_grad

def test_layer_norm_forward_large_values():
    """
    PyTorch equivalent of the Rust test_layer_norm_forward_large_values test.
    Tests numerical stability with large input values [100, 200, 300].
    """
    print("\n=== PyTorch LayerNorm Forward Large Values Test ===")
    
    # Create LayerNorm layer with 3 features (same as Rust test)
    layer_norm = nn.LayerNorm(3, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Test with large values to check numerical stability - same as Rust test
    input_tensor = torch.tensor([[100.0, 200.0, 300.0]])  # shape [1, 3]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    print(f"LayerNorm epsilon: {layer_norm.eps}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    
    # Manual calculation for verification
    print(f"\nManual calculation:")
    input_flat = input_tensor.flatten()
    mean = input_flat.mean()
    var = input_flat.var(unbiased=False)  # Population variance like LayerNorm
    eps = layer_norm.eps
    std = torch.sqrt(var + eps)
    
    print(f"  Mean: {mean}")
    print(f"  Variance: {var}")
    print(f"  Std: {std}")
    print(f"  Centered: {input_flat - mean}")
    print(f"  Normalized: {(input_flat - mean) / std}")
    
    # Check normalization worked (mean should be ~0)
    output_mean = output.mean()
    print(f"  Output mean: {output_mean} (should be ~0)")
    
    # All values should be finite and reasonable
    all_finite = torch.all(torch.isfinite(output))
    max_abs_value = torch.max(torch.abs(output))
    all_reasonable = torch.all(torch.abs(output) < 10.0)
    
    print(f"  All output values finite: {all_finite}")
    print(f"  Max absolute output value: {max_abs_value}")
    print(f"  All values reasonable (< 10.0): {all_reasonable}")
    
    # Test numerical stability - large inputs should still produce normalized outputs
    print(f"\nNumerical Stability Analysis:")
    print(f"  Input range: [{input_tensor.min()}, {input_tensor.max()}]")
    print(f"  Output range: [{output.min()}, {output.max()}]")
    print(f"  Input magnitude: {torch.norm(input_tensor)}")
    print(f"  Output magnitude: {torch.norm(output)}")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_forward_large_values.gguf",
        "layer_norm_forward_large_values"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_forward_large_values_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_forward_large_values_output", output)
    add_tensor_safe(writer, "layer_norm_forward_large_values_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_forward_large_values_bias", layer_norm.bias)
    
    # Save intermediate values for verification
    add_tensor_safe(writer, "layer_norm_forward_large_values_mean", mean)
    add_tensor_safe(writer, "layer_norm_forward_large_values_variance", var)
    add_tensor_safe(writer, "layer_norm_forward_large_values_std", std)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_forward_large_values.gguf")
    
    return input_tensor, output, layer_norm.weight, layer_norm.bias

def test_layer_norm_backward_numerical_stability():
    """
    PyTorch equivalent of the Rust test_layer_norm_backward_numerical_stability test.
    Tests gradient stability with large input values [1000, 2000, 3000, 4000].
    """
    print("\n=== PyTorch LayerNorm Backward Numerical Stability Test ===")
    
    # Create LayerNorm layer with 4 features (same as Rust test)
    layer_norm = nn.LayerNorm(4, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Test with large values (potential numerical issues) - same as Rust test
    input_tensor = torch.tensor([[1000.0, 2000.0, 3000.0, 4000.0]], requires_grad=True)  # shape [1, 4]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input requires_grad: {input_tensor.requires_grad}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    print(f"LayerNorm epsilon: {layer_norm.eps}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    print(f"Output requires_grad: {output.requires_grad}")
    
    # Create loss (sum all dimensions like Rust test: sum(vec![0, 1], true))
    loss = output.sum()
    print(f"Loss: {loss}")
    
    # Backward pass
    loss.backward()
    
    # Get gradients
    input_grad = input_tensor.grad
    weight_grad = layer_norm.weight.grad
    bias_grad = layer_norm.bias.grad
    
    print(f"Input gradients: {input_grad}")
    print(f"Weight gradients: {weight_grad}")
    print(f"Bias gradients: {bias_grad}")
    
    # Check all gradients are finite (no NaN/Inf) - same as Rust test
    print(f"\nNumerical Stability Analysis:")
    
    # Input gradients
    input_grad_finite = torch.all(torch.isfinite(input_grad))
    print(f"  Input gradients finite: {input_grad_finite}")
    
    # Weight and bias gradients
    weight_grad_finite = torch.all(torch.isfinite(weight_grad))
    bias_grad_finite = torch.all(torch.isfinite(bias_grad))
    print(f"  Weight gradients finite: {weight_grad_finite}")
    print(f"  Bias gradients finite: {bias_grad_finite}")
    
    # Additional analysis for numerical stability
    max_input_grad = torch.max(torch.abs(input_grad))
    max_weight_grad = torch.max(torch.abs(weight_grad))
    max_bias_grad = torch.max(torch.abs(bias_grad))
    
    print(f"  Max |input_grad|: {max_input_grad}")
    print(f"  Max |weight_grad|: {max_weight_grad}")
    print(f"  Max |bias_grad|: {max_bias_grad}")
    
    # Test that gradients are reasonable despite large inputs
    input_grad_reasonable = torch.all(torch.abs(input_grad) < 1e10)
    weight_grad_reasonable = torch.all(torch.abs(weight_grad) < 1e10)
    bias_grad_reasonable = torch.all(torch.abs(bias_grad) < 1e10)
    
    print(f"  Input gradients reasonable (< 1e10): {input_grad_reasonable}")
    print(f"  Weight gradients reasonable (< 1e10): {weight_grad_reasonable}")
    print(f"  Bias gradients reasonable (< 1e10): {bias_grad_reasonable}")
    
    # Manual analysis of gradient stability
    print(f"\nGradient Stability Analysis:")
    print(f"  Input range: [{input_tensor.min()}, {input_tensor.max()}]")
    print(f"  Output range: [{output.min()}, {output.max()}]")
    print(f"  Input gradient range: [{input_grad.min()}, {input_grad.max()}]")
    print(f"  Weight gradient range: [{weight_grad.min()}, {weight_grad.max()}]")
    print(f"  Bias gradient range: [{bias_grad.min()}, {bias_grad.max()}]")
    
    # Expected behavior: despite large inputs, gradients should remain finite and reasonable
    print(f"\nExpected Behavior:")
    print(f"  Despite large inputs (1000-4000), all gradients should be finite")
    print(f"  LayerNorm normalization should prevent gradient explosion")
    print(f"  Bias gradients should be simple (just 1's from sum)")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_backward_numerical_stability.gguf",
        "layer_norm_backward_numerical_stability"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_backward_numerical_stability_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_backward_numerical_stability_output", output)
    add_tensor_safe(writer, "layer_norm_backward_numerical_stability_loss", loss)
    add_tensor_safe(writer, "layer_norm_backward_numerical_stability_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_backward_numerical_stability_bias", layer_norm.bias)
    
    # Save gradients
    add_tensor_safe(writer, "layer_norm_backward_numerical_stability_input_grad", input_grad)
    add_tensor_safe(writer, "layer_norm_backward_numerical_stability_weight_grad", weight_grad)
    add_tensor_safe(writer, "layer_norm_backward_numerical_stability_bias_grad", bias_grad)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_backward_numerical_stability.gguf")
    
    return input_tensor, output, input_grad, weight_grad, bias_grad

def test_layer_norm_forward_preserves_batch_independence():
    """
    PyTorch equivalent of the Rust test_layer_norm_forward_preserves_batch_independence test.
    Tests that LayerNorm processes each sample in a batch independently.
    """
    print("\n=== PyTorch LayerNorm Forward Preserves Batch Independence Test ===")
    
    # Create LayerNorm layer with 2 features (same as Rust test)
    layer_norm = nn.LayerNorm(2, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Create batch where each sample has different scale - same as Rust test
    input_tensor = torch.tensor([
        [1.0, 2.0],      # Sample 1: small values
        [10.0, 20.0],    # Sample 2: large values (10x scale)
    ])  # shape [2, 2]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    print(f"LayerNorm epsilon: {layer_norm.eps}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    
    # Manual calculation for verification of batch independence
    print(f"\nManual calculation verification:")
    
    # Sample 1: [1, 2] -> mean=1.5, centered=[-0.5, 0.5], normalized=[-1, 1]
    sample1 = input_tensor[0]
    sample1_mean = sample1.mean()
    sample1_var = sample1.var(unbiased=False)
    sample1_std = torch.sqrt(sample1_var + layer_norm.eps)
    sample1_normalized = (sample1 - sample1_mean) / sample1_std
    
    print(f"  Sample 1 [{sample1[0]}, {sample1[1]}]:")
    print(f"    Mean: {sample1_mean}")
    print(f"    Variance: {sample1_var}")
    print(f"    Std: {sample1_std}")
    print(f"    Centered: {sample1 - sample1_mean}")
    print(f"    Normalized: {sample1_normalized}")
    print(f"    Expected: [-1, 1]")
    
    # Sample 2: [10, 20] -> mean=15, centered=[-5, 5], normalized=[-1, 1]
    sample2 = input_tensor[1]
    sample2_mean = sample2.mean()
    sample2_var = sample2.var(unbiased=False)
    sample2_std = torch.sqrt(sample2_var + layer_norm.eps)
    sample2_normalized = (sample2 - sample2_mean) / sample2_std
    
    print(f"  Sample 2 [{sample2[0]}, {sample2[1]}]:")
    print(f"    Mean: {sample2_mean}")
    print(f"    Variance: {sample2_var}")
    print(f"    Std: {sample2_std}")
    print(f"    Centered: {sample2 - sample2_mean}")
    print(f"    Normalized: {sample2_normalized}")
    print(f"    Expected: [-1, 1]")
    
    # Verify batch independence - both samples should normalize to [-1, 1]
    print(f"\nBatch Independence Verification:")
    print(f"  Sample 1 output: [{output[0, 0]:.6f}, {output[0, 1]:.6f}]")
    print(f"  Sample 2 output: [{output[1, 0]:.6f}, {output[1, 1]:.6f}]")
    
    # Check that both samples normalize to approximately [-1, 1]
    sample1_match = torch.allclose(output[0], torch.tensor([-1.0, 1.0]), atol=1e-4)
    sample2_match = torch.allclose(output[1], torch.tensor([-1.0, 1.0]), atol=1e-4)
    
    print(f"  Sample 1 matches [-1, 1]: {sample1_match}")
    print(f"  Sample 2 matches [-1, 1]: {sample2_match}")
    
    # Verify that despite different input scales, both samples are normalized identically
    print(f"\nKey Insight: Despite 10x scale difference in inputs,")
    print(f"both samples are normalized to the same range [-1, 1]")
    print(f"This demonstrates batch independence - each sample processed separately")
    
    # Additional verification
    all_finite = torch.all(torch.isfinite(output))
    print(f"\nAll output values finite: {all_finite}")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_forward_preserves_batch_independence.gguf",
        "layer_norm_forward_preserves_batch_independence"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_forward_preserves_batch_independence_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_forward_preserves_batch_independence_output", output)
    add_tensor_safe(writer, "layer_norm_forward_preserves_batch_independence_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_forward_preserves_batch_independence_bias", layer_norm.bias)
    
    # Save per-sample statistics for verification
    add_tensor_safe(writer, "layer_norm_forward_preserves_batch_independence_sample1_mean", sample1_mean)
    add_tensor_safe(writer, "layer_norm_forward_preserves_batch_independence_sample1_var", sample1_var)
    add_tensor_safe(writer, "layer_norm_forward_preserves_batch_independence_sample1_std", sample1_std)
    add_tensor_safe(writer, "layer_norm_forward_preserves_batch_independence_sample2_mean", sample2_mean)
    add_tensor_safe(writer, "layer_norm_forward_preserves_batch_independence_sample2_var", sample2_var)
    add_tensor_safe(writer, "layer_norm_forward_preserves_batch_independence_sample2_std", sample2_std)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_forward_preserves_batch_independence.gguf")
    
    return input_tensor, output, layer_norm.weight, layer_norm.bias

def test_layer_norm_forward_negative_values():
    """
    PyTorch equivalent of the Rust test_layer_norm_forward_negative_values test.
    Tests LayerNorm with negative input values [-2, -1, 0].
    """
    print("\n=== PyTorch LayerNorm Forward Negative Values Test ===")
    
    # Create LayerNorm layer with 3 features (same as Rust test)
    layer_norm = nn.LayerNorm(3, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Test with negative values - same as Rust test
    input_tensor = torch.tensor([[-2.0, -1.0, 0.0]])  # shape [1, 3]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    print(f"LayerNorm epsilon: {layer_norm.eps}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    
    # Manual calculation for verification
    print(f"\nManual calculation verification:")
    input_flat = input_tensor.flatten()
    mean = input_flat.mean()
    var = input_flat.var(unbiased=False)  # Population variance like LayerNorm
    eps = layer_norm.eps
    std = torch.sqrt(var + eps)
    
    print(f"  Input: {input_flat}")
    print(f"  Mean: {mean}")
    print(f"  Variance: {var}")
    print(f"  Std: {std}")
    print(f"  Centered: {input_flat - mean}")
    print(f"  Normalized: {(input_flat - mean) / std}")
    
    # Expected calculation for [-2, -1, 0]:
    # Mean = (-2 + -1 + 0) / 3 = -1
    # Centered = [-2 - (-1), -1 - (-1), 0 - (-1)] = [-1, 0, 1]
    # Variance = ((-1)^2 + 0^2 + 1^2) / 3 = 2/3 ≈ 0.667
    # Std = sqrt(2/3 + eps) ≈ sqrt(2/3) ≈ 0.816
    # Normalized = [-1/0.816, 0/0.816, 1/0.816] ≈ [-1.225, 0, 1.225]
    
    print(f"\nExpected calculation:")
    print(f"  Mean should be -1.0")
    print(f"  Centered should be [-1, 0, 1]")
    print(f"  Variance should be ~0.667")
    print(f"  Normalized should be ~[-1.225, 0, 1.225]")
    
    # Check normalization worked (mean should be ~0)
    output_mean = output.mean()
    print(f"  Output mean: {output_mean} (should be ~0)")
    
    # All values should be finite
    all_finite = torch.all(torch.isfinite(output))
    print(f"  All output values finite: {all_finite}")
    
    # Check that LayerNorm handles negative values correctly
    print(f"\nNegative Value Handling:")
    print(f"  Input contains negative values: {torch.any(input_tensor < 0)}")
    print(f"  Output range: [{output.min()}, {output.max()}]")
    print(f"  LayerNorm should normalize regardless of input sign")
    
    # Verify expected behavior: despite negative inputs, normalization should work
    print(f"\nKey Insight: LayerNorm handles negative values correctly,")
    print(f"normalizing them the same way as positive values.")
    print(f"The sign doesn't affect the normalization process.")
    
    # Additional checks
    output_std = output.std()
    print(f"  Output standard deviation: {output_std} (should be ~1)")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_forward_negative_values.gguf",
        "layer_norm_forward_negative_values"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_forward_negative_values_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_forward_negative_values_output", output)
    add_tensor_safe(writer, "layer_norm_forward_negative_values_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_forward_negative_values_bias", layer_norm.bias)
    
    # Save intermediate values for verification
    add_tensor_safe(writer, "layer_norm_forward_negative_values_mean", mean)
    add_tensor_safe(writer, "layer_norm_forward_negative_values_variance", var)
    add_tensor_safe(writer, "layer_norm_forward_negative_values_std", std)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_forward_negative_values.gguf")
    
    return input_tensor, output, layer_norm.weight, layer_norm.bias

def test_layer_norm_forward_different_feature_sizes():
    """
    PyTorch equivalent of the Rust test_layer_norm_forward_different_feature_sizes test.
    Tests LayerNorm with different feature dimensions: 1, 2, 5, 10.
    """
    print("=== PyTorch LayerNorm Different Feature Sizes Test ===")
    
    # Test with different feature dimensions (same as Rust test)
    feature_sizes = [1, 2, 5, 10]
    
    for num_features in feature_sizes:
        print(f"\nTesting with {num_features} features:")
        
        # Create LayerNorm layer with current feature size
        layer_norm = nn.LayerNorm(num_features, elementwise_affine=True)
        
        # Initialize weights to ones and bias to zeros (same as Rust implementation)
        with torch.no_grad():
            layer_norm.weight.fill_(1.0)
            layer_norm.bias.fill_(0.0)
        
        # Create input with values 1 to num_features (same as Rust test)
        input_data = list(range(1, num_features + 1))  # [1, 2, 3, ..., num_features]
        input_tensor = torch.tensor([input_data], dtype=torch.float32)  # shape [1, num_features]
        
        print(f"  Input: {input_tensor}")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  LayerNorm weight: {layer_norm.weight}")
        print(f"  LayerNorm bias: {layer_norm.bias}")
        
        # Forward pass
        output = layer_norm(input_tensor)
        
        print(f"  Output: {output}")
        print(f"  Output shape: {output.shape}")
        
        # Check all outputs are finite
        all_finite = torch.all(torch.isfinite(output))
        print(f"  All output values finite: {all_finite}")
        
        # Check mean is approximately zero (if more than 1 feature)
        if num_features > 1:
            output_mean = output.mean()
            print(f"  Output mean: {output_mean} (should be ~0)")
            
            # Calculate expected values manually for verification
            input_flat = input_tensor.flatten()
            mean = input_flat.mean()
            var = input_flat.var(unbiased=False)  # Use population variance like LayerNorm
            eps = layer_norm.eps
            
            print(f"  Manual calculation:")
            print(f"    Mean: {mean}")
            print(f"    Variance: {var}")
            print(f"    Std: {torch.sqrt(var + eps)}")
    
    # Save tensors to GGUF file for the largest test case (10 features)
    # This provides the most comprehensive test data
    test_features = 10
    layer_norm = nn.LayerNorm(test_features, elementwise_affine=True)
    
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    input_data = list(range(1, test_features + 1))  # [1, 2, 3, ..., 10]
    input_tensor = torch.tensor([input_data], dtype=torch.float32)  # shape [1, 10]
    output = layer_norm(input_tensor)
    
    # Calculate intermediate values for verification
    input_flat = input_tensor.flatten()
    mean = input_flat.mean()
    var = input_flat.var(unbiased=False)
    std = torch.sqrt(var + layer_norm.eps)
    
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_forward_different_feature_sizes.gguf",
        "layer_norm_forward_different_feature_sizes"
    )
    
    # Save input, output, and parameters
    add_tensor_safe(writer, "layer_norm_forward_different_feature_sizes_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_forward_different_feature_sizes_output", output)
    add_tensor_safe(writer, "layer_norm_forward_different_feature_sizes_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_forward_different_feature_sizes_bias", layer_norm.bias)
    
    # Save intermediate values for verification
    add_tensor_safe(writer, "layer_norm_forward_different_feature_sizes_mean", mean)
    add_tensor_safe(writer, "layer_norm_forward_different_feature_sizes_variance", var)
    add_tensor_safe(writer, "layer_norm_forward_different_feature_sizes_std", std)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print(f"\nGGUF file saved to: ./models/tests/layer_norm/layer_norm_forward_different_feature_sizes.gguf")
    print(f"Test data based on {test_features} features case")
    
    return input_tensor, output, layer_norm.weight, layer_norm.bias

def test_layer_norm_forward_multiple_calls():
    """
    PyTorch equivalent of the Rust test_layer_norm_forward_multiple_calls test.
    Tests that multiple calls with the same input give the same output (deterministic behavior).
    """
    print("=== PyTorch LayerNorm Multiple Calls Test ===")
    
    # Create LayerNorm layer with 3 features (same as Rust test)
    layer_norm = nn.LayerNorm(3, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Test input: [1, 2, 3] (same as Rust test)
    input_tensor = torch.tensor([[1.0, 2.0, 3.0]])  # shape [1, 3]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    
    # Forward pass - call twice with same input
    output1 = layer_norm(input_tensor.clone())
    output2 = layer_norm(input_tensor.clone())
    
    print(f"Output1: {output1}")
    print(f"Output2: {output2}")
    print(f"Output shape: {output1.shape}")
    
    # Check that outputs are identical
    outputs_equal = torch.allclose(output1, output2, atol=1e-10)
    print(f"Outputs identical: {outputs_equal}")
    
    # Calculate difference to verify deterministic behavior
    diff = torch.abs(output1 - output2)
    max_diff = torch.max(diff)
    print(f"Max difference: {max_diff}")
    
    # Check all outputs are finite
    all_finite1 = torch.all(torch.isfinite(output1))
    all_finite2 = torch.all(torch.isfinite(output2))
    print(f"All output1 values finite: {all_finite1}")
    print(f"All output2 values finite: {all_finite2}")
    
    # Calculate expected values manually for verification
    input_flat = input_tensor.flatten()
    mean = input_flat.mean()
    var = input_flat.var(unbiased=False)  # Use population variance like LayerNorm
    eps = layer_norm.eps
    std = torch.sqrt(var + eps)
    
    print(f"Manual calculation:")
    print(f"  Mean: {mean}")
    print(f"  Variance: {var}")
    print(f"  Std: {std}")
    print(f"  Expected normalized: {(input_flat - mean) / std}")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_forward_multiple_calls.gguf",
        "layer_norm_forward_multiple_calls"
    )
    
    # Save input, both outputs, and parameters
    add_tensor_safe(writer, "layer_norm_forward_multiple_calls_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_forward_multiple_calls_output1", output1)
    add_tensor_safe(writer, "layer_norm_forward_multiple_calls_output2", output2)
    add_tensor_safe(writer, "layer_norm_forward_multiple_calls_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_forward_multiple_calls_bias", layer_norm.bias)
    
    # Save intermediate values for verification
    add_tensor_safe(writer, "layer_norm_forward_multiple_calls_mean", mean)
    add_tensor_safe(writer, "layer_norm_forward_multiple_calls_variance", var)
    add_tensor_safe(writer, "layer_norm_forward_multiple_calls_std", std)
    
    # Save difference tensor to verify deterministic behavior
    add_tensor_safe(writer, "layer_norm_forward_multiple_calls_diff", diff)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_forward_multiple_calls.gguf")
    
    return input_tensor, output1, output2, layer_norm.weight, layer_norm.bias

def test_layer_norm_backward_gradient_flow():
    """
    PyTorch equivalent of the Rust test_layer_norm_backward_gradient_flow test.
    Tests gradient flow through LayerNorm with different inputs to verify sensitivity.
    """
    print("=== PyTorch LayerNorm Backward Gradient Flow Test ===")
    
    # Create LayerNorm layer with 2 features (same as Rust test)
    layer_norm = nn.LayerNorm(2, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Test inputs: [1, 2] and [1.1, 2.1] (same as Rust test)
    input1 = torch.tensor([[1.0, 2.0]], requires_grad=True)  # shape [1, 2]
    input2 = torch.tensor([[1.1, 2.1]], requires_grad=True)  # shape [1, 2]
    
    print(f"Input1: {input1}")
    print(f"Input2: {input2}")
    print(f"Input1 shape: {input1.shape}")
    print(f"Input2 shape: {input2.shape}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    
    # Forward pass for input1
    output1 = layer_norm(input1)
    loss1 = output1.sum()  # Sum all elements (equivalent to sum([0, 1], true))
    
    # Forward pass for input2
    output2 = layer_norm(input2)
    loss2 = output2.sum()  # Sum all elements
    
    print(f"Output1: {output1}")
    print(f"Output2: {output2}")
    print(f"Loss1: {loss1}")
    print(f"Loss2: {loss2}")
    
    # Test first input - compute gradients
    layer_norm.zero_grad()
    if input1.grad is not None:
        input1.grad.zero_()
    
    loss1.backward(retain_graph=True)
    
    input1_grad = input1.grad.clone()
    weight_grad1 = layer_norm.weight.grad.clone()
    bias_grad1 = layer_norm.bias.grad.clone()
    
    print(f"Input1 gradients: {input1_grad}")
    print(f"Weight gradients 1: {weight_grad1}")
    print(f"Bias gradients 1: {bias_grad1}")
    
    # Test second input - compute gradients
    layer_norm.zero_grad()
    if input2.grad is not None:
        input2.grad.zero_()
    
    loss2.backward(retain_graph=True)
    
    input2_grad = input2.grad.clone()
    weight_grad2 = layer_norm.weight.grad.clone()
    bias_grad2 = layer_norm.bias.grad.clone()
    
    print(f"Input2 gradients: {input2_grad}")
    print(f"Weight gradients 2: {weight_grad2}")
    print(f"Bias gradients 2: {bias_grad2}")
    
    # Gradients should be different for different inputs
    grad_diff = torch.abs(input1_grad - input2_grad)
    max_grad_diff = torch.max(grad_diff)
    print(f"Max gradient difference: {max_grad_diff}")
    
    # Check that gradients are different (showing sensitivity)
    gradient_sensitivity = max_grad_diff > 1e-6
    print(f"Gradients differ for different inputs: {gradient_sensitivity}")
    
    # All gradients should be finite
    input1_finite = torch.all(torch.isfinite(input1_grad))
    input2_finite = torch.all(torch.isfinite(input2_grad))
    weight1_finite = torch.all(torch.isfinite(weight_grad1))
    weight2_finite = torch.all(torch.isfinite(weight_grad2))
    bias1_finite = torch.all(torch.isfinite(bias_grad1))
    bias2_finite = torch.all(torch.isfinite(bias_grad2))
    
    print(f"All gradients finite:")
    print(f"  Input1: {input1_finite}")
    print(f"  Input2: {input2_finite}")
    print(f"  Weight1: {weight1_finite}")
    print(f"  Weight2: {weight2_finite}")
    print(f"  Bias1: {bias1_finite}")
    print(f"  Bias2: {bias2_finite}")
    
    # Calculate expected gradient values manually for verification
    print(f"\nManual gradient analysis:")
    print(f"Input1 LayerNorm forward: mean={(input1.mean()):.6f}, std={(input1.std()):.6f}")
    print(f"Input2 LayerNorm forward: mean={(input2.mean()):.6f}, std={(input2.std()):.6f}")
    print(f"Different inputs should produce different gradients due to LayerNorm sensitivity")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_backward_gradient_flow.gguf",
        "layer_norm_backward_gradient_flow"
    )
    
    # Save inputs, outputs, and losses
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_input1", input1)
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_input2", input2)
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_output1", output1)
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_output2", output2)
    
    # Save losses as 1D tensors to match Rust sum() output format
    loss1_1d = torch.tensor([loss1.item()], dtype=torch.float32)
    loss2_1d = torch.tensor([loss2.item()], dtype=torch.float32)
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_loss1", loss1_1d)
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_loss2", loss2_1d)
    
    # Save parameters
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_bias", layer_norm.bias)
    
    # Save gradients - ensure correct shapes
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_input1_grad", input1_grad)
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_input2_grad", input2_grad)
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_weight_grad1", weight_grad1)
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_weight_grad2", weight_grad2)
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_bias_grad1", bias_grad1)
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_bias_grad2", bias_grad2)
    
    # Debug: Print shapes for troubleshooting
    print(f"DEBUG: input1_grad shape: {input1_grad.shape}")
    print(f"DEBUG: input2_grad shape: {input2_grad.shape}")
    print(f"DEBUG: weight_grad1 shape: {weight_grad1.shape}")
    print(f"DEBUG: weight_grad2 shape: {weight_grad2.shape}")
    print(f"DEBUG: bias_grad1 shape: {bias_grad1.shape}")
    print(f"DEBUG: bias_grad2 shape: {bias_grad2.shape}")
    
    # Save gradient difference for verification
    add_tensor_safe(writer, "layer_norm_backward_gradient_flow_grad_diff", grad_diff)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_backward_gradient_flow.gguf")
    
    return input1, input2, output1, output2, input1_grad, input2_grad, weight_grad1, weight_grad2

def test_layer_norm_backward_parameter_gradients():
    """
    PyTorch equivalent of the Rust test_layer_norm_backward_parameter_gradients test.
    Tests parameter gradient computation for LayerNorm weight and bias.
    """
    print("=== PyTorch LayerNorm Backward Parameter Gradients Test ===")
    
    # Create LayerNorm layer with 3 features (same as Rust test)
    layer_norm = nn.LayerNorm(3, elementwise_affine=True)
    
    # Initialize weights to ones and bias to zeros (same as Rust implementation)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Input that will produce non-zero normalized values: [1, 4, 7] (same as Rust test)
    input_tensor = torch.tensor([[1.0, 4.0, 7.0]], requires_grad=True)  # shape [1, 3]
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"LayerNorm weight: {layer_norm.weight}")
    print(f"LayerNorm bias: {layer_norm.bias}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    loss = output.sum()  # Sum all elements (equivalent to sum([0, 1], true))
    
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss}")
    
    # Zero gradients and compute gradients
    layer_norm.zero_grad()
    if input_tensor.grad is not None:
        input_tensor.grad.zero_()
    
    loss.backward()
    
    weight_grad = layer_norm.weight.grad.clone()
    bias_grad = layer_norm.bias.grad.clone()
    input_grad = input_tensor.grad.clone()
    
    print(f"Weight gradients: {weight_grad}")
    print(f"Bias gradients: {bias_grad}")
    print(f"Input gradients: {input_grad}")
    
    # Verify gradients are finite and non-zero
    weight_finite = torch.all(torch.isfinite(weight_grad))
    bias_finite = torch.all(torch.isfinite(bias_grad))
    input_finite = torch.all(torch.isfinite(input_grad))
    
    print(f"All gradients finite:")
    print(f"  Weight: {weight_finite}")
    print(f"  Bias: {bias_finite}")
    print(f"  Input: {input_finite}")
    
    # Check if gradients are non-zero
    weight_nonzero = torch.all(torch.abs(weight_grad) > 1e-8)
    bias_nonzero = torch.all(torch.abs(bias_grad) > 1e-8)
    
    print(f"All gradients non-zero:")
    print(f"  Weight: {weight_nonzero}")
    print(f"  Bias: {bias_nonzero}")
    
    # Manual analysis of expected gradients
    print(f"\nManual gradient analysis:")
    input_flat = input_tensor.flatten()
    mean = input_flat.mean()
    var = input_flat.var(unbiased=False)
    std = torch.sqrt(var + layer_norm.eps)
    normalized = (input_flat - mean) / std
    
    print(f"  Mean: {mean}")
    print(f"  Variance: {var}")
    print(f"  Std: {std}")
    print(f"  Normalized: {normalized}")
    print(f"  Expected bias gradients: {normalized} (d/d_bias = normalized output)")
    print(f"  Expected weight gradients: {normalized * normalized} (d/d_weight = normalized * normalized)")
    
    # Verify the mathematical relationship
    print(f"\nGradient relationships:")
    print(f"  Bias grad should equal normalized output: {torch.allclose(bias_grad, normalized, atol=1e-5)}")
    print(f"  Weight grad should equal normalized * normalized: {torch.allclose(weight_grad, normalized * normalized, atol=1e-5)}")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_backward_parameter_gradients.gguf",
        "layer_norm_backward_parameter_gradients"
    )
    
    # Save input, output, and loss
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_output", output)
    
    # Save loss as 1D tensor to match Rust sum() output format
    loss_1d = torch.tensor([loss.item()], dtype=torch.float32)
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_loss", loss_1d)
    
    # Save parameters
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_bias", layer_norm.bias)
    
    # Save gradients
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_input_grad", input_grad)
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_weight_grad", weight_grad)
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_bias_grad", bias_grad)
    
    # Save intermediate values for verification
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_mean", torch.tensor([mean.item()]))
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_variance", torch.tensor([var.item()]))
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_std", torch.tensor([std.item()]))
    add_tensor_safe(writer, "layer_norm_backward_parameter_gradients_normalized", normalized)
    
    # Debug: Print shapes for troubleshooting
    print(f"DEBUG: normalized shape: {normalized.shape}")
    print(f"DEBUG: input_grad shape: {input_grad.shape}")
    print(f"DEBUG: weight_grad shape: {weight_grad.shape}")
    print(f"DEBUG: bias_grad shape: {bias_grad.shape}")
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_backward_parameter_gradients.gguf")
    
    return input_tensor, output, weight_grad, bias_grad, normalized

def test_layer_norm_backward_in_network():
    """
    PyTorch equivalent of the Rust test_layer_norm_backward_in_network test.
    Tests LayerNorm as part of a network: Linear -> LayerNorm.
    """
    print("\n=== PyTorch LayerNorm Backward In Network Test ===")
    
    # Create Linear layer (2 inputs, 3 outputs) + LayerNorm (3 features)
    linear = nn.Linear(2, 3, bias=True)
    layer_norm = nn.LayerNorm(3, elementwise_affine=True)
    
    # Initialize LayerNorm parameters to ones and zeros (like Rust)
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.fill_(0.0)
    
    # Enable gradients for all parameters
    linear.weight.requires_grad = True
    linear.bias.requires_grad = True
    layer_norm.weight.requires_grad = True
    layer_norm.bias.requires_grad = True
    
    # Input [1, 2] shape [1, 2] - same as Rust test
    input_tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)
    
    print(f"Input: {input_tensor}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Linear weight shape: {linear.weight.shape}")
    print(f"Linear bias shape: {linear.bias.shape}")
    print(f"LayerNorm weight shape: {layer_norm.weight.shape}")
    print(f"LayerNorm bias shape: {layer_norm.bias.shape}")
    
    # Forward pass: input -> linear -> layer_norm
    hidden = linear(input_tensor)
    output = layer_norm(hidden)
    loss = output.sum()
    
    print(f"Hidden (after Linear): {hidden}")
    print(f"Output (after LayerNorm): {output}")
    print(f"Loss: {loss}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"\nGradients:")
    print(f"Input grad: {input_tensor.grad}")
    print(f"Linear weight grad: {linear.weight.grad}")
    print(f"Linear bias grad: {linear.bias.grad}")
    print(f"LayerNorm weight grad: {layer_norm.weight.grad}")
    print(f"LayerNorm bias grad: {layer_norm.bias.grad}")
    
    # Verify gradients are finite
    all_finite = (
        torch.all(torch.isfinite(input_tensor.grad)) and
        torch.all(torch.isfinite(linear.weight.grad)) and
        torch.all(torch.isfinite(linear.bias.grad)) and
        torch.all(torch.isfinite(layer_norm.weight.grad)) and
        torch.all(torch.isfinite(layer_norm.bias.grad))
    )
    print(f"All gradients finite: {all_finite}")
    
    # Check gradient flow - all gradients should be non-zero
    input_grad_nonzero = torch.any(torch.abs(input_tensor.grad) > 1e-8)
    linear_weight_grad_nonzero = torch.any(torch.abs(linear.weight.grad) > 1e-8)
    linear_bias_grad_nonzero = torch.any(torch.abs(linear.bias.grad) > 1e-8)
    layernorm_weight_grad_nonzero = torch.any(torch.abs(layer_norm.weight.grad) > 1e-8)
    layernorm_bias_grad_nonzero = torch.any(torch.abs(layer_norm.bias.grad) > 1e-8)
    
    print(f"Input grad non-zero: {input_grad_nonzero}")
    print(f"Linear weight grad non-zero: {linear_weight_grad_nonzero}")
    print(f"Linear bias grad non-zero: {linear_bias_grad_nonzero}")
    print(f"LayerNorm weight grad non-zero: {layernorm_weight_grad_nonzero}")
    print(f"LayerNorm bias grad non-zero: {layernorm_bias_grad_nonzero}")
    
    # Save tensors to GGUF file
    writer = GGUFWriter(
        "./models/tests/layer_norm/layer_norm_backward_in_network.gguf",
        "layer_norm_backward_in_network"
    )
    
    # Save input, intermediate, and output tensors
    add_tensor_safe(writer, "layer_norm_backward_in_network_input", input_tensor)
    add_tensor_safe(writer, "layer_norm_backward_in_network_hidden", hidden)
    add_tensor_safe(writer, "layer_norm_backward_in_network_output", output)
    add_tensor_safe(writer, "layer_norm_backward_in_network_loss", loss)
    
    # Save layer parameters
    add_tensor_safe(writer, "layer_norm_backward_in_network_linear_weight", linear.weight)
    add_tensor_safe(writer, "layer_norm_backward_in_network_linear_bias", linear.bias)
    add_tensor_safe(writer, "layer_norm_backward_in_network_layernorm_weight", layer_norm.weight)
    add_tensor_safe(writer, "layer_norm_backward_in_network_layernorm_bias", layer_norm.bias)
    
    # Save gradients
    add_tensor_safe(writer, "layer_norm_backward_in_network_input_grad", input_tensor.grad)
    add_tensor_safe(writer, "layer_norm_backward_in_network_linear_weight_grad", linear.weight.grad)
    add_tensor_safe(writer, "layer_norm_backward_in_network_linear_bias_grad", linear.bias.grad)
    add_tensor_safe(writer, "layer_norm_backward_in_network_layernorm_weight_grad", layer_norm.weight.grad)
    add_tensor_safe(writer, "layer_norm_backward_in_network_layernorm_bias_grad", layer_norm.bias.grad)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print("GGUF file saved to: ./models/tests/layer_norm/layer_norm_backward_in_network.gguf")
    
    return input_tensor, hidden, output, loss

if __name__ == "__main__":
    # Run the test that's currently being worked on
    test_layer_norm_backward_in_network()