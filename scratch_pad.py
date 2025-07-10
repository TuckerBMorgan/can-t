import torch
import torch.nn as nn

def test_layer_norm_backward_parameter_gradients():
    """Python version of test_layer_norm_backward_parameter_gradients"""
    
    # Create LayerNorm with 3 features
    layer_norm = nn.LayerNorm(3)
    
    # Input that will produce non-zero normalized values
    input_tensor = torch.tensor([[1.0, 4.0, 7.0]], requires_grad=True)
    
    print("=== LayerNorm Parameter Gradients Test ===")
    print(f"Input: {input_tensor}")
    print(f"Initial weight (gamma): {layer_norm.weight}")
    print(f"Initial bias (beta): {layer_norm.bias}")
    
    # Forward pass
    output = layer_norm(input_tensor)
    print(f"Output: {output}")
    
    # Sum to create scalar loss
    loss = output.sum()
    print(f"Loss: {loss}")
    
    # Backward pass
    loss.backward()
    
    print(f"\nGradients:")
    print(f"Input gradient: {input_tensor.grad}")
    print(f"Weight gradient: {layer_norm.weight.grad}")
    print(f"Bias gradient: {layer_norm.bias.grad}")
    
    # Manual computation for verification
    print(f"\n=== Manual Verification ===")
    x = input_tensor.detach().numpy().flatten()
    mean = x.mean()
    variance = x.var(ddof=0)  # Population variance
    std = (variance + 1e-5) ** 0.5
    normalized = (x - mean) / std
    
    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print(f"Std: {std}")
    print(f"Normalized values: {normalized}")
    
    print(f"\nExpected bias gradient (should equal normalized): {normalized}")
    print(f"Actual bias gradient: {layer_norm.bias.grad.detach().numpy()}")
    
    # For weight gradient: d/d_weight = normalized * d_loss/d_output = normalized * 1
    print(f"Expected weight gradient (should equal normalized): {normalized}")
    print(f"Actual weight gradient: {layer_norm.weight.grad.detach().numpy()}")

if __name__ == "__main__":
    test_layer_norm_backward_parameter_gradients()