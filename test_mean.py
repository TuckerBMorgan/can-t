import torch

def test_mean_gradient():
    print("=== Testing Mean Gradient ===")
    
    # Create a simple tensor that matches our Rust test case
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=True)
    print(f"Input x: {x.flatten()}")
    
    # Compute mean along dimension 0 (same as Rust mean(vec![0]))
    y = x.mean(dim=0)
    print(f"Mean result y: {y}")
    
    # Compute a simple loss
    loss = y.sum()
    print(f"Loss: {loss}")
    
    # Backward pass
    loss.backward()
    
    print(f"Gradient of x: {x.grad.flatten()}")
    print(f"Expected gradient: [0.25, 0.25, 0.25, 0.25]")
    
    # The gradient should be 1/n for each element where n is the number of elements averaged
    # Since we averaged 4 elements, each should get gradient 1/4 = 0.25

if __name__ == "__main__":
    test_mean_gradient()