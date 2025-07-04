import torch

def test_xor_network():
    print("=== PyTorch XOR Network Test ===")
    
    # XOR dataset (same as Rust test)
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # shape [4, 2]
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # shape [4, 1]
    
    print(f"XOR Dataset:")
    print(f"x: {x}")
    print(f"y: {y.flatten()}")
    
    # Two layer network: 2 -> 4 -> 1 (same architecture as Rust test)
    # Use same random seed approach as Rust (though different implementation)
    torch.manual_seed(12345)
    w1 = torch.randn(2, 4) * 0.5
    w1.requires_grad = True
    b1 = torch.zeros(1, 4)
    b1.requires_grad = True
    
    w2 = torch.randn(4, 1) * 0.5
    w2.requires_grad = True
    b2 = torch.zeros(1, 1)
    b2.requires_grad = True
    
    print(f"Initial w1 sample values: {w1[0, :2]}")
    print(f"Initial w2 sample values: {w2[:2, 0]}")
    
    # Training loop (exact same as Rust test)
    for epoch in range(1000):
        # Zero gradients
        if w1.grad is not None:
            w1.grad.zero_()
        if b1.grad is not None:
            b1.grad.zero_()
        if w2.grad is not None:
            w2.grad.zero_()
        if b2.grad is not None:
            b2.grad.zero_()
        
        # Forward pass (same as Rust: h1 = ((x << w1) + b1).tanh(); output = (h1 << w2) + b2)
        h1 = torch.tanh(x @ w1 + b1)
        output = h1 @ w2 + b2
        
        # Loss (same as Rust: (output - y).pow(2.0).mean(vec![0]))
        # Rust mean(vec![0]) means average along axis 0 only
        loss = ((output - y) ** 2).mean(dim=0)
        
        if epoch % 100 == 0:
            print(f"PyTorch XOR Epoch {epoch}: Loss = {loss.item():.6f}")
            print(f"  output: {output.flatten().tolist()}")
        
        # Backward pass
        loss.backward()
        
        # Parameter update (same learning rate as Rust test: -0.1)
        with torch.no_grad():
            w1 -= 0.1 * w1.grad
            b1 -= 0.1 * b1.grad
            w2 -= 0.1 * w2.grad
            b2 -= 0.1 * b2.grad
    
    # Final test (same as Rust test)
    print("\nFinal XOR test:")
    with torch.no_grad():
        test_h1 = torch.tanh(x @ w1 + b1)
        test_output = test_h1 @ w2 + b2
        print(f"PyTorch Results: {test_output.flatten().tolist()} (should be close to [0, 1, 1, 0])")
        
        # Check if XOR was learned
        predictions = (test_output > 0.5).float().flatten()
        expected = torch.tensor([0., 1., 1., 0.])
        correct = (predictions == expected).all().item()
        print(f"XOR learned correctly: {correct}")
    
    return test_output.flatten().tolist()

if __name__ == "__main__":
    result = test_xor_network()
    print(f"\nFinal result: {result}")