import torch

def debug_xor_single_step():
    print("=== Debug XOR Single Step ===")
    
    # Use the exact same setup as the Rust test
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    
    # Use fixed weights for debugging (instead of random)
    w1 = torch.tensor([[0.1, 0.2, 0.3, 0.4], 
                       [0.5, 0.6, 0.7, 0.8]], requires_grad=True)
    b1 = torch.zeros(1, 4, requires_grad=True)
    
    w2 = torch.tensor([[0.1], [0.2], [0.3], [0.4]], requires_grad=True)
    b2 = torch.zeros(1, 1, requires_grad=True)
    
    print(f"w1:\n{w1}")
    print(f"w2:\n{w2}")
    print(f"x:\n{x}")
    print(f"y:\n{y}")
    
    # Forward pass
    h1 = torch.tanh(x @ w1 + b1)
    output = h1 @ w2 + b2
    loss = ((output - y) ** 2).mean(dim=0)
    
    print(f"\nForward pass:")
    print(f"h1:\n{h1}")
    print(f"output:\n{output}")
    print(f"loss: {loss}")
    
    # Backward pass
    loss.backward()
    
    print(f"\nGradients:")
    print(f"w1.grad:\n{w1.grad}")
    print(f"b1.grad:\n{b1.grad}")
    print(f"w2.grad:\n{w2.grad}")
    print(f"b2.grad:\n{b2.grad}")

if __name__ == "__main__":
    debug_xor_single_step()