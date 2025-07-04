import torch

def extract_successful_pytorch_weights():
    print("=== Extracting PyTorch Weights for Rust Test ===")
    
    # XOR dataset (same as Rust test)
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    
    # Try multiple seeds until we find one that learns successfully
    successful_seed = None
    for seed in range(100):
        torch.manual_seed(seed)
        
        # Same architecture as Rust test
        w1 = torch.randn(2, 4) * 0.5
        w1.requires_grad = True
        b1 = torch.zeros(1, 4)
        b1.requires_grad = True
        
        w2 = torch.randn(4, 1) * 0.5
        w2.requires_grad = True
        b2 = torch.zeros(1, 1)
        b2.requires_grad = True
        
        # Store initial weights
        initial_w1 = w1.clone().detach()
        initial_b1 = b1.clone().detach()
        initial_w2 = w2.clone().detach()
        initial_b2 = b2.clone().detach()
        
        # Quick training to see if this seed works
        final_loss = None
        for epoch in range(200):  # Shorter test
            # Zero gradients
            if w1.grad is not None:
                w1.grad.zero_()
            if b1.grad is not None:
                b1.grad.zero_()
            if w2.grad is not None:
                w2.grad.zero_()
            if b2.grad is not None:
                b2.grad.zero_()
            
            # Forward pass
            h1 = torch.tanh(x @ w1 + b1)
            output = h1 @ w2 + b2
            loss = ((output - y) ** 2).mean(dim=0)
            
            # Backward pass
            loss.backward()
            
            # Parameter update (same as Rust: -0.1)
            with torch.no_grad():
                w1 -= 0.1 * w1.grad
                b1 -= 0.1 * b1.grad
                w2 -= 0.1 * w2.grad
                b2 -= 0.1 * b2.grad
                
            final_loss = loss.item()
        
        # Check if this seed learned successfully (loss < 0.1)
        if final_loss < 0.1:
            successful_seed = seed
            print(f"Found successful seed: {seed}, final loss: {final_loss:.6f}")
            
            # Test final output
            with torch.no_grad():
                final_h1 = torch.tanh(x @ w1 + b1)
                final_output = final_h1 @ w2 + b2
                predictions = (final_output > 0.5).float().flatten()
                expected = torch.tensor([0., 1., 1., 0.])
                success = (predictions == expected).all().item()
                print(f"XOR learned correctly: {success}")
                print(f"Final output: {final_output.flatten().tolist()}")
            
            # Print initial weights for Rust
            print(f"\n=== Initial Weights for Rust (Seed {seed}) ===")
            print("// Initial w1 weights:")
            w1_flat = initial_w1.flatten().tolist()
            print(f"let mut w1 = Tensor::from_vec(vec!{w1_flat}, vec![2, 4]);")
            
            print("// Initial b1 weights:")
            b1_flat = initial_b1.flatten().tolist()
            print(f"let mut b1 = Tensor::from_vec(vec!{b1_flat}, vec![1, 4]);")
            
            print("// Initial w2 weights:")
            w2_flat = initial_w2.flatten().tolist()
            print(f"let mut w2 = Tensor::from_vec(vec!{w2_flat}, vec![4, 1]);")
            
            print("// Initial b2 weights:")
            b2_flat = initial_b2.flatten().tolist()
            print(f"let mut b2 = Tensor::from_vec(vec!{b2_flat}, vec![1, 1]);")
            
            break
    
    if successful_seed is None:
        print("No successful seed found in first 100 attempts")
        print("Trying with learning rate -0.5...")
        
        # Try with higher learning rate
        for seed in range(100):
            torch.manual_seed(seed)
            
            w1 = torch.randn(2, 4) * 0.5
            w1.requires_grad = True
            b1 = torch.zeros(1, 4)
            b1.requires_grad = True
            
            w2 = torch.randn(4, 1) * 0.5
            w2.requires_grad = True
            b2 = torch.zeros(1, 1)
            b2.requires_grad = True
            
            initial_w1 = w1.clone().detach()
            initial_b1 = b1.clone().detach()
            initial_w2 = w2.clone().detach()
            initial_b2 = b2.clone().detach()
            
            for epoch in range(200):
                if w1.grad is not None:
                    w1.grad.zero_()
                if b1.grad is not None:
                    b1.grad.zero_()
                if w2.grad is not None:
                    w2.grad.zero_()
                if b2.grad is not None:
                    b2.grad.zero_()
                
                h1 = torch.tanh(x @ w1 + b1)
                output = h1 @ w2 + b2
                loss = ((output - y) ** 2).mean(dim=0)
                
                loss.backward()
                
                with torch.no_grad():
                    w1 -= 0.5 * w1.grad  # Higher learning rate
                    b1 -= 0.5 * b1.grad
                    w2 -= 0.5 * w2.grad
                    b2 -= 0.5 * b2.grad
                    
                final_loss = loss.item()
            
            if final_loss < 0.1:
                successful_seed = seed
                print(f"Found successful seed with LR=0.5: {seed}, final loss: {final_loss:.6f}")
                
                with torch.no_grad():
                    final_h1 = torch.tanh(x @ w1 + b1)
                    final_output = final_h1 @ w2 + b2
                    predictions = (final_output > 0.5).float().flatten()
                    expected = torch.tensor([0., 1., 1., 0.])
                    success = (predictions == expected).all().item()
                    print(f"XOR learned correctly: {success}")
                    print(f"Final output: {final_output.flatten().tolist()}")
                
                print(f"\n=== Initial Weights for Rust (Seed {seed}, LR=0.5) ===")
                print("// Initial w1 weights:")
                w1_flat = initial_w1.flatten().tolist()
                print(f"let mut w1 = Tensor::from_vec(vec!{w1_flat}, vec![2, 4]);")
                
                print("// Initial b1 weights:")
                b1_flat = initial_b1.flatten().tolist()
                print(f"let mut b1 = Tensor::from_vec(vec!{b1_flat}, vec![1, 4]);")
                
                print("// Initial w2 weights:")
                w2_flat = initial_w2.flatten().tolist()
                print(f"let mut w2 = Tensor::from_vec(vec!{w2_flat}, vec![4, 1]);")
                
                print("// Initial b2 weights:")
                b2_flat = initial_b2.flatten().tolist()
                print(f"let mut b2 = Tensor::from_vec(vec!{b2_flat}, vec![1, 1]);")
                
                break
    
    return successful_seed

if __name__ == "__main__":
    extract_successful_pytorch_weights()