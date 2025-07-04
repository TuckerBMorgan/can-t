import numpy as np
import torch
import matplotlib.pyplot as plt

def analyze_weight_initialization():
    print("=== Analysis of Weight Initialization Strategies ===")
    
    # Parameters for a 2x4 layer (like our XOR network's first layer)
    fan_in = 2
    fan_out = 4
    
    print(f"Layer shape: [{fan_in}, {fan_out}]")
    print(f"fan_in: {fan_in}, fan_out: {fan_out}")
    
    # 1. Current Rust implementation: Normal(0, 0.01)
    current_std = 0.01
    current_weights = np.random.normal(0, current_std, (fan_in, fan_out))
    
    # 2. Xavier/Glorot Uniform: U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
    xavier_limit = np.sqrt(6.0 / (fan_in + fan_out))
    xavier_weights = np.random.uniform(-xavier_limit, xavier_limit, (fan_in, fan_out))
    
    # 3. Xavier/Glorot Normal: N(0, sqrt(2/(fan_in + fan_out)))
    xavier_normal_std = np.sqrt(2.0 / (fan_in + fan_out))
    xavier_normal_weights = np.random.normal(0, xavier_normal_std, (fan_in, fan_out))
    
    # 4. He initialization: N(0, sqrt(2/fan_in)) - good for ReLU, but let's test
    he_std = np.sqrt(2.0 / fan_in)
    he_weights = np.random.normal(0, he_std, (fan_in, fan_out))
    
    # 5. PyTorch default for linear layers
    pytorch_std = 1.0 / np.sqrt(fan_in)  # This is what PyTorch uses
    pytorch_weights = np.random.normal(0, pytorch_std, (fan_in, fan_out))
    
    # 6. What we used in successful XOR (randn * 0.5)
    successful_std = 0.5
    successful_weights = np.random.normal(0, successful_std, (fan_in, fan_out))
    
    strategies = [
        ("Current Rust (std=0.01)", current_weights, current_std),
        ("Xavier Uniform", xavier_weights, xavier_limit),
        ("Xavier Normal", xavier_normal_weights, xavier_normal_std),
        ("He Normal", he_weights, he_std),
        ("PyTorch Default", pytorch_weights, pytorch_std),
        ("Successful (std=0.5)", successful_weights, successful_std),
    ]
    
    print("\n=== Initialization Strategy Analysis ===")
    for name, weights, scale in strategies:
        mean = np.mean(weights)
        std = np.std(weights)
        min_val = np.min(weights)
        max_val = np.max(weights)
        
        print(f"\n{name}:")
        print(f"  Scale parameter: {scale:.4f}")
        print(f"  Actual mean: {mean:.6f}")
        print(f"  Actual std: {std:.6f}")
        print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
        
        # Calculate variance of forward pass (assuming input variance = 1)
        forward_var = np.sum(weights**2, axis=0).mean()  # Variance of output
        print(f"  Forward pass output variance: {forward_var:.6f}")
        
        # For tanh networks, we want initial activations in the linear region
        # tanh'(0) = 1, but tanh'(x) drops quickly for |x| > 1
        linear_region_fraction = np.mean(np.abs(weights) < 1.0)
        print(f"  Fraction in tanh linear region (|w| < 1): {linear_region_fraction:.3f}")
    
    print("\n=== Recommendations ===")
    print("Problems with current Rust implementation (std=0.01):")
    print("1. TOO SMALL: Weights are tiny, leading to very small gradients")
    print("2. VANISHING GRADIENTS: Network can't learn effectively")
    print("3. SYMMETRY: All weights start very close to zero")
    
    print("\nFor tanh networks, good initialization should:")
    print("1. Break symmetry between neurons")
    print("2. Keep initial activations in tanh's linear region")
    print("3. Maintain reasonable gradient flow")
    
    print("\nBest strategies for XOR with tanh:")
    print("1. Xavier/Glorot Normal (designed for tanh)")
    print("2. Successful approach (std=0.5) - empirically proven")
    print("3. PyTorch default (std=1/sqrt(fan_in))")

def test_xor_success_rates():
    """Test XOR learning success rates with different initializations"""
    print("\n=== XOR Success Rate Analysis ===")
    
    # This would ideally test multiple seeds, but we'll show the concept
    strategies = {
        "Current (std=0.01)": 0.01,
        "Xavier Normal": np.sqrt(2.0 / (2 + 4)),  # sqrt(2/(fan_in + fan_out))
        "PyTorch Default": 1.0 / np.sqrt(2),       # 1/sqrt(fan_in)
        "Successful (std=0.5)": 0.5,
    }
    
    print("Theoretical analysis (based on activation scaling):")
    for name, std in strategies.items():
        # For XOR input [0,1] or [1,0], typical activation magnitude
        typical_activation = std * np.sqrt(2)  # Rough estimate
        
        print(f"\n{name} (std={std:.4f}):")
        print(f"  Typical pre-activation magnitude: {typical_activation:.4f}")
        
        if typical_activation < 0.1:
            print("  Risk: TOO SMALL - vanishing gradients likely")
        elif typical_activation > 2.0:
            print("  Risk: TOO LARGE - saturation likely")
        else:
            print("  Status: Good range for tanh")

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible analysis
    analyze_weight_initialization()
    test_xor_success_rates()