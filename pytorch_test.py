import torch
import torch.nn.functional as F

# Linear regression test equivalent to the Rust test
def test_linear_regression():
    print("=== PyTorch Linear Regression Test ===")
    
    # Generate simple y = 2x + 1 data (same as Rust test)
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # shape [4, 1]
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])  # shape [4, 1]
    
    # Verify the data is correct: y = 2x + 1
    print("Data verification:")
    for i in range(4):
        x_val = x[i, 0].item()
        y_val = y[i, 0].item()
        expected = 2.0 * x_val + 1.0
        print(f"x={x_val}, y={y_val}, expected={expected}")
    
    # Initialize parameters (same as Rust test)
    w = torch.tensor([[0.5]], requires_grad=True)  # shape [1, 1]
    b = torch.tensor([[0.0]], requires_grad=True)  # shape [1, 1]
    
    # Training loop
    for epoch in range(100):
        # Forward pass
        pred = x @ w + b  # matrix multiplication + bias
        loss = ((pred - y) ** 2).sum() / 4.0  # MSE loss, manually averaged
        
        if epoch % 20 == 0:
            print(f"PyTorch Epoch {epoch}: Loss = {loss.item():.6f}")
            print(f"  pred: {pred.flatten().tolist()}")
            print(f"  y: {y.flatten().tolist()}")
            print(f"  w before backward: {w.item():.6f}")
            print(f"  b before backward: {b.item():.6f}")
        
        # Backward pass
        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
            
        loss.backward()
        
        if epoch % 20 == 0:
            print(f"  w.grad: {w.grad.item():.6f}")
            print(f"  b.grad: {b.grad.item():.6f}")
        
        # Parameter update (same learning rate as Rust test)
        with torch.no_grad():
            w -= 0.01 * w.grad
            b -= 0.01 * b.grad
            
        if epoch % 20 == 0:
            print(f"  w after update: {w.item():.6f}")
            print(f"  b after update: {b.item():.6f}")
            print()
    
    # Final results
    final_w = w.item()
    final_b = b.item()
    print(f"Final w: {final_w:.6f}, Final b: {final_b:.6f} (should be ~2.0, ~1.0)")
    
    # Check if we learned the right parameters
    w_correct = abs(final_w - 2.0) < 0.1
    b_correct = abs(final_b - 1.0) < 0.1
    print(f"Weight correct: {w_correct}")
    print(f"Bias correct: {b_correct}")
    
    return final_w, final_b

def test_xor_network():
    print("\n=== PyTorch XOR Network Test ===")
    
    # XOR dataset (same as Rust test)
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # shape [4, 2]
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # shape [4, 1]
    
    print("XOR Dataset:")
    print(f"x: {x}")
    print(f"y: {y.flatten()}")
    
    # Two layer network: 2 -> 4 -> 1 (same architecture as Rust test)
    torch.manual_seed(42)  # For reproducibility
    w1 = torch.randn(2, 4, requires_grad=True) * 0.5
    b1 = torch.zeros(1, 4, requires_grad=True)
    w2 = torch.randn(4, 1, requires_grad=True) * 0.5  
    b2 = torch.zeros(1, 1, requires_grad=True)
    
    print(f"Initial w1: {w1}")
    print(f"Initial w2: {w2}")
    
    # Training loop
    for epoch in range(1000):
        # Forward pass
        h1 = torch.tanh(x @ w1 + b1)
        output = h1 @ w2 + b2
        loss = ((output - y) ** 2).mean()  # MSE loss
        
        if epoch % 100 == 0:
            print(f"PyTorch XOR Epoch {epoch}: Loss = {loss.item():.6f}")
            print(f"  output: {output.flatten().tolist()}")
        
        # Backward pass
        for param in [w1, b1, w2, b2]:
            if param.grad is not None:
                param.grad.zero_()
        
        loss.backward()
        
        # Parameter update (same learning rate as Rust test: -0.5)
        with torch.no_grad():
            w1 -= 0.5 * w1.grad
            b1 -= 0.5 * b1.grad
            w2 -= 0.5 * w2.grad  
            b2 -= 0.5 * b2.grad
    
    # Final test
    print("\nFinal XOR test:")
    with torch.no_grad():
        h1 = torch.tanh(x @ w1 + b1)
        final_output = h1 @ w2 + b2
        print(f"Final output: {final_output.flatten().tolist()}")
        print(f"Expected: [0, 1, 1, 0]")
        
        # Check if XOR was learned (threshold at 0.5)
        predictions = (final_output > 0.5).float().flatten()
        expected = torch.tensor([0., 1., 1., 0.])
        correct = (predictions == expected).all().item()
        print(f"XOR learned correctly: {correct}")
    
    return final_output.flatten().tolist()

def test_masked_fill_basic():
    """PyTorch equivalent of the Rust test_masked_fill_basic test"""
    print("\n=== PyTorch Masked Fill Basic Test ===")
    
    # Test basic masked fill functionality - matching the actual Rust implementation
    data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([False, True, False, True])  # Same as Rust test
    
    # Rust implementation: mask==1.0 means FILL, mask==0.0 means KEEP
    # PyTorch: mask==True means FILL, mask==False means KEEP
    # So we use mask directly (convert to boolean)

    
    result = data.masked_fill(mask, -999.0)
    
    print(f"Original data: {data}")
    print(f"Original mask: {mask}")
    print(f"PyTorch mask (boolean): {mask}")
    print(f"Result: {result}")
    
    # Expected behavior (matching actual Rust implementation):
    # Where mask is 0.0, keep original values; where mask is 1.0, fill with -999.0
    expected = torch.tensor([1.0, -999.0, 3.0, -999.0])
    print(f"Expected: {expected}")
    
    # Check results
    tolerance = 1e-6
    matches = torch.abs(result - expected) < tolerance
    print(f"Matches expected: {matches}")
    print(f"All match: {matches.all().item()}")
    
    # Individual checks (matching actual Rust behavior, not the wrong comments)
    print(f"result[0] = {result[0]:.6f} (mask=0.0 -> keep original)")
    print(f"result[1] = {result[1]:.6f} (mask=1.0 -> fill)")  
    print(f"result[2] = {result[2]:.6f} (mask=0.0 -> keep original)")
    print(f"result[3] = {result[3]:.6f} (mask=1.0 -> fill)")
    
    return result

if __name__ == "__main__":
    test_masked_fill_basic()
    exit()

'''
def add_tensor_safe(writer: GGUFWriter, name: str, t: torch.Tensor) -> None:
    """
    Add a tensor to the GGUF file guaranteeing it always has at least one dim.
    Scalars (ndim == 0) are reshaped to shape (1,).
    """
    arr = t.detach().cpu().numpy()
    if arr.ndim == 0:          # scalar → make it length-1
        arr = arr.reshape((1,))
    writer.add_tensor(name, arr)

words = open('data/bigram/names.txt', 'r').read().splitlines()
words[:8]

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%

# MLP revisited
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5) #* 0.2
#b1 = torch.randn(n_hidden,                        generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0

# BatchNorm parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
# ----- write everything -----

writer = GGUFWriter(
    "./models/tests/bigram/bigram_simple.gguf",
    "bigram_simple",
)
add_tensor_safe(writer, "bigram_simple_C", C)
add_tensor_safe(writer, "bigram_simple_W1", W1)
add_tensor_safe(writer, "bigram_simple_W2", W2)
add_tensor_safe(writer, "bigram_simple_b2", b2)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()

'''