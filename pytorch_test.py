from gguf import GGUFWriter
import torch
import torch.nn.functional as F


test = torch.tensor(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
print(test.std((0, 1)))

exit()
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

