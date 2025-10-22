import torch

a = torch.randn(4, 3, 2)

print(a)

b = a.diagonal(0, 0, 2)

print(b)
print(b.shape)