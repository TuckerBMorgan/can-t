import torch

a = torch.arange(12).reshape(3, 2, 2)
b = torch.arange(6).reshape(3, 2)
print(a)
print(b)
print(torch.einsum("bec,be->bc", a, b))