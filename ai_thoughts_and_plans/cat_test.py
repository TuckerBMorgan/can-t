import torch

a = torch.arange(4)#.reshape(2)
b = torch.arange(start=5, end=9, step=1)#.reshape(2, 2)
print(torch.einsum("i,j -> ij", a, b))