import torch

a = torch.arange(24)
a = a.reshape(2, 3, 4)

print(a.diagonal(0, 1, 2))