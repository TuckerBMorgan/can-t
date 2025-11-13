import torch


a = torch.randn(2, 3, 4)

print(a)
maxes = a.max(1)

print(maxes.values)
print(maxes.indices)