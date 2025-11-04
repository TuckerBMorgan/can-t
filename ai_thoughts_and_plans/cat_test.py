import torch
total = 3 * 4 * 3
x = torch.rand(3, 4, 3)#.reshape(2, 2)#, 2, 1)
print(x)
print(torch.topk(x, 2, 1))