import torch

a = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[11, 22, 33], [44, 55, 66]]])
b = torch.tensor([[[10, 21, 31], [41, 51, 61]], [[12, 29, 39], [49, 59, 69]]])
print(a.shape)


print(torch.cat((a, b), dim=0))