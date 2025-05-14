import torch

tensor1 = torch.randn(15, 15, 6, 2)
tensor2= torch.randn(1, 2, 6)
print(torch.matmul(tensor1, tensor2).size())