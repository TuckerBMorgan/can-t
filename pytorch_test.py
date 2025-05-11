import torch

tensor1 = torch.randn(10, 10)
tensor2 = torch.randn(10, 10)
print(torch.matmul(tensor1, tensor2).size())

tensor1 = torch.randn(1, 10)
tensor2 = torch.randn(10, 1)
print(torch.matmul(tensor1, tensor2).size())

tensor1 = torch.randn(2, 1, 10)
tensor2 = torch.randn(10, 1)
print(torch.matmul(tensor1, tensor2).size())