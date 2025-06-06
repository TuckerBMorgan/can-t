from gguf import GGUFWriter
import torch


tensor_a = torch.rand(10, 5)

tensor_b = torch.rand(10, 5)
tensor_c = tensor_a + tensor_b
first_sum = tensor_c.sum(1, keepdim=True)
print(first_sum.shape)
second_sum = first_sum.sum(0, keepdim=True)
print(second_sum.shape)
second_sum.requires_grad = True
second_sum.backward()