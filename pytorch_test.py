from gguf import GGUFWriter
import torch


tensor_a = torch.rand(2, 2, 3, 4)
tensor_b = torch.rand(1, 2, 4, 3)
tensor_c = tensor_a @ tensor_b
print(tensor_c)