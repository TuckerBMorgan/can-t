from gguf import GGUFWriter
import torch


tensor_a = torch.rand(1, 2, 3, 4)
tensor_b = tensor_a.reshape([1, 3, 2, 4])
