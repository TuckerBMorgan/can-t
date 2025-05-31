from gguf import GGUFWriter
import torch

tensor = torch.randn(4, 4)
f32_data = tensor.numpy()

writer = GGUFWriter("./tests/test.gguf", "TestModel")
writer.add_tensor("test_tensor", f32_data)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()