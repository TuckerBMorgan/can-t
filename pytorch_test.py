from gguf import GGUFWriter
import torch


tensor_a = torch.rand(1, 2, 3, 4)
tensor_b = torch.rand(4)

tensor_x = tensor_a @ tensor_b
print(tensor_x.shape)

writer = GGUFWriter("./models/tests/matmul/matmul_4x1.gguf", "matmul_4x1")

writer.add_tensor("matmul_4x1_tensor_a", tensor_a.detach().numpy())
writer.add_tensor("matmul_4x1_tensor_b", tensor_b.detach().numpy())
writer.add_tensor("matmul_4x1_tensor_x", tensor_x.detach().numpy())

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()