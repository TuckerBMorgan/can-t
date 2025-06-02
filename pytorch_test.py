from gguf import GGUFWriter
import torch


tensor_a = torch.rand(1)
tensor_b = torch.rand(2)
tensor_c = tensor_a + tensor_b


writer = GGUFWriter("./models/tests/add/add_broadcast_test.gguf", "add_broadcast_test")

writer.add_tensor("add_broadcast_test_tensor_a", tensor_a.detach().numpy())
writer.add_tensor("add_broadcast_test_tensor_b", tensor_b.detach().numpy())
writer.add_tensor("add_broadcast_test_tensor_c", tensor_c.detach().numpy())

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()