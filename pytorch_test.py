from gguf import GGUFWriter
import torch

tensor = torch.randn(1, 2, 3, 4)
f32_data = tensor.numpy()

summed = tensor.sum((1, 3))
summed_data = summed.numpy()

writer = GGUFWriter("./models/tests/sum/double_index_with_skip_sum__test_file_container.gguf", "double_index_with_skip_sum__test_file_container")
writer.add_tensor("sum_test_pre_double_index_with_skip_sum_model", f32_data)
writer.add_tensor("sum_test_post_double_index_with_skip_sum__model", summed_data)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()