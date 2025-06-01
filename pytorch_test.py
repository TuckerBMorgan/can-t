from gguf import GGUFWriter
import torch


tensor_a = torch.rand(1, 4, 4, 4)
tensor_a.requires_grad = True
tensor_a.retain_grad()
tensor_b = torch.rand(1, 4, 4, 4)
tensor_b.requires_grad = True
tensor_b.retain_grad()
multiplied = tensor_a * tensor_b
multiplied.retain_grad()
summed = multiplied.sum((1, 2, 3))
summed.retain_grad()
summed.backward()




writer = GGUFWriter("./models/tests/sum/sum_backward_test_container.gguf", "sum_backward_test_container")

writer.add_tensor("sum_backward_test_container_tensor_a", tensor_a.detach().numpy())
writer.add_tensor("sum_backward_test_container_tensor_a_grad", tensor_a.grad.detach().numpy())
writer.add_tensor("sum_backward_test_container_tensor_b", tensor_b.detach().numpy())
writer.add_tensor("sum_backward_test_container_tensor_b_grad", tensor_b.grad.numpy())
writer.add_tensor("sum_backward_test_container_tensor_multiplied", multiplied.detach().numpy())
writer.add_tensor("sum_backward_test_container_tensor_multiplied_grad", multiplied.grad.detach().numpy())
writer.add_tensor("sum_backward_test_container_tensor_summed", summed.detach().numpy())


writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
