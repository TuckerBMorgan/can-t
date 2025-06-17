from gguf import GGUFWriter
import torch


def add_tensor_safe(writer: GGUFWriter, name: str, t: torch.Tensor) -> None:
    """
    Add a tensor to the GGUF file guaranteeing it always has at least one dim.
    Scalars (ndim == 0) are reshaped to shape (1,).
    """
    arr = t.detach().cpu().numpy()
    if arr.ndim == 0:          # scalar → make it length-1
        arr = arr.reshape((1,))
    writer.add_tensor(name, arr)


# ----- build a little computation graph -----
tensor_a = torch.rand(1, 2, 3, 4, requires_grad=True)
tensor_b = torch.rand(4, requires_grad=True)

tensor_c = tensor_a @ tensor_b          # (2, 3, 4, 4)
first_sum = tensor_c.sum(0)             # (3, 4, 4)
first_sum.retain_grad()

tensor_c = first_sum.sum(0)             # (4, 4)
middle_sum = tensor_c.sum(0)            # (4,)
middle_sum.retain_grad()

tensor_c = middle_sum.sum(0)            # scalar (rank-0)
tensor_c.backward()

# ----- write everything -----
writer = GGUFWriter(
    "./models/tests/matmul/matmul_backward_test_4x1.gguf",
    "matmul_backward_test_4x1",
)

add_tensor_safe(writer, "matmul_backward_test_4x1_tensor_a", tensor_a)
add_tensor_safe(writer, "matmul_backward_test_4x1_tensor_a_grad", tensor_a.grad)
add_tensor_safe(writer, "matmul_backward_test_4x1_tensor_b", tensor_b)
add_tensor_safe(writer, "matmul_backward_test_4x1_middle_sum_grad", middle_sum.grad)
add_tensor_safe(writer, "matmul_backward_test_4x1_first_sum_grad", first_sum.grad)
add_tensor_safe(writer, "matmul_backward_test_4x1_tensor_c", tensor_c)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
