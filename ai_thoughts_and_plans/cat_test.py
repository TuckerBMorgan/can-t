import torch

# 1. Create two tensors of shape (4,) filled with 1s, and make them trainable
x1 = torch.ones(4, requires_grad=True)
x2 = torch.ones(4, requires_grad=True)

# 2. Define an Adam optimizer over these tensors
optimizer = torch.optim.Adam([x1, x2], lr=0.1)

# 3. Target tensor of four 3s
target = torch.full((4,), 3.0)

# 4. Forward pass: add the two tensors
prediction = x1 + x2

# 5. Compute loss (MSE) against the target tensor
loss = torch.nn.functional.mse_loss(prediction, target)

print("Prediction before step:", prediction.detach().numpy())
print("Loss before step:", loss.item())

# 6. Backward + one optimization step
optimizer.zero_grad()  # Clear previous gradients
loss.backward()        # Compute gradients
optimizer.step()       # Take one Adam step

# 7. Show updated tensors and new prediction
with torch.no_grad():
    new_prediction = x1 + x2
    new_loss = torch.nn.functional.mse_loss(new_prediction, target)

print("Prediction after step:", new_prediction.numpy())
print("Loss after step:", new_loss.item())
print("x1:", x1.data.numpy())
print("x2:", x2.data.numpy())
