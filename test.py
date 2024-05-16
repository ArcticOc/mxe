import torch

# 创建形状为[128, 1]的张量
tensor_a = torch.randn(128, 1)

# 创建形状为[128]的张量
tensor_b = torch.randn(128)

c = tensor_a / tensor_b.unsqueeze(1)

print(c.shape)
