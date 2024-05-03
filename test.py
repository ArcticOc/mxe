import torch

A = torch.randn(128, 1)  # 创建一个128x1的张量
B = torch.randn(128)  # 创建一个长度为128的一维张量

# 通过广播相加
result = A + B

print(result.shape)  # 输出: torch.Size([128, 128])
