import torch

A = torch.randn(91, 128)  # 创建一个128x1的张量
B = torch.randn(91)  # 创建一个长度为128的一维张量

# 通过广播相加
result = torch.tensor([1, 2, 3])

print(result.shape)  # 输出: torch.Size([128, 128])
