import torch

b = torch.randn(1, 128)
a = torch.randn(1, 128)

# a, b = F.normalize(a, p=0.1, dim=1), F.normalize(b, p=0.1, dim=1)

d = torch.cdist(a, b, p=2)

print(d)
