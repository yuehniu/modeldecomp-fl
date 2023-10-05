import torch
from torch.nn import functional as F

for i in range(100000):
    b = 1
    tocuda = torch.device('cuda')
    img = torch.rand(3*b, 1, 450, 450)
    img = img.to(tocuda)
    kernel = torch.rand(b, 1, 51, 51)
    kernel = kernel.to(tocuda)

    F.conv2d(img, kernel, padding=0)
    # F.conv2d(img, kernel, padding=0).view(b, 3, 400, 400)

print('finished')