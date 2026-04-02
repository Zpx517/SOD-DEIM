import torch
import torch.nn as nn
class Shift_channel_mix(nn.Module):
    def __init__(self, shift_size=1):
        super(Shift_channel_mix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):  # x的张量 [B,C,H,W]
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        x1 = torch.roll(x1, self.shift_size, dims=2)  # [:,:,1:,:]

        x2 = torch.roll(x2, -self.shift_size, dims=2)  # [:,:,:-1,:]

        x3 = torch.roll(x3, self.shift_size, dims=3)  # [:,:,:,1:]

        x4 = torch.roll(x4, -self.shift_size, dims=3)  # [:,:,:,:-1]

        x = torch.cat([x1, x2, x3, x4], 1)

        return x


class HSLConv(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()
        self.k_size = k_size
        if k_size == 7:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=((3 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=((3 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=4, groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=11, stride=1, padding=15, groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=13, stride=1, padding=18, groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=17, stride=1, padding=24, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim * 2, dim, 1)
        self.scm = Shift_channel_mix()
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(self.scm(torch.cat([x, self.conv_spatial(x)], dim=1)))
        return x



if __name__ == '__main__':
    ch = 32
    model = HSLConv(ch,k_size=7)
    input = torch.randn(1, ch , 32, 32)
    output = model(input)

