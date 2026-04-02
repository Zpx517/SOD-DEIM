import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvAtt2(nn.Module):
    def __init__(self, in_channels, att_channels=16, lk_size=13, sk_size=3, reduction=2):

        super().__init__()
        self.in_channels = in_channels
        self.att_channels = att_channels
        self.idt_channels = in_channels - att_channels
        self.lk_size = lk_size
        self.sk_size = sk_size

        # 动态卷积核生成器
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(att_channels, att_channels // reduction, 1),
            nn.GELU(),#AIFHG
            nn.Conv2d(att_channels // reduction, att_channels * sk_size * sk_size, 1)
        )
        nn.init.zeros_(self.kernel_gen[-1].weight)
        nn.init.zeros_(self.kernel_gen[-1].bias)

        # 共享静态大核卷积核：定义为参数，非卷积层
        self.lk_filter = nn.Parameter(torch.randn(att_channels, att_channels, lk_size, lk_size))
        nn.init.kaiming_normal_(self.lk_filter, mode='fan_out', nonlinearity='relu')

        # 融合层
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.att_channels + self.idt_channels, f"Input channel {C} must match att + idt ({self.att_channels} + {self.idt_channels})"

        # 通道拆分
        F_att, F_idt = torch.split(x, [self.att_channels, self.idt_channels], dim=1)

        # 生成动态卷积核 [B * att, 1, 3, 3]
        kernel = self.kernel_gen(F_att).reshape(B * self.att_channels, 1, self.sk_size, self.sk_size)

        # 动态卷积操作
        F_att_re = rearrange(F_att, 'b c h w -> 1 (b c) h w')
        out_dk = F.conv2d(F_att_re, kernel, padding=self.sk_size // 2, groups=B * self.att_channels)
        out_dk = rearrange(out_dk, '1 (b c) h w -> b c h w', b=B, c=self.att_channels)

        # 静态大核卷积
        out_lk = F.conv2d(F_att, self.lk_filter, padding=self.lk_size // 2)

        # 融合（两个卷积结果加和）
        out_att = out_lk + out_dk

        # 拼接 F_idt（保留通道）
        out = torch.cat([out_att, F_idt], dim=1)

        # 1x1 融合
        out = self.fusion(out)
        return out
class MSDAM(nn.Module): #Multi-Scale Dynamic Attention Module
    def __init__(self, in_channels, out_channels, levels=4,lk_size=13): # levels 代表特征金字塔的层数，可以灵活修改做消融实验。lk_size 这个参数是控制ConvAtt模块的卷积核，可以自己灵活修改。
        super(MSDAM, self).__init__()

        self.levels = levels
        # 下采样（Down）层：每个尺度级别应用卷积 -> ReLU -> 卷积
        self.downsample_layers = nn.ModuleList()
        for i in range( levels):
            # 下采样：卷积 -> ReLU -> 卷积
            i = i + 1
            if i == 1:
                self.downsample_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, dilation=1, padding=1),  # C2D (3)
                    nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)  # C2D (1)
                ))
            else:
                self.downsample_layers.append(nn.Sequential(
                    nn.AvgPool2d(kernel_size=2**(i-1), stride=2**(i-1)),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, dilation=2*i+1, padding=2*i+1),  # C2D (3)
                    nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)  # C2D (1)
                ))

        # 上采样（Up）层
        self.upsample_layers = nn.ModuleList()
        for i in range( levels):
            i=i+1
            if i == 1:
                self.upsample_layers.append(nn.Identity())  # Up
            else:
                self.upsample_layers.append(nn.Upsample(scale_factor=2**(i-1), mode='nearest'))  # Up

        # 最后的输出卷积层
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()
        self.rule = nn.ReLU()
        self.DyAtt = ConvAtt2(in_channels,lk_size = lk_size)

    def forward(self, x):
        # 存储每一层的特征
        features = []
        for i in range(self.levels):
            downsampled = self.downsample_layers[i](x)  # 下采样
            features.append(downsampled)

        # 多尺度特征融合
        output = features[0]
        output = self.DyAtt(output)
        for i in range(1, self.levels):
            output +=  self.upsample_layers[i](self.DyAtt(features[i])) # 上采样并融合特征

        # 最后的卷积输出
        output = self.final_conv(output)

        return output


if __name__ == '__main__':
    input =  torch.randn(1, 64, 256, 256)  # 输入大小为[1, 64, 256, 256]
    MSDAM = MSDAM(in_channels=64, out_channels=128, levels=4,lk_size=13)
    output = MSDAM(input)


