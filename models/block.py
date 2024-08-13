import torch
import torch.nn as nn



class Conv_GroupNorm(nn.Module):
    """
    nn.BatchNorm2d 适用于每个通道独立归一化的场景。
    nn.GroupNorm 适用于通道数较多时，希望减少参数数量并提高模型泛化能力的场景。
    """
    def __init__(
            self,
            in_c,
            out_c,
            kernel_size,
            stride,
            padding,
            is_activate=True,
    ):
        super(Conv_GroupNorm, self).__init__()
        self.is_activate = is_activate
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gn = GroupNorm(out_c)
        self.swish = Swish()

    def forward(self, x):
        x = self.gn(self.conv(x))
        if self.is_activate:
            return self.swish(x)
        return x

class Conv_SubPixel(nn.Module):
    def __init__(
            self,
            in_c,
            out_c,
            kernel_size=3,
            stride=1,
            scale_factor=2,
    ):
        super(Conv_SubPixel, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c * scale_factor ** 2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)  # 上采样

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

class UpSample_SubPix(nn.Module):
    def __init__(
            self,
            in_c,
            out_c,
            kernel_size,
    ):
        super(UpSample_SubPix, self).__init__()
        self.subPix = Conv_SubPixel(in_c, out_c, kernel_size=kernel_size)
        self.gn = GroupNorm(out_c)
        self.swish = Swish()

    def forward(self, x):
        return self.swish(self.gn(self.subPix(x)))

class UpSample_ConvT(nn.Module):
    def __init__(
            self,
            in_c,
            out_c,
            kernel_size,
            stride,
            padding,
    ):
        super(UpSample_ConvT, self).__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gn = GroupNorm(out_c)
        self.swish = Swish()

    def forward(self, x):
        return self.swish(self.gn(self.conv))

class GroupNorm(nn.Module):
    def __init__(self, channels, num_groups=16, groups_eps=1e-6):
        super(GroupNorm, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups, channels, groups_eps)

    def forward(self, x):
        return self.group_norm(x)

class Swish(nn.Module):
    """
    非单调激活函数，通过 Sigmoid 函数对输入进行缩放，可以控制激活值的大小，有助于防止梯度消失或爆炸问题。
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
