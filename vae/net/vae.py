import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vgg16_bn
from defines import DEVICE
from models import (
    Swish,
    Conv_SubPixel,
    Conv_GroupNorm,
    UpSample_SubPix,
)

class VAE(nn.Module):
    def __init__(self, middle_c, out_put_act="Identity"):
        super(VAE, self).__init__()
        self.encoder = Encoder(middle_c, is_vae=True).to(DEVICE)
        self.decoder = Decoder(middle_c, out_put_act=out_put_act).to(DEVICE)

    def forward(self, x):
        result = self.encoder(x)
        z = self.reParameterRize(result, result)
        x = self.decoder(z)
        return x, result, result

    def reParameterRize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.normal(0, 1, std.size()).to(DEVICE)
        x = eps * std + mu
        # x = self.nor(x)
        return x

    # 释放encoder
    def Res_encoder(self):
        del self.encoder
        torch.cuda.empty_cache()

class Encoder(nn.Module):
    def __init__(
            self,
            out_c,
            is_vae: bool
    ):
        super(Encoder, self).__init__()
        self.is_vae = is_vae
        self.blocks = nn.Sequential(
            # 下采样调整尺寸
            EncoderBlock(3, 16, kernel_size=3, stride=2),
            EncoderBlock(16, 32, kernel_size=3, stride=2),
            EncoderBlock(32, 64, kernel_size=3, stride=2),
            EncoderBlock(64, 128, kernel_size=3, stride=2),

            EncoderBlock(128, 256, kernel_size=3, stride=1)
        )
        self.to_vectors = nn.Conv2d(256, out_c, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.blocks(x)
        return self.to_vectors(x)

class Decoder(nn.Module):
    def __init__(
            self,
            in_c,
            out_put_act,
    ):
        super(Decoder, self).__init__()
        self.from_vectors = nn.Conv2d(in_c, 256, kernel_size=1, stride=1)
        self.blocks = nn.Sequential(
            EncoderBlock(256, 128, kernel_size=3, stride=1),
            EncoderBlock(128, 64, kernel_size=3, stride=1),
            DecoderBlock(64, 32, kernel_size=3),
            DecoderBlock(32, 16, kernel_size=3),
            DecoderBlock(16, 16, kernel_size=3),
        )
        self.Conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.act1 = Swish()
        self.Conv2 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        if out_put_act == "tanh":
            self.act2 = nn.Tanh()
        else:
            self.act2 = nn.Identity()

    def forward(self, x):
        x = self.from_vectors(x)
        x = self.blocks(x)
        x = self.Conv1(x)
        x = self.act1(x)
        x = self.Conv2(x)
        x = self.act2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_c,
            out_c,
            kernel_size,
            stride,
            deep=3,
    ):
        super(EncoderBlock, self).__init__()
        padding = kernel_size // 2
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(
            Conv_GroupNorm(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        )
        for _ in range(deep-2):
            self.conv_blocks.append(
                Conv_GroupNorm(out_c, out_c, kernel_size=kernel_size, stride=1, padding=padding)
            )
        self.conv_blocks.append(
            Conv_GroupNorm(out_c, out_c, kernel_size=kernel_size, stride=1, padding=padding, is_activate=False)
        )
        if in_c != out_c or stride != 1:
            self.conv_skil = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0)    # 下采样调整尺寸
        else:
            self.conv_skil = nn.Identity()
        self.act_skip = Swish()

    def forward(self, x):
        skip = self.conv_skil(x)
        for block in self.conv_blocks:
            x = block(x)
        return self.act_skip(x + skip)      # 残差


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_c,
            out_c,
            kernel_size,
            deep=3,
    ):
        super(DecoderBlock, self).__init__()
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(
            UpSample_SubPix(in_c, out_c, kernel_size=kernel_size)
        )
        for _ in range(deep-2):
            self.conv_blocks.append(
                Conv_GroupNorm(out_c, out_c, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            )
        self.conv_blocks.append(
            Conv_GroupNorm(out_c, out_c, kernel_size=kernel_size, stride=1, padding=kernel_size//2, is_activate=False)
        )
        self.conv_skip = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.up_sample = Conv_SubPixel(in_c, in_c, kernel_size=3)
        self.act_skip = Swish()

    def forward(self, x):
        skip = self.up_sample(x)
        skip = self.conv_skip(skip)
        for block in self.conv_blocks:
            x = block(x)
        return self.act_skip(x + skip)


class Discriminator(nn.Module):
    def __init__(
            self,
            use_vgg=False,
    ):
        super(Discriminator, self).__init__()
        self.use_vgg = use_vgg
        if self.use_vgg:
            self.model = vgg16_bn(pretrained=False)
            classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(True),
                nn.Linear(512, 1)
            )
            self.model.classifier = classifier
        else:
            cfgs = [64, "M", 128, "M", 256, "M", 256, 256, "M", 256, 256, "M"]
            layers: list[nn.Module] = []
            in_c = 3
            for v in cfgs:
                if v == "M":
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_c, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.ReLU(inplace=True)]
                    in_c = v
            self.classifier = nn.Sequential(
                nn.Linear(256 * 8 * 8, 256),
                nn.ReLU(True),
                nn.Linear(256, 1),
            )
            self.model = nn.Sequential(*layers)
            self.classifier.to(DEVICE)
        self.model.to(DEVICE)

    def forward(self, x):
        x = self.model(x)
        if not self.use_vgg:
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x



if __name__ == "__main__":
    model = VAE(middle_c=8)
    print("vae总参数", sum(i.numel() for i in model.parameters()) / 10000, "单位:万")
    data = np.zeros((1, 3, 256, 256), dtype=np.float32)
    data = torch.Tensor(data).to(DEVICE)
    out = model(data)
    print(out[0].shape)


    D = Discriminator(use_vgg=False)
    print("dis总参数", sum(i.numel() for i in D.parameters()) / 10000, "单位:万")
    data = torch.Tensor(data).to(DEVICE)
    out = D(data)
    print(out[0].shape)
    input()