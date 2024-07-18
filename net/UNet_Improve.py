"""
基于基础UNet++模型结构改进
1-每层卷积块自定义repeats-conv来加强特征提取
2-每层卷积头新增QKV-Attention注意力机制
3-Conv后bn归一化替换为group_norm,减少参数数量并提高模型泛化能力
4-激活函数使用[非单调激活函数 x * Sigmoid(x)]
"""


import math
import torch
import torch.nn as nn
import numpy as np


class UNet(nn.Module):
    def __init__(self,
            en_out_cs,
            en_downs,
            en_skips,
            en_att_heads,

            de_out_cs,
            de_ups,
            de_skips,
            de_att_heads,

            t_out_c,
            vae_c,
            block_deep,
    ):
        super(UNet, self).__init__()
        # 编码器
        self.encoder = Encoder(
            model_in_c=vae_c,
            out_cs=en_out_cs,
            down_samples=en_downs,
            skip_outs=en_skips,
            att_num_heads=en_att_heads,
            t_in_c=t_out_c,
            block_deep=block_deep,
        )
        # 解码器
        self.decoder = Decoder(
            in_c=en_out_cs[-1],
            model_out_c=vae_c,
            out_cs=de_out_cs,
            up_samples=de_ups,
            skip_outs=de_skips,
            att_num_heads=de_att_heads,
            t_in_c=t_out_c,
            block_deep=block_deep,
        )
        self.t_encoder = TEncoder(t_out_c)

    def forward(self, x, t):
        t = self.t_encoder(t)
        # print("encoded_t:", torch.mean(t), torch.std(t))
        # print("t:", t.shape)
        encoder_out = self.encoder(x, t)
        # print("encode:")
        # for e in encoder_out:
        #     print(e.shape)
        decoder_out = self.decoder(encoder_out, t)
        # print("decoder:", decoder_out.shape)
        return decoder_out

class TEncoder(nn.Module):
    def __init__(
            self,
            out_c,
            scale=30.,
    ):
        super(TEncoder, self).__init__()
        self.out_c = out_c
        self.W = nn.Parameter(torch.randn(out_c//2)*scale, requires_grad=False)
        self.linear = nn.Sequential(
            nn.Linear(out_c, out_c),
            Swish(),
            nn.Linear(out_c, out_c),
        )

    def forward(self, x):
        proj = self.timestep_embedding(x)[:, 0, :]
        return self.linear(proj)

    def timestep_embedding(self, time_steps, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param time_steps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = self.out_c // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(time_steps.device)
        args = time_steps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.out_c % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class Encoder(nn.Module):
    """
    编码器
    """
    def __init__(
            self,
            model_in_c,
            out_cs,
            down_samples,
            skip_outs,     # UNet的条链
            att_num_heads,
            t_in_c,
            block_deep,
    ):
        super(Encoder, self).__init__()
        self.skip_outs = skip_outs
        self.model_blocks = nn.ModuleList()
        for i, (out_c, down_sample, att_nun_head) in enumerate(zip(out_cs, down_samples, att_num_heads)):
            in_c = model_in_c if i == 0 else out_cs[i-1]
            self.model_blocks.append(
                EncoderBlock(in_c, out_c, kernel_size=3, stride=down_sample+1, t_in_c=t_in_c, att_num_head=att_nun_head, block_deep=block_deep)
            )

    def forward(self, x, t):
        results = []
        for i, block in enumerate(self.model_blocks):
            x = block(x, t)
            if self.skip_outs[i] == 1:
                results.append(x)
        return results

class Decoder(nn.Module):
    def __init__(
            self,
            in_c,
            model_out_c,
            out_cs,
            up_samples,
            skip_outs,
            att_num_heads,
            t_in_c,
            block_deep,
    ):
        super(Decoder, self).__init__()
        self.skip_outs = skip_outs
        self.decoder_blocks = nn.ModuleList()
        for i, (out_c, up_sample, att_num_head) in enumerate(zip(out_cs, up_samples, att_num_heads)):
            if self.skip_outs[i] == 1 and i > 0:
                in_c *= 2   # 以stride=2来进行卷积、
            self.decoder_blocks.append(
                DecoderBlock(in_c, out_c, kernel_size=3, up_sample=up_sample, t_in_c=t_in_c, att_num_head=att_num_head, block_deep=block_deep)
            )
            in_c = out_c
        self.conv = nn.Conv2d(out_cs[-1], model_out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t):
        result = None
        for i, block in enumerate(self.decoder_blocks):
            if self.skip_outs[i] == 1:
                if i == 0:
                    result = x.pop()
                else:
                    result = torch.cat([result, x.pop()], dim=1)    # 链条内容合并
            result = block(result, t)
        return self.conv(result)

class EncoderBlock(nn.Module):
    """
    # 编码块
    """
    def __init__(
            self,
            in_c,
            out_c,
            kernel_size,
            stride,
            t_in_c,
            att_num_head,
            block_deep,
    ):
        super(EncoderBlock, self).__init__()
        self.deep = block_deep
        padding = kernel_size // 2

        # 注意力机制
        if att_num_head != 0:
            self.att_block = AttentionBlock(out_c, num_head=att_num_head)
        else:
            self.att_block = nn.Identity()  # 不作处理，返回输入值的结构
        # 卷积块
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(
            Conv_GroupNorm(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        )
        for _ in range(self.deep - 2):  # -2:去头、去尾
            self.conv_blocks.append(
                Conv_GroupNorm(out_c, out_c, kernel_size=kernel_size, stride=1, padding=padding)
            )
        self.conv_blocks.append(
            Conv_GroupNorm(out_c, out_c, kernel_size=kernel_size, stride=1, padding=padding, is_activate=False)
        )
        # 编码时间-t，全连接
        self.encode_t = nn.ModuleList(
            [nn.Linear(t_in_c, out_c) for _ in range(len(self.conv_blocks) - 1)]
        )
        # 链条卷积
        if in_c != out_c or stride != 1:
            self.conv_skip = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0)
        else:
            self.conv_skip = nn.Identity()
        # 链条激活函数
        self.act_skip = Swish()

    def forward(self, x, t):
        skip = self.conv_skip(x)
        for i, conv in enumerate(self.conv_blocks):
            x = conv(x)
            if i == 0:
                x = self.att_block(x)   # 添加注意力机制
            if i < self.deep - 1:
                t_ = self.encode_t[i](t)    # 全连接层
                t_ = t_[:, :, None, None]
                x = x + t_
        return self.act_skip(x + skip)  # 残差

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_c,
            out_c,
            kernel_size,
            up_sample,
            t_in_c,
            att_num_head,
            block_deep,
    ):
        super(DecoderBlock, self).__init__()
        self.block_deep = block_deep
        self.conv_blocks = nn.ModuleList()

        if up_sample == "subpix":
            self.conv_blocks.append(
                UpSample_SubPix(in_c, out_c, kernel_size=3)
            )
            self.upSample = Conv_SubPixel(in_c, in_c, kernel_size=3)
        elif up_sample == "convt":
            self.conv_blocks.append(
                UpSample_ConvT(in_c, out_c, kernel_size=4, stride=2, padding=1)
            )
            self.upSample = nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, padding=1)
        else:
            self.conv_blocks.append(
                Conv_GroupNorm(in_c, out_c, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            )
            self.upSample = nn.Identity()
        for _ in range(self.block_deep-2):
            self.conv_blocks.append(
                Conv_GroupNorm(out_c, out_c, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            )
        self.conv_blocks.append(
            Conv_GroupNorm(out_c, out_c, kernel_size=kernel_size, stride=1, padding=kernel_size//2, is_activate=False)
        )

        if att_num_head != 0:
            self.att_block = AttentionBlock(out_c, num_head=att_num_head)
        else:
            self.att_block = nn.Identity()

        self.encode_t = nn.ModuleList(
            [nn.Linear(t_in_c, out_c) for _ in range(len(self.conv_blocks)-1)]
        )

        self.conv_skip = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.act_skip = Swish()

    def forward(self, x, t):
        skip = self.conv_skip(self.upSample(x))     # 结果特征先上采样，再卷积会原本大小
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            if i == 0:
                x = self.att_block(x)
            if i < self.block_deep-1:
                t_ = self.encode_t[i](t)
                t_ = t_[:, :, None, None]
                x = x + t_
        return self.act_skip(x + skip)

class AttentionBlock(nn.Module):
    """
    允许空间位置相互关注的注意力块。
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(self, channels, num_head=-1, use_checkpoint=False):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.num_head = num_head if num_head != -1 else min(channels // 32, 8)
        self.use_checkpoint = use_checkpoint

        self.norm = nn.GroupNorm(16, channels, eps=1e-6)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = ZeroModule(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_head, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])

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
    非单调激活函数
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def ZeroModule(module):
    """
    将Module中模块参数权重置0
    """
    for p in module.parameters():
        p.detach().zero_()
    return module



if __name__ == "__main__":
    import cv2, onnx

    def merge_images(images: np.ndarray):
        """
        合并图像
        :param images: 图像数组
        :return: 合并后的图像数组
        """
        n, h, w, c = images.shape
        nn = int(np.ceil(n ** 0.5))
        merged_image = np.zeros((h * nn, w * nn, 3), dtype=images.dtype)
        for i in range(n):
            row = i // nn
            col = i % nn
            merged_image[row * h:(row + 1) * h, col * w:(col + 1) * w, :] = images[i]

        merged_image = np.clip(merged_image, 0, 255)
        merged_image = np.array(merged_image, dtype=np.uint8)
        return merged_image


    config = {  # 模型结构相关
        # 编码器参数
        "en_out_c": (256, 256, 256, 320, 320, 320, 576, 576, 576, 704, 704, 704),
        "en_down": (0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0),
        "en_skip": (0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1),
        "en_att_heads": (8, 8, 8, 0, 8, 8, 0, 8, 8, 0, 8, 8),
        # 解码器参数
        "de_out_c": (704, 576, 576, 576, 320, 320, 320, 256, 256, 256, 256),
        "de_up": ("none", "subpix", "none", "none", "subpix", "none", "none", "subpix", "none", "none", "none"),
        "de_skip": (1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0),
        "de_att_heads": (8, 8, 0, 8, 8, 0, 8, 8, 0, 8, 8),  # skip的地方不做self-attention

        "t_out_c": 256,
        "vae_c": 4,
        "block_deep": 3,
    }
    device = "cuda"
    total_step = 1000

    unet = UNet(config["en_out_c"], config["en_down"], config["en_skip"], config["en_att_heads"],
                config["de_out_c"], config["de_up"], config["de_skip"], config["de_att_heads"],
                config["t_out_c"], config["vae_c"], config["block_deep"]).to(device)

    print("总参数", sum(i.numel() for i in unet.parameters()) / 10000, "单位:万")
    print("encoder", sum(i.numel() for i in unet.encoder.parameters()) / 10000, "单位:万")
    print("decoder", sum(i.numel() for i in unet.decoder.parameters()) / 10000, "单位:万")
    print("t", sum(i.numel() for i in unet.t_encoder.parameters()) / 10000, "单位:万")

    batch_size = 2
    x = np.random.random((batch_size, config["vae_c"], 32, 32))
    t = np.random.uniform(1, total_step + 0.9999, size=(batch_size, 1))
    t = np.array(t, dtype=np.int16)
    t = t / total_step

    with torch.no_grad():
        x = torch.Tensor(x).to(device)
        t = torch.Tensor(t).to(device)
        y = unet(x, t)
        print(y.shape)

        z = y[0].cpu().numpy()
        # z = (z - np.mean(z)) / (np.max(z) - np.min(z))
        z = np.clip(np.asarray((z + 1) * 127.5), 0, 255)
        z = np.asarray(z, dtype=np.uint8)

        z = [np.tile(z[ii, :, :, np.newaxis], (1, 1, 3)) for ii in range(z.shape[0])]
        noise = merge_images(np.array(z))

        noise = cv2.resize(noise, None, fx=2, fy=2)
        cv2.imshow("noise", noise)
        cv2.waitKey(0)

    # 增加维度信息
    model_file = 'unet.onnx'
    onnx_model = onnx.load(model_file)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)