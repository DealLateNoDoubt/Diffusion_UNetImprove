import os
import argparse

import common
from defines import *
from net import UNet
from utils import EMA
from vae import (
    PretrainVae,
    VAE,
)


def parse_opt():
    parser = argparse.ArgumentParser()
    # 训练相关
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--start_use_scheduler', default=30)
    parser.add_argument('--lr_scheduler_step', default=[500, 1000])
    parser.add_argument('--lr_scheduler_step_scale', default=0.1)
    parser.add_argument('--eta_min', default=5e-5)
    parser.add_argument('--optimizer', default="lion")
    parser.add_argument('--start_epoch', default=0)
    parser.add_argument('--epochs', default=20000)
    parser.add_argument('--batch_size', default=16 if not CHECK_LEARN else 1)
    parser.add_argument('--clip_grad_norm', default=2)
    # EMA
    parser.add_argument("--use_ema", default=True)
    parser.add_argument("--ema_rate", default=0.9999)
    # UNET
    parser.add_argument("--en_out_c", default=[256, 256, 256, 320, 320, 320, 576, 576, 576, 704, 704, 704])
    parser.add_argument("--en_down", default=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    parser.add_argument("--en_skip", default=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
    parser.add_argument("--en_att_heads", default=[8, 8, 8, 0, 8, 8, 0, 8, 8, 0, 8, 8])
    parser.add_argument("--de_out_c", default=[704, 576, 576, 576, 320, 320, 320, 256, 256, 256, 256])
    parser.add_argument("--de_up", default=["none", "subpix", "none", "none", "subpix", "none", "none", "subpix", "none", "none", "none"])
    parser.add_argument("--de_skip", default=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    parser.add_argument("--de_att_heads", default=[8, 8, 0, 8, 8, 0, 8, 8, 0, 8, 8])
    parser.add_argument("--t_out_c", default=256)
    parser.add_argument("--vae_c", default=4)
    parser.add_argument("--block_deep", default=3)
    # SAVE
    parser.add_argument("--save_model_step", default=1)
    parser.add_argument("--model_save_path", default="./weight/nvae_uncontrol2_lion")
    parser.add_argument("--model_name", default="unc_unet")
    parser.add_argument("--save_every_check_point", default=False)
    # 加噪
    parser.add_argument("--total_step", default=1000)
    parser.add_argument("--beta_schedule_name", default="scaled_linear")
    parser.add_argument("--normal_t", default=True)
    parser.add_argument("--random_t", default=True)
    # 样图
    parser.add_argument("--is_sample", default=USE_WANDB)
    parser.add_argument("--save_img_step", default=1)
    parser.add_argument("--sample_img_step", default=1000)
    parser.add_argument("--use_pretrain_vae", default=True)

    return parser.parse_args()


if __name__ == "__main__":
    # 加入wandb
    if USE_WANDB:
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    # 参数
    opt = parse_opt()
    # unet
    unet = UNet(
        opt.en_out_c, opt.en_down, opt.en_skip, opt.en_att_heads,
        opt.de_out_c, opt.de_up, opt.de_skip, opt.de_att_heads,
        opt.t_out_c, opt.vae_c, opt.block_deep,
    ).to(DEVICE)
    # ema
    if opt.use_ema:
        ema = EMA(unet, opt.ema_rate)
        ema.Register()  # 登记可梯度更新的权重
    else:
        ema = None
    # vae
    if opt.use_pretrain_vae:    # 使用官方预训练模型（自己再训练一个效率较低）
        vae = PretrainVae()
    else:
        vae_middle_c = 8
        vae = VAE(vae_middle_c).to(DEVICE).eval()
        vae = common.modelLoad(vae, os.path.join("./vae/weight/vgg_perce_gram_l2_vae_small_kl_f8_c8_bigger_model", "vae_50.pth"))
        vae.Res_encoder()

    print("DOWN")