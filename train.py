import sys
sys.path.append('..')  # 添加上级父目录到搜索路径

import os
import torch
import wandb
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader

import common
from defines import *
from net import UNet
from utils import (
    EMA,
    CosineAnnealingWarmBootingLR,
)
from vae import (
    PretrainVae,
    VAE,
)
from dataloader import UDataset, DataIter
from test_sample import sample

# pip list --format=freeze > requirements.txt
# nohup python train.py > output.log &


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
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--batch_size', default=8 if not CHECK_LEARN else 1)
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
    parser.add_argument("--save_best_loss", default="./weight/best_loss.txt")
    parser.add_argument("--load_check_point", default="./weight/nvae_uncontrol1_adam/unc_unet_ep_last.pth")
    parser.add_argument("--ema_check_point", default="./weight/nvae_uncontrol1_adam/unc_unet_ema_last.pth")
    parser.add_argument("--optimizer_check_point", default="./weight/nvae_uncontrol1_adam/optimizer_last.pth")

    # 加噪
    parser.add_argument("--total_step", default=1000)
    parser.add_argument("--beta_schedule_name", default="scaled_linear")
    parser.add_argument("--normal_t", default=True)
    parser.add_argument("--random_t", default=True)
    # 样图
    parser.add_argument("--is_sample", default=False)
    parser.add_argument("--save_img_step", default=1)
    parser.add_argument("--sample_img_step", default=1000)
    parser.add_argument("--use_pretrain_vae", default=True)

    return parser.parse_args()

def wandb_init(config):
    if not USE_WANDB:
        return
    name = "nvae_d2"
    wandb.init(
        project="diffusion_unet",
        config=config,
        name=name,
        id=wandb.util.generate_id(),
    )


def get_best_loss(loss_path):
    if not os.path.exists(loss_path):
        with open(loss_path, 'w') as file:
            file.write(str(np.float32(100.0)))
            file.close()
    with open(loss_path, 'r') as file:
        loss = file.read()
        file.close()
    return np.float32(loss)


def val_loss_and_sample(unet, ema, val_dataloaders, vae, wandb_log, epoch, opt):
    if ema is not None:
        ema.Apply_shadow()  # 换成ema的参数
        print("ema sample...")
        if opt.is_sample:
            for shape in [32, 64]:
                imgs = sample(unet,
                              vae,
                              batch_size=8 if shape == 32 else 4,
                              latten_shape=shape,
                              epoch=1,
                              step=opt.sample_img_step,
                              beta_schedule_name=opt.beta_schedule_name,
                              hold_xt=True,
                              normal_t=opt.normal_t,
                              istrain=True, device=DEVICE)
                img = common.merge_images(imgs)
                img = wandb.Image(img, caption="epoch:{}".format(epoch))
                wandb_log.update({f"sample_img_ema{'' if shape == 32 else 512}": img})

        # 计算ema的loss
        print("计算ema的loss...")
        with torch.no_grad():
            ema_loss = 0.
            count = 30
            data_iter = DataIter(val_dataloaders)
            for c, (x_t, t, noise) in enumerate(tqdm(data_iter), start=1):
                x_t = x_t.float().to(DEVICE)
                t = t.float().to(DEVICE)
                noise = noise.float().to(DEVICE)
                # t要归一化
                if opt.normal_t:
                    t = t / opt.total_step
                n_ = unet(x_t, t)
                loss = loss_fn(n_, noise)

                print("ema-loss:", loss.item())
                ema_loss += loss.item()
                if c == count:
                    break

            wandb_log.update({"ema loss": ema_loss / c})
            print(f"ema loss:{ema_loss / c}")
            ema.Restore()  # 换回更新的参数

    # if opt.is_sample:
    #     print("sample...")
    #     for shape in [32, 64]:
    #         imgs = sample(unet,
    #                       vae,
    #                       batch_size=8 if shape == 32 else 4,
    #                       latten_shape=shape,
    #                       epoch=1,
    #                       step=opt.sample_img_step,
    #                       beta_schedule_name=opt.beta_schedule_name,
    #                       hold_xt=True,
    #                       normal_t=opt.normal_t,
    #                       istrain=True, device=DEVICE)
    #         img = common.merge_images(imgs)
    #         img = wandb.Image(img, caption="epoch:{}".format(epoch))
    #         wandb_log.update({f"sample_img_model{'' if shape == 32 else 512}": img})

    # # 计算val的loss
    # print("计算val的loss...")
    # with torch.no_grad():
    #     val_loss = 0.
    #     count = 30
    #     for c, (x_t, t, noise) in enumerate(tqdm(val_dataloader), start=1):
    #         x_t = x_t.float().to(device)
    #         t = t.float().to(device)
    #         noise = noise.float().to(device)
    #         # t要归一化
    #         if config["normal_t"]:
    #             t = t / config["total_step"]
    #         n_ = unet(x_t, t)
    #         loss = loss_fn(n_, noise)
    #
    #         print("loss:", loss.item())
    #         val_loss += loss.item()
    #         if c == count:
    #             break
    #
    #     wandb_log.update({"val loss": val_loss / c})
    #     print(f"val loss:{val_loss / c}")

    return wandb_log


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

    # 损失函数
    loss_fn = torch.nn.MSELoss()
    loss_best = get_best_loss(opt.save_best_loss)

    # 优化器
    if opt.optimizer == "adamw":
        optimizer = torch.optim.AdamW(unet.parameters(), lr=opt.learning_rate)
    elif opt.optimizer == "lion":
        from lion_pytorch import Lion
        optimizer = Lion(unet.parameters(), lr=opt.learning_rate)
    elif opt.optimizer == "prodigy":
        from prodigyopt import Prodigy
        optimizer = Prodigy(unet.parameters(), lr=opt.learning_rate)
    else:
        raise f"{opt.optimizer}未知optimizer"

    # learning_rate调度器（结合余弦退火（cosine-annealing））
    scheduler = CosineAnnealingWarmBootingLR(
        optimizer,
        epochs=opt.epochs,
        eta_min=opt.eta_min,
        steps=opt.lr_scheduler_step,
        step_scale=opt.lr_scheduler_step_scale,
        last_epoch=opt.start_epoch,
    )
    print("now lr:", float(optimizer.param_groups[0]['lr']))

    # 手动加载模型
    if opt.load_check_point is not None and False:
        if opt.use_ema and opt.ema_check_point is not None:
            unet = common. modelLoad(unet, opt.ema_check_point)
            ema.Register()
        unet = common.modelLoad(unet, opt.load_check_point)

    # 加载优化器参数
    if opt.optimizer_check_point and False:
        optimizer.load_state_dict(torch.load(opt.optimizer_check_point))
        if opt.optimizer_init_lr:
            for p in optimizer.param_groups:
                p["lr"] = opt.learning_rate
                p["initial_lr"] = opt.learning_rate
    print("start-lr:", float(optimizer.param_groups[0]['lr']))

    wandb_init(config=vars(opt))  # vars: parse_args转dict)

    dataset256 = UDataset(
        opt.total_step,
        opt.beta_schedule_name,
        data_size=256,
        random_t=opt.random_t,
    )
    dataloader256 = DataLoader(dataset256, batch_size=opt.batch_size, shuffle=True)

    dataset512 = UDataset(
        opt.total_step,
        opt.beta_schedule_name,
        data_size=512,
        random_t=opt.random_t,
    )
    dataloader512 = DataLoader(dataset512, batch_size=opt.batch_size//5, shuffle=True)

    val_dataset256 = UDataset(
        opt.total_step,
        opt.beta_schedule_name,
        data_size=256,
        random_t=opt.random_t,
    )
    val_dataloader256 = DataLoader(val_dataset256, batch_size=opt.batch_size, shuffle=True)

    val_dataset512 = UDataset(
        opt.total_step,
        opt.beta_schedule_name,
        data_size=512,
        random_t=opt.random_t,
    )
    val_dataloader512 = DataLoader(val_dataset512, batch_size=opt.batch_size//5, shuffle=True)

    for epoch in range(opt.start_epoch, opt.epochs):
        e_st = time()
        epoch_loss = 0
        data_iter = DataIter([dataloader256, dataloader512])
        data_len = len(data_iter)

        print("now epoch", epoch)
        show_GPU = True
        for x_t, t, noise in tqdm(data_iter):
            x_t = x_t.float().to(DEVICE)
            t = t.float().to(DEVICE)
            noise = noise.float().to(DEVICE)

            # 显示GPU占用情况
            if show_GPU:
                common.ShowGPU()
                show_GPU = False

            if opt.normal_t:
                t = t / opt.total_step
            n_ = unet(x_t, t)
            loss = loss_fn(n_, noise)

            print("\n loss:", loss.item())
            epoch_loss += loss.item()

            if CHECK_LEARN: # 可视化loss
                common.View_loss(n_, noise, t)
                optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), opt.clip_grad_norm)
                optimizer.step()
                if opt.use_ema:
                    ema.Update()
                optimizer.zero_grad()

        if opt.start_use_scheduler <= epoch:
            scheduler.step()
            print(" scheduler.step ")

        loss_mean = epoch_loss / data_len
        print("loss mean:", epoch_loss / data_len)
        print("now lr:", optimizer.param_groups[0]["lr"])
        wandb_log = {"loss": loss_mean, "lr": optimizer.param_groups[0]["lr"]}

        # 记录best_loss
        if loss_best > loss_mean:
            with open(opt.save_best_loss, 'w') as file:
                file.write(str(loss_mean))
                file.close()
            common.modelSave(unet, opt.model_save_path, f"{opt.model_name}_best.pth")
            common.modelSave(optimizer, opt.model_save_path, "optimizer_best.pth")

        # 测试模型并产生图片
        if epoch % opt.save_img_step == 0:
            wandb_log = val_loss_and_sample(unet, ema, [val_dataloader256, val_dataloader512], vae, wandb_log, epoch, opt)
        if USE_WANDB:
            wandb.log(wandb_log)

        # 保存模型参数
        if epoch > 0 and epoch % opt.save_model_step == 0:
            if opt.use_ema:
                ema.Apply_shadow()
                common.modelSave(unet, opt.model_save_path, f"{opt.model_name}_ema.pth")
                ema.Restore()  # 换回更新的参数
            common.modelSave(unet, opt.model_save_path, f"{opt.model_name}_ep{epoch if opt.save_every_check_point else 'n'}.pth")
            common.modelSave(optimizer, opt.model_save_path, "optimizer.pth")

    # 保存最后的模型
    if opt.use_ema:
        ema.Apply_shadow()
        common.modelSave(unet, opt.model_save_path, f"{opt.model_name}_ema_last.pth")
        ema.Restore()  # 换回更新的参数
    common.modelSave(unet, opt.model_save_path, f"{opt.model_name}_ep_last.pth")
    common.modelSave(optimizer, opt.model_save_path, "optimizer_last.pth")

    if USE_WANDB:
        wandb_log = val_loss_and_sample(unet, ema, [val_dataloader256, val_dataloader512], vae, {}, epoch, opt)
        wandb.log(wandb_log)
        wandb.finish()