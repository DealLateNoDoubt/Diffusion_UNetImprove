import os
import cv2
import torch
import wandb
import GPUtil
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from defines import DEVICE


def modelSave(model, save_path, save_name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


def modelLoad(model, model_path, data_parallel=False):
    model.load_state_dict(torch.load(model_path), strict=True)
    if data_parallel:
        model = nn.DataParallel(model)
    return model


def merge_images(images: np.ndarray):
    n, h, w, c = images.shape
    n_n = int(np.ceil(n ** 0.5))
    merge_image = np.zeros((h * n_n, w * n_n, 3), dtype=images.dtype)
    for i in range(n):
        row = i // n_n
        col = i % n_n
        merge_image[row * h:(row + 1) * h, col * w:(col + 1) * w, :] = images[i]

    merge_image = np.clip(merge_image, 0, 255)
    merge_image = np.array(merge_image, dtype=np.uint8)
    return merge_image


def View_loss(y_, target, t):
    import cv2
    y_ = y_[0].detach().cpu().numpy()
    target = target[0].detach().cpu().numpy()
    loss = y_ - target
    print(f"myl2loss:{np.mean(loss ** 2)}, t:{t}")
    y_ = show_model_output(y_, std=True)
    target = show_model_output(target, std=True)
    loss_view = show_model_output(loss, std=True)

    cv2.imshow("loss", loss_view)
    cv2.imshow("y_", y_)
    cv2.imshow("target", target)
    cv2.waitKey(0)

def ShowGPU():
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"\nGPU {gpu.id}:")
        print(f"Free Memory: {gpu.memoryFree} bytes")
        print(f"Used Memory: {gpu.memoryUsed} bytes")
        print(f"Total Memory: {gpu.memoryTotal} bytes")

def show_model_output(z, std=False):
    z = (z - np.mean(z))  # / np.std(z)
    if std:
        z /= np.std(z)
    z = np.clip(np.asarray((z + 1) * 127.5), 0, 255)
    z = np.asarray(z, dtype=np.uint8)

    z = [np.tile(z[ii, :, :, np.newaxis], (1, 1, 3)) for ii in range(z.shape[0])]
    img = merge_images(np.array(z))
    img = cv2.resize(img, None, fx=2, fy=2)
    return img