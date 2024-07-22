"""
将dataset训练数据，每30*8为1份(减缓内存压力)，合并成多份npy，最后再合并npy堆叠成一份
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from defines import DEVICE
from vae import PretrainVae

class ImgDataset(Dataset):
    def __init__(
            self,
            path,
    ):
        super(ImgDataset, self).__init__()
        self.data_path = self.read_photo_path(path)

    def read_photo_path(self, path):
        file_paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().split(".")[-1] in ["png", "jpg"]:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        return file_paths

    def to_tenser(self, plt_img):
        img = np.array(plt_img, dtype=np.float32)
        img_mirrored = plt_img.transpose(Image.FLIP_LEFT_RIGHT)
        img_mirrored = np.array(img_mirrored, dtype=np.float32)
        data = np.array(img / 255, dtype=np.float32)
        data_mirrored = np.asarray(img_mirrored / 255, dtype=np.float32)
        return np.transpose(data, (2, 0, 1)), np.transpose(data_mirrored, (2, 0, 1))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        img = cv2.imread(self.data_path[item])
        h, w, c = img.shape
        size = min(h, w)

        # 计算裁剪区域
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        cropped_image = img[start_y: start_y+size, start_x: start_x+size]
        # 转为PIL
        plt_img = Image.fromarray(cropped_image[:, :, ::-1])
        # resize
        plt_img256 = plt_img.resize((256, 256), Image.Resampling.LANCZOS)
        plt_img512 = plt_img.resize((512, 512), Image.Resampling.LANCZOS)

        data256, data_mirrored256 = self.to_tenser(plt_img256)
        data512, data_mirrored512 = self.to_tenser(plt_img512)

        return data256, data_mirrored256, data512, data_mirrored512, int(size >= 256), int(size >= 512)


def encode(vae_model, image_paths, save_paths):
    print("encode")
    for path, save_path in zip(image_paths, save_paths):
        dataset = ImgDataset(path)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        encode_data256, encode_data512 = None, None
        with torch.no_grad():
            save_count_256 = 0
            save_count_512 = 0
            for data256, mirror256, data512, mirror512, flag256, flag512 in tqdm(dataloader):
                if len(data512.shape) > 1 and torch.sum(flag512) > 1:
                    save_count_512 += 1
                    data512 = data512[flag512 == 1]
                    encode_data512 = get_encode_data(vae_model, data512, mirror512, encode_data512)
                    if save_count_512 % 30 == 0:
                        np.save(save_path.format(size=512, count=save_count_512), encode_data512)
                        encode_data512 = None

                if len(data256.shape) > 1 and torch.sum(flag256) > 1:
                    data256 = data256[flag256 == 1]
                    encode_data256 = get_encode_data(vae_model, data256, mirror256, encode_data256)
                    save_count_256 += 1
                    if save_count_256 % 30 == 0:
                        np.save(save_path.format(size=256, count=save_count_256), encode_data256)
                        encode_data256 = None

            if not (encode_data256 is None):
                np.save(save_path.format(size=256, count=save_count_256), encode_data256)
            if not (encode_data512 is None):
                np.save(save_path.format(size=512, count=save_count_512), encode_data512)

def get_encode_data(vae_model, data, mirror_data, encode_data):
    data = data.to(DEVICE)
    mirror_data = mirror_data.to(DEVICE)
    mu, _var = vae_model.encoder(data)
    mu = mu.data.cpu().numpy()
    mu_mirror, _var_mirror = vae_model.encoder(mirror_data)
    mu_mirror = mu_mirror.data.cpu().numpy()
    mu = np.concatenate((mu, mu_mirror))
    encode_data = mu if encode_data is None else np.concatenate((encode_data, mu))
    return encode_data


def merge_npy():
    roots = ["../datasets/face-x-1.5-npy", "../datasets/face-x-1.8-npy"]
    for root in roots:
        npy_data = None
        for name in os.listdir(root):
            if name.startswith("512"):
                data = np.load(os.path.join(root, name))
                if npy_data is None:
                    npy_data = data
                else:
                    npy_data = np.concatenate([npy_data, data])
        np.save(f"{root}.npy", npy_data)


if __name__ == "__main__":
    image_paths = [r"../datasets/face-x-1.5/", r"../datasets/face-x-1.8/"]
    save_paths = ["../datasets/face-x-1.5-npy/{size}_{count}.npy", "../datasets/face-x-1.8-npy/{size}_{count}.npy"]

    vae_model = PretrainVae()
    encode(vae_model, image_paths, save_paths)
    # merge_npy()