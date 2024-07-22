import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

def betas_for_alpha_bar(num_diffusion_time_steps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_time_steps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_time_steps):
        t1 = i / num_diffusion_time_steps
        t2 = (i + 1) / num_diffusion_time_steps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_time_steps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_time_steps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_time_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.002
        return np.linspace(beta_start, beta_end, num_diffusion_time_steps, dtype=np.float64)
    elif schedule_name == "scaled_linear":
        scale = 1000 / num_diffusion_time_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.002
        return np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_time_steps, dtype=np.float64) ** 2
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_time_steps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class UDataset(Dataset):
    def __init__(
            self,
            total_step,
            beta_schedule_name,
            data_size=256,
            random_t=True,
            train=True,
            val_size=0.3,
    ):
        """
        :param total_step: 总共多少步
        :param beta_schedule_name: 学习率调度器的名称
        :param random_t: 是否随机t,若否,t递增
        :param train: 是否训练集
        """
        super(Dataset, self).__init__()
        self.data = np.load(f"./datasets/data15_sdvae_{data_size}.npy")
        self.data = np.concatenate([self.data, np.load(f"./datasets/data18_sdvae_{data_size}.npy")], axis=0)
        if not train:
            val_size = int(val_size * len(self.data))
            indices = np.random.choice(len(self.data), size=val_size, replace=False)
            self.data = self.data[indices]

        self.data = self.data * 0.18215  # vae scaling_factor 归一化处理

        self.total_step = total_step
        self.afa_bars, self.betas = self.get_afa_bars(beta_schedule_name)
        self.random_t = random_t
        self.big_small_t_switch = True  # 用于平衡t的大小,采用一个大一个小的策略
        self.last_t = None

    def get_afa_bars(self, beta_schedule_name):
        """
        生成afa bar的列表,列表长度为total_step
        :param beta_schedule_name: beta_schedule
        :return: afa_bars和betas
        """
        betas = get_named_beta_schedule(beta_schedule_name, self.total_step)
        afas = 1 - betas
        afas_cumprod = np.cumprod(afas)
        # afas_cumprod = np.concatenate((np.array([1]), afas_cumprod[:-1]), axis=0)
        return afas_cumprod, betas


    def apple_noise(self, data, step):
        """
        添加噪声,返回xt和噪声
        :param data: 数据,就是x0
        :param step: 选择的步数
        :return:
        """
        noise = np.random.normal(size=data.shape)
        afa_bar_t = self.afa_bars[step - 1]
        x_t = np.sqrt(afa_bar_t) * data + np.sqrt(1 - afa_bar_t) * noise
        return x_t, noise


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, item):
        if self.random_t:
            if self.big_small_t_switch:
                t = np.random.uniform(1, self.total_step + 0.9999, size=(1,))  # t [1,total_step]
                self.last_t = t
            else:
                t = self.total_step - self.last_t + 1.9999
            self.big_small_t_switch = not self.big_small_t_switch
        else:
            t = [self.total_step]
            self.t = max(1, self.total_step - 1)
        t = np.array(t, dtype=np.int16)
        x_t, noise = self.apple_noise(self.data[item if self.random_t else 3], t)
        return x_t, t, noise



class DataIter(object):
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.iter_list = [i for i in range(len(self.dataloaders)) for _ in range(len(self.dataloaders[i]))]
        self.data_len = len(self.iter_list)
        np.random.shuffle(self.iter_list)
        self.iters = [dl.__iter__() for dl in self.dataloaders]
        self.cursor = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor + 1 < self.data_len:
            self.cursor += 1
            return next(self.iters[self.iter_list[self.cursor]])
        else:
            raise StopIteration()

    def __len__(self):
        return self.data_len