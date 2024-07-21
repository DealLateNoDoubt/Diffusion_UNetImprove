import torch
from diffusers import AutoencoderKL
from defines import DEVICE


class PretrainVae(object):
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained(   # 加载官方已训练模型
            "gsdf/Counterfeit-V2.5",  # segmind/small-sd
            subfolder="vae",
            cache_dir="./vae/pretrain_vae"
        ).to(DEVICE)
        self.vae.requires_grad_(False)
        self.middle_c = 4

    def encoder(self, x):
        latents = self.vae.encode(x)
        latents = latents.latent_dist
        mean = latents.mean
        var = latents.var
        return mean, var

    def decoder(self, latents):
        latents = latents / 0.18215
        output = self.vae.decode(latents).sample
        return output

    # 释放encoder
    def Res_encoder(self):
        del self.vae.encoder
        torch.cuda.empty_cache()




if __name__ == "__main__":
    test = PretrainVae()
    print(test)