import onnx
import onnxruntime
import argparse
import numpy as np
import torch.onnx
from PIL import Image
from tqdm.auto import trange

import common
from net import UNet
from defines import *
from utils import EMA
from vae import PretrainVae
from test_sample import sample

def parse_opt():
    parser = argparse.ArgumentParser()

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
    # EMA
    parser.add_argument("--ema_rate", default=0.9999)

    # path
    parser.add_argument("--unet_pt", default="./weight/unc_unet_ep.pth")
    parser.add_argument("--ema_pt", default="./weight/unc_unet_em.pth")

    return parser.parse_args()

def get_afa_bars(total_step):
    """
    生成afa bar的列表,列表长度为total_step
    :param beta_schedule_name: beta_schedule
    :return: afa_bars和betas
    """
    scale = 1000 / total_step
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, total_step, dtype=np.float64) ** 2

    afas = 1 - betas
    afas_cumprod = np.cumprod(afas)
    # afas_cumprod = np.concatenate((np.array([1]), afas_cumprod[:-1]), axis=0)
    return afas_cumprod, betas

# region onnx-predict
def to_onnx(unet, ema):
    # ema-权重
    ema.Apply_shadow()
    unet.eval()     # 推理模式

    model_file = "./weight/unet-ema.onnx"
    x = torch.randn((16, 4, 32, 32)).to(DEVICE)
    t = np.random.uniform(1, 1000 + 0.9999, size=(16, 1))
    t = np.array(t, dtype=np.int16)
    t = t / 1000
    x = torch.Tensor(x).to(DEVICE)
    t = torch.Tensor(t).to(DEVICE)

    # with torch.no_grad():
    #     y = unet(x, t)
    #     print(y.shape, type(y))
    #
    # ort_session = onnxruntime.InferenceSession("./weight/unet-ema.onnx")
    # z_t = ort_session.run(None, {
    #     ort_session.get_inputs()[0].name: x.cpu().numpy(),
    #     ort_session.get_inputs()[1].name: t.cpu().numpy(),
    # })
    # z_t = torch.Tensor(z_t[0])
    # print(z_t.shape, type(z_t))

    torch.onnx.export(
        unet,
        (x, t),
        model_file,
        export_params=True,     # 是否可训练
    )
    onnx_model = onnx.load(model_file)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)

def to_onnx_predict(vae):
    # unet-predict
    unet_img = to_onnx_sample("./weight/unet.onnx", vae)
    unet_img.save("unet_img.png")

    # ema-predict
    ema_img = to_onnx_sample("./weight/unet-ema.onnx", vae)
    ema_img.save("ema_img.png")

def to_onnx_sample(onnx_path, vae):
    images = onnx_sample(
        onnx_path,
        vae,
        batch_size=16,
        middle_c=4,
        latten_shape=32,
        epochs=1,
        step=1000,  # 步长1000，与dataset设置的加噪一致
    )
    img = common.merge_images(images)
    im = Image.fromarray(img)
    return im

def onnx_sample(onnx_path, vae, batch_size, middle_c, latten_shape, epochs, step):
    # 随机噪声
    xt = torch.randn((batch_size, middle_c, latten_shape, latten_shape)).to(DEVICE)
    # 高斯分布
    afa_bars, betas = get_afa_bars(step)

    ort_session = onnxruntime.InferenceSession(onnx_path)

    res = None
    for epoch in range(epochs):
        print(f"onnx-sample-epoch: {epoch}/{epochs}")

        # 通过simply_sampler函数生成一个去噪后的潜在表示x0
        x0 = onnx_simply_sampler(ort_session, xt, betas, afa_bars)
        # 使用VAE的解码器将这个潜在表示解码成可视化的图像
        x0 = vae.decoder(x0)

        if res is None:
            res = x0
        else:
            res = torch.concatenate((res, x0), dim=0)

    print("simply_sampler-down")
    res = res.cpu().numpy()
    res = np.asarray(res * 255)
    res = np.transpose(res, [0, 2, 3, 1])  # RGB
    return res

def onnx_simply_sampler(ort_session, xt, betas, afa_bars):
    total_step = len(betas)
    batch_size = xt.shape[0]

    betas = torch.tensor(betas).to(DEVICE)
    afa_bars = torch.tensor(afa_bars).to(DEVICE)

    """
    使用 for 循环从 total_step 到 1 逆序迭代。在每次迭代中：
        计算当前步的时间参数 t。
        计算系数 coeff，这个系数将用于调整模型输出 z_t。
        通过模型获取当前步的输出 z_t。
        如果是第一步，直接使用均值更新 x；如果是后续步骤，使用一个随机过程更新 x，该过程考虑了方差 v。
    """
    z_t_s = []
    for i in trange(total_step, 0, -1):
        t = torch.tensor([i / total_step])[None, :].to(DEVICE)
        t = torch.tile(t, (batch_size, 1))

        coeff = betas[i - 1] / (torch.sqrt(1 - afa_bars[i - 1]))  # + 1e-5

        z_t = ort_session.run(None, {
            ort_session.get_inputs()[0].name: xt.cpu().numpy(),
            ort_session.get_inputs()[1].name: t.cpu().numpy(),
        })
        z_t = torch.from_numpy(z_t[0]).to(DEVICE)
        z_t_s.append(z_t.cpu().numpy())
        mean = (1 / torch.sqrt(1 - betas[i - 1])) * (xt - coeff * z_t)

        if i == 1:
            xt = mean
        else:
            v = (1 - afa_bars[i - 2]) / (1 - afa_bars[i - 1]) * betas[i - 1]
            xt = torch.sqrt(v) * torch.randn_like(xt) + mean

    print("=" * 50)
    z_t_s = np.array(z_t_s)
    z_t_s = np.reshape(z_t_s, (z_t_s.shape[0], -1))
    print("z_t方差:", np.std(z_t_s, axis=0))
    print("=" * 50)

    return xt
# endregion


def to_predict(unet, ema, vae):
    # unet-predict
    unet_img = to_sample(unet, vae)
    unet_img.save("unet_img.png")

    # ema-predict
    ema.Apply_shadow()
    ema_img = to_sample(unet, vae)
    ema_img.save("ema_img.png")

def to_sample(unet, vae):
    images = sample(
        unet,
        vae,
        batch_size=16,
        latten_shape=32,
        epoch=1,
        step=1000,  # 步长1000，与dataset设置的加噪一致
        beta_schedule_name="scaled_linear",
        hold_xt=True,
        normal_t=True,
        istrain=True,
        device=DEVICE,
    )
    img = common.merge_images(images)
    im = Image.fromarray(img)
    return im



if __name__ == "__main__":
    opt = parse_opt()

    # unet
    unet = UNet(
        opt.en_out_c, opt.en_down, opt.en_skip, opt.en_att_heads,
        opt.de_out_c, opt.de_up, opt.de_skip, opt.de_att_heads,
        opt.t_out_c, opt.vae_c, opt.block_deep,
    ).to(DEVICE)

    # ema
    ema = EMA(unet, opt.ema_rate)
    ema.Register()

    # vae
    vae = PretrainVae()

    # 加载训练权重
    unet = common.modelLoad(unet, opt.ema_pt)
    ema.Register()
    unet = common.modelLoad(unet, opt.unet_pt)

    # to_onnx(unet, ema)
    # to_onnx_predict(vae)

    to_predict(unet, ema, vae)
