"""
根据训练完毕的权重unet模型数据，使用ddim去噪扩散隐式模型来实现生成图片
"""


from tqdm.auto import trange
from plotly import graph_objects as go

from common import *
from net import UNet
from vae import PretrainVae

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "scaled_linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def get_afa_bars(beta_schedule_name, total_step):
    """
    生成afa bar的列表,列表长度为total_step
    :param beta_schedule_name: beta_schedule
    :return: afa_bars和betas
    """

    # if linear:
    #     # 线性
    #     betas = np.linspace(1e-5, 0.1, self.total_step)
    #
    # else:
    #     # sigmoid
    #     betas = np.linspace(-6, 6, self.total_step)
    #     betas = 1 / (1 + np.exp(betas)) * (afa_max - afa_min) + afa_min

    betas = get_named_beta_schedule(beta_schedule_name, total_step)

    afas = 1 - betas
    afas_cumprod = np.cumprod(afas)
    # afas_cumprod = np.concatenate((np.array([1]), afas_cumprod[:-1]), axis=0)
    return afas_cumprod, betas

@torch.no_grad()
def simply_sampler(model, x, betas, afa_bars, normal_t, istrain=False): # 简单采样器
    """
    :param model: 模型
    :param x: 最初的输入
    :param betas: 加噪时的beta
    :param afa_bars: 加噪时的afa_bar
    :param istrain: 如果是true则只在最后有输出
    :return:
    """
    total_step = len(betas)
    batch_size = x.shape[0]

    betas = torch.tensor(betas).to(x.device)
    afa_bars = torch.tensor(afa_bars).to(x.device)

    z_t_s = []

    if not istrain:
        # 画图用
        coeffs = []
        model_outputs = []
        xs = []
        x_coffsOuts = []  # x - coeff * z_t
        coeff_z_t = []  # coeff * z_t
        vs = []  # 方差
    """
    使用 for 循环从 total_step 到 1 逆序迭代。在每次迭代中：
        计算当前步的时间参数 t。
        计算系数 coeff，这个系数将用于调整模型输出 z_t。
        通过模型获取当前步的输出 z_t。
        如果是第一步，直接使用均值更新 x；如果是后续步骤，使用一个随机过程更新 x，该过程考虑了方差 v。
    """
    for i in trange(total_step, 0, -1):  # , disable=None
        if normal_t:
            t = torch.tensor([i / total_step])[None, :].to(x.device)
        else:
            t = torch.tensor([i])[None, :].to(x.device)
        t = torch.tile(t, (batch_size, 1))

        coeff = betas[i - 1] / (torch.sqrt(1 - afa_bars[i - 1]))  # + 1e-5
        z_t = model(x, t)
        # z_t.clamp(-1, 1)
        z_t_s.append(z_t.cpu().numpy())
        mean = (1 / torch.sqrt(1 - betas[i - 1])) * (x - coeff * z_t)

        if i == 1:
            x = mean
        else:
            v = (1 - afa_bars[i - 2]) / (1 - afa_bars[i - 1]) * betas[i - 1]
            x = torch.sqrt(v) * torch.randn_like(x) + mean

        if not istrain:
            # print("coeff:", coeff)
            # print("(x - coeff * z_t):", torch.mean((x - coeff * z_t)))
            # print("1 / torch.sqrt(1 - betas[i - 1]):", 1 / torch.sqrt(1 - betas[i - 1]))
            # print(i, "x_mean:", torch.mean(x).item(), torch.std(x).item())
            # print("z_mean", torch.mean(z_t))
            # print("=" * 60)
            coeffs.append(torch.mean(coeff).item())
            model_outputs.append(torch.mean(z_t).item())
            xs.append(torch.mean(x).item())
            coeff_z_t.append(torch.mean(coeff * z_t).item())
            x_coffsOuts.append(torch.mean((x - coeff * z_t)).item())
            vs.append(torch.mean(v).item())
            view_x_t(x)

    print("=" * 50)
    z_t_s = np.array(z_t_s)
    # print("z_t mean:", np.mean(z_t_s, axis=(2, 3)))
    z_t_s = np.reshape(z_t_s, (z_t_s.shape[0], -1))
    print("z_t方差:", np.std(z_t_s, axis=0))
    print("=" * 50)

    if not istrain:
        fig = go.Figure()
        scatter_x = list(range(len(coeffs)))
        fig.add_trace(go.Scatter(x=scatter_x, y=coeffs, name="coeffs"))
        fig.add_trace(go.Scatter(x=scatter_x, y=model_outputs, name="model_outputs"))
        fig.add_trace(go.Scatter(x=scatter_x, y=xs, name="xt"))
        fig.add_trace(go.Scatter(x=scatter_x, y=x_coffsOuts, name="x - coeff * z"))
        fig.add_trace(go.Scatter(x=scatter_x, y=coeff_z_t, name="coeff * z"))
        fig.add_trace(go.Scatter(x=scatter_x, y=vs, name="v"))
        fig.show()
    # x = (x - torch.mean(x)) / torch.std(x)
    return x

def view_x_t(x):
    # print("x_mean:", torch.mean(x).item(), torch.std(x).item())
    z = x[0].cpu().numpy()
    z = (z - np.mean(z)) / np.std(z)
    z = np.clip(np.asarray((z + 1) * 127.5), 0, 255)
    z = np.asarray(z, dtype=np.uint8)

    z = [np.tile(z[ii, :, :, np.newaxis], (1, 1, 3)) for ii in range(z.shape[0])]
    noise = merge_images(np.array(z))

    noise = cv2.resize(noise, None, fx=2, fy=2)
    cv2.imshow("模型输出", noise)
    cv2.waitKey(1)


@torch.no_grad()
def sample(unet, vae, batch_size, latten_shape, epoch, step, beta_schedule_name, hold_xt, normal_t, istrain, device):
    """

    :param unet: 模型
    :param vae: vae
    :param batch_size:
    :param epoch:
    :param step: 运行的step
    :param beta_schedule_name:
    :param hold_xt: 是否固定最初的输入
    :param normal_t: 是否归一化t
    :param istrain: 如果是true则只在最后有输出
    :param device:
    :return:
    """
    afa_bars, betas = get_afa_bars(beta_schedule_name, step)
    res = None
    holded_xt = torch.randn((batch_size, vae.middle_c, latten_shape, latten_shape)).to(device)

    # data_mean = torch.Tensor(np.load("../data/data_mean.npy")).to(device)

    for e in range(epoch):
        print(f"sample-epoch: {e}/{epoch}")
        # 给定一个噪声xt
        xt = torch.randn((batch_size, vae.middle_c, latten_shape, latten_shape)).to(device)
        # xt += data_mean
        if hold_xt:
            xt = holded_xt
        # xt = torch.clip(xt, -1, 1)
        # 通过simply_sampler函数生成一个去噪后的潜在表示x0
        x0 = simply_sampler(unet, xt, betas, afa_bars, normal_t, istrain=istrain)

        # afa_bars = afa_bars[::-1].copy()
        # x0 = sample_dpmpp_2m_test(unet, xt, afa_bars)
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


if __name__ == '__main__':
    import train
    opt = train.parse_opt()

    unet = UNet(
        opt.en_out_c, opt.en_down, opt.en_skip, opt.en_att_heads,
        opt.de_out_c, opt.de_up, opt.de_skip, opt.de_att_heads,
        opt.t_out_c, opt.vae_c, opt.block_deep,
    ).to(DEVICE)

    if opt.use_ema or True:
        unet = modelLoad(unet, "./weight/nvae_uncontrol2_lion/unc_unet_ema_best.pth")
    unet = modelLoad(unet, "./weight/nvae_uncontrol2_lion/unc_unet_best.pth")

    vae = PretrainVae()

    imgs = sample(
        unet,
        vae,
        batch_size=1,
        latten_shape=32,
        epoch=2,
        step=opt.sample_img_step,
        beta_schedule_name=opt.beta_schedule_name,
        hold_xt=True,
        normal_t=opt.normal_t,
        istrain=True,
        device=DEVICE,
    )

    print("sample-down")

    img = merge_images(imgs)
    img = img[:, :, ::-1]
    cv2.imshow("imgs", img)
    cv2.waitKey(0)