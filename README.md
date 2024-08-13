# Diffusion_UNetImprove
## 基于DDPM原理，实现Unet + vae 实现动漫人物头像生成
![fed9cecaf6d29688e02ce5abf57fb24](https://github.com/user-attachments/assets/f66eece1-4d0d-4193-be38-90ee38e64ce2)
DDPM流程：
1：初始化：从带噪声的图像开始
2：正向扩散：逐步向数据添加高斯噪声，直到数据完全转化为无结果的噪声
3：反向去噪：通过模型预测并逐渐去掉每一步(t)加入的噪声，还原得到无噪声的图像
4：训练：使用反向传播更新模型参数，以最小化正向和反向过程之间的差异
5：生成：生成新的噪声，应用训练好的模型来进行去噪，再使用vae来解码模型结果将数据可视化生成为图片


# Unet模型：
### 基于基础UNet++模型结构改进
#### 1-每层卷积块自定义repeats-conv来加深网络深度；
#### 2-每层卷积头新增QKV-Attention注意力机制，增强特征表示；
#### 3-Conv后bn归一化替换为group_norm, 使用group轻量化概念，减少参数数量并提高模型泛化能力；
#### 4-激活函数使用[非单调激活函数 x * Sigmoid(x)]，通过 Sigmoid 函数对输入进行缩放，可以控制激活值的大小，有助于防止梯度消失或爆炸问题；
#### 模型详情：net/unet_improve.py

# 模型训练：
#### · 基于yolov8-pose训练出能够提取头部特征的模型来获取训练集数据；
#### · 基于基础np.linspace生成步长1000的平均噪声；
#### · loss函数使用均方损失函数（MSELoss）；
#### · optimizer使用adamW，吸取动量变化；
#### · 使用预热启动和余弦退火实现对学习率(LearningRate)在训练过程中动态调整;
#### · 实现EMA(ExponentialMovingAverage)模型可梯度变化参数平均移动，实现记录unet模型中可梯度变化参数和更新;
#### · 基于EMA记录的权重来进行推理；

#### 模型训练1000+后，loss值在0.15左右，任由下降趋势；
![e37e342a1e2acac6c2e68ec03f99e40](https://github.com/user-attachments/assets/749dc6e3-9b74-43ea-a85a-195d1485dfe1)

# onnx部署
#### 将训练完毕得到的unet权重和ema权重文件，更新到Unet模型中，使用torch.onnx来将Unet模型转化为onnx文件；

# 推理：
#### 1-使用torch.randn来随机生成噪声信息，基于模型训练时步长信息生成高斯分布信息；
#### 2-使用unet-onnx部署，基于步长，将噪声信息实现去噪，得到一个去噪后的潜在表示的数据信息；
#### 3-使用vae解码器，将潜在表示的数据信息解码成可视化图像信息；
#### 4-transpose图像信息并将生成图像保存到本地从而实现动漫头像生成；
