# Diffusion_UNetImprove
## 基于Diffusion原理，实现Unet + vae 实现动漫人物头像生成
![fed9cecaf6d29688e02ce5abf57fb24](https://github.com/user-attachments/assets/f66eece1-4d0d-4193-be38-90ee38e64ce2)

# Unet模型：
### 基于基础UNet++模型结构改进
#### 1-每层卷积块自定义repeats-conv来加强特征提取；
#### 2-每层卷积头新增QKV-Attention注意力机制；
#### 3-Conv后bn归一化替换为group_norm, 使用group轻量化概念，减少参数数量并提高模型泛化能力；
#### 4-激活函数使用[非单调激活函数 x * Sigmoid(x)]
#### 模型详情：net/unet_improve.py

# 模型训练：
#### · 基于yolov8-pose训练出能够提取头部特征的模型来获取训练集数据；
#### · loss函数使用均方损失函数（MSELoss）；
#### · optimizer使用adamW，吸取动量变化；
#### · 使用预热启动和余弦退火实现对学习率(LearningRate)在训练过程中动态调整;
#### · 实现EMA(ExponentialMovingAverage)模型可梯度变化参数平均移动，实现记录unet模型中可梯度变化参数和更新;
#### · 基于EMA记录的权重来进行推理；

#### 模型训练1000+后，loss值在0.15左右，任由下降趋势；
![e37e342a1e2acac6c2e68ec03f99e40](https://github.com/user-attachments/assets/749dc6e3-9b74-43ea-a85a-195d1485dfe1)


# 推理：


