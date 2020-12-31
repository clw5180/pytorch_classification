### pytorch 图像分类竞赛框架
参考：spytensor 

### 0.分类模型提调参技巧


### 1. 依赖库
- pretrainedmodels
- progress
- efficientnet-pytorch
- apex

### 2. 支持功能

- [x] pytorch官网模型
- [x] [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) 复现的部分模型
- [x] [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) 
- [x] fp16混合精度训练
- [x] TTA
- [x] 固定验证集/随机划分验证集
- [x] 多种优化器：adam、radam、novograd、sgd、ranger、ralamb、over9000、lookahead、lamb
- [x] OneCycle训练策略
- [x] LabelSmoothLoss
- [x] Focal Loss
- [ ] AutoAgument
  
### 3. 使用方法
1、训练
#### 训练执行 `python train.py`，
更改`config.py`中的参数， 主要修改：dataset(路径), bs, lr, epochs，input_size, gpu_id, model_name, fp16, loss_func,
注意：务必使用ImageNet预训练模型，否则最后结果会差很多

2、验证（分析验证集结果，根据真实标签）
#### 执行 `python evaluate.py`

3、预测（验证集或测试集结果输出，可视化）
#### 执行 `python inference.py`


### 4. submit_example.csv 
每一行：filename,label
样例：
```
0001.jpg,dog
0002.jpg,dog
0003.jpg,dog
```
注：预测图像可能没有label，所以label可以随意给个临时的，但一些比赛平台对都会给个提交样例，随意给个label


### 5.TODO

- [ ] 优化模型融合策略
- [ ] 优化online数据增强
- [ ] 优化pytorch官方模型调用接口
- [ ] 增加模型全连接层初始化
- [ ] 增加更多学习率衰减策略
- [ ] 增加find lr
- [ ] 增加dali
- [ ] 增加wsl模型
- [ ] 增加tensorboardX
- [ ] 优化文件夹创建
