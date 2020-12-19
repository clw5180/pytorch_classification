### pytorch 图像分类竞赛框架

### 1. 更新日志
- (2020年5月2日) 基础版本上线

### 2. 依赖库
- pretrainedmodels
- progress
- efficientnet-pytorch
- apex

### 3. 支持功能

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
  
### 4. 使用方法
#### 1、训练
更改`config.py`中的参数，训练执行 `python main.py`，
主要修改：dataset(路径), bs, lr, epochs(如果是step，还需指定step_size)，input_size, num_classes, gpu_id, model_name, fp16, loss_func,
务必使用预训练模型，否则最后结果会差很多
没用：
Epoch: [1 | 50] LR: 0.100000
Training:  |################################| (358/358) Data: 0.000s | Batch: 0.059s | Total: 0:01:13 | ETA: 0:00:01 | Loss: 0.6746 | top1:  66.2439 
Validating:  |################################| (28/28) Data: 0.110s | Batch: 0.168s | Total: 0:00:04 | ETA: 0:00:01 | Loss: 0.7655 | top1:  47.9215 
train_loss:0.674563, val_loss:0.765460, train_acc:66.243877, train_5:1.000000, val_acc:47.921478, val_5:1.000000

Epoch: [2 | 50] LR: 0.100000
Training:  |################################| (358/358) Data: 0.000s | Batch: 0.060s | Total: 0:01:12 | ETA: 0:00:01 | Loss: 0.5050 | top1:  75.5423 
Validating:  |################################| (28/28) Data: 0.127s | Batch: 0.180s | Total: 0:00:05 | ETA: 0:00:01 | Loss: 0.7250 | top1:  50.8083 
train_loss:0.505044, val_loss:0.724990, train_acc:75.542337, train_5:1.000000, val_acc:50.808314, val_5:1.000000

Epoch: [3 | 50] LR: 0.100000
Training:  |################################| (358/358) Data: 0.000s | Batch: 0.059s | Total: 0:01:12 | ETA: 0:00:01 | Loss: 0.4726 | top1:  77.6679 
Validating:  |################################| (28/28) Data: 0.123s | Batch: 0.177s | Total: 0:00:04 | ETA: 0:00:01 | Loss: 0.7703 | top1:  52.4249 
train_loss:0.472609, val_loss:0.770282, train_acc:77.667950, train_5:1.000000, val_acc:52.424942, val_5:1.000000

...

Epoch: [47 | 50] LR: 0.001000
Training:  |################################| (358/358) Data: 0.000s | Batch: 0.056s | Total: 0:01:09 | ETA: 0:00:01 | Loss: 0.1415 | top1:  94.8915 
Validating:  |################################| (28/28) Data: 0.124s | Batch: 0.175s | Total: 0:00:04 | ETA: 0:00:01 | Loss: 0.6887 | top1:  72.7483 
train_loss:0.141452, val_loss:0.688665, train_acc:94.891533, train_5:1.000000, val_acc:72.748268, val_5:1.000000


用了：
Epoch: [1 | 50] LR: 0.100000
Training:  |################################| (358/358) Data: 0.000s | Batch: 0.058s | Total: 0:01:09 | ETA: 0:00:01 | Loss: 0.5490 | top1:  78.7001 
Validating:  |################################| (28/28) Data: 0.124s | Batch: 0.178s | Total: 0:00:04 | ETA: 0:00:01 | Loss: 0.6237 | top1:  74.3649 
train_loss:0.548992, val_loss:0.623722, train_acc:78.700140, train_5:1.000000, val_acc:74.364896, val_5:1.000000

Epoch: [2 | 50] LR: 0.100000
Training:  |################################| (358/358) Data: 0.000s | Batch: 0.060s | Total: 0:01:13 | ETA: 0:00:01 | Loss: 0.4685 | top1:  88.8383 
Validating:  |################################| (28/28) Data: 0.136s | Batch: 0.192s | Total: 0:00:05 | ETA: 0:00:01 | Loss: 0.7281 | top1:  61.3164 
train_loss:0.468477, val_loss:0.728094, train_acc:88.838348, train_5:1.000000, val_acc:61.316397, val_5:1.000000

Epoch: [3 | 50] LR: 0.100000
Training:  |################################| (358/358) Data: 0.000s | Batch: 0.059s | Total: 0:01:17 | ETA: 0:00:01 | Loss: 0.4539 | top1:  90.1417 
Validating:  |################################| (28/28) Data: 0.169s | Batch: 0.223s | Total: 0:00:06 | ETA: 0:00:01 | Loss: 0.6116 | top1:  75.1732 
train_loss:0.453887, val_loss:0.611589, train_acc:90.141707, train_5:1.000000, val_acc:75.173210, val_5:1.000000

...

Epoch: [47 | 50] LR: 0.001000
Training:  |################################| (358/358) Data: 0.000s | Batch: 0.056s | Total: 0:01:07 | ETA: 0:00:01 | Loss: 0.3543 | top1:  97.9706 
Validating:  |################################| (28/28) Data: 0.120s | Batch: 0.170s | Total: 0:00:04 | ETA: 0:00:01 | Loss: 0.5316 | top1:  81.0624 
train_loss:0.354261, val_loss:0.531626, train_acc:97.970609, train_5:1.000000, val_acc:81.062356, val_5:1.000000


#### 2、预测
执行`python test.py`




### 5. submit_example.csv 
每一行：filename,label
样例：
```
0001.jpg,dog
0002.jpg,dog
0003.jpg,dog
```
注：预测图像可能没有label，所以label可以随意给个临时的，但一些比赛平台对都会给个提交样例，随意给个label


### 6.TODO

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
