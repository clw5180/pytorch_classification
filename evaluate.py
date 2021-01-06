# clw note: 根据验证集结果 和 真实标签，用于对验证集的结果进行分析

import os
from torchvision import models
import time
import torch
from utils.misc import AverageMeter, accuracy, get_files
from progress.bar import Bar
from utils.reader import WeatherDataset
from config import configs
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pretrainedmodels
import torch.nn as nn
import timm
from torch.utils.data.sampler import *


# 绘制混淆矩阵  参考：https://www.jianshu.com/p/cd59aed787cf?open_source=weibo_search
def plot_confusion_matrix(cm, classes, title=None, cmap=plt.cm.Reds):  # plt.cm.Blues
    '''
    cm - 混淆矩阵的数值， 是一个二维numpy数组
    classes - 各个类别的标签（label）
    title - 图片标题
    cmap - 颜色图
    '''
    plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print("Normalized confusion matrix")
    print(cm)
    # str_cm = cm.astype(np.str).tolist()
    # for row in str_cm:
    #     print('\t'.join(row))

    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #plt.savefig('cm.jpg', dpi=300)
    plt.show()


def validate_and_analysis(val_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    #top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    label_predict_matrix = np.zeros((configs.num_classes, configs.num_classes))  # clw note: 创建混淆矩阵，用于统计比如predict类别1但是预测成了类别4;
    batch_nums = len(val_loader)  # clw add
    end = time.time()
    bar = Bar('Validating: ', max=len(val_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda()  # .half()

            # compute output
            #outputs = model(inputs)
            #feature_1, feature_2, feature_3, feature_4, outputs = model(inputs)  # clw modify
            outputs = model(inputs)
            predict_class_ids = torch.argmax(outputs.data, dim=1).cpu().numpy()
            true_label_class_ids = torch.argmax(targets.data, dim=1).cpu().numpy()
            for i in range( inputs.shape[0] ):
                label_predict_matrix[ true_label_class_ids[i] ][ predict_class_ids[i] ] += 1

            # measure accuracy and record loss
            #prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            top1.update(prec1.item(), inputs.size(0))
            #top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} '.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        top1=top1.avg,
                        )
            bar.next()

    bar.finish()
    #return (losses.avg, top1.avg, top5.avg)
    return (label_predict_matrix, top1.avg, 1)


if __name__ == "__main__":
    model_root_path = '/home/user/pytorch_classification/checkpoints'
    #model_file_name = 'resnet50_2021_01_01_10_22_18-checkpoint.pth.tar'
    #model_file_name = 'resnet50_2020_12_31_20_46_33-checkpoint.pth.tar'
    #model_file_name = 'resnet50_2020_12_31_20_29_11-checkpoint.pth.tar'
    #model_file_name = 'se_resnext50_32x4d_2021_01_01_20_23_36-checkpoint.pth.tar'

    '''
    model_file_name = 'efficientnet-b3_2021_01_05_21_21_49-best_model.pth.tar'  # 89.327417  cutmix reflect101
        [[0.67889908 0.05045872 0.02293578 0.03211009 0.21559633]              # but again 89.3046
        [0.0456621  0.82876712 0.02054795 0.043379   0.06164384]
        [0.01464435 0.02301255 0.77196653 0.12761506 0.06276151]
        [0.0018997  0.00379939 0.00987842 0.97758359 0.00683891]
        [0.08333333 0.05620155 0.04069767 0.09883721 0.72093023]]

   model_file_name = 'efficientnet-b3_2021_01_05_19_22_28-best_model.pth.tar'  # 89.280710    cutmix  
        [[0.60550459 0.05504587 0.02752294 0.02752294 0.28440367]
         [0.043379   0.81506849 0.02968037 0.03652968 0.07534247]
         [0.0209205  0.01882845 0.77196653 0.10669456 0.08158996]
         [0.0018997  0.00265957 0.01253799 0.97606383 0.00683891]
         [0.0755814  0.03875969 0.04069767 0.07751938 0.76744186]]
         
   model_file_name = 'efficientnet-b3_2021_01_05_15_43_49-best_model.pth.tar'  # 88.953760, but online:0.8946??      
        [[0.71100917 0.02293578 0.01834862 0.01834862 0.2293578 ]       .half:  88.930406  apex same...
         [0.04794521 0.74885845 0.03881279 0.04109589 0.12328767]
         [0.01882845 0.01046025 0.79916318 0.10460251 0.06694561]
         [0.00455927 0.00151976 0.01785714 0.96808511 0.00797872]
         [0.10077519 0.02325581 0.06007752 0.04844961 0.76744186]]
         
    model_file_name = 'efficientnet-b3_2021_01_05_13_31_01-best_model.pth.tar'   # 89.654367 线上89.1
        [[0.68807339 0.04587156 0.02293578 0.02752294 0.21559633]
         [0.03196347 0.82191781 0.02511416 0.04109589 0.07990868]
         [0.0167364  0.01464435 0.78870293 0.10460251 0.07531381]
         [0.00265957 0.00569909 0.01367781 0.97074468 0.00721884]
         [0.09689922 0.03682171 0.04457364 0.05232558 0.76937984]]         
    '''

    #model_file_name = 'efficientnet-b3_2021_01_06_09_42_16-best_model.pth.tar'
    model_file_name = 'efficientnet-b3_2021_01_06_21_58_50-best_model.pth.tar'


    my_state_dict = torch.load(os.path.join(model_root_path, model_file_name))['state_dict']
    if 'se_resnext50' in model_file_name:
        model = pretrainedmodels.se_resnext50_32x4d(pretrained="imagenet")
        model.last_linear=nn.Linear(2048, configs.num_classes)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
    elif "efficientnet-b0" in model_file_name:
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif "efficientnet-b2" in model_file_name:
        model = timm.create_model('tf_efficientnet_b2_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif "efficientnet-b3" in model_file_name:
        model = timm.create_model('tf_efficientnet_b3_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif "efficientnet-b4" in model_file_name:
        model = timm.create_model('tf_efficientnet_b4_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    else:
        model = models.resnet50(pretrained=False, num_classes=configs.num_classes)  # clw note: fc.weight: (num_class, 2048)
    model.cuda()   # .half()
    model.load_state_dict(my_state_dict)

    from apex import amp
    from utils.misc import get_optimizer
    optimizer = get_optimizer(model)
    model, _ = amp.initialize(model, optimizer,
                                      opt_level=configs.opt_level,
                                      keep_batchnorm_fp32=None if configs.opt_level == "O1" else False,
                                      # verbosity=0  # 不打印amp相关的日志
                                      )



    val_files = get_files(configs.dataset + "/val/", "val")
    val_dataset = WeatherDataset(val_files, mode="val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=configs.bs, shuffle=False, sampler=SequentialSampler(val_dataset),
        num_workers=configs.workers, pin_memory=True
    )

    # 绘制混淆矩阵并打印acc
    label_predict_matrix, val_acc, _ = validate_and_analysis(val_loader, model)
    print('Test Acc: %.6f' % val_acc)
    plot_confusion_matrix(label_predict_matrix, [i for i in range(configs.num_classes)])


