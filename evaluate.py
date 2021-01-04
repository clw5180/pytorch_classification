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

    # print("Normalized confusion matrix")
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

    predict_label_matrix = np.zeros((configs.num_classes, configs.num_classes))  # clw note: 创建混淆矩阵，用于统计比如predict类别1但是预测成了类别4;
    batch_nums = len(val_loader)  # clw add
    end = time.time()
    bar = Bar('Validating: ', max=len(val_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda()
            #inputs = inputs.half()

            # compute output
            #outputs = model(inputs)
            #feature_1, feature_2, feature_3, feature_4, outputs = model(inputs)  # clw modify
            outputs = model(inputs)
            predict_class_ids = torch.argmax(outputs.data, dim=1).cpu().numpy()
            true_label_class_ids = torch.argmax(targets.data, dim=1).cpu().numpy()
            for i in range( inputs.shape[0] ):
                predict_label_matrix[ predict_class_ids[i] ][ true_label_class_ids[i] ] += 1

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
    return (predict_label_matrix, top1.avg, 1)

if __name__ == "__main__":
    model_root_path = '/home/user/pytorch_classification/checkpoints'
    #model_file_name = 'resnet50_2021_01_01_10_22_18-checkpoint.pth.tar'
    #model_file_name = 'resnet50_2020_12_31_20_46_33-checkpoint.pth.tar'
    #model_file_name = 'resnet50_2020_12_31_20_29_11-checkpoint.pth.tar'
    #model_file_name = 'se_resnext50_32x4d_2021_01_01_20_23_36-checkpoint.pth.tar'
    model_file_name = 'efficientnet-b0_2021_01_04_20_34_44-checkpoint.pth.tar'

    my_state_dict = torch.load(os.path.join(model_root_path, model_file_name))['state_dict']
    if 'se_resnext50' in model_file_name:
        model = pretrainedmodels.se_resnext50_32x4d(pretrained="imagenet")
        model.last_linear=nn.Linear(2048, configs.num_classes)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
    elif "efficientnet-b0" in model_file_name:
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif "efficientnet-b4" in model_file_name:
        model = timm.create_model('tf_efficientnet_b4_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    else:
        model = models.resnet50(pretrained=False, num_classes=configs.num_classes)  # clw note: fc.weight: (num_class, 2048)
    model.load_state_dict(my_state_dict)
    #model.half()
    model.cuda()


    val_files = get_files(configs.dataset + "/val/", "val")
    val_dataset = WeatherDataset(val_files, mode="val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=configs.bs, shuffle=False,
        num_workers=configs.workers, pin_memory=True
    )

    # 绘制混淆矩阵并打印acc
    predict_label_matrix, val_acc, _ = validate_and_analysis(val_loader, model)
    print('Test Acc: %.4f' % val_acc)
    plot_confusion_matrix(predict_label_matrix, [i for i in range(configs.num_classes)])


