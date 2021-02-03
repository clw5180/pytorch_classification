# clw note: 根据验证集结果 和 真实标签，用于对验证集的结果进行分析

import os
from torchvision import models
import time
import torch
from utils.misc import AverageMeter, accuracy, get_files
from progress.bar import Bar
from utils.reader import CassavaValDataset
from config import configs
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pretrainedmodels
import torch.nn as nn
import timm
from torch.utils.data.sampler import *
from models import get_model_no_pretrained
import torch.nn.functional as F
import random

# set random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)

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

            # img0_tensor = img_tensor[:, :, :, :384]
            # img1_tensor = img_tensor[:, :, :, 384:768]
            # img2_tensor = img_tensor[:, :, :, 768:1152]
            # img3_tensor = img_tensor[:, :, :, 1152:1536]
            # img4_tensor = img_tensor[:, :, :, 1536:1920]
            # img0_tensor = inputs[:, :, :, :512]
            # img1_tensor = inputs[:, :, :, 512:1024]
            # img2_tensor = inputs[:, :, :, 1024:1536]
            # img3_tensor = inputs[:, :, :, 1536:2048]
            # img4_tensor = inputs[:, :, :, 2048:2560]

            img0_tensor = inputs


            p = []
            ####
            logit = model(img0_tensor)              #feature_1, feature_2, feature_3, feature_4, outputs = model(inputs)  # clw modify
            p.append(F.softmax(logit, -1))
            # logit = model(img1_tensor)
            # p.append(F.softmax(logit, -1))
            # logit = model(img2_tensor)
            # p.append(F.softmax(logit, -1))
            # logit = model(img3_tensor)
            # p.append(F.softmax(logit, -1))
            # logit = model(img4_tensor)
            # p.append(F.softmax(logit, -1))
            ####

            p = torch.stack(p).mean(0)

            predict_class_ids = torch.argmax(p, dim=1).cpu().numpy()
            true_label_class_ids = torch.argmax(targets.data, dim=1).cpu().numpy()
            for i in range( inputs.shape[0] ):
                label_predict_matrix[ true_label_class_ids[i] ][ predict_class_ids[i] ] += 1

            # measure accuracy and record loss
            #prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            prec1 = accuracy(p, targets.data, topk=(1,))[0]
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

    #model_file_name = 'efficientnet-b3_2021_01_11_16_03_30-checkpoint.pth.tar'
    #model_file_name = 'efficientnet-b3_2021_01_12_00_06_11-best_model.pth.tar'
    #model_file_name = 'efficientnet-b3_2021_01_30_22_17_34-best_model.pth.tar'
    #model_file_name = 'efficientnet-b3_2021_01_30_22_17_34-checkpoint.pth.tar'
    #model_file_name = 'efficientnet-b3_2021_01_31_21_53_10-best_model.pth.tar'
    #model_file_name = 'efficientnet-b3_2021_02_02_20_02_13-best_model.pth.tar'
    model_file_name = 'efficientnet-b3_2021_02_02_20_02_13-checkpoint.pth.tar'

    if "ckpt" in model_file_name:  # from train_holychen.py
        model_root_path = '/home/user/pytorch_classification'
        state_dict = torch.load(os.path.join(model_root_path, model_file_name))
        my_state_dict = {}
        for k, v in state_dict.items():
            my_state_dict[k[6:]] = v
    else:
        model_root_path = '/home/user/pytorch_classification/checkpoints'
        my_state_dict = torch.load(os.path.join(model_root_path, model_file_name))['state_dict']

    model = get_model_no_pretrained(model_file_name, my_state_dict)
    model.cuda()


    val_files = get_files(configs.dataset + "/val/", "val")
    val_dataset = CassavaValDataset(val_files)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, sampler=SequentialSampler(val_dataset),
        num_workers=configs.workers, pin_memory=True
    )

    # 绘制混淆矩阵并打印acc
    label_predict_matrix, val_acc, _ = validate_and_analysis(val_loader, model)
    print('Test Acc: %.6f' % val_acc)
    plot_confusion_matrix(label_predict_matrix, [i for i in range(configs.num_classes)])


