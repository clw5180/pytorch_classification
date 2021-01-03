import random
import time
import warnings
import os

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from PIL import ImageFile
from config import configs
from models.model import get_model
from sklearn.model_selection import train_test_split
from utils.misc import get_files, accuracy, AverageMeter, get_lr, adjust_learning_rate, save_checkpoint, get_optimizer
from utils.logger import *
from utils.losses import *
from progress.bar import Bar
from utils.reader import TrainDataset, albu_transforms
from utils.scheduler import WarmupCosineAnnealingLR, WarmUpCosineAnnealingLR2, WarmupCosineLR3
from utils.sampler.imbalanced import ImbalancedDatasetSampler
from utils.sampler.utils import make_weights_for_balanced_classes

######## clw modify
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter   # clw modify: it's quicker than   #from torch.utils.tensorboard import SummaryWriter
tb_logger = SummaryWriter()  # clw modify



# for train fp16
if configs.fp16:
    try:
        from apex import amp
        #from apex.parallel import DistributedDataParallel as DDP
        #from apex.fp16_utils import *
        #from apex import amp, optimizers
        #from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id

# set random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(configs.seed)

# make dir for use
def makdir():
    if not os.path.exists(configs.checkpoints):
        os.makedirs(configs.checkpoints)
    if not os.path.exists(configs.log_dir):
        os.makedirs(configs.log_dir)
    if not os.path.exists(configs.submits):
        os.makedirs(configs.submits)
makdir()


def main():
    n_fold = 5
    logger = Logger(os.path.join(configs.log_dir, '%s_%s_%d_fold_log.txt' % (configs.model_name, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), n_fold)), title=configs.model_name, resume=configs.resume)
    logger.info(str(configs))
    logger.info(str(albu_transforms))


    target_col = 'label'
    #train_df_merge = pd.read_csv('/home/user/dataset/kaggle_cassava_merge/merged.csv')
    train_df_merge = pd.read_csv('/home/user/dataset/kaggle_cassava_merge/2020.csv')  # only use 2020 dataset

    folds = train_df_merge.copy()
    Fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=configs.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[target_col])):
        folds.loc[val_index, 'fold'] = int(n)  # clw note: 相当于给folds加了一列, 表明数据集里面的某个样本是属于哪个fold的,
    folds['fold'] = folds['fold'].astype(int)   #          比如10个样本,0 4属于fold0, 1,7属于fold1....8,6属于fold4;
    # print(folds.groupby(['fold', target_col]).size())  # 统计每个fold的各类别样本数量

    start_epoch = configs.start_epoch

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # clw note: r g b
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #for fold in range(n_fold):
    for fold in [0, 1, 2, 3, 4]:  # clw modify
        best_acc = 0  # best test accuracy
        logger.info(f"========== fold: {fold} training ==========")

        # # Data loading code
        # ====================================================
        # loader
        # ====================================================
        trn_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        train_folds = folds.loc[trn_idx].reset_index(drop=True)
        valid_folds = folds.loc[val_idx].reset_index(drop=True)

        train_dataset = TrainDataset(train_folds,
                                     transform=transform_train)
        valid_dataset = TrainDataset(valid_folds,
                                     transform=transform_val)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                  batch_size=configs.bs,
                                  shuffle=True,
                                  num_workers=configs.workers, pin_memory=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset,
                                  batch_size=configs.bs,
                                  shuffle=False,
                                  num_workers=configs.workers, pin_memory=True, drop_last=False)

        # get model
        model = get_model()
        model.cuda()

        optimizer = get_optimizer(model)
        # set lr scheduler method
        if configs.lr_scheduler == "step":
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(configs.epochs*0.3), gamma=0.1)   # clw note: 注意调用step_size这么多次学习率开始变化，如果每个epoch结束后执行scheduler.step(),那么就设置成比如epochs*0.3;
                                                                                                                    #           最好不放在mini-batch下，否则还要设置成len(train_dataloader)*epoches*0.3
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)   # clw note: 学习率每step_size变为之前的0.1
        elif configs.lr_scheduler == "cosine_change_per_batch":
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, configs.epochs, eta_min=1e-6, last_epoch=-1)  # clw modify
            #scheduler = WarmupCosineAnnealingLR(optimizer, max_iters=configs.epochs * len(train_loader), delay_iters=1000, eta_min_lr=1e-5)
            # scheduler = WarmUpCosineAnnealingLR2(optimizer=optimizer, T_max=configs.epochs * len(train_loader), T_warmup= 3 * len(train_loader), eta_min=1e-5)
            scheduler = WarmupCosineLR3(optimizer, total_iters=configs.epochs * len(train_loader), warmup_iters=0, eta_min=1e-6)  # clw note: 默认cosine是按batch来更新 ; warmup_iters=500, eta_min=1e-7
        elif configs.lr_scheduler == "cosine_change_per_epoch":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=configs.epochs, T_mult=1, eta_min=1e-6)  # clw note: usually 1e-6
        elif configs.lr_scheduler == "on_loss":
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=False)
        elif configs.lr_scheduler == "on_acc":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=False)
        elif configs.lr_scheduler == "adjust":
            pass
        else:
            raise Exception("Not implement this lr_scheduler, please modify config.py !!!")
        # for fp16
        if configs.fp16:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level=configs.opt_level,
                                              keep_batchnorm_fp32= None if configs.opt_level == "O1" else False,
                                              # verbosity=0  # 不打印amp相关的日志
                                              )
        if configs.resume:
                # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(configs.resume), 'Error: no checkpoint directory found!'
            configs.checkpoint = os.path.dirname(configs.resume)
            checkpoint = torch.load(configs.resume)
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            logger.set_names(['Learning Rate', 'Train Loss', 'Train Acc.', 'Valid Acc.'])


        ################################################### clw modify: loss function
        if configs.loss_func == "LabelSmoothCELoss":
            criterion = LabelSmoothingLoss(configs.label_smooth_epsilon, configs.num_classes)  # now better than 0.05 and 0.1  TODO
        elif configs.loss_func == "CELoss":
            criterion = nn.CrossEntropyLoss()
        elif configs.loss_func == "BCELoss":
            criterion = nn.BCEWithLogitsLoss()
        elif configs.loss_func == "FocalLoss":
            criterion = FocalLoss(gamma=2)
        elif configs.loss_func == "FocalLoss_clw":  # clw modify
            criterion = FocalLoss_clw()
        else:
            raise Exception("No this loss type, please check config.py !!!")

        ###################################################

        # Train and val
        for epoch in range(start_epoch, configs.epochs):
            #print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, configs.epochs, optimizer.param_groups[0]['lr']))
            print('\nEpoch: [%d | %d] ' % (epoch + 1, configs.epochs))
            if configs.lr_scheduler == "adjust":
                adjust_learning_rate(optimizer, epoch)

            train_loss, train_acc, train_5 = train(train_loader, model, criterion, optimizer, epoch, scheduler=scheduler if configs.lr_scheduler == "cosine_change_per_batch" else None) # clw modify: 暂时默认cosine按mini-batch调整学习率
            val_loss, val_acc, test_5 = validate(val_loader, model, criterion, epoch)
            tb_logger.add_scalar('loss_val', val_loss, epoch)  # clw note: 观察训练集loss曲线

            # adjust lr
            if configs.lr_scheduler == "on_acc":
                scheduler.step(val_acc)
            elif configs.lr_scheduler == "on_loss":
                scheduler.step(val_loss)
            elif configs.lr_scheduler == "step":
                scheduler.step()
            elif configs.lr_scheduler == "cosine_change_per_epoch":
                scheduler.step()


            # append logger file
            lr_current = get_lr(optimizer)
            logger.append([lr_current,train_loss, train_acc, val_acc])
            print('train_loss:%f, train_acc:%f, train_5:%f, val_acc:%f, val_5:%f' % (train_loss, train_acc, train_5, val_acc, test_5))

            # save model
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_checkpoint({
                'fold': fold,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'train_acc': train_acc,
                'acc': val_acc,
                'best_acc': best_acc,
                #'optimizer': optimizer.state_dict(),   # TODO, 可以不保存优化器
            }, is_best)

        print('Best acc:')
        print(best_acc)

    logger.close()


def train(train_loader, model, criterion, optimizer, epoch, scheduler=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    batch_nums = len(train_loader)  # clw add
    bar = Bar('Training: ', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        if configs.loss_func == "BCELoss":
            targets_one_hot = torch.zeros(len(targets), configs.num_classes).cuda()  # clw note：这里不能写configs.bs，因为最后一个batch可能不是完整的
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)  # one hot
            loss = criterion(outputs, targets_one_hot)
        else:
            loss = criterion(outputs, targets)

        curr_step = batch_nums * epoch + batch_idx
        tb_logger.add_scalar('loss_train', loss.item(), curr_step)   # clw note: 观察训练集loss曲线

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]  # clw note: 这里计算acc； 如果只有两个类，此时top5会报错;
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if configs.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)  # clw note: max_norm top方案1000, or 10, 20；TODO

        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # clw modify

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | LR: {lr:.6f} | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lr=optimizer.param_groups[0]['lr'],
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, 1)


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    batch_nums = len(val_loader)  # clw add
    end = time.time()
    bar = Bar('Validating: ', max=len(val_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            losses.update(val_loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        )
            bar.next()

    bar.finish()
    return (losses.avg, top1.avg, 1)



if __name__ == '__main__':
    main()
