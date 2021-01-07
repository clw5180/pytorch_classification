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
from utils.reader import WeatherDataset, albu_transforms_train
from utils.scheduler import WarmupCosineAnnealingLR, WarmUpCosineAnnealingLR2, WarmupCosineLR3
from utils.sampler.imbalanced import ImbalancedDatasetSampler
from utils.sampler.utils import make_weights_for_balanced_classes
from utils.utils import rand_bbox
from tqdm import tqdm
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter   # clw modify: it's quicker than   #from torch.utils.tensorboard import SummaryWriter
tb_logger = SummaryWriter()  # clw modify

######## clw modify: items for training
do_cutmix_prob = 0.0



# for train fp16
if configs.fp16:
    try:
        import apex
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
        from apex import amp, optimizers
        from apex.multi_tensor_apply import multi_tensor_applier
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
    logger = Logger(os.path.join(configs.log_dir, '%s_%s_log.txt' % (configs.model_name, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))), title=configs.model_name, resume=configs.resume)
    logger.info(str(configs))
    logger.info(str(albu_transforms_train))

    best_acc = 0  # best test accuracy
    start_epoch = configs.start_epoch


    # Data loading code
    train_data_df = get_files(configs.dataset+"/train/",   "train")  # DataFrame: ( image_nums, 2 )
    train_dataset = WeatherDataset(train_data_df, "train")
    val_data_df = get_files(configs.dataset+"/val/",   "val")
    val_dataset = WeatherDataset(val_data_df, "val")

    if train_dataset.do_cutmix_prob > 0:
        logger.info('\ndo cutmix in __get_item()__ !\n')
    if train_dataset.do_mixup_prob > 0:
        logger.info('\ndo mixup in __get_item()__ !\n')
    if do_cutmix_prob > 0:
        logger.info('\ndo cutmix in a batch !\n')
    assert (train_dataset.do_cutmix_prob == 0 or do_cutmix_prob == 0)  # can't >0 both



    if configs.sampler == "WeightedSampler":          # TODO：解决类别不平衡问题：根据不同类别样本数量给予不同的权重
        train_data_list = np.array(train_data_df).tolist()
        weight_for_all_images = make_weights_for_balanced_classes(train_data_list, configs.num_classes)
        weightedSampler = torch.utils.data.sampler.WeightedRandomSampler(weight_for_all_images, len(weight_for_all_images))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=configs.bs,
                                                   #shuffle=True,
                                                   sampler=weightedSampler,
                                                   num_workers=configs.workers,
                                                   pin_memory=True)
    elif configs.sampler == "imbalancedSampler":
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=configs.bs,
                                                   #shuffle=True,
                                                   sampler=ImbalancedDatasetSampler(train_dataset),
                                                   num_workers=configs.workers,
                                                   pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=configs.bs,
                                                   shuffle=True,
                                                   num_workers=configs.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=configs.bs,
                                             shuffle=False,
                                             num_workers=configs.workers,
                                             pin_memory=True,
                                             drop_last=False)

    # get model
    model = get_model()
    model.cuda()

    # set lr scheduler methget_optimizerod
    optimizer = get_optimizer(model)
    # set lr scheduler method
    if configs.lr_scheduler == "step":
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(configs.epochs*0.3), gamma=0.1)   # clw note: 注意调用step_size这么多次学习率开始变化，如果每个epoch结束后执行scheduler.step(),那么就设置成比如epochs*0.3;
        #           最好不放在mini-batch下，否则还要设置成len(train_dataloader)*epoches*0.3
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 19],
                                                         gamma=0.3)  # clw note: 学习率每step_size变为之前的0.1
    elif configs.lr_scheduler == "cosine_change_per_batch":
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, configs.epochs, eta_min=1e-6, last_epoch=-1)  # clw modify
        # scheduler = WarmupCosineAnnealingLR(optimizer, max_iters=configs.epochs * len(train_loader), delay_iters=1000, eta_min_lr=1e-5)
        # scheduler = WarmUpCosineAnnealingLR2(optimizer=optimizer, T_max=configs.epochs * len(train_loader), T_warmup= 3 * len(train_loader), eta_min=1e-5)
        scheduler = WarmupCosineLR3(optimizer, total_iters=configs.epochs * len(train_loader), warmup_iters=len(train_loader),
                                    eta_min=1e-6)  # clw note: 默认cosine是按batch来更新 ; warmup_iters=500, eta_min=1e-7
    elif configs.lr_scheduler == "cosine_change_per_epoch":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=configs.epochs, T_mult=1,
                                                                         eta_min=1e-6)  # clw note: usually 1e-6
    elif configs.lr_scheduler == "on_loss":
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4,
                                                               verbose=False)
    elif configs.lr_scheduler == "on_acc":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5,
                                                               verbose=False)
    elif configs.lr_scheduler == "adjust":
        pass
    else:
        raise Exception("Not implement this lr_scheduler, please modify config.py !!!")
    # for fp16
    if configs.fp16:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=configs.opt_level,
                                          keep_batchnorm_fp32=None if configs.opt_level == "O1" else False,
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
    if configs.loss_func == "LabelSmoothingLoss":
        criterion = LabelSmoothingLoss(configs.label_smooth_epsilon, configs.num_classes)  # now better than 0.05 and 0.1  TODO
    elif configs.loss_func == "LabelSmoothingLoss_clw":
        criterion = LabelSmoothingLoss_clw(configs.label_smooth_epsilon,
                                           configs.num_classes)  # now better than 0.05 and 0.1  TODO
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
            adjust_learning_rate(optimizer,epoch)

        #train_loss, train_acc, train_5 = train(train_loader, model, criterion, optimizer, epoch)
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
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'train_acc': train_acc,
            'acc': val_acc,
            'best_acc': best_acc,
            #'optimizer': optimizer.state_dict(),   # TODO, 可以不保存优化器
        }, is_best)

    logger.close()
    print('Best acc:')
    print(best_acc)


def train(train_loader, model, criterion, optimizer, epoch, scheduler=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #top5 = AverageMeter()
    end = time.time()

    batch_nums = len(train_loader)  # clw add


    bar = Bar('Training: ', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.cuda(), targets.cuda()
        # compute output
        # feature_1:(bs, 256, 1/4, 1/4)  feature_2:(bs, 512, 1/8, 1/8)    feature_3: (bs, 1024, 1/16, 1/16)   feature_3: (bs, 2048, 1/32, 1/32)
        #feature_1, feature_2, feature_3, feature_4, outputs = model(inputs)  # clw note: inputs: (32, 3, 224, 224)  # 在这里可以把所有stage的feature map返回，便于下面可视化；

        ### clw added: cutmix, same as official -  https://github.com/clovaai/CutMix-PyTorch
        r = np.random.rand(1)
        if r < do_cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)



        ################################################### clw modify: tensorboard
        curr_step = batch_nums * epoch + batch_idx
        tb_logger.add_scalar('loss_train', loss.item(), curr_step)   # clw note: 观察训练集loss曲线
        # tb_logger.add_image('image_0', make_grid(inputs[0], normalize=True), curr_step)  # 注意这样会导致训练速度下降很多,gpu利用率明显降低 !!!从97左右降到60 70左右
        #                                                                                 # 因为在Dataloader里面对输入图片做了Normalize，导致此时的图像已经有正有负，所以这里要用到make_grid，再归一化到0～1之间；
        # tb_logger.add_image('image_1', make_grid(inputs[1], normalize=True), curr_step)
        # tb_logger.add_image('image_2', make_grid(inputs[2], normalize=True), curr_step)
        # tb_logger.add_image('image_3', make_grid(inputs[3], normalize=True), curr_step)
        # tb_logger.add_image('feature_111', make_grid(torch.sum(feature_1[0], dim=0), normalize=True), curr_step)
        # tb_logger.add_image('feature_222', make_grid(torch.sum(feature_2[0], dim=0), normalize=True), curr_step)
        # tb_logger.add_image('feature_333', make_grid(torch.sum(feature_3[0], dim=0), normalize=True), curr_step)
        # tb_logger.add_image('feature_444', make_grid(torch.sum(feature_4[0], dim=0), normalize=True), curr_step)

        ### tb_logger.add_image('feature_1', make_grid(feature_1[0].unsqueeze(dim=1), normalize=False), curr_step)
        ### tb_logger.add_image('feature_2', make_grid(feature_2[0].unsqueeze(dim=1), normalize=False), curr_step)
        ### tb_logger.add_image('feature_3', make_grid(feature_3[0].unsqueeze(dim=1), normalize=False), curr_step)
        ### tb_logger.add_image('feature_4', make_grid(feature_4[0].unsqueeze(dim=1), normalize=False), curr_step)


        ####################################################

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]  # clw note: 这里计算acc； 如果只有两个类，此时top5会报错;
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        #top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if configs.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0, norm_type=2)  # TODO
        optimizer.step()
        if scheduler is not None:  # clw modify
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        # bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
        #             batch=batch_idx + 1,
        #             size=len(train_loader),
        #             data=data_time.val,
        #             bt=batch_time.val,
        #             total=bar.elapsed_td,
        #             eta=bar.eta_td,
        #             loss=losses.avg,
        #             top1=top1.avg,
        #             top5=top5.avg,
        #             )
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
    #return (losses.avg, top1.avg, top5.avg)
    return (losses.avg, top1.avg, 1)


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #top5 = AverageMeter()

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
            #feature_1, feature_2, feature_3, feature_4, outputs = model(inputs)  # clw modify
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            curr_step = batch_nums * epoch + batch_idx


            # measure accuracy and record loss
            #prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            losses.update(val_loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            #top5.update(prec5.item(), inputs.size(0))

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
    #return (losses.avg, top1.avg, top5.avg)
    return (losses.avg, top1.avg, 1)



if __name__ == '__main__':
    main()
