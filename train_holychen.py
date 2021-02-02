import os

# gpu_info = os.system('nvidia-smi')
# print(gpu_info)

import pandas as pd
import numpy as np
import random
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import timm
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from utils.losses import BiTemperedLogisticLoss
import pretrainedmodels

import argparse


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class Config:
    seed = 42
    #data_dir = '../input/cassava-leaf-disease-classification/'
    data_dir = '/home/user/dataset/kaggle_cassava_merge/'
    train_data_dir = data_dir + 'train/'
    train_csv_path = data_dir + 'merged.csv'
    #train_csv_path = data_dir + '2020.csv'
    #arch = 'vit_base_patch16_384'  ## model name
    arch = 'efficientnet-b3'
    #arch = 'efficientnet-b4'
    #arch = 'efficientnet-b5'
    #arch = 'se_resnext50_32x4d_timm'
    device = 'cuda'
    debug = False  ## clw modify

    #image_size = 384
    image_size = 512
    #train_batch_size = 16
    train_batch_size = 32
    val_batch_size = 32
    epochs = 12  ## total train epochs
    milestones = [7, 11]
    #freeze_bn_epochs = 5  ## freeze bn weights before epochs
    freeze_bn_epochs = 0  # clw modify

    #lr = 1e-4  ## init learning rate
    lr = 1e-1

    weight_decay = 1e-6
    num_workers = 4
    num_splits = 5  ## numbers splits
    num_classes = 5  ## numbers classes

    # min_lr = 1e-6  ## min learning rate
    T_0 = 10
    T_mult = 1
    #accum_iter = 2
    accum_iter = 1  # clw modify
    verbose_step = 1

    #criterion = 'LabelSmoothingCrossEntropy'
    criterion = 'bitempered'
    label_smoothing = 0.3

    train_id = [0, 1, 2, 3, 4]


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class CassavaDataset(Dataset):
    def __init__(self, data_dir, df, transforms=None, output_label=True):
        self.data_dir = data_dir
        self.df = df
        self.transforms = transforms
        self.output_label = output_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_infos = self.df.iloc[index]
        image_path = self.data_dir + image_infos.image_id

        image = load_image(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        ### augment
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        else:
            image = torch.from_numpy(image)

        if self.output_label:
            return image, image_infos.label
        else:
            return image


class CassavaClassifier(nn.Module):
    def __init__(self, model_arch, num_classes, pretrained=False):
        super().__init__()

        ### vit
        ###self.model = timm.create_model(model_arch, pretrained=pretrained)
        #num_features = self.model.head.in_features
        #self.model.head = nn.Linear(num_features, num_classes)

        ### efficientnet
        if "efficientnet-b3" in model_arch:
            self.model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        elif "efficientnet-b4" in model_arch:
            self.model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        elif 'se_resnext50_pretrainedmodels' in model_arch:
            self.model = pretrainedmodels.se_resnext50_32x4d(pretrained="imagenet")
            self.model.last_linear = nn.Linear(2048, num_classes)
            self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif 'se_resnext50_timm' in model_arch:
            self.model = timm.create_model('seresnext50_32x4d', pretrained=True)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, num_classes)
        elif 'vit_base_patch16_384' in model_arch:
            self.model = timm.create_model('vit_base_patch16_384', pretrained=True,
                                      num_classes=num_classes)  # , drop_rate=0.1)
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)

        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(num_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(num_features, num_classes, bias=True)
        )
        '''

    def forward(self, x):
        x = self.model(x)
        return x

def get_train_transforms(CFG):
    return A.Compose([
            # A.Resize(height=600, width=800),  # clw modify
            # A.RandomResizedCrop(height=CFG.image_size, width=CFG.image_size, p=0.5),
            # A.Transpose(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            # A.ShiftScaleRotate(p=0.5),
            # A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            # A.CenterCrop(CFG.image_size, CFG.image_size),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            # A.CoarseDropout(p=0.5),
            # A.Cutout(p=0.5),
            # ToTensorV2(),

            A.Resize(height=600, width=800),
            A.RandomResizedCrop(height=CFG.image_size, width=CFG.image_size, scale=(0.8, 1.0), p=1),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=1),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1)], p=0.7
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            A.CoarseDropout(p=0.5, max_height=32, max_width=32),
            # A.Cutout(p=0.5),
            ToTensorV2(),
        ],p=1.0)

def get_val_transforms(CFG):
    return A.Compose([
            # A.Resize(height=600, width=800),  # clw modify
            # A.CenterCrop(CFG.image_size, CFG.image_size, p=0.5),
            # A.Resize(CFG.image_size, CFG.image_size),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            # ToTensorV2(),
            A.Resize(height=CFG.image_size, width=CFG.image_size),
            #A.RandomResizedCrop(height=CFG.image_size, width=CFG.image_size, scale=(0.8, 1.0), p=1),
            A.Normalize(),  # A.Normalize(mean=(0.43032, 0.49673, 0.31342), std=(0.237595, 0.240453, 0.228265)),
            ToTensorV2(),
        ],p=1.0)


def load_dataloader(CFG, df, train_idx, val_idx):
    df_train = df.loc[train_idx, :].reset_index(drop=True)
    df_val = df.loc[val_idx, :].reset_index(drop=True)

    train_dataset = CassavaDataset(
        CFG.train_data_dir,
        df_train,
        transforms=get_train_transforms(CFG),
        output_label=True)

    val_dataset = CassavaDataset(
        CFG.train_data_dir,
        df_val,
        transforms=get_val_transforms(CFG),
        output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=CFG.num_workers,
        # sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.val_batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader


from utils.utils import rand_bbox
def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()
    lr = optimizer.state_dict()['param_groups'][0]['lr']

    running_loss = None
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (images, targets) in pbar:
        images = images.to(device).float()
        targets = targets.to(device).long()

        with autocast():
            if np.random.rand() < 0.5:
                # generate mixed sample
                #lam = np.random.beta(1.0, 1.0)
                lam = np.clip(np.random.beta(1, 1), 0.3, 0.4)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
                preds = model(images)
                # loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                loss = loss_fn(preds, target_a) * lam + loss_fn(preds, target_b) * (1. - lam)  #### no label smooth when using cutmix
            else:
                preds = model(images)
                loss = loss_fn(preds, targets)
            ###### clw modify
            #preds = model(images)
            #loss = loss_fn(preds, targets)

            scaler.scale(loss).backward()
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * 0.99 + loss.item() * 0.01

            if ((step + 1) % CFG.accum_iter == 0) or ((step + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()
            if ((step + 1) % CFG.accum_iter == 0) or ((step + 1) == len(train_loader)):
                description = f'Train epoch {epoch} loss: {running_loss:.5f}'
                pbar.set_description(description)

    if scheduler is not None and schd_batch_update:
        scheduler.step()


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    loss_sum = 0
    sample_num = 0
    preds_all = []
    targets_all = []
    scores = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (images, targets) in pbar:
        images = images.to(device).float()
        targets = targets.to(device).long()
        preds = model(images)

        preds_all += [torch.argmax(preds, 1).detach().cpu().numpy()]
        targets_all += [targets.detach().cpu().numpy()]

        loss = loss_fn(preds, targets)
        loss_sum += loss.item() * targets.shape[0]
        sample_num += targets.shape[0]

        if ((step + 1) % CFG.accum_iter == 0) or ((step + 1) == len(train_loader)):
            description = f'Val epoch {epoch} loss: {loss_sum / sample_num:.5f}'
            pbar.set_description(description)

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    accuracy = (preds_all == targets_all).mean()
    print(f'Validation multi-class accuracy = {accuracy:.5f}')

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()

    return accuracy


################ freeze bn
def freeze_batchnorm_stats(net):
    try:
        for m in net.modules():
            if isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.LayerNorm):
                m.eval()
    except ValueError:
        print('error with batchnorm2d or layernorm')
        return


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


if __name__ == '__main__':
    CFG = Config
    parser = argparse.ArgumentParser(description="PyTorch Template Training")
    parser.add_argument("--model_arch", default=CFG.arch, help="", type=str)
    parser.add_argument("--lr", default=CFG.lr, help="", type=float)
    parser.add_argument("--image_size", default=CFG.image_size, help="", type=int)
    args = parser.parse_args()

    CFG.arch = args.model_arch
    CFG.lr = args.lr
    CFG.image_size = args.image_size

    print('model_arch:', CFG.arch)
    print('lr:', CFG.lr)
    print('image_size:', CFG.image_size)

    train = pd.read_csv(CFG.train_csv_path)

    if CFG.debug:
        CFG.epochs = 1
        train = train.sample(100, random_state=CFG.seed).reset_index(drop=True)

    print('CFG seed is ', CFG.seed)
    if CFG.seed is not None:
        seed_everything(CFG.seed)

    folds = StratifiedKFold(
        n_splits=CFG.num_splits,
        shuffle=True,
        random_state=CFG.seed).split(np.arange(train.shape[0]), train.label.values)

    cross_accuracy = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        ########
        # load data
        #######
        train_loader, val_loader = load_dataloader(CFG, train, train_idx, val_idx)

        device = torch.device(CFG.device)
        #         assert(CFG.num_classes ==  train.label.nunique())
        model = CassavaClassifier(CFG.arch, train.label.nunique(), pretrained=True).to(device)

        scaler = GradScaler()
        # optimizer = torch.optim.Adam(
        #     model.parameters(),
        #     lr=CFG.lr,
        #     weight_decay=CFG.weight_decay)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=CFG.lr,
            momentum=0.9,
            weight_decay=CFG.weight_decay,
            nesterov=True)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=CFG.T_0,
        #     T_mult=CFG.T_mult,
        #     eta_min=CFG.min_lr,
        #     last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=CFG.milestones, gamma=0.1)


        ########
        # criterion
        #######
        if CFG.criterion == 'LabelSmoothingCrossEntropy':  #### label smoothing cross entropy
            loss_train = LabelSmoothingCrossEntropy(smoothing=CFG.label_smoothing)
        elif CFG.criterion == 'bitempered':
            loss_train = BiTemperedLogisticLoss(t1=0.3, t2=1.0, smoothing=CFG.label_smoothing)
        else:
            loss_train = nn.CrossEntropyLoss().to(device)
        loss_val = nn.CrossEntropyLoss().to(device)

        best_accuracy = 0
        best_epoch = 0
        for epoch in range(CFG.epochs):
            if epoch < CFG.freeze_bn_epochs:
                freeze_batchnorm_stats(model)
            train_one_epoch(
                epoch,
                model,
                loss_train,
                optimizer,
                train_loader,
                device,
                scheduler=scheduler,
                schd_batch_update=False)

            with torch.no_grad():
                epoch_accuracy = valid_one_epoch(
                    epoch,
                    model,
                    loss_val,
                    val_loader,
                    device,
                    scheduler=None,
                    schd_loss_update=False)

            if epoch_accuracy > best_accuracy:
                torch.save(model.state_dict(), '{}_fold{}_best.ckpt'.format(CFG.arch, fold))
                best_accuracy = epoch_accuracy
                best_epoch = epoch
                print('Best model is saved')
        cross_accuracy += [best_accuracy]
        print('Fold{} best accuracy = {} in epoch {}'.format(fold, best_accuracy, best_epoch))
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()
    print('{} folds cross validation CV = {:.5f}'.format(CFG.num_splits, np.average(cross_accuracy)))