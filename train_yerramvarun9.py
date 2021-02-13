import os
import cv2
import copy
import time
import random
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torch.cuda import amp
#from tqdm.notebook import tqdm
from tqdm import tqdm  # clw modify

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
import pretrainedmodels
from utils.utils import rand_bbox

#ROOT_DIR = "/home/user/dataset/kaggle2020-leaf-disease-classification"
ROOT_DIR = "/home/user/dataset/kaggle2020_leaf_v1"
TRAIN_DIR = "/home/user/dataset/kaggle2020_leaf_v1/train_images"
df = pd.read_csv(f"{ROOT_DIR}/train.csv")

class CFG:
    model_name = 'tf_efficientnet_b3_ns'
    img_size = 512

    T_max = 10
    T_0 = 10
    optim = 'sgd'
    if optim == 'adam':
        num_epochs = 10
        lr = 1e-4
        scheduler = 'CosineAnnealingWarmRestarts'
    else:
        num_epochs = 12
        milestones = [6, 10, 11]
        lr = 1e-1
        scheduler = 'step'
    min_lr = 1e-6
    batch_size = 32
    weight_decay = 1e-6
    seed = 42
    num_classes = 5

    n_fold = 5
    #NUM_FOLDS_TO_RUN = [2, ]
    NUM_FOLDS_TO_RUN = [0,1,2,3,4]
    smoothing = 0.3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('model:', model_name)
    print('optim:', optim)
    print('lr:', lr)
    print('scheduler:', scheduler)
    print('batchsize:', batch_size)
    print('epochs:', num_epochs)

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(CFG.seed)


#skf = StratifiedKFold(n_splits=CFG.n_fold)
skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)  # Otherwise your folds could intersect during different launching for training and it will lead to decreasing of the accuracy if you will use k-fold ansable for the submittion.
for fold, (_, val_) in enumerate(skf.split(X=df, y=df.label)):
    df.loc[val_, "kfold"] = int(fold) # clw note: add new column kfold in df

df['kfold'] = df['kfold'].astype(int)


class CassavaLeafDataset(nn.Module):
    def __init__(self, root_dir, df, transforms=None):
        self.root_dir = root_dir
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.df.iloc[index, 1]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return img, label


data_transforms = {
    "train": A.Compose([
        # A.Resize(height=600, width=800),  # clw note: if add 2019, need this
        A.RandomResizedCrop(CFG.img_size, CFG.img_size, scale=(0.4, 1.0), p=1.),  # clw add scale=(0.8, 1.0)
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.2,
            sat_shift_limit=0.2,
            val_shift_limit=0.2,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        #A.CoarseDropout(p=0.5, max_height=32, max_width=32),  # clw modify
        A.CoarseDropout(p=0.5),
        A.Cutout(p=0.5),
        ToTensorV2()], p=1.),


    "valid": A.Compose([
        A.Resize(600, 800),
        #A.CenterCrop(CFG.img_size, CFG.img_size, p=1.),  # clw delete
        A.CenterCrop(CFG.img_size, CFG.img_size, p=1.),  # clw delete
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
}


# implementations reference - https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
# paper - https://www.ijcai.org/Proceedings/2020/0305.pdf

class TaylorSoftmax(nn.Module):

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n + 1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out


class LabelSmoothingLoss(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        """Taylor Softmax and log are already applied on the logits"""
        # pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TaylorCrossEntropyLoss(nn.Module):

    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.2):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(CFG.num_classes, smoothing=smoothing)

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        # loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels)
        return loss


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = defaultdict(list)
    scaler = amp.GradScaler()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if (phase == 'train'):
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(CFG.device)
                labels = labels.to(CFG.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                if phase == 'train':
                    cut_mix_prob = 0
                else:
                    cut_mix_prob = 0
                with torch.set_grad_enabled(phase == 'train'):
                    with amp.autocast():
                        ##############################
                        if np.random.rand() < cut_mix_prob:
                            # generate mixed sample
                            # lam = np.random.beta(1.0, 1.0)
                            lam = np.clip(np.random.beta(1, 1), 0.3, 0.4)
                            rand_index = torch.randperm(inputs.size()[0]).cuda()
                            target_a = labels
                            target_b = labels[rand_index]
                            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                            # adjust lambda to exactly match pixel ratio
                            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                            # compute output
                            outputs = model(inputs)
                            # loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (
                                        1. - lam)  #### no label smooth when using cutmix
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        ################
                        # outputs = model(inputs)
                        # _, preds = torch.max(outputs, 1)
                        # loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).double().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            history[phase + ' loss'].append(epoch_loss)
            history[phase + ' acc'].append(epoch_acc)

            if phase == 'train' and scheduler != None:
                scheduler.step()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                PATH = f"Fold{fold}_{best_acc}_epoch{epoch}_v2.bin"
                torch.save(model.state_dict(), PATH)

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Accuracy ", best_acc)

    # load best model weights
    #model.load_state_dict(best_model_wts)  # clw delete
    #return model, history, best_acc
    return history, best_acc  # clw modify


def run_fold(model, criterion, optimizer, scheduler, device, fold, num_epochs=10):
    valid_df = df[df.kfold == fold]
    train_df = df[df.kfold != fold]

    train_data = CassavaLeafDataset(TRAIN_DIR, train_df, transforms=data_transforms["train"])
    valid_data = CassavaLeafDataset(TRAIN_DIR, valid_df, transforms=data_transforms["valid"])

    dataset_sizes = {
        'train': len(train_data),
        'valid': len(valid_data)
    }

    train_loader = DataLoader(dataset=train_data, batch_size=CFG.batch_size, num_workers=4, pin_memory=True,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=CFG.batch_size, num_workers=4, pin_memory=True,
                              shuffle=False)

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    #model, history, best_acc = train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders,dataset_sizes, device, fold)
    history, best_acc = train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders,dataset_sizes, device, fold)  # clw modify

    #return model, history, best_acc
    return history, best_acc  # clw modify




def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr)
    elif CFG.scheduler == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=CFG.milestones, gamma=0.1)
    elif CFG.scheduler == None:
        return None

    return scheduler




accs = []
for fold in CFG.NUM_FOLDS_TO_RUN:
    ### clw modify: move these in the for loop
    model = timm.create_model(CFG.model_name, pretrained=True)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, CFG.num_classes)
    model.to(CFG.device)

    if CFG.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    else:
        optimizer = optim.SGD(model.parameters(), lr=CFG.lr, momentum=0.9, weight_decay=CFG.weight_decay, nesterov=True)

    criterion = TaylorCrossEntropyLoss(n=2, smoothing=CFG.smoothing)
    scheduler = fetch_scheduler(optimizer)
    ###

    print(f"\n\nFOLD: {fold}\n\n")
    #model, history, ba = run_fold(model, criterion, optimizer, scheduler, device=CFG.device, fold=fold, num_epochs=CFG.num_epochs)
    history, ba = run_fold(model, criterion, optimizer, scheduler, device=CFG.device, fold=fold, num_epochs=CFG.num_epochs)  # clw modify
    accs.append(ba)

print(f"MEAN_ACC - {sum(accs)/len(accs)}")

