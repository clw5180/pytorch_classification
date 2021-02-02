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

ROOT_DIR = "/home/user/dataset/kaggle2020-leaf-disease-classification"
TRAIN_DIR = "/home/user/dataset/kaggle2020-leaf-disease-classification/train_images"
TEST_DIR = "/home/user/dataset/kaggle2020-leaf-disease-classification/test_images"

class CFG:
    model_name = 'tf_efficientnet_b3_ns'
    img_size = 512
    scheduler = 'CosineAnnealingWarmRestarts'
    T_max = 10
    T_0 = 10
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 16
    #batch_size = 4
    weight_decay = 1e-6
    seed = 42
    num_classes = 5
    num_epochs = 10
    n_fold = 5
    #NUM_FOLDS_TO_RUN = [2, ]
    NUM_FOLDS_TO_RUN = [0,1,2,3,4]
    smoothing = 0.2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
df = pd.read_csv(f"{ROOT_DIR}/train.csv")

#skf = StratifiedKFold(n_splits=CFG.n_fold)
skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)  # Otherwise your folds could intersect during different launching for training and it will lead to decreasing of the accuracy if you will use k-fold ansable for the submittion.
for fold, (_, val_) in enumerate(skf.split(X=df, y=df.label)):
    df.loc[val_, "kfold"] = int(fold)

df['kfold'] = df['kfold'].astype(int)


from utils.utils import rand_bbox_clw
albu_transforms_train_cutmix =  [
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, interpolation=cv2.INTER_LINEAR, border_mode=0, p=0.85),
                A.ShiftScaleRotate(p=0.5),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=1),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1)], p = 0.7
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                A.CoarseDropout(p=0.5, max_height=32, max_width=32),
                #A.Cutout(p=0.5),
                ToTensorV2(),
            ]
train_aug_cutmix = A.Compose(albu_transforms_train_cutmix)


class CassavaLeafDataset(nn.Module):
    def __init__(self, root_dir, df, transforms=None, mode="train"):
        self.root_dir = root_dir
        self.df = df
        self.transforms = transforms
        # clw modify
        self.mode = mode
        self.random_crop = A.Compose( [A.RandomResizedCrop(CFG.img_size, CFG.img_size, scale=(0.8, 1.0), ratio=(0.75, 1.333333))])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.df.iloc[index, 1]

        # clw note: origin version
        # if self.transforms:
        #     img = self.transforms(image=img)["image"]
        # print(label)
        # return img, label

        label = torch.tensor(label).long()
        if self.mode == "train":
            if random.random() < 0.5:  # do cutmix
                img, label = self.do_cutmix(img, label, index)
            else:
                img = self.transforms(image=img)['image']
                label = torch.zeros(CFG.num_classes).scatter_(0, label, 1)  # one hot
        elif self.mode == "val":
            img = self.transforms(image=img)["image"]
            label = torch.zeros(CFG.num_classes).scatter_(0, label, 1)  # one hot
        return img, label

    def do_cutmix(self, img, label, index):
        '''
        Args:
            img: img to mixup
            label: label to mixup
            index: cutmix with other imgs in dataset, exclude itself( index )
        '''

        r_idx = random.choice(np.delete(np.arange(self.df.shape[0]), index))

        r_img_path = os.path.join(self.root_dir, self.df.iloc[r_idx, 0])
        r_img = cv2.imread(r_img_path)
        r_img = r_img[:, :, ::-1]

        img = self.random_crop(image=img)['image']
        r_img = self.random_crop(image=r_img)['image']
        ####
        img_h, img_w = r_img.shape[:2]

        lam = np.clip(np.random.beta(1, 1), 0.3, 0.4)
        ###lam = np.random.beta(1, 1)
        bbx1, bby1, bbx2, bby2 = rand_bbox_clw(img_w, img_h, lam)
        img_new = img.copy()
        img_new[bby1:bby2, bbx1:bbx2, :] = r_img[bby1:bby2, bbx1:bbx2, :]
        #cv2.imwrite(str(index) + '.jpg', img_new[:, :, ::-1])

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_h * img_w))
        label_one_hot = torch.zeros(CFG.num_classes).scatter_(0, label, 1)
        r_label = self.df.iloc[r_idx, 1]
        r_label = torch.tensor(r_label).long()
        r_label_one_hot = torch.zeros(CFG.num_classes).scatter_(0, r_label, 1)
        label_new = label_one_hot * lam + r_label_one_hot * (1 - lam)
        #img_new = train_aug(image=img_new)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边
        img_new = train_aug_cutmix(image=img_new)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边

        return img_new, label_new


data_transforms = {
    "train": A.Compose([
        # A.Resize(height=600, width=800),  # clw note: if add 2019, need this
        A.RandomResizedCrop(CFG.img_size, CFG.img_size, scale=(0.8, 1.0), p=1.),  # clw add scale=(0.8, 1.0)
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
        A.CoarseDropout(p=0.5, max_height=32, max_width=32),  # clw modify
        ToTensorV2()], p=1.),


    "valid": A.Compose([
        #A.CenterCrop(CFG.img_size, CFG.img_size, p=1.),  # clw delete
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
}



from utils.losses.taylorceloss import TaylorCrossEntropyLoss



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
                with torch.set_grad_enabled(phase == 'train'):
                    with amp.autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                labels = torch.argmax(labels, dim=1)  # one-hot to label
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

    train_data = CassavaLeafDataset(TRAIN_DIR, train_df, transforms=data_transforms["train"], mode="train")
    valid_data = CassavaLeafDataset(TRAIN_DIR, valid_df, transforms=data_transforms["valid"], mode="val")

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

    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    criterion = TaylorCrossEntropyLoss(n=2, smoothing=0.3)
    scheduler = fetch_scheduler(optimizer)
    ###

    print(f"\n\nFOLD: {fold}\n\n")
    #model, history, ba = run_fold(model, criterion, optimizer, scheduler, device=CFG.device, fold=fold, num_epochs=CFG.num_epochs)
    history, ba = run_fold(model, criterion, optimizer, scheduler, device=CFG.device, fold=fold, num_epochs=CFG.num_epochs)  # clw modify
    accs.append(ba)

print(f"MEAN_ACC - {sum(accs)/len(accs)}")

