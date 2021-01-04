from torch.utils.data import Dataset
from PIL import Image
import cv2  # clw modify
from config import configs  # clw modify
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torch
import random
import numpy as np

input_size = configs.input_size if isinstance(configs.input_size, tuple) else (configs.input_size, configs.input_size)

albu_transforms_train =  [
                ### single r50 89.1 solution
                # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT101, p=0.8),  # border_mode=cv2.BORDER_REPLICATE
                # A.VerticalFlip(p=0.5),
                # A.HorizontalFlip(p=0.5),
                # A.OneOf([A.RandomBrightness(limit=0.1, p=1), A.RandomContrast(limit=0.1, p=1)]),   # #A.RandomBrightnessContrast( brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                # A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3)], p=0.5),

                ### 2019 top1 solution
                # A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=30, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT101, p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.HorizontalFlip(p=0.5),
                # A.RandomResizedCrop(600, 800, scale=(0.6, 1.0), ratio=(0.6, 1.666666), p=0.5)

                ### new try
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT101, p=0.8),  # border_mode=cv2.BORDER_REPLICATE
                A.OneOf([A.VerticalFlip(p=1), A.HorizontalFlip(p=1)], p=0.5),
                A.OneOf([A.RandomBrightness(limit=0.1, p=1), A.RandomContrast(limit=0.1, p=1)]),   # #A.RandomBrightnessContrast( brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3)], p=0.5),
                A.CoarseDropout(max_holes=32, p=0.3),
                A.OneOf([A.RandomRotate90(p=1), A.Transpose(p=1)], p=0.5),
                A.Normalize(),
                ToTensorV2()

                #A.RandomResizedCrop(600, 800, scale=(0.8, 1.2), ratio=(0.75, 1.3333), p=0.5),  # #A.RandomCrop( int(input_size[1]*0.8), int(input_size[0]*0.8), p=0.5 ),  # clw note：注意这里顺序是 h, w;
                #A.RandomCrop( int(input_size[1]*0.8), int(input_size[0]*0.8), p=0.5 ),
                #A.CoarseDropout()
                #A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.5),  # A.Cutout(num_holes=16, max_h_size=32, max_w_size=32, p=0.5)
                #A.RandomRotate90(p=0.5)
                ###dict(type='CLAHE', p=0.5)  # clw note：gunzi 掉点明显
            ]

albu_transforms_val = [
                A.Normalize(),
                ToTensorV2()
            ]
train_aug = A.Compose(albu_transforms_train)
val_aug = A.Compose(albu_transforms_val)


class WeatherDataset(Dataset):
    # define dataset
    def __init__(self,label_list, mode="train"):
        super(WeatherDataset,self).__init__()
        self.label_list = label_list
        self.mode = mode
        imgs = []
        if self.mode == "test":
            for index,row in label_list.iterrows():
                imgs.append((row["filename"]))
            self.imgs = imgs
        else:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"],row["label"]))
            self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,index):
        if self.mode == "test":  # no label
            filename = self.imgs[index]
            img = cv2.imread(filename)
            input_size = configs.input_size if isinstance(configs.input_size, tuple) else (configs.input_size, configs.input_size)
            img = cv2.resize(img, input_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = val_aug(image=img)['image']
            return img, filename

        else:  # train or val, all need label
            filename, label = self.imgs[index]
            img = cv2.imread(filename)
            input_size = configs.input_size if isinstance(configs.input_size, tuple) else (configs.input_size, configs.input_size)
            img = cv2.resize(img, input_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            label = torch.tensor(label).long()
            if self.mode == "train":

                ### mixup
                if random.random() < 0.5:
                    mixup_ratio = np.random.beta(1.5, 1.5)

                    img = train_aug(image=img)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边

                    r_idx = random.choice(np.delete(np.arange(len(self.imgs)), index))
                    r_filename, r_label = self.imgs[r_idx]
                    r_img = cv2.imread(os.path.join(configs.dataset+"/train/", r_filename))
                    r_img = cv2.resize(r_img, input_size)
                    r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
                    r_img = train_aug(image=r_img)['image']

                    img = img * mixup_ratio + r_img * (1 - mixup_ratio)
                    ## cv2.imwrite(os.path.join("/home/user", self.file_names[idx] + '_' + self.file_names[r_idx]), img)

                    ### one-hot
                    label_one_hot = torch.zeros(configs.num_classes).scatter_(0, label, 1)
                    r_label = torch.tensor(r_label).long()
                    r_label_one_hot = torch.zeros(configs.num_classes).scatter_(0, r_label, 1)
                    label = label_one_hot * mixup_ratio + r_label_one_hot * (1 - mixup_ratio)
                else:
                    img = train_aug(image=img)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边
                    label = torch.zeros(configs.num_classes).scatter_(0, label, 1)
            else:
                img = val_aug(image=img)['image']
                label = torch.zeros(configs.num_classes).scatter_(0, label, 1)

            return img, label




# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, mode="train"):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).long()
        img = cv2.imread( os.path.join(configs.dataset_merge_csv, self.file_names[idx]) )
        input_size = configs.input_size if isinstance(configs.input_size, tuple) else (configs.input_size, configs.input_size)
        img = cv2.resize(img, input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == "train":
            ### mixup
            if random.random() < 0.5:
                mixup_ratio = np.random.beta(1.5, 1.5)

                img = train_aug(image=img)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边

                r_idx = random.choice(np.delete(np.arange(len(self.file_names)), idx))
                r_img = cv2.imread( os.path.join(configs.dataset_merge_csv, self.file_names[r_idx]))
                r_img = cv2.resize(r_img, input_size)
                r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
                r_img = train_aug(image=r_img)['image']

                img = img * mixup_ratio + r_img * (1 - mixup_ratio)
                ## cv2.imwrite(os.path.join("/home/user", self.file_names[idx] + '_' + self.file_names[r_idx]), img)

                ### one-hot
                label_one_hot = torch.zeros(configs.num_classes).scatter_(0, label, 1)
                r_label = torch.tensor(self.labels[r_idx]).long()
                r_label_one_hot = torch.zeros(configs.num_classes).scatter_(0, r_label, 1)
                label = label_one_hot * mixup_ratio + r_label_one_hot * (1 - mixup_ratio)
            else:
                img = train_aug(image=img)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边
                label = torch.zeros(configs.num_classes).scatter_(0, label, 1)
        else:
            img = val_aug(image=img)['image']
            label = torch.zeros(configs.num_classes).scatter_(0, label, 1)

        return img, label

