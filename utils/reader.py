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
from utils.utils import rand_bbox_clw, RandomErasing, RandomErasing2

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
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.05, rotate_limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT101, p=0.5),
 # border_mode=cv2.BORDER_REPLICATE  BORDER_REFLECT101 BORDER_CONSTANT
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
                #A.Lambda(image=RandomErasing),
                A.OneOf([A.RandomRotate90(p=1), A.Transpose(p=1)], p=0.5),
                A.Normalize(),   #A.Normalize(mean=(0.43032, 0.49673, 0.31342), std=(0.237595, 0.240453, 0.228265)),
                ToTensorV2(),

                ######### HolyCHen Vit !!!
                # A.RandomResizedCrop(height=input_size[0], width=input_size[1], p=0.5),
                # A.Transpose(p=0.5),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                # A.ShiftScaleRotate(p=0.5),
                # A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                # A.CenterCrop(input_size[0], input_size[1]),
                # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                # A.CoarseDropout(p=0.5),
                # A.Cutout(p=0.5),
                # ToTensorV2(),
            ]

albu_transforms_val = [
                A.Normalize(),   #A.Normalize(mean=(0.43032, 0.49673, 0.31342), std=(0.237595, 0.240453, 0.228265)),
                ToTensorV2(),

                ######## HolyCHen Vit !!!
                # A.CenterCrop(input_size[0], input_size[1], p=0.5),
                # A.Resize(input_size[0], input_size[1]),
                # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                # ToTensorV2(),

            ]
train_aug = A.Compose(albu_transforms_train)
random_crop = A.Compose([A.RandomResizedCrop(input_size[0], input_size[1], scale=(0.6, 1.0), ratio=(0.6, 1.66666))])
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


            self.do_mixup_prob = configs.do_mixup_in_dataset
            self.do_cutmix_prob = configs.do_cutmix_in_dataset
            assert (self.do_mixup_prob == 0 or self.do_cutmix_prob == 0)  # can't >0 both


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if not self.mode == "test":  # train or val, all need label
            filename, label = self.imgs[index]
            img = cv2.imread(filename)
            ### 1 resize
            input_size = configs.input_size if isinstance(configs.input_size, tuple) else (configs.input_size, configs.input_size)
            img = cv2.resize(img, input_size)
            ### 2 RandomResizedCrop
            #img = random_crop(image=img)['image']
            ####
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = torch.tensor(label).long()
            if self.mode == "train":
                if random.random() < self.do_mixup_prob:
                    img, label = self.do_mixup(img, label, index)
                elif random.random() < self.do_cutmix_prob:
                    img, label = self.do_cutmix(img, label, index)
                else:
                    img = train_aug(image=img)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边
                    label = torch.zeros(configs.num_classes).scatter_(0, label, 1)
            elif self.mode == "val":
                img = val_aug(image=img)['image']
                label = torch.zeros(configs.num_classes).scatter_(0, label, 1)
            return img, label

        else:  # no label
            filename = self.imgs[index]
            img = cv2.imread(filename)
            input_size = configs.input_size if isinstance(configs.input_size, tuple) else (configs.input_size, configs.input_size)
            img = cv2.resize(img, input_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = val_aug(image=img)['image']
            return img, filename


    def do_mixup(self, img, label, index):
        '''
        Args:
            img: img to mixup
            label: label to mixup
            index: mixup with other imgs in dataset, exclude itself( index )
        '''
        mixup_ratio = np.random.beta(1.5, 1.5)

        img = train_aug(image=img)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边

        r_idx = random.choice(np.delete(np.arange(len(self.imgs)), index))
        r_filename, r_label = self.imgs[r_idx]
        r_img = cv2.imread(os.path.join(configs.dataset + "/train/", r_filename))
        r_img = cv2.resize(r_img, input_size)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        r_img = train_aug(image=r_img)['image']
        img_new = img * mixup_ratio + r_img * (1 - mixup_ratio)
        label_one_hot = torch.zeros(configs.num_classes).scatter_(0, label, 1)
        r_label = torch.tensor(r_label).long()
        r_label_one_hot = torch.zeros(configs.num_classes).scatter_(0, r_label, 1)
        label_new = label_one_hot * mixup_ratio + r_label_one_hot * (1 - mixup_ratio)
        return img_new, label_new


    def do_cutmix(self, img, label, index):
        '''
        Args:
            img: img to mixup
            label: label to mixup
            index: cutmix with other imgs in dataset, exclude itself( index )
        '''
        img_h, img_w = img.shape[:2]

        r_idx = random.choice(np.delete(np.arange(len(self.imgs)), index))
        r_filename, r_label = self.imgs[r_idx]
        r_img = cv2.imread(os.path.join(configs.dataset + "/train/", r_filename))
        r_img = cv2.resize(r_img, input_size)
        ###r_img = random_crop(image=r_img)['image']

        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)

        lam = np.clip(np.random.beta(1, 1), 0.3, 0.4)
        #lam = 0.9
        bbx1, bby1, bbx2, bby2 = rand_bbox_clw(img_w, img_h, lam)
        img_new = img.copy()
        img_new[bby1:bby2, bbx1:bbx2, :] = r_img[bby1:bby2, bbx1:bbx2, :]


        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_h * img_w))
        label_one_hot = torch.zeros(configs.num_classes).scatter_(0, label, 1)
        r_label = torch.tensor(r_label).long()
        r_label_one_hot = torch.zeros(configs.num_classes).scatter_(0, r_label, 1)
        label_new = label_one_hot * lam + r_label_one_hot * (1 - lam)
        img_new = train_aug(image=img_new)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边

        return img_new, label_new




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

