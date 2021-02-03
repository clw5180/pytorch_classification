from torch.utils.data import Dataset
import cv2  # clw modify
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torch
import random
import numpy as np
from utils.utils import rand_bbox_clw, RandomErasing, RandomErasing2
from config import configs  # clw modify


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

                ### now top
                # A.ShiftScaleRotate(shift_limit=0, scale_limit=0.05, rotate_limit=20, interpolation=cv2.INTER_LINEAR, border_mode=0, p=0.5),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.Transpose(p=0.5),
                # A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
                # A.Normalize(),   #A.Normalize(mean=(0.43032, 0.49673, 0.31342), std=(0.237595, 0.240453, 0.228265)),
                # ToTensorV2(),

                ### new try
                # A.Transpose(p=0.5),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # #A.RandomBrightness(limit=0.1, p=0.5),
                # #A.RandomContrast(limit=0.1, p=0.5),
                # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, interpolation=cv2.INTER_LINEAR, border_mode=0, p=0.85),
                # #A.OneOf([A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=1), A.Lambda(name='aaa', image=RandomErasing)], p=0.3),
                # A.Cutout(max_h_size=192, max_w_size=192, num_holes=1, p=0.7),
                # A.Normalize(),   #A.Normalize(mean=(0.43032, 0.49673, 0.31342), std=(0.237595, 0.240453, 0.228265)),
                # ToTensorV2(),

                ######### HolyCHen Vit !!!
                A.Resize(height=600, width=800),
                A.RandomResizedCrop(height=configs.input_size[1], width=configs.input_size[0], scale=(0.6, 1.0), p=1),
                A.CenterCrop(height=configs.input_size[1], width=configs.input_size[0]),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=1),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1)], p = 0.7
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                A.CoarseDropout(p=0.5, max_height=32, max_width=32),
                ToTensorV2(),
            ]

albu_transforms_train_cutmix =  [
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, interpolation=cv2.INTER_LINEAR, border_mode=0, p=0.85),
                A.ShiftScaleRotate(p=0.5),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=1),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1)], p = 0.7
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                A.CoarseDropout(p=0.5, max_height=32, max_width=32),
                # A.CoarseDropout(max_holes=12, max_height=int(0.11 * configs.input_size[1]),
                #                 max_width=int(0.11 * configs.input_size[0]),
                #                 min_holes=1, min_height=int(0.03 * configs.input_size[1]),
                #                 min_width=int(0.03 * configs.input_size[0]),
                #                 always_apply=False, p=0.5),
                #A.Cutout(p=0.5),
                ToTensorV2(),
            ]

albu_transforms_val = [
                A.Resize(height=configs.input_size[1], width=configs.input_size[0]),  # clw note: CenterCrop not good
                A.Normalize(),   #A.Normalize(mean=(0.43032, 0.49673, 0.31342), std=(0.237595, 0.240453, 0.228265)),
                ToTensorV2(),
                ######## HolyCHen Vit !!!
                # A.CenterCrop(input_size[0], input_size[1], p=0.5),
                # A.Resize(input_size[0], input_size[1]),
                # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                # ToTensorV2(),
            ]



train_aug = A.Compose(albu_transforms_train)
train_aug_cutmix = A.Compose(albu_transforms_train_cutmix)
val_aug = A.Compose(albu_transforms_val)


class CassavaTrainingDataset(Dataset):
    # define dataset
    def __init__(self,label_list, mode="train"):
        super(CassavaTrainingDataset,self).__init__()
        self.label_list = label_list
        self.mode = mode
        self.input_size = configs.input_size if isinstance(configs.input_size, tuple) else (configs.input_size, configs.input_size)
        self.Resize_Crop = A.Compose( [A.RandomResizedCrop(self.input_size[1], self.input_size[0], scale=(0.6, 1.0), ratio=(0.75, 1.333333))])
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row["filename"]))
        self.imgs = imgs
        if self.mode == "train" or self.mode == "val":
            self.do_mixup_prob = configs.do_mixup_in_dataset
            self.do_cutmix_prob = configs.do_cutmix_in_dataset
            assert (self.do_mixup_prob == 0 or self.do_cutmix_prob == 0)  # can't >0 both


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "val":  # train or val, all need label
            filename, label = self.imgs[index]
            label = torch.tensor(label).long()
            img = cv2.imread(filename)
            img = img[:, :, ::-1]

            if self.mode == "train":
                # if random.random() < self.do_mixup_prob:
                #     img, label = self.do_mixup(img, label, index)
                if random.random() < self.do_cutmix_prob:
                    img, label = self.do_cutmix(img, label, index)
                else:
                    img = train_aug(image=img)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边
                    label = torch.zeros(configs.num_classes).scatter_(0, label, 1)
            elif self.mode == "val":
                img = val_aug(image=img)['image']
                label = torch.zeros(configs.num_classes).scatter_(0, label, 1)
            return img, label

        elif self.mode == "test":  # no label
            filename = self.imgs[index]
            img = cv2.imread(filename)
            input_size = configs.input_size if isinstance(configs.input_size, tuple) else (configs.input_size, configs.input_size)
            img = cv2.resize(img, input_size)
            img = img[:, :, ::-1]
            img = val_aug(image=img)['image']
            return img, filename


    def do_mixup(self, img, label, index):
        '''
        Args:
            img: img to mixup
            label: label to mixup
            index: mixup with other imgs in dataset, exclude itself( index )
        '''
        input_size = configs.input_size if isinstance(configs.input_size, tuple) else (configs.input_size, configs.input_size)
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

        r_idx = random.choice(np.delete(np.arange(len(self.imgs)), index))
        r_filename, r_label = self.imgs[r_idx]
        r_img = cv2.imread(os.path.join(configs.dataset + "/train/", r_filename))
        r_img = r_img[:, :, ::-1]

        img = self.Resize_Crop(image=img)['image']
        r_img = self.Resize_Crop(image=r_img)['image']
        ####
        img_h, img_w = r_img.shape[:2]

        lam = np.clip(np.random.beta(1, 1), 0.3, 0.4)
        ###lam = np.random.beta(1, 1)
        bbx1, bby1, bbx2, bby2 = rand_bbox_clw(img_w, img_h, lam)
        img_new = img.copy()
        img_new[bby1:bby2, bbx1:bbx2, :] = r_img[bby1:bby2, bbx1:bbx2, :]
        #cv2.imwrite(str(index) + '.jpg', img_new[:, :, ::-1])

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_h * img_w))
        label_one_hot = torch.zeros(configs.num_classes).scatter_(0, label, 1)
        r_label = torch.tensor(r_label).long()
        r_label_one_hot = torch.zeros(configs.num_classes).scatter_(0, r_label, 1)
        label_new = label_one_hot * lam + r_label_one_hot * (1 - lam)
        #img_new = train_aug(image=img_new)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边
        img_new = train_aug_cutmix(image=img_new)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边

        return img_new, label_new


# ====================================================
# Val Dataset
# ====================================================

class CassavaValDataset(Dataset):  # evaluate才使用,训练不用
    # define dataset
    def __init__(self, label_list):
        super(CassavaValDataset,self).__init__()
        self.label_list = label_list
        self.input_size = (512, 512)

        imgs = []
        for index,row in label_list.iterrows():
            imgs.append((row["filename"],row["label"]))
            self.imgs = imgs

        self.val_aug0 = A.Compose([
            A.Resize(self.input_size[0], self.input_size[1], p=1.0),  # clw note:
            #A.CenterCrop(self.input_size[0], self.input_size[1], p=1.0),
            #A.RandomResizedCrop(self.input_size[0], self.input_size[1], scale=(0.8, 1.0), ratio=(0.75, 1.333333), p=1.0),
            A.Normalize(),  # A.Normalize(mean=(0.430, 0.497, 0.313), std=(0.238, 0.240, 0.228)),
            ToTensorV2()
        ])
        self.val_aug1 = A.Compose([
            A.Resize(self.input_size[0], self.input_size[1], p=1.0),
            #A.RandomResizedCrop(self.input_size[0], self.input_size[1], scale=(0.8, 1.0), ratio=(0.75, 1.333333), p=1.0),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=15, interpolation=cv2.INTER_LINEAR,
                               border_mode=4, p=1),

            A.Normalize(),  # A.Normalize(mean=(0.430, 0.497, 0.313), std=(0.238, 0.240, 0.228)),
            ToTensorV2()
        ])
        self.val_aug2 = A.Compose([
            A.Resize(self.input_size[0], self.input_size[1], p=1.0),
            # A.RandomResizedCrop(self.input_size[0], self.input_size[1], scale=(0.8, 1.0), ratio=(0.75, 1.333333), p=1.0),
            # A.HorizontalFlip(p=1),
            # A.VerticalFlip(p=1),
            A.Transpose(p=1),
            A.Normalize(),  # A.Normalize(mean=(0.430, 0.497, 0.313), std=(0.238, 0.240, 0.228)),
            ToTensorV2()
        ])
        self.val_aug3 = A.Compose([
            #A.Resize(self.input_size[0], self.input_size[1], p=1.0),
            A.RandomResizedCrop(self.input_size[0], self.input_size[1], scale=(0.8, 1.0), ratio=(0.75, 1.333333), p=1.0),
            A.HorizontalFlip(p=1),
            A.Normalize(),  # A.Normalize(mean=(0.430, 0.497, 0.313), std=(0.238, 0.240, 0.228)),
            ToTensorV2()
        ])
        self.val_aug4 = A.Compose([
            #A.Resize(self.input_size[0], self.input_size[1], p=1.0),
            A.RandomResizedCrop(self.input_size[0], self.input_size[1], scale=(0.8, 1.0), ratio=(0.75, 1.333333), p=1.0),
            A.VerticalFlip(p=1),
            A.Normalize(),  # A.Normalize(mean=(0.430, 0.497, 0.313), std=(0.238, 0.240, 0.228)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        label = torch.tensor(label).long()
        label = torch.zeros(configs.num_classes).scatter_(0, label, 1)
        img = cv2.imread(filename)
        img = img[:, :, ::-1]

        ########################################
        # img0_tensor = self.val_aug0(image=img)['image']
        # img1_tensor = self.val_aug1(image=img)['image']
        # img2_tensor = self.val_aug2(image=img)['image']
        # img3_tensor = self.val_aug3(image=img)['image']
        # img4_tensor = self.val_aug4(image=img)['image']  # (c, h, w)
        # img_tensor = torch.cat((img0_tensor, img1_tensor, img2_tensor, img3_tensor, img4_tensor), 2)
        # return img_tensor, label
        #####################################

        img0_tensor = self.val_aug0(image=img)['image']
        return img0_tensor, label


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

