from torch.utils.data import Dataset
from PIL import Image
import cv2  # clw modify
from config import configs  # clw modify
import albumentations as A
import os
import torch

input_size = configs.input_size if isinstance(configs.input_size, tuple) else (configs.input_size, configs.input_size)

albu_transforms =  [
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
                A.CoarseDropout(max_holes=512, p=0.3),
                A.OneOf([A.RandomRotate90(p=1), A.Transpose(p=1)], p=0.5),


                #A.RandomResizedCrop(600, 800, scale=(0.8, 1.2), ratio=(0.75, 1.3333), p=0.5),  # #A.RandomCrop( int(input_size[1]*0.8), int(input_size[0]*0.8), p=0.5 ),  # clw note：注意这里顺序是 h, w;
                #A.RandomCrop( int(input_size[1]*0.8), int(input_size[0]*0.8), p=0.5 ),
                #A.CoarseDropout()
                #A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.5),  # A.Cutout(num_holes=16, max_h_size=32, max_w_size=32, p=0.5)
                #A.RandomRotate90(p=0.5)
                ###dict(type='CLAHE', p=0.5)  # clw note：gunzi 掉点明显
            ]
aug = A.Compose(albu_transforms)


class WeatherDataset(Dataset):
    # define dataset
    def __init__(self,label_list,transforms=None,mode="train"):
        super(WeatherDataset,self).__init__()
        self.label_list = label_list
        self.transforms = transforms
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
        if self.mode == "test":
            filename = self.imgs[index]
            #img = Image.open(filename).convert('RGB')  # clw modify：因为前向推理要用cv2，无法复现transform.resize的结果（取值一个是0~255，一个是float64 0~1)
            img = cv2.imread(filename)
            input_size = configs.input_size
            if isinstance(input_size, tuple):
                img = cv2.resize(img, configs.input_size)
            else:
                img = cv2.resize(img, (configs.input_size, configs.input_size))

            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )  ## 转成pil格式，做transform提供的增强
            img = self.transforms(img_pil)
            return img, filename
        elif self.mode == "val":
            filename, label = self.imgs[index]
            #img = Image.open(filename).convert('RGB')  # clw modify：因为前向推理要用cv2，无法复现transform.resize的结果（取值一个是0~255，一个是float64 0~1)
            img = cv2.imread(filename)
            input_size = configs.input_size
            if isinstance(input_size, tuple):
                img = cv2.resize(img, configs.input_size)
            else:
                img = cv2.resize(img, (configs.input_size, configs.input_size))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)              ## 转成pil格式，做transform提供的增强
            img = self.transforms(img_pil)
            return img, label
        else:
            filename,label = self.imgs[index]
            #img = Image.open(filename).convert('RGB')  # clw modify：因为前向推理要用cv2，无法复现transform.resize的结果（取值一个是0~255，一个是float64 0~1)
            img = cv2.imread(filename)
            input_size = configs.input_size

            img_augmented = aug(image=img)['image']   # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边
            if isinstance(input_size, tuple):
                img_augmented = cv2.resize(img_augmented, configs.input_size)
            else:
                img_augmented = cv2.resize(img_augmented, (configs.input_size, configs.input_size))

            ## 转成pil格式，做transform提供的增强
            img_augmented = cv2.cvtColor(img_augmented, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_augmented)
            ###############
            img = self.transforms(img_pil)
            return img,label


# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(configs.dataset_merge_csv, file_name)   # f'{TRAIN_PATH}/{file_name}'
        # image = cv2.imread(file_path)
        # if self.transform:
        #     augmented = self.transform(image=image)
        #     image = augmented['image']
        # label = torch.tensor(self.labels[idx]).long()
        # return image, label

        ### clw modify
        img = cv2.imread(file_path)
        input_size = configs.input_size
        img_augmented = aug(image=img)['image']  # clw note: 考虑到这里有crop等导致输入尺寸不同的操作，把resize放在后边
        if isinstance(input_size, tuple):
            img_augmented = cv2.resize(img_augmented, configs.input_size)
        else:
            img_augmented = cv2.resize(img_augmented, (configs.input_size, configs.input_size))

        ## 转成pil格式，做transform提供的增强
        img_augmented = cv2.cvtColor(img_augmented, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_augmented)
        ###############
        img = self.transform(img_pil)
        label = torch.tensor(self.labels[idx]).long()
        return img, label


# class TestDataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.df = df
#         self.file_names = df['image_id'].values
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         file_name = self.file_names[idx]
#         file_path = f'{TEST_PATH}/{file_name}'
#         image = cv2.imread(file_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         if self.transform:
#             augmented = self.transform(image=image)
#             image = augmented['image']
#         return image