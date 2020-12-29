from torch.utils.data import Dataset
from PIL import Image
import cv2  # clw modify
from config import configs  # clw modify
import albumentations as A

albu_transforms =  [
                #A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0, rotate_limit=15, interpolation=1, border_mode=1, p=0.5),  # border_mode=cv2.BORDER_REPLICATE
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT101, p=0.8),  # border_mode=cv2.BORDER_REPLICATE
                #A.RandomBrightnessContrast( brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.OneOf([A.RandomBrightness(limit=0.1, p=1), A.RandomContrast(limit=0.1, p=1)]),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3)], p=0.5)
                #A.RandomRotate90(p=0.5)
                ###dict(type='CLAHE', p=0.5)  # clw note：掉点明显
                ###dict(type='MotionBlur', blur_limit=12, p=0.5)  # clw note：掉点明显
            ]
aug = A.Compose(albu_transforms)
print(str(albu_transforms) + '\n')

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
            #img = Image.open(filename).convert('RGB')

            ######## clw modify：因为前向推理要用cv2，无法复现transform.resize的结果（取值一个是0~255，一个是float64 0~1)
            img = cv2.imread(filename)
            input_size = configs.input_size
            if isinstance(input_size, tuple):
                img = cv2.resize(img, configs.input_size)
            else:
                img = cv2.resize(img, (configs.input_size, configs.input_size))

            ## 转成pil格式，做transform提供的增强
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img = self.transforms(img_pil)

            return img, filename
        elif self.mode == "val":
            filename, label = self.imgs[index]
            #img = Image.open(filename).convert('RGB')

            ######## clw modify：因为前向推理要用cv2，无法复现transform.resize的结果（取值一个是0~255，一个是float64 0~1)
            img = cv2.imread(filename)
            input_size = configs.input_size
            if isinstance(input_size, tuple):
                img = cv2.resize(img, configs.input_size)
            else:
                img = cv2.resize(img, (configs.input_size, configs.input_size))

            ## 转成pil格式，做transform提供的增强
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img = self.transforms(img_pil)

            return img, label
        else:
            filename,label = self.imgs[index]
            #img = Image.open(filename).convert('RGB')

            ######## clw modify：因为前向推理要用cv2，无法复现transform.resize的结果（取值一个是0~255，一个是float64 0~1)
            img = cv2.imread(filename)
            input_size = configs.input_size
            if isinstance(input_size, tuple):
                img = cv2.resize(img, configs.input_size)
            else:
                img = cv2.resize(img, (configs.input_size, configs.input_size))

            img_augmented = aug(image=img)['image']

            ## 转成pil格式，做transform提供的增强
            img_augmented = cv2.cvtColor(img_augmented, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_augmented)
            ###############
            img = self.transforms(img_pil)
            return img,label


