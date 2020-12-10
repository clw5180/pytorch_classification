from torch.utils.data import Dataset
from PIL import Image
import cv2  # clw modify
from config import configs  # clw modify


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
            img = cv2.resize(img, (configs.input_size, configs.input_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            ###############
            img = self.transforms(img_pil)
            return img, filename
        else:
            filename,label = self.imgs[index]
            #img = Image.open(filename).convert('RGB')

            ######## clw modify：因为前向推理要用cv2，无法复现transform.resize的结果（取值一个是0~255，一个是float64 0~1)
            img = cv2.imread(filename)
            img = cv2.resize(img, (configs.input_size, configs.input_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            ###############
            img = self.transforms(img_pil)
            return img,label


