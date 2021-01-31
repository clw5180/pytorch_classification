import torch
import os
import cv2
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import numpy as np
import pretrainedmodels
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data.sampler import *
import torch.nn.functional as F
import timm
import shutil


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


checkpoint0 = [
    "/home/user/pytorch_classification/checkpoints/se_resnext50_32x4d_2021_01_03_00_12_40_fold0-checkpoint.pth.tar",
    "/home/user/pytorch_classification/checkpoints/se_resnext50_32x4d_2021_01_03_00_12_40_fold1-checkpoint.pth.tar",
    "/home/user/pytorch_classification/checkpoints/se_resnext50_32x4d_2021_01_03_00_12_40_fold2-checkpoint.pth.tar",
    "/home/user/pytorch_classification/checkpoints/se_resnext50_32x4d_2021_01_03_00_12_40_fold3-checkpoint.pth.tar",
    "/home/user/pytorch_classification/checkpoints/se_resnext50_32x4d_2021_01_03_00_12_40_fold4-checkpoint.pth.tar"
               ]
# checkpoint0 = [
#     "/home/user/pytorch_classification/checkpoints/se_resnext50_32x4d_2021_01_03_00_12_40_fold0-checkpoint.pth.tar"
#                ]

img_path = "/home/user/dataset/kaggle2020_leaf/all/"
img_sub_folder = os.listdir(img_path)
img_dirs = [os.path.join(img_path, s) for s in img_sub_folder]
save_path = 'output'
if not os.path.exists('output'):
    os.makedirs('output')
classes_nums = 5
for i in range(classes_nums):
    if not os.path.exists(os.path.join(save_path, str(i))):
        os.makedirs(os.path.join(save_path, str(i)))




input_size = (512, 512)


albu_transforms_val = [
                A.Normalize(), #A.Normalize(mean=(0.430, 0.497, 0.313), std=(0.238, 0.240, 0.228)),
                ToTensorV2()
            ]
val_aug = A.Compose(albu_transforms_val)


class CassavaValDataset(Dataset):
    def __init__(self, ):
        self.img_paths = []
        self.labels = []
        for img_dir in img_dirs:
            class_id = int(img_dir.split('/')[-1])
            img_names = os.listdir(img_dir)
            for img_name in tqdm(img_names):
                self.img_paths.append(os.path.join(img_dir, img_name) )
                self.labels.append(class_id)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        label = self.labels[index]
        img_path = self.img_paths[index]
        img0 = cv2.imread(img_path)
        img = cv2.resize(img0, input_size)
        img = img[:, :, ::-1]  # clw note: faster than   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = val_aug(image=img)['image']
        return img, label, img_path



if __name__ == '__main__':

    # model ---
    net0 = []
    for f in checkpoint0:
        # model = timm.create_model('tf_efficientnet_b3_ns', pretrained=False)
        # model.classifier = nn.Linear(model.classifier.in_features, 5)
        model = pretrainedmodels.se_resnext50_32x4d(num_classes=5, pretrained=None)
        model.last_linear = nn.Linear(2048, 5)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)

        model.cuda()
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage)['state_dict'], strict=True)
        net0.append(model)

    dataset = CassavaValDataset()
    data_loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=1,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    print('total img nums:', len(dataset))

    # start here! ------------------
    probability = []
    with torch.no_grad():
        for i, (image, label, img_path) in enumerate(tqdm(data_loader)):
            img_path = img_path[0]
            if '681602202.jpg' in img_path:
                print('aaa')

            image = image.cuda()

            ps = []
            for net in net0:  #
                net.eval()
                logit = net(image)
                p = F.softmax(logit, -1)
                p = p[0]  # bs = 1
                ps.append(p)


            # ---------
            # ---------
            aaa = torch.stack(ps)
            bbb = aaa[:, label.item()] < 0.2
            ccc = bbb.sum()
            ddd = torch.nonzero(bbb)
            ddd = ddd.squeeze(dim=1)
            eee = aaa[ddd]
            if(ccc >= 2 ):
                fff = eee.mean(0)
                # ps_mean = torch.stack(ps).mean(0)
                # if ps_mean[label.item()] < 0.2:  # or other strategy to filter
                predicted_classid = fff.argmax(-1).cpu().item()
                print(img_path, 'labeled:', label.item(), '  predicted:', predicted_classid, '  score:',
                      fff.cpu().numpy())
                #shutil.copy(img_path, os.path.join(save_path, str(predicted_classid)))
                shutil.move(img_path, os.path.join(save_path, str(predicted_classid)))


'''
                ggg = eee.max(dim=1).values
                if ggg.min() > 0.4:  # 0.5:
                    fff = eee.mean(0)
                    #ps_mean = torch.stack(ps).mean(0)
                    #if ps_mean[label.item()] < 0.2:  # or other strategy to filter
                    predicted_classid = fff.argmax(-1).cpu().item()
                    print(img_path, 'labeled:', label.item(), '  predicted:', predicted_classid, '  score:', fff.cpu().numpy())
                    shutil.copy(img_path, os.path.join(save_path, str(predicted_classid)))
                else:
                    fff = eee.mean(0)
                    predicted_classid = fff.argmax(-1).cpu().item()
                    print('====== not assure: ', img_path, 'labeled:', label.item(), '  predicted:', predicted_classid, '  score:', fff.cpu().numpy())
'''