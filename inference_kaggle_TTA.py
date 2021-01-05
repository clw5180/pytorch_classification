DEBUG = True

import torch
import os
import cv2
from tqdm import tqdm
import pandas as pd
import timm
import torch.nn as nn
import numpy as np
import psutil
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data.sampler import *
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


print('ram:', '%dGB' % int(psutil.virtual_memory()[0] / 1024 / 1024 / 1024))
print('cpu:', '%dcore' % psutil.cpu_count())
print('gpu:', str(torch.cuda.get_device_properties(0))[22:-1])
print('')

if DEBUG:
    img_path = "/home/user/dataset/kaggle2020_leaf/val/" + '1'
    checkpoint0 = [
        "/home/user/pytorch_classification/checkpoints/efficientnet-b3_2021_01_05_15_43_49-checkpoint.pth.tar"
    ]
    checkpoint1 = []
else:
    import sys

    sys.path.append('../input/pretrainedmodels')
    sys.path.append('../input/pytorchimagemodels')
    img_path = "/kaggle/input/cassava-leaf-disease-classification/test_images"
    checkpoint0 = [
        "../input/20210105-effb3-modify-normalize/efficientnet-b3_2021_01_05_15_43_49-checkpoint.pth.tar"  # clw note: need modify
    ]
    checkpoint1 = []



input_size = (512, 512)
image_id = list(os.listdir(img_path))

albu_transforms_val = [
                A.Normalize(), #A.Normalize(mean=(0.430, 0.497, 0.313), std=(0.238, 0.240, 0.228)),
                ToTensorV2()
            ]
val_aug = A.Compose(albu_transforms_val)


class CassavaDataset(Dataset):
    def __init__(self, ):
        pass

    def __len__(self):
        return len(image_id)

    def __getitem__(self, index):
        img = cv2.imread(img_path + '/%s' % image_id[index])
        img = cv2.resize(img, input_size)
        img = img[:, :, ::-1]  # clw note: faster than   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = val_aug(image=img)['image']

        return img



if __name__ == '__main__':


    # if 'se_resnext50' in model_path:
    #     model = pretrainedmodels.se_resnext50_32x4d(num_classes=5, pretrained=None)
    #     model.last_linear = nn.Linear(2048, 5)
    #     model.avg_pool = nn.AdaptiveAvgPool2d(1)
    # elif "efficientnet-b2" in model_path:
    #     model = timm.create_model('tf_efficientnet_b2_ns', pretrained=False)
    #     model.classifier = nn.Linear(model.classifier.in_features, 5)
    # elif "efficientnet-b3" in model_path:
    #     model = timm.create_model('tf_efficientnet_b3_ns', pretrained=False)
    #     model.classifier = nn.Linear(model.classifier.in_features, 5)
    # elif "efficientnet-b4" in model_path:
    #     model = timm.create_model('tf_efficientnet_b4_ns', pretrained=False)
    #     model.classifier = nn.Linear(model.classifier.in_features, 5)
    # else:
    #     model = models.resnet50(pretrained=False, num_classes=5)  # clw note: 默认是1000个类别的imagenet数据集，而我这里是2个


    # model ---
    net0 = []
    for f in checkpoint0:
        model = timm.create_model('tf_efficientnet_b3_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 5)
        model.cuda()
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage)['state_dict'], strict=True)
        net0.append(model)

    net1 = []
    for f in checkpoint1:
        model = timm.create_model('tf_efficientnet_b3_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 5)
        model.cuda()
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage)['state_dict'], strict=True)
        net1.append(model)

    dataset = CassavaDataset()
    data_loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=32,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    # start here! ------------------
    probability = []
    with torch.no_grad():
        for t, image in enumerate(tqdm(data_loader)):
            image = image.cuda()

            p = []
            for net in net0 + net1:  #
                net.eval()
                logit = net(image)
                aaa = F.softmax(logit, -1)
                p.append(aaa)

                # tta ----
                if 0:
                    logit = net(torch.flip(image, dims=(2,)).contiguous())
                    p.append(F.softmax(logit, -1))

                    logit = net(torch.flip(image, dims=(3,)).contiguous())
                    p.append(F.softmax(logit, -1))

                    logit = net(torch.flip(image, dims=(2, 3)).contiguous())
                    p.append(F.softmax(logit, -1))

                    logit = net(image.permute(0, 1, 3, 2).contiguous())
                    p.append(F.softmax(logit, -1))

            # ---------
            p = torch.stack(p).mean(0)
            probability.append(p.data.cpu().numpy())


    probability = np.concatenate(probability)
    predict = probability.argmax(-1)

    # submission,写csv
    df_submit = pd.DataFrame({'image_id': image_id, 'label': predict})
    print('df_submit', df_submit.shape)
    print(df_submit)


    df_submit.to_csv('submission.csv', index=False)

    print(df_submit.head())
    print(predict)

    if DEBUG:
        true_label = int(img_path[-1])
        print('acc: ', sum(predict == true_label) / len(predict)  )

