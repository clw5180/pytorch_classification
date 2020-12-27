DEBUG = True


import torch
import os
import cv2
import numpy as np
from torchvision import models
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from config import configs
import pandas as pd
import torch.backends.cudnn as cudnn
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True

img_path = "/home/user/dataset/kaggle2020_leaf/val/3"
OUTPUT_DIR = './'
if DEBUG:
    test = pd.read_csv('/home/user/dataset/sample_submission_file.csv')
else:
    test = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
print(test.head())



if __name__ == '__main__':

    input_size = 512

    #model_path = "/home/user/pytorch_classification/checkpoints/shufflenet_v2_x1_0-checkpoint.pth.tar"
    # my_state_dict_origin = torch.load(model_path)['state_dict']
    # my_state_dict = {}
    # for k, v in my_state_dict_origin.items():
    #     if 'tracked' in k:
    #         continue
    #     if 'backbone.' in k:
    #         k = k.replace('backbone.', '')
    #         # my_state_dict[k[9:]] = v  # clw note：去掉“backbone_”
    #     elif 'head.' in k:
    #         k = k.replace('head.', '')
    #         # my_state_dict[k[5:]] = v
    #     elif 'last_linear' in k:
    #         k = k.replace('last_linear', 'fc')
    #     my_state_dict[k] = v

    model_path = '/home/user/pytorch_classification/checkpoints/resnet50_2020_12_28_01_38_12-checkpoint.pth.tar'
    my_state_dict = torch.load(model_path)['state_dict']
    model = models.resnet50(pretrained=False, num_classes=5)   # clw note: 默认是1000个类别的imagenet数据集，而我这里是2个
    model.load_state_dict(my_state_dict)
    model.to('cuda:0')
    model.eval()

    result_dict = {}
    result_label = []
    result_image_names = []
    class_0_cnt = 0
    class_1_cnt = 0
    class_2_cnt = 0
    class_3_cnt = 0
    class_4_cnt = 0
    total_cnt = 0

    img_names = os.listdir(img_path)
    for img_name in tqdm(img_names):
        img_file_path = os.path.join(img_path, img_name)
        img_origin = cv2.imread(img_file_path)
        img = img_origin.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # img = img[:, :, ::-1]
        img = cv2.resize(img, (input_size, input_size),
                         interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC = 2  INTER_LINEAR = 1

        # opencv
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.float() / 255.0  # img_tensor = img_tensor.float()
        img_tensor[:, :, 0] = (img_tensor[:, :,
                               0] - 0.485) / 0.229  # img_tensor[:, :, 0] = (img_tensor[:, :, 0] - 123.675) / 58.395
        img_tensor[:, :, 1] = (img_tensor[:, :,
                               1] - 0.456) / 0.224  # img_tensor[:, :, 1] = (img_tensor[:, :, 1] - 116.28) / 57.12
        img_tensor[:, :, 2] = (img_tensor[:, :,
                               2] - 0.406) / 0.225  # img_tensor[:, :, 2] = (img_tensor[:, :, 2] - 103.53) / 57.375
        img_tensor = img_tensor.unsqueeze(0)  # (h, w, c) -> (1, h, w, c)
        img_tensor = img_tensor.permute((0, 3, 1, 2))  # (1, h, w, c) -> (n, c, h, w)

        img_tensor = img_tensor.to('cuda:0')
        output = model(img_tensor)
        output = torch.nn.functional.softmax(output, dim=1)  # clw note: (batchsize, class_nums)
        pred_score, pred_label = torch.max(output, 1)
        # pred_score = pred_score.item()  # clw note: assume bs=1
        if pred_label == 0:
            class_0_cnt += 1
        elif pred_label == 1:
            class_1_cnt += 1
        elif pred_label == 2:
            class_2_cnt += 1
        elif pred_label == 3:
            class_3_cnt += 1
        elif pred_label == 4:
            class_4_cnt += 1
        total_cnt += 1

        result_label.append(pred_label.item())
        result_image_names.append(img_name)

    # submission,写csv
    result_dict['image_id'] = result_image_names
    result_dict['label'] = result_label

    df_test = pd.DataFrame(result_dict)
    df_test.to_csv( os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)

    #print('acc: ', correct_cnt / total_cnt)
    print('total_cnt: ', total_cnt)
    print('class_0_cnt: ', class_0_cnt)
    print('class_1_cnt: ', class_1_cnt)
    print('class_2_cnt: ', class_2_cnt)
    print('class_3_cnt: ', class_3_cnt)
    print('class_4_cnt: ', class_4_cnt)
    # print(test)




