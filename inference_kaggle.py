import torch
import os
import cv2
from torchvision import models
from tqdm import tqdm
import pandas as pd
import pretrainedmodels
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

img_path = "/home/user/dataset/test_toy"
model_path = "/home/user/pytorch_classification/checkpoints"
model_name = "resnet50_2021_01_02_15_15_10_fold{}_-checkpoint.pth.tar"

model_paths = []
for i in range(5):
    model_paths.append(os.path.join(model_path, model_name.format(str(i)) ) )

state_dicts = []
for i in range(5):
    state_dicts.append( torch.load(model_paths[i])['state_dict']  )

OUTPUT_CSV_DIR = './'
if not os.path.exists(OUTPUT_CSV_DIR):
    os.makedirs(OUTPUT_CSV_DIR)


if __name__ == '__main__':
    input_size = (512, 512)

    if 'se_resnext50' in model_path:
        model = pretrainedmodels.se_resnext50_32x4d(num_classes=5, pretrained=None)
        model.last_linear = nn.Linear(2048, 5)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
    else:
        model = models.resnet50(pretrained=False, num_classes=5)  # clw note: 默认是1000个类别的imagenet数据集，而我这里是2个

    model.to('cuda:0')
    model.eval()

    result_dict = {}
    result_label = []
    result_image_names = []

    img_names = os.listdir(img_path)

    for img_name in tqdm(img_names):
        img_file_path = os.path.join(img_path, img_name)
        img_origin = cv2.imread(img_file_path)
        img = img_origin.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # img = img[:, :, ::-1]
        img = cv2.resize(img, input_size,
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

        outputs = []
        for state_dict in state_dicts:
            model.load_state_dict(state_dict)

            with torch.no_grad():
                output = model(img_tensor)  # clw note: (batchsize, class_nums)
            outputs.append(output.softmax(dim=1).to('cpu').numpy())
        avg_preds = np.mean(outputs, axis=0)
        pred_label = np.argmax(avg_preds, 1)

        result_label.append(pred_label.item())
        result_image_names.append(img_name)

    # submission,写csv
    result_dict['image_id'] = result_image_names
    result_dict['label'] = result_label

    df_test = pd.DataFrame(result_dict)
    df_test.to_csv(os.path.join(OUTPUT_CSV_DIR, 'submission.csv'), index=False)
    print(df_test.head())