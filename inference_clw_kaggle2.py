DEBUG = True

import torch
import os
import cv2
from torchvision import models
from tqdm import tqdm
import pandas as pd
import torch.backends.cudnn as cudnn
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True

model_name = "resnet50_2020_12_31_20_29_11-checkpoint.pth.tar"
if DEBUG:
    model_root_path = "/home/user/pytorch_classification/checkpoints"
else:
    model_root_path = "/kaggle/input/cassava-resnet50-weights/"

model_path = os.path.join(model_root_path, model_name)

OUTPUT_CSV_DIR = './'


##########################################################
from utils.reader import WeatherDataset
from config import configs
import torchvision.transforms as transforms
from glob import glob
from itertools import chain

folder = '0'

def get_files(root,mode):
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files


if __name__ == '__main__':

    val_files = get_files(configs.dataset + "/val/" + folder + '/', "test")
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = WeatherDataset(val_files, transform_val, mode="test")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=configs.bs, shuffle=False,
        num_workers=configs.workers, pin_memory=True
    )


    input_size = 512

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

    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(val_loader)):
            #img_tensor = img_tensor.to('cuda:0')
            img_tensor = inputs[0].cuda()
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

    print('total_cnt: ', total_cnt)
    print('acc: ', class_0_cnt / total_cnt)
    print('class_0_cnt: ', class_0_cnt)
    print('class_1_cnt: ', class_1_cnt)
    print('class_2_cnt: ', class_2_cnt)
    print('class_3_cnt: ', class_3_cnt)
    print('class_4_cnt: ', class_4_cnt)




