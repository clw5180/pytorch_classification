# 读取 torch.jit.save保存的模型（不仅有state_dict，还有网络结构不再需要cfg加载），直接前传

import torch
import os
import cv2
import numpy as np
from torchvision import models
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import shutil
from config import configs

from tensorboardX import SummaryWriter   # clw modify: it's quicker than   #from torch.utils.tensorboard import SummaryWriter
tb_logger = SummaryWriter()  # clw modify
from torchvision.utils import make_grid

import torch.backends.cudnn as cudnn
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True


def draw_result(img,
                result,  # clw note: such as  {'pred_label':1, 'pred_score':0.98}
                out_file,
                text_color=(0, 255, 0),
                font_scale=0.5,
                row_width=20,
                ):
    """Draw `result` over `img`.

    Args:
        img (ndarray): The image to be displayed.
        result (dict): The classification results to draw over `img`.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        font_scale (float): Font scales of texts.
        row_width (int): width between each row of results on the image.
        show (bool): Whether to show the image.
            Default: False.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """

    # write results on left-top of the image
    x, y = 0, row_width
    for k, v in result.items():
        if isinstance(v, float):
            v = f'{v:.2f}'
        label_text = f'{k}: {v}'
        cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_COMPLEX,
                    font_scale, text_color)
        y += row_width

    cv2.imwrite(out_file, img)


if __name__ == '__main__':
    #model_path = "/home/user/pytorch_classification/checkpoints/resnet50-checkpoint.pth.tar"
    #model_path = "/home/user/pytorch_classification/checkpoints/resnet50-best_loss.pth.tar"
    # model_path = "/home/user/pytorch_classification/checkpoints/resnet50-best_model.pth.tar"

    model_path = "/home/user/pytorch_classification/checkpoints/shufflenet_v2_x1_0-checkpoint.pth.tar"
    #model_path = "/home/user/pytorch_classification/checkpoints/shufflenet_v2_x1_0-best_loss.pth.tar"
    #model_path = "/home/user/pytorch_classification/checkpoints/shufflenet_v2_x1_0-best_model.pth.tar"

    input_size = 512

    my_state_dict_origin = torch.load(model_path)['state_dict']
    my_state_dict = {}
    for k,v in my_state_dict_origin.items():
        if 'tracked' in k:
            continue
        if 'backbone.' in k:
            k = k.replace('backbone.', '')
            #my_state_dict[k[9:]] = v  # clw note：去掉“backbone_”
        elif 'head.' in k:
            k = k.replace('head.', '')
            #my_state_dict[k[5:]] = v
        elif 'last_linear' in k:
            k=k.replace('last_linear', 'fc')
        my_state_dict[k] = v

    official_state_dict = torch.load('/home/user/.cache/torch/checkpoints/resnet50-19c8e357.pth', map_location='cpu')
    #model = models.resnet50(pretrained=False, num_classes=2)   # clw note: 默认是1000个类别的imagenet数据集，而我这里是2个
    model = models.shufflenet_v2_x1_0(pretrained=False, num_classes=configs.num_classes)

    model.load_state_dict(my_state_dict)
    model.to('cuda:0')
    model.eval()

    save_path = './output'

    #img_path = "/home/user/dataset/gunzi/test_ng"
    #img_path = "/home/user/dataset/gunzi/val"
    #img_path = "/home/user/dataset/gunzi/test_toy"
    img_path = configs.test_folder
    img_names = os.listdir(img_path)

    img_names_ok = []
    img_names_ng = []
    for i, img_name in tqdm(enumerate(img_names)):
        img_file_path = os.path.join(img_path, img_name)
        img_origin = cv2.imread(img_file_path)
        img = img_origin.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # img = img[:, :, ::-1]
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR) # INTER_CUBIC = 2  INTER_LINEAR = 1


        # PIL
        # img_pil = Image.open(img_file_path).convert('RGB')
        # img_pil = Image.fromarray(img)
        # img_pil = transforms.Resize((224, 224))(img_pil)
        # img = np.asarray(img_pil)
        # img_tensor = transforms.ToTensor()(img)
        # img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
        # img_tensor = img_tensor.unsqueeze(0)

        # opencv
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.float() / 255.0                               # img_tensor = img_tensor.float()
        img_tensor[:, :, 0] = (img_tensor[:, :, 0] - 0.485) / 0.229           # img_tensor[:, :, 0] = (img_tensor[:, :, 0] - 123.675) / 58.395
        img_tensor[:, :, 1] = (img_tensor[:, :, 1] - 0.456) / 0.224           # img_tensor[:, :, 1] = (img_tensor[:, :, 1] - 116.28) / 57.12
        img_tensor[:, :, 2] = (img_tensor[:, :, 2] - 0.406) / 0.225           # img_tensor[:, :, 2] = (img_tensor[:, :, 2] - 103.53) / 57.375
        img_tensor = img_tensor.unsqueeze(0) # (h, w, c) -> (1, h, w, c)
        img_tensor = img_tensor.permute((0, 3, 1, 2))  #  (1, h, w, c) -> (n, c, h, w)

        img_tensor = img_tensor.to('cuda:0')
        #output = model(img_tensor)
        ################################################### clw modify ####################################################
        feature_1, feature_2, feature_3, feature_4, output = model(img_tensor)
        aaa = feature_4.squeeze()
        bbb = torch.sum(aaa, dim=0)
        ccc = bbb.cpu().detach().numpy()
        ddd = (ccc * 255.0 / ccc.max() ).astype(np.uint8)
        feature_h = ddd.shape[0]
        feature_w = ddd.shape[1]
        img_out = img.copy()  # 注意必须是resize之后的，不要用原图...
        for i in range(feature_h):
            for j in range(feature_w):
                if ddd[i][j] > 128:
                    x = j*32  # clw note: 如果用的feature map4且对应网络是下采样32倍，可以这么写...
                    y = i*32
                    cv2.rectangle(img_out, (x, y), (x+32, y+32), (0, 0, 255), thickness=2)
        cv2.imwrite(os.path.join(save_path, img_name[:-4] + '_show_localization.jpg'), img_out)



        ################################################### clw modify ####################################################
        tb_logger.add_image('feature_1', make_grid(feature_1[0].unsqueeze(dim=1), normalize=False), i)
        tb_logger.add_image('feature_2', make_grid(feature_2[0].unsqueeze(dim=1), normalize=False), i)
        tb_logger.add_image('feature_3', make_grid(feature_3[0].unsqueeze(dim=1), normalize=False), i)
        tb_logger.add_image('feature_4', make_grid(feature_4[0].unsqueeze(dim=1), normalize=False), i)

        tb_logger.add_image('feature_111', make_grid(torch.sum(feature_1[0], dim=0), normalize=True), i)
        tb_logger.add_image('feature_222', make_grid(torch.sum(feature_2[0], dim=0), normalize=True), i)
        tb_logger.add_image('feature_333', make_grid(torch.sum(feature_3[0], dim=0), normalize=True), i)
        tb_logger.add_image('feature_444', make_grid(torch.sum(feature_4[0], dim=0), normalize=True), i)

        ## tb_logger.add_image('feature_1', make_grid(feature_1[0].unsqueeze(dim=1), normalize=False), curr_step)
        ## tb_logger.add_image('feature_2', make_grid(feature_2[0].unsqueeze(dim=1), normalize=False), curr_step)
        ## tb_logger.add_image('feature_3', make_grid(feature_3[0].unsqueeze(dim=1), normalize=False), curr_step)
        ## tb_logger.add_image('feature_4', make_grid(feature_4[0].unsqueeze(dim=1), normalize=False), curr_step)

        tb_logger.add_image('image', make_grid(img_tensor[0], normalize=True), i)  # 因为在Dataloader里面对输入图片做了Normalize，导致此时的图像已经有正有负，
                                                                                        # 所以这里要用到make_grid，再归一化到0～1之间；
        ####################################################################################################################

        output = torch.nn.functional.softmax(output, dim=1)  # clw note: (batchsize, class_nums)
        pred_score, pred_label = torch.max(output, 1)
        pred_label = pred_label.item()
        pred_score = pred_score.item()  # clw note: assume bs=1

        #if pred_label == 1 or (pred_label == 0 and pred_score < 0.8) :  # 概率低的，干到ng里面
        if pred_label != 0:
            img_names_ng.append(img_name)
        elif pred_label == 0 :
            img_names_ok.append(img_name)

        result = {'pred_label':pred_label, 'pred_score':pred_score}

        draw_result(img_origin, result, os.path.join(save_path, img_name))

    print('ok品总数：', len(img_names_ok))
    print(img_names_ok)
