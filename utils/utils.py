import numpy as np
import random
import math
import torch.nn as nn

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_bbox_clw(img_w, img_h, lam):
    W = img_w
    H = img_h
    cut_rat = np.sqrt(1 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

#def RandomErasing(img, p=0.3, sl=0.02, sh=0.3, r1=0.3, **kwargs):
def RandomErasing(img, p=1, sl=0.02, sh=0.3, r1=0.3, **kwargs):
    """Random erasing the an rectangle region in Image.
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    Args:
        img: opencv numpy array in form of [w, h, c] range from [0, 255]
        sl: min erasing area region
        sh: max erasing area region
        r1: min aspect ratio range of earsing region
        p: probability of performing random erasing

    Returns:
        erased img
    """

    s = (sl, sh)
    r = (r1, 1 / r1)

    assert len(img.shape) == 3, 'image should be a 3 dimension numpy array'

    if random.random() > p:
        return img
    else:
        img_out = img.copy()
        while True:
            Se = random.uniform(*s) * img_out.shape[0] * img_out.shape[1]
            re = random.uniform(*r)

            He = int(round(math.sqrt(Se * re)))
            We = int(round(math.sqrt(Se / re)))

            xe = random.randint(0, img_out.shape[1])
            ye = random.randint(0, img_out.shape[0])

            if xe + We <= img_out.shape[1] and ye + He <= img_out.shape[0]:
                img_out[ye: ye + He, xe: xe + We, :] = np.random.randint(low=0, high=255, size=(He, We, img_out.shape[2]))   # 彩色随机值
                #img_out[ye: ye + He, xe: xe + We, :] = 0  # 黑条

                return img_out

# 官方实现
def RandomErasing2(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[110.162, 127.162, 80.2355 ], **kwargs):  # mean=[123.675, 116.28, 103.52 ]  mean=[0.4914, 0.4822, 0.4465]
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    if random.uniform(0, 1) > probability:
        return img

    img_out = img.copy()
    for attempt in range(100):
        #area = img.size()[1] * img.size()[2]

        area = img_out.shape[0] * img_out.shape[1]  # clw modify: for ndarray

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img_out.shape[1] and h < img_out.shape[0]:
            x1 = random.randint(0, img_out.shape[1] - w)
            y1 = random.randint(0, img_out.shape[0] - h)
            if img_out.shape[2] == 3:
                img_out[y1:y1 + h, x1:x1 + w, 2] = mean[0]
                img_out[y1:y1 + h, x1:x1 + w, 1] = mean[1]
                img_out[y1:y1 + h, x1:x1 + w, 0] = mean[2]
            else:
                img_out[y1:y1 + h, x1:x1 + h, 0] = mean[0]
            return img_out

    return img


################ freeze bn
def freeze_batchnorm_stats(net):
    try:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                #print(m)
                m.eval()
    except ValueError:
        print('error with batchnorm2d')
        return