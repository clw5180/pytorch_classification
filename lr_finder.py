import numpy as np
import os
import pandas as pd
from fastai.vision.all import *
import albumentations
import matplotlib.pyplot as plt
from PIL import Image



set_seed(999,reproducible=True)

torch.backends.cudnn.benchmark = True  # clw modify: for seresnext

dataset_path = Path('../dataset/kaggle2020-leaf-disease-classification')  # clw note: must relative path
print(os.listdir(dataset_path))

train_df = pd.read_csv(dataset_path/'train.csv')
print(train_df.head())


train_df['path'] = train_df['image_id'].map(lambda x:dataset_path/'train_images'/x)
train_df = train_df.drop(columns=['image_id'])
train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
print(train_df.head(10))

len_df = len(train_df)
print(f"There are {len_df} images")

# 画出各类别数量的直方图
#train_df['label'].hist(figsize = (10, 5))
#plt.show()

im = Image.open(train_df['path'][1])
width, height = im.size
print(width,height)
#im.show()

class AlbumentationsTransform(RandTransform):
    "A transform handler for multiple `Albumentation` transforms"
    split_idx, order = None, 2

    def __init__(self, train_aug, valid_aug):
        store_attr()

    def before_call(self, b, split_idx):
        self.idx = split_idx

    def encodes(self, img: PILImage):
        if self.idx == 0:
            aug_img = self.train_aug(image=np.array(img))['image']
        else:
            aug_img = self.valid_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

def get_train_aug(sz): return albumentations.Compose([
            albumentations.RandomResizedCrop(sz,sz),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
            albumentations.CoarseDropout(p=0.5),
            albumentations.Cutout(p=0.5)
])

def get_valid_aug(sz): return albumentations.Compose([
    albumentations.CenterCrop(sz,sz, p=1.),
    albumentations.Resize(sz,sz)
], p=1.)

def get_dls(sz,bs):
    item_tfms = AlbumentationsTransform(get_train_aug(sz), get_valid_aug(sz))
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]
    dls = ImageDataLoaders.from_df(train_df, #pass in train DataFrame
                                   valid_pct=0.2, #80-20 train-validation random split
                                   seed=999, #seed
                                   label_col=0, #label is in the first column of the DataFrame
                                   fn_col=1, #filename/path is in the second column of the DataFrame
                                   bs=bs, #pass in batch size
                                   item_tfms=item_tfms, #pass in item_tfms
                                   batch_tfms=batch_tfms) #pass in batch_tfms
    return dls

#dls = get_dls(456, 16)
dls = get_dls(512, 32)

#dls.show_batch()  #


### training
from timm import create_model
from fastai.vision.learner import _update_first_layer


def create_timm_body(arch: str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i, o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int):
        return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut):
        return cut(model)
    else:
        raise NamedError("cut must be either integer or function")


def create_timm_model(arch: str, n_out, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_,
                      custom_head=None,
                      #concat_pool=True, **kwargs):
                      concat_pool=False, **kwargs):  # clw modify
    "Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library"
    body = create_timm_body(arch, pretrained, None, n_in)
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else:
        head = custom_head
    model = nn.Sequential(body, head)
    if init is not None: apply_init(model[1], init)
    return model


def timm_learner(dls, arch: str, loss_func=None, pretrained=True, cut=None, splitter=None,
                 y_range=None, config=None, n_out=None, normalize=True, **kwargs):
    "Build a convnet style learner from `dls` and `arch` using the `timm` library"
    if config is None: config = {}
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = create_timm_model(arch, n_out, default_split, pretrained, y_range=y_range, **config)
    learn = Learner(dls, model, loss_func=loss_func, splitter=default_split, **kwargs)
    if pretrained: learn.freeze()
    return learn

learn = timm_learner(dls,
                     #'tf_efficientnet_b5_ns',
                     #'tf_efficientnet_b3_ns',
                     'seresnext50_32x4d',
                     #opt_func=ranger,
                     opt_func=Adam,  # clw modify : SGD
                     loss_func=LabelSmoothingCrossEntropy(eps=0.3),
                     cbs=[GradientAccumulation(n_acc=32)],
                     #metrics = [accuracy]).to_native_fp16()
                     metrics = [accuracy]).to_fp16()  # clw modify

learn.lr_find()

plt.show()

