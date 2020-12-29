from pretrainedmodels import models as pm
import pretrainedmodels
from torch import nn
import torchvision
from config import configs
from efficientnet_pytorch import EfficientNet
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torchvision import models

weights = {
        "efficientnet-b3":"/data/dataset/detection/pretrainedmodels/efficientnet-b3-c8376fa2.pth",
        #"efficientnet-b4":"/data/dataset/detection/pretrainedmodels/efficientnet-b4-6ed6700e.pth",
        "efficientnet-b4":"/home/user/.cache/torch/checkpoints/efficientnet-b4-6ed6700e.pth",
        "efficientnet-b5":"/data/dataset/detection/pretrainedmodels/efficientnet-b5-b6417697.pth",
        "efficientnet-b6":"/data/dataset/detection/pretrainedmodels/efficientnet-b6-c76e70fd.pth",
        }

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def get_model():
    if configs.model_name.startswith("resnext50_32x4d"):
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, configs.num_classes)
        model.cuda()
    elif configs.model_name.startswith("resnet50"):
        model = models.resnet50(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, configs.num_classes)
        model.cuda()
    elif configs.model_name.startswith("efficient"):
        # efficientNet
        model_name = configs.model_name[:15]
        model = EfficientNet.from_name(model_name)
        model.load_state_dict(torch.load(weights[model_name]))
        in_features = model._fc.in_features
        model._fc = nn.Sequential(
                        nn.BatchNorm1d(in_features),
                        nn.Dropout(0.5),  # TODO
                        nn.Linear(in_features, configs.num_classes),
                         )
        model.cuda()
    elif configs.model_name == 'shufflenetv2_x0_5':  # clw modify
        model = models.shufflenet_v2_x0_5(pretrained=True)  # clw modify
        model.fc = nn.Linear(model.fc.in_features, configs.num_classes)
        ####
        mean=0
        std=0.01
        bias=0
        nn.init.normal_(model.fc.weight, mean, std)
        if hasattr(model.fc, 'bias') and model.fc.bias is not None:
            nn.init.constant_(model.fc.bias, bias)
        ####
        model.cuda()
    elif configs.model_name == 'shufflenet_v2_x1_0':  # clw modify
        model = models.shufflenet_v2_x1_0(pretrained=False)  # clw modify
        #model = models.shufflenet_v2_x1_0(pretrained=True)  # clw modify
        model.fc = nn.Linear(model.fc.in_features, configs.num_classes)
        ####
        mean=0
        std=0.01
        bias=0
        nn.init.normal_(model.fc.weight, mean, std)
        if hasattr(model.fc, 'bias') and model.fc.bias is not None:
            nn.init.constant_(model.fc.bias, bias)
        ####
        model.cuda()
    elif configs.model_name == 'shufflenetv2_x1_5':  # clw modify
        model = models.shufflenet_v2_x1_5(pretrained=False)  # clw modify
        model.fc = nn.Linear(model.fc.in_features, configs.num_classes)
        ####
        mean=0
        std=0.01
        bias=0
        nn.init.normal_(model.fc.weight, mean, std)
        if hasattr(model.fc, 'bias') and model.fc.bias is not None:
            nn.init.constant_(model.fc.bias, bias)
        ####
        model.cuda()

    return model