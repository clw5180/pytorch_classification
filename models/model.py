from pretrainedmodels import models as pm
import pretrainedmodels
from torch import nn
import torchvision
from config import configs
###from efficientnet_pytorch import EfficientNet
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models
import timm  # clw modify

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

################################################
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class BinaryHead(nn.Module):
    def __init__(self, num_class=4, emb_size=2048, s=16.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit


class se_resnext50_32x4d_clw(nn.Module):
    def __init__(self, num_classes):
        super(se_resnext50_32x4d_clw, self).__init__()

        self.model_ft = nn.Sequential(
            *list(pretrainedmodels.__dict__["se_resnext50_32x4d"](num_classes=1000, pretrained="imagenet").children())[
                :-2
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = None
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(num_classes, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        img_feature = self.model_ft(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        # fea = self.dropout(fea)
        output = self.binary_head(fea)

        return output
################################################


def get_model():
    if configs.model_name.startswith("resnext50_32x4d"):
        # model = torchvision.models.resnext50_32x4d(pretrained=True)
        # model.avgpool = nn.AdaptiveAvgPool2d(1)
        # model.fc = nn.Linear(2048, configs.num_classes)
        ####
        model = timm.create_model('resnext50_32x4d', pretrained=True)
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, configs.num_classes)

    elif configs.model_name.startswith("se_resnext50_32x4d"):
        #model = se_resnext50_32x4d_clw(configs.num_classes)    # 自定义se_resnext50_32x4d, result not good

        model = pretrainedmodels.se_resnext50_32x4d(pretrained="imagenet")
        model.last_linear=nn.Linear(2048, configs.num_classes)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)  # clw note: 在senet.py中，默认是self.avg_pool = nn.AvgPool2d(7, stride=1)，这里的7是根据imagenet输入224来的，所以要改一下，否则输出就不是 32,2048,1,1了

        # model = timm.create_model('seresnext50_32x4d', pretrained=True)   # not good
        # n_features = model.fc.in_features
        # model.fc = nn.Linear(n_features, configs.num_classes)

    elif configs.model_name.startswith("se_resnext101_32x4d"):  # TODO: pretrainedmodels.se_resnext50_32x4d()
        model = pretrainedmodels.se_resnext101_32x4d(pretrained="imagenet")
        model.last_linear = nn.Linear(2048, configs.num_classes)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)

    elif configs.model_name.startswith("pnasnet"):  # TODO: pretrainedmodels.se_resnext50_32x4d()
        model = pretrainedmodels.pnasnet5large(pretrained="imagenet")
        model.last_linear=nn.Linear(4320, configs.num_classes)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)  # clw note: 在senet.py中，默认是self.avg_pool = nn.AvgPool2d(7, stride=1)，这里的7是根据imagenet输入224来的，所以要改一下，否则输出就不是 32,2048,1,1了

    elif configs.model_name.startswith("resnet18"):
        model = models.resnet18(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, configs.num_classes)

    elif configs.model_name.startswith("resnet34"):
        model = models.resnet34(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, configs.num_classes)

    elif configs.model_name.startswith("resnet50"):
        model = models.resnet50(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, configs.num_classes)
        print(model.relu)

        # model = timm.create_model("resnet50", pretrained=True)
        # model.fc = nn.Linear(model.fc.in_features, configs.num_classes)

    elif configs.model_name.startswith("efficientnet-b0"):
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif configs.model_name.startswith("efficientnet-b2"):
        model = timm.create_model('tf_efficientnet_b2_ns', pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif configs.model_name.startswith("efficientnet-b3"):
        #model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True, num_classes=configs.num_classes, drop_path_rate=configs.drop_out_rate, drop_rate=configs.drop_out_rate)
        model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True, num_classes=configs.num_classes, drop_path_rate=configs.drop_out_rate, drop_rate=configs.drop_out_rate)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif configs.model_name.startswith("efficientnet-b4"):
        #model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=configs.num_classes, drop_path_rate=configs.drop_out_rate)  # drop_path_rate=0.2~0.5
        model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=configs.num_classes)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif configs.model_name.startswith("efficientnet-b5"):
        model = timm.create_model('tf_efficientnet_b5_ns', pretrained=True, num_classes=configs.num_classes, drop_path_rate=configs.drop_out_rate)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif configs.model_name.startswith("vit_base_patch16_384"):
        model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=configs.num_classes)  # , drop_rate=0.1)
        model.head = nn.Linear(model.head.in_features, configs.num_classes)
    elif configs.model_name.startswith("vit_large_patch16_384"):
        model = timm.create_model('vit_large_patch16_384', pretrained=True, num_classes=configs.num_classes)
        model.head = nn.Linear(model.head.in_features, configs.num_classes)
    elif configs.model_name.startswith('vit_base_resnet50_384'):
        model = timm.create_model('vit_base_resnet50_384', pretrained=True, num_classes=configs.num_classes)
        model.head = nn.Linear(model.head.in_features, configs.num_classes)

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

    return model


def get_model_no_pretrained(model_file_name, my_state_dict):
    if 'se_resnext50' in model_file_name:
        model = pretrainedmodels.se_resnext50_32x4d(pretrained="imagenet")
        model.last_linear=nn.Linear(2048, configs.num_classes)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
    elif "efficientnet-b0" in model_file_name:
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif "efficientnet-b2" in model_file_name:
        model = timm.create_model('tf_efficientnet_b2_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif "efficientnet-b3" in model_file_name:
        model = timm.create_model('tf_efficientnet_b3_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif "efficientnet-b4" in model_file_name:
        model = timm.create_model('tf_efficientnet_b4_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif "efficientnet-b5" in model_file_name:
        model = timm.create_model('tf_efficientnet_b5_ns', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, configs.num_classes)
    elif configs.model_name.startswith("vit_base_patch16_384"):
        model = timm.create_model('vit_base_patch16_384', pretrained=False)
        model.head = nn.Linear(model.head.in_features, configs.num_classes)
    elif configs.model_name.startswith("vit_large_patch16_384"):
        model = timm.create_model('vit_large_patch16_384', pretrained=True)
        model.head = nn.Linear(model.head.in_features, configs.num_classes)
    else:
        model = models.resnet50(pretrained=False, num_classes=configs.num_classes)  # clw note: fc.weight: (num_class, 2048)

    model.load_state_dict(my_state_dict)
    return model