import torch
import torch.nn as nn
from torchvision import models
import torch.nn.utils.weight_norm as weightnorm


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


res_dict = {"resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101}


class ResNetBase(nn.Module):
    def __init__(self, res_name, feature_dim=512):
        super(ResNetBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.encoder = nn.Sequential(*list(model_resnet.children())[:-1])
        self.projector = nn.Linear(model_resnet.fc.in_features, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim, affine=True)

        self.projector.apply(init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        x = self.bn(x)
        return x


class Classifier(nn.Module):
    def __init__(self, feature_dim, class_num, wn=True):
        super(Classifier, self).__init__()
        if wn:
            self.fc = weightnorm(nn.Linear(feature_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(feature_dim, class_num)

        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x
