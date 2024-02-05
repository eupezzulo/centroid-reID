import torch

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

class EfficientNetWOFC(nn.Module):
    '''
        Defines a wrapper of the official EfficientNetB1 from PyTorch Hub.
        See https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b1.html#torchvision.models.efficientnet_b1
    '''
    def __init__(self):
        super(EfficientNetWOFC, self).__init__()

        self.net = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        # remove classifier
        self.net = nn.Sequential(*list(self.net.children())[:-2]) 

    def forward(self, x):
        return self.net(x)


class ResNetWOFC(nn.Module):
    def __init__(self, backbone_name='resnet50'):
        '''
            Defines a wrapper of the official ResNet50 or ResNet34 from PyTorch Hub.
            See https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
            and https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html#torchvision.models.resnet34
        '''
        super(ResNetWOFC, self).__init__()

        if backbone_name == 'resnet50':
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif backbone_name == 'resnet34':
            self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]) 

    def forward(self, x):
        return self.resnet(x)