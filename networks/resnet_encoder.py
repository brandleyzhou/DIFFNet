from __future__ import absolute_import, division, print_function

import numpy as np
from .CBAM_resnet import ResNet
import torch
import torch.nn as nn
import torchvision.models as models # a subpackage containing different models 
import torch.utils.model_zoo as model_zoo#pretrained network
from hr_layers import *

class ResNetMultiImageInput(ResNet):
#class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        #block:  block_type = models.resnet.BasicBlock or Bottleneck
        #layers: blocks = [2,2,2,2] if blocktype == BasicBlock else [3,4,6,3]
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multiimage_input(num_layers, pretrained=True, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained=True, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        #self.se_block0 = SE_block(self.num_ch_enc[0], visual_weights=True)
        #self.se_block1 = SE_block(self.num_ch_enc[1],1,True)
        #self.se_block2 = SE_block(self.num_ch_enc[2],2,True)
        #self.se_block3 = SE_block(self.num_ch_enc[3],3,True)
        #self.se_block4 = SE_block(self.num_ch_enc[4],4,True)
        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        features = []
        x = (input_image - 0.45) / 0.225 # normalizetion?
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        #features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        #visual_block(features[-1],"feature1")
        #features.append(self.encoder.layer2(features[-1]))
        #visual_block(features[-1],2)
        #features.append(self.encoder.layer3(features[-1]))
        #visual_block(features[-1],3)
        #features.append(self.encoder.layer4(features[-1]))
        #visual_block(features[-1],4)
        #features[-1] = self.se_block0(features[-1])
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        #features[-1] = self.se_block1(features[-1])
        features.append(self.encoder.layer2(features[-1]))
        #features[-1] = self.se_block2(features[-1])
        features.append(self.encoder.layer3(features[-1]))
        #features[-1] = self.se_block3(features[-1])
        features.append(self.encoder.layer4(features[-1]))
        #features[-1] = self.se_block4(features[-1])
        return features# feature has 5 elements
