from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoderv1(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoderv1, self).__init__()
        #num_ch_enc = [64,64,128,256,512]
        #num_input_features = 1
        #num_frames_to_predict_for = 2
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(256, 128, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(128, 64, 3, stride, 1)
        #self.convs[("pose", 1)] = nn.Linear(128*6*20, 6)
        self.convs[("pose", 2)] = nn.Conv2d(64, 32, 3)
        #self.convs[("pose", 2)] = nn.Linear(64*6*20, 6)
        self.convs[("pose", 3)] = nn.Conv2d(32, 16, 3)
        self.convs[("pose", 4)] = nn.Linear(16*32, 6)

        self.relu = nn.ReLU()#in depthdecoder activation function is sigmoid()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        #input_features is a list which just has a element but the element has 5 scales feature maps. 
        last_features = [f[-1] for f in input_features]#only collect last_feature?
        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features,1)
        out = cat_features
        for i in range(5):
            if i == 4:
                out = self.convs[("pose", i)](out.view(8, -1))
                break
            out = self.convs[("pose", i)](out)
            if i != 4:
                out = self.relu(out)
        #out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, 1, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation
