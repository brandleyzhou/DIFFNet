from __future__ import absolute_import, division, print_function
import torch.nn as nn
import torch
from .layers import transformation_from_parameters
from .resnetv1 import ResnetEncoderV1
from .pose_decoder import PoseDecoder
class PoseNet(nn.Module):
    def __init__(self, num_layers=18, pretrained=True,num_input_images=2):
        super(PoseNet,self).__init__()
        self.encoder = ResnetEncoderV1(num_layers, pretrained, num_input_images=num_input_images)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=num_input_images)

    def forward(self, input_images):
        outputs = {}
        for i in [-1,1]:
            if i == -1:
                input_image = torch.cat([input_images[i], input_images[0]],1)
            else:
                input_image = torch.cat([input_images[0], input_images[i]],1)

            out = self.encoder(input_image)
            axisangle, translation = self.decoder(out)
         
            outputs[("axisangle", 0, i)] = axisangle
            outputs[("translation", 0, i)] = translation
            outputs[("cam_T_cam", 0, i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(i < 0))
        return outputs 
