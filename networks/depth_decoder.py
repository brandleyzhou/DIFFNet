# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
#from hr_layers import * 
#from visual_block import visual_block
from matplotlib import pyplot as plt

def visual_feature(features,stage):
    feature_map = features.squeeze(0).cpu()
    n,h,w = feature_map.size()
    list_mean = []
    #sum_feature_map = torch.sum(feature_map,0)
    sum_feature_map,_ = torch.max(feature_map,0)
    for i in range(n):
        list_mean.append(torch.mean(feature_map[i]))
        
    sum_mean = sum(list_mean)
    feature_map_weighted = torch.ones([n,h,w])
    for i in range(n):
        feature_map_weighted[i,:,:] = (torch.mean(feature_map[i]) / sum_mean) * feature_map[i,:,:]
    sum_feature_map_weighted = torch.sum(feature_map_weighted,0)
    plt.imshow(sum_feature_map)
    #plt.savefig('feature_viz/{}_stage.png'.format(a))
    plt.savefig('feature_viz/decoder_{}.png'.format(stage))
    plt.imshow(sum_feature_map_weighted)
    #plt.savefig('feature_viz/{}_stage_weighted.png'.format(a))
    plt.savefig('feature_viz/decoder_{}_weighted.png'.format(stage))

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        #  num_ch_enc = np.array([64, 64, 128, 256, 512])
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        if self.num_ch_enc[1] == 64:
            # for monodepth2
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        elif self.num_ch_enc[1] == 18:
            # for hrnet18
            self.num_ch_dec = np.array([6, 9, 18, 36, 72])
        elif self.num_ch_enc[1] == 32:
            # for hrnet32
            self.num_ch_dec = np.array([8, 16, 32, 64, 128])
        elif self.num_ch_enc[1] == 48:
            # for hrnet48
            self.num_ch_dec = np.array([12, 24, 48, 96, 192])
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        # se_block in depth_map Conv 
        #self.se_block0 = SE_block(self.num_ch_dec[0]) 
        #self.se_block1 = SE_block(self.num_ch_dec[1])
        #self.se_block2 = SE_block(self.num_ch_dec[2])
        #self.se_block3 = SE_block(self.num_ch_dec[3])
        
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1): #i=[4,3,2,1,0]
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)#CONV2D

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()#why not relu?
        self.relu = nn.ReLU()
    
    def forward(self, input_features):
        self.outputs = {}
        #block_list = [self.se_block0, self.se_block1, self.se_block2, self.se_block3]
        # decoder
        #depth_att = input_features.pop(-1)
        x = input_features[-1]
        #visual_feature(input_feature[1])
        for i in range(4, -1, -1):#[4,3,2,1,0]
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]#this function in layers.py
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            #visual_block(x, i)
            if i in self.scales:
                #se_block before depth Conv layer
                #if i == 0:
                #    x = x * depth_att
                #x = block_list[i](x)
                ###################################
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                #self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](self.relu(x)))
        return self.outputs
