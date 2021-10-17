# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
from torchvision import transforms as transforms1
import PIL.Image as pil
import cv2
from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {'l': 'Camera_0','r': '/Camera_1/'}

    def check_depth_vk(self):
        return True
    
    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)
    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class VK2Dataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(VK2Dataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        #f_str = "{:010d}{}".format(frame_index, self.img_ext)
        self.img_ext = '.jpg'
        f_str = "rgb_{:05d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            #self.data_path, folder, "clone/frames/rgb{}".format(self.side_map[side], f_str))
            self.data_path, '{}/clone/frames/rgb'.format(folder), "{}/{}".format(self.side_map[side], f_str))
        return image_path

    def get_depth_vk(self, folder, frame_index, side, do_flip):
        d_str = "depth_{:05d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path, '{}/clone/frames/depth'.format(folder), "{}/{}".format(self.side_map[side], d_str))
        #depth_gt_vk = cv2.imread(depth_path)  
        depth_gt_vk = pil.open(depth_path)  
        resize_depth = transforms1.Resize((192, 640),
                                            interpolation=pil.ANTIALIAS)
        depth_gt_vk = resize_depth(depth_gt_vk)
        if do_flip:
            depth_gt_vk = np.fliplr(depth_gt_vk)
            
        return depth_gt_vk
