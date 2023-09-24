import random
from typing import Any, Dict, Union

import cv2
import numpy as np
import torch
import utils.utils_image as util

from .dataset_ir import DatasetIR


class DatasetDenoising(DatasetIR):
    def __init__(self, opt_dataset: Dict[str, Any]):
        super().__init__(opt_dataset)

        self.tag_x = str(self.sigma_y)

    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:
        # get H image

        img_path = self.img_paths[index]
        img_H = util.imread_uint(img_path, self.n_channels)

        H, W = img_H.shape[:2]

        img_path_m = self.img_paths_m[index]
        img_M = util.imread_uint(img_path_m, self.n_channels)  # 另一个模态的真实图像



        if self.opt['phase'] == 'train':
            self.count += 1

            # crop
            rnd_h = random.randint(0, max(0, H - self.patch_size))

            rnd_w = random.randint(0, max(0, W - self.patch_size))

            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w +
                                                                 self.patch_size, :]

            # augmentation
            a = np.random.randint(0, 8)
            patch_H = util.augment_img(patch_H, mode=a)


            patch_H_m = img_M[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w +
                                                                 self.patch_size, :]

            # augmentation
            patch_H_m = util.augment_img(patch_H_m, mode=a)

            # HWC to CHW, numpy(uint) to tensor
            img_H = util.uint2tensor3(patch_H)
            img_L: torch.Tensor = img_H.clone()

            img_M = util.uint2tensor3(patch_H_m)

            # get noise level
            noise_level: torch.FloatTensor = torch.FloatTensor(
                [np.random.uniform(self.sigma_y[0], self.sigma_y[1])]) / 255.0
            # add noise
            noise = torch.randn(img_L.size()).mul_(noise_level).float()
            img_L.add_(noise)

        else:
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            # add noise
            np.random.seed(seed=0)

            img_L += np.random.normal(0, self.sigma_y / 255.0, img_L.shape)

            noise_level = torch.FloatTensor([self.sigma_y / 255.0])



            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(
                img_L)

            img_M = util.uint2single(img_M)
            img_M = util.single2tensor3(img_M)

        return {
            'y': img_L,
            'm': img_M,
            'y_gt': img_H,
            'sigma_y': noise_level.unsqueeze(1).unsqueeze(1),

            'path': img_path
        }