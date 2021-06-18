import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision.transforms as transforms
import torchvision
import cv2
from torch.utils import data
import sys
from PIL import Image

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

class GTA5TestDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, test_size=(1024, 512), test_scale=1.0, mean=(128, 128, 128),
                 std=(1, 1, 1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.test_h, self.test_w = test_size
        self.scale = scale
        self.test_scale = test_scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []

        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            label_file = osp.join(self.root, label_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)  # OpenCV read image as BGR, not RGB
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        img_name = datafiles["img_name"]
        image = cv2.resize(image, None, fx=self.test_scale, fy=self.test_scale, interpolation=cv2.INTER_CUBIC)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        image -= self.mean  # BGR
        image = image / self.std  # np.reshape(self.std,(1,1,3))
        size = image.shape
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), img_name
