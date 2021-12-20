import os
import cv2
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pycocotools.mask as mask
from skimage import measure

if __name__ == "__main__":

    crop_height = 640

    dataset_path = 'data/synthia'
    # image_path_ori = os.path.join(dataset_path, 'images')
    image_path_ori = os.path.join(dataset_path, 'labels')

    dataset_path = 'data/synthia_crop640'
    # image_path = os.path.join(dataset_path, 'images')
    image_path = os.path.join(dataset_path, 'labels_16cls')
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    image_files = os.listdir(image_path_ori)
    image_files.sort()
    ind = 0
    for image_file in image_files:
        img = cv2.imread(os.path.join(image_path_ori,image_file))
        img = img[img.shape[0]-crop_height:,:,]

        # selct 16 classse (delete 9 'terrain', 14 'truck', 16 'train')
        img[img==9] = 255
        img[img==10] = 9
        img[img==11] = 10
        img[img==12] = 11
        img[img==13] = 12
        img[img==14] = 255
        img[img==15] = 13
        img[img==16] = 255
        img[img==17] = 14
        img[img==18] = 15

        # cv2.imwrite(os.path.join(image_path,image_file),img)
        img = Image.fromarray(img).convert('L')
        img.save(os.path.join(image_path,image_file))

    dataset_path = 'data/synthia'
    image_path_ori = os.path.join(dataset_path, 'images')

    dataset_path = 'data/synthia_crop640'
    image_path = os.path.join(dataset_path, 'images')
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    image_files = os.listdir(image_path_ori)
    image_files.sort()
    ind = 0
    for image_file in image_files:
        img = cv2.imread(os.path.join(image_path_ori,image_file))
        img = img[img.shape[0]-crop_height:,:,]
        cv2.imwrite(os.path.join(image_path,image_file),img)




