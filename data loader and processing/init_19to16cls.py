import os
import cv2
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pycocotools.mask as mask
# from skimage import measure

if __name__ == "__main__":

    set = 'train'
    dataset_path = 'data/cityscapes'
    anno_path = os.path.join(dataset_path, 'annotations')
    anno_file = os.path.join(anno_path, 'cityscapes_fine_val.json')
    with open(anno_file) as f: anno = json.load(f)
    anno_path = os.path.join(dataset_path, 'annotations_7cls')
    anno_file = os.path.join(anno_path, 'cityscapes_fine_val.json')

    # dataset_path = 'data/vistas'
    # anno_file = os.path.join(dataset_path, 'panoptic_val.json')
    # with open(anno_file) as f: anno = json.load(f)
    # anno_file = os.path.join(dataset_path, 'panoptic_val_16cls.json')

    gt = dict()
    gt['images'] = anno['images']
    gt['categories'] = list()
    gt['annotations'] = list()

    # selct 16 classse (delete 9 'terrain', 14 'truck', 16 'train')
    for cat in anno['categories']:
        if cat['id'] not in [9,14,16]:
            if cat['id'] == 10: cat['id'] = 9
            elif cat['id'] == 11: cat['id'] = 10
            elif cat['id'] == 12: cat['id'] = 11
            elif cat['id'] == 13: cat['id'] = 12
            elif cat['id'] == 15: cat['id'] = 13
            elif cat['id'] == 17: cat['id'] = 14
            elif cat['id'] == 18: cat['id'] = 15
            gt['categories'].append(cat)

    for ann in anno['annotations']:
        gt_ann = ann.copy()
        gt_ann['segments_info'] = []
        for seg in ann['segments_info']:
            if seg['category_id'] not in [9,14,16]:
                if seg['category_id'] == 10: seg['category_id'] = 9
                elif seg['category_id'] == 11: seg['category_id'] = 10
                elif seg['category_id'] == 12: seg['category_id'] = 11
                elif seg['category_id'] == 13: seg['category_id'] = 12
                elif seg['category_id'] == 15: seg['category_id'] = 13
                elif seg['category_id'] == 17: seg['category_id'] = 14
                elif seg['category_id'] == 18: seg['category_id'] = 15
                gt_ann['segments_info'].append(seg)
        gt['annotations'].append(gt_ann)

    print(len(gt['images']))
    with open(anno_file, 'w') as f:
        f.write(json.dumps(gt))
        f.close()

    # dataset_path = 'data/vistas'
    # image_path_ori = os.path.join(dataset_path, 'annotations/labels_val')
    # image_path = os.path.join(dataset_path, 'labels_16cls/val')
    # if not os.path.exists(image_path):
    #     os.makedirs(image_path)
    #
    # image_files = os.listdir(image_path_ori)
    # image_files.sort()
    # ind = 0
    # for image_file in image_files:
    #     img = cv2.imread(os.path.join(image_path_ori,image_file))
    #
    #     # selct 16 classse (delete 9 'terrain', 14 'truck', 16 'train')
    #     img[img==9] = 255
    #     img[img==10] = 9
    #     img[img==11] = 10
    #     img[img==12] = 11
    #     img[img==13] = 12
    #     img[img==14] = 255
    #     img[img==15] = 13
    #     img[img==16] = 255
    #     img[img==17] = 14
    #     img[img==18] = 15
    #
    #     # cv2.imwrite(os.path.join(image_path,image_file),img)
    #     img = Image.fromarray(img).convert('L')
    #     img.save(os.path.join(image_path,image_file))
    #
    #     if ind % 100 == 0:
    #         print(image_file)
    #         print(np.unique(img))
    #     ind += 1




