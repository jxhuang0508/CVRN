import os
import cv2
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pycocotools.mask as COCOmask
from skimage import measure

if __name__ == "__main__":

    dataset_path = 'data/vistas'
    set = 'train'
    pano_json = json.load(open(os.path.join(dataset_path, 'annotations', 'panoptic_{}'.format(set), 'panoptic_2018.json')))
    stff_json = json.load(open(os.path.join(dataset_path, 'annotations', 'stuff_{}.json'.format(set))))

    image_dirs = os.path.join(dataset_path, 'images', set)
    LABEL_dirs = os.path.join(dataset_path, 'annotations', 'panoptic_{}_semantic_trainid'.format(set))

    label_dirs = os.path.join(dataset_path, 'annotations', 'labels_{}'.format(set))
    folder = os.path.exists(label_dirs)
    if not folder:
        os.makedirs(label_dirs)

    # id_to_trainid = {7: 0, 8: 0, 10: 0, 13: 0, 14: 0, 23: 0, 24: 0,
    #                  2: 1, 9: 1, 11: 1, 15: 1, 17: 2, 6: 3, 3: 4,
    #                  45: 5, 47: 5, 48: 6, 49: 7, 50: 7, 30: 8, 29: 9,
    #                  27: 10, 19:11, 20:12, 21:12, 22:12, 55:13, 61:14, 54:15, 58:16, 57:17, 52:18,
    #                  0: 255, 1: 255, 4: 255, 5: 255, 12: 255, 16: 255, 18: 255, 25: 255, 26: 255, 28: 255, 31: 255, 32: 255,
    #                  33: 255, 34: 255, 35: 255, 36: 255, 37: 255, 38: 255, 39: 255,40: 255, 41: 255, 42: 255, 43: 255, 44: 255,
    #                  46: 255, 51: 255, 53: 255, 56: 255, 59: 255, 60: 255, 62: 255, 63: 255, 64: 255, }
    id_to_trainid = {7: 0, 8: 0, 10: 0, 13: 0, 14: 0, 23: 0, 24: 0,
                     2: 1, 9: 1, 11: 1, 15: 1, 17: 2, 6: 3, 3: 4,
                     45: 5, 47: 5, 48: 6, 49: 7, 50: 7, 30: 8, 29: 9,
                     27: 10, 19:11, 20:12, 21:12, 22:12, 55:13, 61:14, 54:15, 58:16, 57:17, 52:18,
                     0: 255, 1: 255, 4: 255, 5: 255, 12: 255, 16: 255, 18: 255, 25: 255, 26: 255, 28: 255, 31: 255, 32: 255,
                     33: 255, 34: 255, 35: 255, 36: 255, 37: 255, 38: 255, 39: 255,40: 255, 41: 255, 42: 255, 43: 255, 44: 255,
                     46: 255, 51: 255, 53: 255, 56: 255, 59: 255, 60: 255, 62: 255, 63: 255, 64: 255, 65: 255, }

    id_to_instanceid = {19:1, 20:2, 21:2, 22:2, 55:3, 61:4, 54:5, 58:6, 57:7, 52:8}

    gt = dict()
    gt['images'] = stff_json['images']
    gt['categories'] = [{'id': 1, 'name': 'person'},
                        {'id': 2, 'name': 'rider'},
                        {'id': 3, 'name': 'car'},
                        {'id': 4, 'name': 'truck'},
                        {'id': 5, 'name': 'bus'},
                        {'id': 6, 'name': 'train'},
                        {'id': 7, 'name': 'motorcycle'},
                        {'id': 8, 'name': 'bicycle'}]
    gt['annotations'] = list()

    for im_ind in range(len(pano_json['images'])):
        gt['images'][im_ind]['width'] = pano_json['images'][im_ind]['height']
        gt['images'][im_ind]['height'] = pano_json['images'][im_ind]['width']

    anno_id = 0
    all_stff_id = -1
    for ind in range(len(pano_json['images'])):
        print(ind)
        image_id = pano_json['images'][ind]['id']
        label = cv2.imread(os.path.join(LABEL_dirs,image_id + '.png'),-1)
        label_copy = label.copy()
        for k, v in id_to_trainid.items():
            label_copy[label == k+1] = v
        label_copy_img = Image.fromarray(label_copy)
        label_copy_img.save(os.path.join(label_dirs, image_id + '_labelTrainIds.png'))

        gt['images'][ind]['id'] = ind
        gt['images'][ind]['seg_file_name'] = image_id + '_labelTrainIds.png'

        segments = pano_json['annotations'][ind]['segments_info']
        for stff_id in range(len(segments)):
            all_stff_id += 1
            gt_anno = segments[stff_id].copy()
            category_id = gt_anno['category_id']
            if category_id-1 in id_to_instanceid:
                gt_anno['category_id'] = id_to_instanceid[category_id-1]
                gt_anno['image_id'] = ind
                gt_anno['id'] = anno_id
                anno_id += 1
                RS = stff_json['annotations'][all_stff_id]['segmentation']
                mask = COCOmask.decode(RS)
                gt_anno["segmentation"] = []
                contours = measure.find_contours(mask, 0.5)
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    segmentation = contour.ravel().tolist()
                    gt_anno["segmentation"].append(segmentation)
                gt['annotations'].append(gt_anno)

    anno_file = os.path.join(dataset_path, 'annotations', 'instancesonly_{}.json'.format(set))
    with open(anno_file, 'w') as f:
        f.write(json.dumps(gt, indent=4))
        f.close()

    pass





