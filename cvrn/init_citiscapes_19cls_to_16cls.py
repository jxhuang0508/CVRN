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

    dataset_path = 'UPSNet/data/cityscapes'
    anno_path = os.path.join(dataset_path, 'annotations')
    anno_file = os.path.join(anno_path, 'cityscapes_fine_val.json')
    with open(anno_file) as f: anno = json.load(f)
    anno_path = os.path.join(dataset_path, 'annotations_7cls')
    if not os.path.exists(anno_path):
        os.makedirs(anno_path)
    anno_file = os.path.join(anno_path, 'cityscapes_fine_val.json')

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

    set = 'val'
    anno_path = os.path.join(dataset_path, 'annotations_7cls')
    anno_file = os.path.join(anno_path, 'instancesonly_gtFine_{}.json'.format(set))
    inst_json = json.load(open(os.path.join(dataset_path, 'annotations', 'instancesonly_gtFine_{}.json'.format(set))))

    gt = dict()
    gt['images'] = inst_json['images']
    gt['categories'] = [{'id': 1, 'name': 'person'},
                        {'id': 2, 'name': 'rider'},
                        {'id': 3, 'name': 'car'},
                        {'id': 4, 'name': 'bus'},         #5
                        {'id': 5, 'name': 'motorcycle'},  #7
                        {'id': 6, 'name': 'bicycle'}]     #8
    gt['annotations'] = list()
    id = 0

    annotations = inst_json['annotations']
    for ind in range(len(annotations)):
        if ind % 100 ==0:
            print(ind, '/', len(annotations))
        gt_anno = annotations[ind].copy()
        # selct 6 classse (delete 0 'background', 4 'truck', 6 'train')
        cat = annotations[ind]['category_id']
        if cat==4 or cat==6:
            continue
        if cat==5:
            gt_anno['category_id'] = cat-1
        if cat>6:
            gt_anno['category_id'] = cat-2
        gt_anno['id'] = id
        gt['annotations'].append(gt_anno)
        id += 1

    with open(anno_file, 'w') as f:
        f.write(json.dumps(gt, indent=4))
        f.close()

    image_path_ori = os.path.join(dataset_path, 'labels')
    image_path = os.path.join(dataset_path, 'labels_16cls')
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    image_files = os.listdir(image_path_ori)
    image_files.sort()
    ind = 0
    for image_file in image_files:
        img = cv2.imread(os.path.join(image_path_ori,image_file))

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

        img = Image.fromarray(img).convert('L')
        img.save(os.path.join(image_path,image_file))

        if ind % 100 == 0:
            print(image_file)
            print(np.unique(img))
        ind += 1


