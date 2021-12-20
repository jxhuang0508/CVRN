
from __future__ import print_function, division
import os
import sys
import logging
import pprint
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import cv2
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
from upsnet.config.config import config
from upsnet.config.parse_args import parse_args
from lib.utils.logging import create_logger
from lib.utils.timer import Timer
from upsnet.dataset import *
from upsnet.models import *
from upsnet.bbox.bbox_transform import bbox_transform, clip_boxes, expand_boxes
from lib.utils.callback import Speedometer
from lib.utils.data_parallel import DataParallel
from pycocotools.mask import encode as mask_encode
from pycocotools.mask import decode as mask_decode
from pycocotools.mask import area as mask_area
cv2.ocl.setUseOpenCL(False)
cudnn.enabled = True
cudnn.benchmark = False
import json
from pycocotools.coco import COCO
import pycocotools._mask as maskUtils
from upsnet.config.parse_args import parse_args
args = parse_args()


def upsnet_test():

    # id_to_trainid = {
    #     3: 0,
    #     4: 1,
    #     2: 2,
    #     21: 3,
    #     5: 4,
    #     7: 5,
    #     15: 6,
    #     9: 7,
    #     6: 8,
    #     1: 9,
    #     10: 10,
    #     17: 11,
    #     8: 12,
    #     19: 13,
    #     12: 14,
    #     11: 15,
    # }
    id_to_trainid = {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        23: 9,
        24: 10,
        25: 11,
        26: 12,
        28: 13,
        32: 14,
        33: 15,
    }

    sseg_path = '../ADVENT/CRST/results/synthia_cross_style_ep6/0/prob'  # probability maps dir
    sseg_pseudo_labels_path = '../ADVENT/CRST/results/synthia_cross_style_ep6/0/pseudo_label'  # pseudo label maps dir

    config.iter = args.iter

    config.dataset.dataset = 'Cityscapes_7cls'
    config.dataset.test_image_set = 'train'
    config.test.scales = [800]
    config.test.max_size = 1600

    config.test.rpn_post_nms_top_n = 1000
    config.test.rpn_nms_thresh = 0.7
    config.test.score_thresh = 0.5

    config.network.has_fcn_head = False
    config.network.has_rpn = True
    config.network.has_rcnn = True
    config.network.has_mask_head = True
    config.network.has_panoptic_head = False

    im_height = 1024
    im_width = 2048

    final_output_path = os.path.join(config.output_path, os.path.basename(args.cfg).split('.')[0], config.dataset.test_image_set)
    outputs = pickle.load(open(os.path.join(final_output_path, 'results_iter_'+ args.iter, 'outputs.pkl'),'rb'))

    test_dataset = eval(config.dataset.dataset)(image_sets=config.dataset.test_image_set.split('+'), flip=False,
                                                result_path=final_output_path, phase='test')

    anno_path = 'data/cityscapes/annotations_7cls'
    anno_file = anno_path + '/instancesonly_gtFine_train.json'
    pseudo_labels_path = anno_path + '/pseudo_labels/'

    if pseudo_labels_path is not None:
        os.makedirs(pseudo_labels_path, exist_ok=True)


    all_boxes = outputs['all_boxes'][:config.dataset.num_classes]
    all_masks = outputs['all_masks'][:config.dataset.num_classes]
    scores = [[] for i in range(len(all_boxes))]
    for ind, boxes in enumerate(all_boxes):
        if len(boxes)>0:
            for box in boxes:
                for b in box:
                    scores[ind].append(b[-1])


    thres_inss = []
    for scs in scores[1:]:
        if len(scs) < 2:
            thres_inss.append(0)
            continue
        scs = np.sort(scs)
        scs = scs[scs>config.test.score_thresh]
        thres_inss.append(scs[np.int(np.round(len(scs)*0.5))]) ### CBST Ranking


    thres_ins = [round(th, 3) for th in thres_inss]
    print('thres_ins:', thres_ins)
    thres_ins = np.array(thres_inss)
    thres_ins[thres_ins > 0.9] = 0.9
    thres_ins[thres_ins < 0.5] = 0.5
    thres_ins = [round(th, 3) for th in thres_ins]
    print('thres_ins_09:', thres_ins)

    anno = json.load(open(anno_file))
    categories = anno['categories']
    for category_id in range(len(categories)):
        categories[category_id]['id'] = category_id + 1
    gt = dict()
    gt['images'] = list()
    gt['categories'] = categories
    gt['annotations'] = list()
    id = 0
    for image_id, roidb in enumerate(test_dataset.roidb):
        file_name = roidb['image'].split('/')[-1]
        print(image_id, file_name)

        sseg_prob = np.load(sseg_path + '/' + file_name.replace('png','npy'))
        sseg_pseudo_label = np.array(Image.open(sseg_pseudo_labels_path + '/' + file_name))
        sseg_pseudo_label_id2train = 255 * np.ones(sseg_pseudo_label.shape, dtype=np.float32)
        for k, v in id_to_trainid.items():
            sseg_pseudo_label_id2train[sseg_pseudo_label == k] = v

        images=dict()
        images['id'] = image_id
        images['file_name'] = file_name
        images['height'] = im_height
        images['width'] = im_width
        gt['images'].append(images)
        anns = []
        for category_id, (boxes, masks) in enumerate(zip(all_boxes, all_masks)):
            if category_id == 0:
                continue
            category_id_seg = category_id + 9
            if len(boxes) == 0:
                continue
            box = boxes[image_id]
            mask = masks[image_id]
            for i, (b, rle) in enumerate(zip(box,mask)):
                score = b[-1]
                if score < config.test.score_thresh:
                    continue

                anno = dict()
                anno['id'] = id
                id += 1
                anno['image_id'] = image_id
                anno['category_id'] = category_id
                bbox = b[:-1]
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                anno['bbox'] = bbox.tolist()
                mask_p = maskUtils.decode([rle])[:, :, 0]
                _, contours, hier = cv2.findContours(mask_p.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                anno["segmentation"] = list()
                if contours == []:
                    print('Warning: empty mask.')
                    continue

                for contour in contours:
                    if len(contour) <= 4:
                        print('Warning: empty contour.')
                        continue
                    segmentation = contour.ravel().tolist()
                    anno["segmentation"].append(segmentation)

                box_ent = -score*np.log2(score + 1e-10)
                if mask_p.sum()==0:
                    continue
                sseg_prob_overlap_mask = sseg_prob[:,:,category_id_seg].reshape(-1)[mask_p.reshape(-1)==1]
                sseg_prob_overlap_mask_ent = (-sseg_prob_overlap_mask*np.log2(sseg_prob_overlap_mask+1e-10)).max()

                sseg_pseudo_label_overlap_mask = sseg_pseudo_label_id2train.reshape(-1)[mask_p.reshape(-1)==1]
                intersecton_judgement = (sseg_pseudo_label_overlap_mask == category_id_seg).sum() / sseg_pseudo_label_overlap_mask.shape[0]

                if intersecton_judgement > 0.9:
                    score = 1

                if score > thres_ins[category_id - 1] and box_ent < sseg_prob_overlap_mask_ent:
                    anno['ignore'] = 0
                    anno['iscrowd'] = 0
                else:
                    anno['ignore'] = 1
                    anno['iscrowd'] = 1
                anno['area'] = int(mask_area(rle))
                anns.append(anno)
        gt['annotations'] = gt['annotations'] + anns
    save_file = pseudo_labels_path + 'instancesonly_gtFine_train_pseudo_labels_rle_fuse.json'
    with open(save_file, 'w') as f: f.write(json.dumps(gt)); f.close()


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete

if __name__ == "__main__":
    upsnet_test()
