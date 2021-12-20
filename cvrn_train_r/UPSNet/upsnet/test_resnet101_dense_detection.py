
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

import glob
import json
import scipy

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from upsnet.config.config import config
from upsnet.config.parse_args import parse_args
from lib.utils.logging import create_logger
from lib.utils.timer import Timer

args = parse_args()
logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)

from upsnet.dataset import *
from upsnet.models import *
from upsnet.bbox.bbox_transform import bbox_transform, clip_boxes, expand_boxes
from lib.utils.callback import Speedometer
from lib.utils.data_parallel import DataParallel
from pycocotools.mask import encode as mask_encode

from pycocotools import mask as COCOmask
# from sklearn.metrics import average_precision_score, precision_score,  recall_score, roc_curve

from pycocotools.coco import COCO

cv2.ocl.setUseOpenCL(False)

cudnn.enabled = True
cudnn.benchmark = False

def detection_test():

    gpus = [0]

    test_iteration_start = int(args.iter)

    # config.dataset.dataset = 'Cityscapes'
    # config.dataset.dataset = 'Cityscapes_7cls'

    # config.test.scales = [1024]
    # config.test.max_size = 2048
    config.test.scales = [800]
    config.test.max_size = 1600

    config.test.rpn_post_nms_top_n = 1000
    config.test.rpn_nms_thresh = 0.7
    config.test.nms_thresh = 0.5
    config.test.score_thresh = 0.05

    config.network.has_rpn = True
    config.network.has_rcnn = True
    config.network.has_mask_head = False
    config.network.has_fcn_head = False
    config.network.has_panoptic_head = False

    pprint.pprint(config)
    logger.info('test config:{}\n'.format(pprint.pformat(config)))

    # create data loader
    test_dataset = eval(config.dataset.dataset)(image_sets=config.dataset.test_image_set.split('+'), flip=False,
                                                result_path=final_output_path, phase='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False,
                                              num_workers=2, drop_last=False, pin_memory=False, collate_fn=test_dataset.collate)

    print('Prapare data')
    i_iter = 0
    test_iter = test_loader.__iter__()
    data_timer = Timer()
    data_all = []
    # labels_all = []
    while i_iter < len(test_loader):
        data_timer.tic()
        data, label, _ = test_iter.next()
        data_all.append(data)
        # labels_all.append(label)
        data_time = data_timer.toc()
        i_iter += 1
        if i_iter % 50 == 0:
            print('Iter_data %d/%d, data_time:%.3f' % (i_iter, len(test_loader), data_time))

    # with open(os.path.join(final_output_path, 'labels_all.pkl'), 'wb') as f:
    #     pickle.dump(labels_all, f, protocol=2)

    results_dir = os.path.join(final_output_path, 'results_detection_all')
    os.makedirs(results_dir, exist_ok=True)

    # train_dir = os.path.join(config.output_path,os.path.basename(args.cfg).split('.')[0],'train')
    # train_models = glob.glob(os.path.join(train_dir,'*0.pth'))
    # train_iter = [int(os.path.basename(x).split('.')[0].replace(config.model_prefix,'')) for x in train_models]
    # train_iter.sort()

    all_mAP = []
    ind = 0
    step = config.train.snapshot_step
    max_iter = config.train.max_iteration
    for curr_iter in range(step, max_iter+step, step):
        if curr_iter < test_iteration_start:
            continue
        restore_from = os.path.join(os.path.join(os.path.join(config.output_path, os.path.basename(args.cfg).split('.')[0]),
                                   '_'.join(config.dataset.image_set.split('+')), config.model_prefix+str(curr_iter)+'.pth'))
        if not os.path.exists(restore_from):
            print('Waiting for model..!')
            while not os.path.exists(restore_from):
                time.sleep(5)

        test_model = resnet_101_upsnet_test().cuda(device=gpus[0])
        test_model.load_state_dict(torch.load(restore_from, map_location='cuda:0'), resume=True)
        for p in test_model.parameters():
            p.requires_grad = False

        test_model = DataParallel(test_model, device_ids=gpus, gather_output=False).to(gpus[0])
        test_model.eval()

        batch_num = 0
        i_iter = 0

        all_boxes = [[] for _ in range(test_dataset.num_classes)]
        outputs = []

        net_timer = Timer()
        post_timer = Timer()
        while i_iter < len(test_loader):
            batch = []
            # labels = []
            for gpu_id in gpus:
                data = data_all[i_iter].copy()
                for k, v in data.items():
                    data[k] = v.pin_memory().to(gpu_id, non_blocking=True) if torch.is_tensor(v) else v
                batch.append((data, None))
                i_iter += 1
            batch_num += 1

            im_infos = [_[0]['im_info'][0] for _ in batch]

            net_timer.tic()
            with torch.no_grad():
                torch.cuda.synchronize()
                output = test_model(*batch)
                net_time = net_timer.toc()
                output = im_detect(output, batch, im_infos)
            post_timer.tic()
            outputs.append(output)
            for score, box, cls_idx in zip(output['scores'], output['boxes'], output['cls_inds']):
                im_post(all_boxes, score, box, cls_idx, test_dataset.num_classes)
            post_time = post_timer.toc()
            if i_iter % 100 == 0:
                print('Curr_iter %d/%d, Batch %d/%d, net_time:%.3f, post_time:%.3f' % (curr_iter, max_iter, batch_num, len(test_loader)/len(gpus), net_time, post_time))

        # trim redundant predictions
        for i in range(1, test_dataset.num_classes):
            all_boxes[i] = all_boxes[i][:len(test_loader)]
        #
        mAP, ap_all = test_dataset.evaluate_boxes_all(all_boxes, results_dir, curr_iter)
        mAP = round(mAP,1)
        all_mAP.append(mAP)
        print(all_mAP)

        # os.makedirs(os.path.join(final_output_path, 'outputs_all'), exist_ok=True)
        # with open(os.path.join(final_output_path, 'outputs_all', 'outputs_iter_' + str(curr_iter) + '.pkl'), 'wb') as f:
        #     pickle.dump(outputs, f, protocol=2)

        del test_model
        ind += 1

def im_detect(output_all, data, im_infos):

    scores_all = []
    pred_boxes_all = []
    pred_masks_all = []
    pred_ssegs_all = []
    pred_panos_all = []
    pred_pano_cls_inds_all = []
    cls_inds_all = []

    if len(data) == 1:
        output_all = [output_all]

    output_all = [{k: v.data.cpu().numpy() for k, v in output.items()} for output in output_all]

    for i in range(len(data)):
        im_info = im_infos[i]
        scores_all.append(output_all[i]['cls_probs'])
        pred_boxes_all.append(output_all[i]['pred_boxes'][:, 1:] / im_info[2])
        cls_inds_all.append(output_all[i]['cls_inds'])

        if config.network.has_mask_head:
            pred_masks_all.append(output_all[i]['mask_probs'])
        if config.network.has_fcn_head:
            pred_ssegs_all.append(output_all[i]['fcn_outputs'])
        if config.network.has_panoptic_head:
            pred_panos_all.append(output_all[i]['panoptic_outputs'])
            pred_pano_cls_inds_all.append(output_all[i]['panoptic_cls_inds'])

    return {
        'scores': scores_all,
        'boxes': pred_boxes_all,
        'masks': pred_masks_all,
        'ssegs': pred_ssegs_all,
        'panos': pred_panos_all,
        'cls_inds': cls_inds_all,
        'pano_cls_inds': pred_pano_cls_inds_all,
    }

def im_post(boxes_all, scores, pred_boxes, cls_inds, num_classes):
    for idx in range(1, num_classes):
        cls_boxes = np.hstack([pred_boxes[idx == cls_inds, :], scores.reshape(-1, 1)[idx == cls_inds]])
        boxes_all[idx].append(cls_boxes)

def bbox_results_one_category(self, boxes, cat_id):
    results = []
    image_ids = self.dataset.COCO.getImgIds()
    image_ids.sort()
    assert len(boxes) == len(image_ids)
    for i, image_id in enumerate(image_ids):
        dets = boxes[i]
        if isinstance(dets, list) and len(dets) == 0:
            continue
        dets = dets.astype(np.float)
        scores = dets[:, -1]
        xywh_dets = bbox_transform.xyxy_to_xywh(dets[:, 0:4])
        xs = xywh_dets[:, 0]
        ys = xywh_dets[:, 1]
        ws = xywh_dets[:, 2]
        hs = xywh_dets[:, 3]
        results.extend(
            [{'image_id': image_id,
              'category_id': cat_id,
              'bbox': [xs[k], ys[k], ws[k], hs[k]],
              'score': scores[k]} for k in range(dets.shape[0])])
    return results

if __name__ == "__main__":
    detection_test()
