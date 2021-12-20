
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

cv2.ocl.setUseOpenCL(False)

cudnn.enabled = True
cudnn.benchmark = False

def upsnet_test():

    gpus = [0]
    config.iter = args.iter

    # config.dataset.dataset = 'Cityscapes'
    config.dataset.dataset = 'Cityscapes_7cls'
    # config.test.scales = [512]
    # config.test.max_size = 1024
    config.test.scales = [800]
    config.test.max_size = 1600

    config.test.rpn_post_nms_top_n = 1000
    config.test.rpn_nms_thresh = 0.7
    config.test.nms_thresh = 0.5
    config.test.score_thresh = 0.05

    config.network.has_fcn_head = False
    config.network.has_rpn = True
    config.network.has_rcnn = True
    config.network.has_mask_head = True
    config.network.has_panoptic_head = False

    im_height = 1024
    im_width = 2048

    print('test iter:{}\n'.format(pprint.pformat(args.iter)))
    logger.info('test config:{}\n'.format(pprint.pformat(config)))

    final_output_path = os.path.join(config.output_path, os.path.basename(args.cfg).split('.')[0], config.dataset.test_image_set)
    test_dataset = eval(config.dataset.dataset)(image_sets=config.dataset.test_image_set.split('+'), flip=False,
                                                result_path=final_output_path, phase='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False,
                                              num_workers=2, drop_last=False, pin_memory=False, collate_fn=test_dataset.collate)

    restore_from = os.path.join(os.path.join(os.path.join(config.output_path, os.path.basename(args.cfg).split('.')[0]),
                                             '_'.join(config.dataset.image_set.split('+')), config.model_prefix + str(args.iter) + '.pth'))

    test_model = resnet_101_upsnet_test().cuda(device=gpus[0])
    test_model.load_state_dict(torch.load(restore_from, map_location='cuda:0'), resume=True)
    for p in test_model.parameters():
        p.requires_grad = False

    test_model = DataParallel(test_model, device_ids=gpus, gather_output=False).to(gpus[0])
    test_model.eval()

    all_boxes = [[] for _ in range(test_dataset.num_classes)]
    all_masks = [[] for _ in range(test_dataset.num_classes)]

    i_iter = 0
    test_iter = test_loader.__iter__()
    while i_iter < len(test_loader):
        batch = []
        labels = []
        for gpu_id in gpus:
            try:
                data, label, _ = test_iter.next()
                if label is not None:
                    data['roidb'] = label['roidb']
                for k, v in data.items():
                    data[k] = v.pin_memory().to(gpu_id, non_blocking=True) if torch.is_tensor(v) else v
            except StopIteration:
                data = data.copy()
                for k, v in data.items():
                    data[k] = v.pin_memory().to(gpu_id, non_blocking=True) if torch.is_tensor(v) else v
            batch.append((data, None))
            labels.append(label)
            i_iter += 1

        im_infos = [_[0]['im_info'][0] for _ in batch]

        net_timer = Timer()
        post_timer = Timer()
        with torch.no_grad():
            net_timer.tic()
            output = test_model(*batch)
            torch.cuda.synchronize()
            net_time = net_timer.toc()
            output = im_detect(output, batch, im_infos)
            post_timer.tic()
            for score, box, mask, cls_idx in zip(output['scores'], output['boxes'], output['masks'], output['cls_inds']):
                im_post(all_boxes, all_masks, score, box, mask, cls_idx, test_dataset.num_classes, (im_height, im_width))
            post_time = post_timer.toc()
            print('Batch %d/%d, net_time:%.3f, post_time:%.3f' % (i_iter, len(test_loader), net_time, post_time))

    output_dir = os.path.join(final_output_path, 'results_iter_'+ args.iter)
    os.makedirs(output_dir, exist_ok=True)

    # import pdb
    # pdb.set_trace()

    outputs = {'all_boxes': all_boxes,'all_masks': all_masks,}
    with open(os.path.join(output_dir, 'outputs.pkl'), 'wb') as f: pickle.dump(outputs, f, protocol=2)

    test_dataset.evaluate_boxes(all_boxes, output_dir)
    test_dataset.evaluate_masks(all_boxes, all_masks, output_dir)
    # test_dataset.vis_all_mask(all_boxes, all_masks,os.path.join(output_dir, 'inssegs'))

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

def im_post(boxes_all, masks_all, scores, pred_boxes, pred_masks, cls_inds, num_classes, im_info):

    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0

    M = config.network.mask_size

    scale = (M + 2.0) / M

    ref_boxes = expand_boxes(pred_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    for idx in range(1, num_classes):
        segms = []
        cls_boxes = np.hstack([pred_boxes[idx == cls_inds, :], scores.reshape(-1, 1)[idx == cls_inds]])
        cls_pred_masks = pred_masks[idx == cls_inds]
        cls_ref_boxes = ref_boxes[idx == cls_inds]
        for _ in range(cls_boxes.shape[0]):

            if pred_masks.shape[1] > 1:
                padded_mask[1:-1, 1:-1] = cls_pred_masks[_, idx, :, :]
            else:
                padded_mask[1:-1, 1:-1] = cls_pred_masks[_, 0, :, :]
            ref_box = cls_ref_boxes[_, :]

            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > 0.5, dtype=np.uint8)
            im_mask = np.zeros((im_info[0], im_info[1]), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_info[1])
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_info[0])

            im_mask[y_0:y_1, x_0:x_1] = mask[
                                        (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                        (x_0 - ref_box[0]):(x_1 - ref_box[0])
                                        ]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            rle['counts'] = rle['counts'].decode()
            segms.append(rle)

            mask_ind += 1

        cls_segms[idx] = segms
        boxes_all[idx].append(cls_boxes)
        masks_all[idx].append(segms)

if __name__ == "__main__":
    upsnet_test()
