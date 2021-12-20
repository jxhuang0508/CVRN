
from __future__ import print_function

import os
import sys
import torch
import torch.utils.data

import pickle, gzip
import numpy as np
import scipy.io as sio
import cv2
import json
from PIL import Image, ImageDraw
from collections import defaultdict, Sequence
from pycocotools.cocoeval import COCOeval

from upsnet.config.config import config
from upsnet.dataset.json_dataset import JsonDataset, extend_with_flipped_entries, filter_for_training, add_bbox_regression_targets
from upsnet.dataset.base_dataset import BaseDataset
from upsnet.rpn.assign_anchor import add_rpn_blobs
from upsnet.bbox.sample_rois import sample_rois
from lib.utils.logging import logger
import pycocotools.mask as mask_util

class Cityscapes_7cls_ST(BaseDataset):

    def __init__(self, image_sets, flip=False, proposal_files=None, phase='train', result_path=''):

        super(Cityscapes_7cls_ST, self).__init__()

        self.image_dirs = {
            'train': os.path.join(config.dataset.dataset_path, 'images'),
            'val': os.path.join(config.dataset.dataset_path, 'images'),
            'test': os.path.join(config.dataset.dataset_path, 'images'),
            'train_extra': os.path.join(config.dataset.dataset_path, 'images'),
            'debug': os.path.join(config.dataset.dataset_path, 'images'),
        }

        self.anno_files = {
            # 'train': 'instancesonly_gtFine_train.json',
            'train': 'pseudo_labels/instancesonly_gtFine_train_pseudo_labels_rle_fuse.json',
            'val': 'instancesonly_gtFine_val.json',
            'test': 'image_info_test.json',
            'train_extra': 'instancesonly_gtCoarse_train_extra.json',
            'debug': 'instancesonly_gtFine_debug.json',
        }

        self.panoptic_json_file = os.path.join(config.dataset.dataset_path, 'annotations_7cls', 'cityscapes_fine_val.json')
        self.panoptic_gt_folder = 'data/cityscapes/panoptic'

        self.flip = flip
        self.result_path = result_path
        self.num_classes = 9
        self.phase = phase
        self.image_sets = image_sets

        if image_sets[0] == 'demoVideo':
            assert len(image_sets) == 1
            assert phase == 'test'
            im_path = [_.strip() for _ in open('data/cityscapes/split/demoVideo_img.txt', 'r').readlines()]
            self.roidb = [{'image': _, 'flipped': False} for _ in im_path]
            return

        if proposal_files is None:
            proposal_files = [None] * len(image_sets)

        if phase == 'train' and len(image_sets) > 1:
            # combine multiple datasets
            roidbs = []
            for image_set, proposal_file in zip(image_sets, proposal_files):
                dataset = JsonDataset('cityscapes_' + image_set,
                                      image_dir=self.image_dirs[image_set],
                                      anno_file=os.path.join(config.dataset.dataset_path, 'annotations_7cls', self.anno_files[image_set]))
                roidb = dataset.get_roidb(gt=True, proposal_file=proposal_file, crowd_filter_thresh=config.train.crowd_filter_thresh)
                if flip:
                    if logger:
                        logger.info('Appending horizontally-flipped training examples...')
                    extend_with_flipped_entries(roidb, dataset)
                roidbs.append(roidb)
            roidb = roidbs[0]
            for r in roidbs[1:]:
                roidb.extend(r)
            roidb = filter_for_training(roidb)
            add_bbox_regression_targets(roidb)

        else:
            assert len(image_sets) == 1
            self.dataset = JsonDataset('cityscapes_' + image_sets[0],
                                       image_dir=self.image_dirs[image_sets[0]],
                                       anno_file=os.path.join(config.dataset.dataset_path, 'annotations_7cls',
                                       self.anno_files[image_sets[0]]))
            roidb = self.dataset.get_roidb(gt=True, proposal_file=proposal_files[0],
                                           crowd_filter_thresh=config.train.crowd_filter_thresh if phase != 'test' else 0)
            if flip:
                if logger:
                    logger.info('Appending horizontally-flipped training examples...')
                extend_with_flipped_entries(roidb, self.dataset)
            if phase != 'test':
                roidb = filter_for_training(roidb)
                add_bbox_regression_targets(roidb)
            else:
                if logger:
                    logger.info('Filtered roidb entries: {}'.format(len(roidb)))
                else:
                    print('Filtered roidb entries: {}'.format(len(roidb)))

        self.roidb = roidb

    def __getitem__(self, index):
        blob = defaultdict(list)
        im_blob, im_scales = self.get_image_blob([self.roidb[index]])
        if config.network.has_rpn:
            if self.phase != 'test':
                add_rpn_blobs(blob, im_scales, [self.roidb[index]])
                data = {'data': im_blob,
                        'im_info': blob['im_info']}
                label = {'roidb': blob['roidb'][0]}
                for stride in config.network.rpn_feat_stride:
                    label.update({
                        'rpn_labels_fpn{}'.format(stride): blob['rpn_labels_int32_wide_fpn{}'.format(stride)].astype(
                            np.int64),
                        'rpn_bbox_targets_fpn{}'.format(stride): blob['rpn_bbox_targets_wide_fpn{}'.format(stride)],
                        'rpn_bbox_inside_weights_fpn{}'.format(stride): blob[
                            'rpn_bbox_inside_weights_wide_fpn{}'.format(stride)],
                        'rpn_bbox_outside_weights_fpn{}'.format(stride): blob[
                            'rpn_bbox_outside_weights_wide_fpn{}'.format(stride)]
                    })
            else:
                data = {'data': im_blob,
                        'im_info': np.array([[im_blob.shape[-2],
                                              im_blob.shape[-1],
                                              im_scales[0]]], np.float32),
                        }
                # label = {'roidb': self.roidb[index]}
                label = None
        else:
            if self.phase != 'test':
                data = {'data': im_blob,
                        'im_info': blob['im_info']}
                label = {'roidb': self.roidb[index]}
            else:
                data = {'data': im_blob,
                        'rois': np.hstack(
                            (np.zeros((self.roidb[index]['boxes'].shape[0], 1)), self.roidb[index]['boxes'])).astype(
                            np.float32),
                        'im_info': np.array([[im_blob.shape[-2],
                                              im_blob.shape[-1],
                                              im_scales[0]]], np.float32),
                        'id': self.roidb[index]['id']}
                label = None
        if config.network.has_fcn_head:
            if self.phase != 'test':
                seg_gt = np.array(Image.open(self.roidb[index]['image'].replace('images', 'labels_16cls').replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')))
                if self.roidb[index]['flipped']:
                    seg_gt = np.fliplr(seg_gt)
                seg_gt = cv2.resize(seg_gt, None, None, fx=im_scales[0], fy=im_scales[0], interpolation=cv2.INTER_NEAREST)
                label.update({'seg_gt': seg_gt})
                label.update({'gt_classes': label['roidb']['gt_classes']})
                label.update({'mask_gt': np.zeros((len(label['gt_classes']), im_blob.shape[-2], im_blob.shape[-1]))})
                for i in range(len(label['gt_classes'])):
                    img = Image.new('L', (int(im_blob.shape[-1] / im_scales[0]), int(im_blob.shape[-2] / im_scales[0])), 0)
                    for j in range(len(label['roidb']['segms'][i])):
                        ImageDraw.Draw(img).polygon(tuple(label['roidb']['segms'][i][j]), outline=1, fill=1)
                    label['mask_gt'][i] = cv2.resize(np.array(img), None, None, fx=im_scales[0], fy=im_scales[0], interpolation=cv2.INTER_NEAREST)
                if config.train.fcn_with_roi_loss:
                    gt_boxes = label['roidb']['boxes'][np.where(label['roidb']['gt_classes'] > 0)[0]]
                    gt_boxes = np.around(gt_boxes * im_scales[0]).astype(np.int32)
                    label.update({'seg_roi_gt': np.zeros((len(gt_boxes), config.network.mask_size, config.network.mask_size), dtype=np.int64)})
                    for i in range(len(gt_boxes)):
                        if gt_boxes[i][3] == gt_boxes[i][1]:
                            gt_boxes[i][3] += 1
                        if gt_boxes[i][2] == gt_boxes[i][0]:
                            gt_boxes[i][2] += 1
                        label['seg_roi_gt'][i] = cv2.resize(seg_gt[gt_boxes[i][1]:gt_boxes[i][3], gt_boxes[i][0]:gt_boxes[i][2]], (config.network.mask_size, config.network.mask_size), interpolation=cv2.INTER_NEAREST)
            else:
                pass

        return data, label, index
        # return data.copy(), label, index

    def get_image_blob(self, roidb):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        num_images = len(roidb)
        # Sample random scales to use for each image in this batch
        if self.phase == 'train':
            scale_inds = np.random.randint(
                0, high=len(config.train.scales), size=num_images
            )
        else:
            scale_inds = np.random.randint(
                0, high=len(config.test.scales), size=num_images
            )
        processed_ims = []
        im_scales = []
        for i in range(num_images):
            im = cv2.imread(roidb[i]['image'])

            # print(roidb[i]['image'])

            assert im is not None, \
                'Failed to read image \'{}\''.format(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            if self.phase == 'train':
                target_size = config.train.scales[scale_inds[i]]
                im, im_scale = self.prep_im_for_blob(
                    im, config.network.pixel_means, [target_size], config.train.max_size
                )
            else:
                target_size = config.test.scales[scale_inds[i]]
                im, im_scale = self.prep_im_for_blob(
                    im, config.network.pixel_means, [target_size], config.test.max_size
                )
            im_scales.append(im_scale[0])
            processed_ims.append(im[0].transpose(2, 0, 1))

        # Create a blob to hold the input images
        assert len(processed_ims) == 1
        blob = processed_ims[0]

        return blob, im_scales

    def vis_all_mask(self, all_boxes, all_masks, save_path=None):
        """
        visualize all detections in one image
        :param im_array: [b=1 c h w] in rgb
        :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
        :param class_names: list of names in imdb
        :param scale: visualize the scaled image
        :return:
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        import random
        import cv2
        palette = {
            'person': (220, 20, 60),
            'rider': (255, 0, 0),
            'car': (0, 0, 142),
            'bus': (0, 60, 100),
            'motorcycle': (0, 0, 230),
            'bicycle': (119, 11, 32),
            #
            'road': (128, 64, 128),
            'sidewalk': (244, 35, 232),
            'building': (70, 70, 70),
            'wall': (102, 102, 156),
            'fence': (190, 153, 153),
            'pole': (153, 153, 153),
            'sky': (70, 130, 180),
            'traffic light': (250, 170, 30),
            'traffic sign': (220, 220, 0),
            'vegetation': (107, 142, 35)
        }
        name2id = {
            'road': 0,
            'sidewalk': 1,
            'building': 2,
            'wall': 3,
            'fence': 4,
            'pole': 5,
            'traffic light': 6,
            'traffic sign': 7,
            'vegetation': 8,
            'terrain': 9
        }
        self.classes = [
            '__background__',
            'person',
            'rider',
            'car',
            'bus',
            'motorcycle',
            'bicycle',
        ]

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

        for i in range(len(self.roidb)):

            im = np.array(Image.open(self.roidb[i]['image']))
            fig = plt.figure(frameon=False)

            fig.set_size_inches(im.shape[1] / 200, im.shape[0] / 200)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            ax.imshow(im)
            for j, name in enumerate(self.classes):
                if name == '__background__':
                    continue
                boxes = all_boxes[j][i]
                segms = all_masks[j][i]
                if segms == []:
                    continue
                masks = mask_util.decode(segms)
                for k in range(boxes.shape[0]):
                    score = boxes[k, -1]
                    mask = masks[:, :, k]
                    if score < 0.5:
                        continue
                    bbox = boxes[k, :]
                    ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                      fill=False, edgecolor='g', linewidth=2, alpha=0.5)
                    )
                    # ax.text(bbox[0], bbox[1] - 2, name + '{:0.2f}'.format(score).lstrip('0'), fontsize=6, family='serif',
                    #         bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'), color='white')
                    # contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                    _, contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                    color = (palette[name][0] / 255, palette[name][1] / 255, palette[name][2] / 255)
                    for c in contour:
                        ax.add_patch(
                            Polygon(
                                c.reshape((-1, 2)),
                                fill=True, facecolor=color, edgecolor='w', linewidth=1, alpha=0.5
                            )
                        )
            if save_path is None:
                plt.show()
            else:
                fig.savefig(os.path.join(save_path, '{}.png'.format(self.roidb[i]['image'].split('/')[-1][:-16])), dpi=100)
            plt.close('all')

    def flip_vis_all_mask(self, all_boxes, all_masks, save_path=None):
        """
        visualize all detections in one image
        :param im_array: [b=1 c h w] in rgb
        :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
        :param class_names: list of names in imdb
        :param scale: visualize the scaled image
        :return:
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        import random
        import cv2
        # palette = {
        #     'person': (220, 20, 60),
        #     'rider': (255, 0, 0),
        #     'car': (0, 0, 142),
        #     'truck': (0, 0, 70),
        #     'bus': (0, 60, 100),
        #     'train': (0, 80, 100),
        #     'motorcycle': (0, 0, 230),
        #     'bicycle': (119, 11, 32),
        #     #
        #     'road': (128, 64, 128),
        #     'sidewalk': (244, 35, 232),
        #     'building': (70, 70, 70),
        #     'wall': (102, 102, 156),
        #     'fence': (190, 153, 153),
        #     'pole': (153, 153, 153),
        #     'sky': (70, 130, 180),
        #     'traffic light': (250, 170, 30),
        #     'traffic sign': (220, 220, 0),
        #     'vegetation': (107, 142, 35),
        #     'terrain': (152, 251, 152)
        # }
        # name2id = {
        #     'road': 0,
        #     'sidewalk': 1,
        #     'building': 2,
        #     'wall': 3,
        #     'fence': 4,
        #     'pole': 5,
        #     'traffic light': 6,
        #     'traffic sign': 7,
        #     'vegetation': 8,
        #     'terrain': 9,
        #     'sky': 10
        # }
        # self.classes = [
        #     '__background__',
        #     'person',
        #     'rider',
        #     'car',
        #     'bus',
        #     'motorcycle',
        #     'bicycle',
        # ]
        palette = {
            'person': (220, 20, 60),
            'rider': (255, 0, 0),
            'car': (0, 0, 142),
            'bus': (0, 60, 100),
            'motorcycle': (0, 0, 230),
            'bicycle': (119, 11, 32),
            #
            'road': (128, 64, 128),
            'sidewalk': (244, 35, 232),
            'building': (70, 70, 70),
            'wall': (102, 102, 156),
            'fence': (190, 153, 153),
            'pole': (153, 153, 153),
            'sky': (70, 130, 180),
            'traffic light': (250, 170, 30),
            'traffic sign': (220, 220, 0),
            'vegetation': (107, 142, 35)
        }
        name2id = {
            'road': 0,
            'sidewalk': 1,
            'building': 2,
            'wall': 3,
            'fence': 4,
            'pole': 5,
            'traffic light': 6,
            'traffic sign': 7,
            'vegetation': 8,
            'sky': 9
        }
        self.classes = [
            '__background__',
            'person',
            'rider',
            'car',
            'bus',
            'motorcycle',
            'bicycle',
        ]

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

        for i in range(len(self.roidb)):

            im = np.array(Image.open(self.roidb[i]['image']))
            im = im[:, ::-1, :]
            fig = plt.figure(frameon=False)

            fig.set_size_inches(im.shape[1] / 200, im.shape[0] / 200)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            ax.imshow(im)
            for j, name in enumerate(self.classes):
                if name == '__background__':
                    continue
                boxes = all_boxes[j][i]
                segms = all_masks[j][i]
                if segms == []:
                    continue
                masks = mask_util.decode(segms)
                for k in range(boxes.shape[0]):
                    score = boxes[k, -1]
                    mask = masks[:, :, k]
                    if score < 0.5:
                        continue
                    bbox = boxes[k, :]
                    ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                      fill=False, edgecolor='g', linewidth=2, alpha=0.5)
                    )
                    # ax.text(bbox[0], bbox[1] - 2, name + '{:0.2f}'.format(score).lstrip('0'), fontsize=6, family='serif',
                    #         bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'), color='white')
                    # contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                    _, contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                    color = (palette[name][0] / 255, palette[name][1] / 255, palette[name][2] / 255)
                    for c in contour:
                        ax.add_patch(
                            Polygon(
                                c.reshape((-1, 2)),
                                fill=True, facecolor=color, edgecolor='w', linewidth=1, alpha=0.5
                            )
                        )
            if save_path is None:
                plt.show()
            else:
                fig.savefig(os.path.join(save_path, '{}.png'.format(self.roidb[i]['image'].split('/')[-1][:-16])), dpi=100)
            plt.close('all')


    def evaluate_masks(
            self,
            all_boxes,
            all_segms,
            output_dir,
    ):
        res_file = os.path.join(
            output_dir, 'segmentations_' + self.dataset.name + '_results.json'
        )
        results = []
        for cls_ind, cls in enumerate(self.dataset.classes):
            if cls == '__background__':
                continue
            if cls_ind >= len(all_boxes):
                break
            cat_id = self.dataset.category_to_id_map[cls]
            results.extend(self.segms_results_one_category(all_boxes[cls_ind], all_segms[cls_ind], cat_id))
        if logger:
            logger.info(
            'Writing segmentation results json to: {}'.format(
                os.path.abspath(res_file)))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)
        coco_dt = self.dataset.COCO.loadRes(str(res_file))
        coco_eval = COCOeval(self.dataset.COCO, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        # self.log_detection_eval_metrics(coco_eval, os.path.join(output_dir, 'instance_segmentation_results.txt'))
        test_iter = config.iter
        mAP, AP_all = self.log_detection_eval_metrics_ap50(coco_eval, os.path.join(output_dir, 'instance_segmentation_results_iter' + str(test_iter) + '.txt'), test_iter)
        return mAP, AP_all

    def evaluate_masks_folder(self, output_dir,):
        res_file = os.path.join(
            output_dir, 'segmentations_' + self.dataset.name + '_results.json'
        )
        coco_dt = self.dataset.COCO.loadRes(str(res_file))
        coco_eval = COCOeval(self.dataset.COCO, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        # self.log_detection_eval_metrics(coco_eval, os.path.join(output_dir, 'instance_segmentation_results.txt'))
        test_iter = config.iter
        mAP, AP_all = self.log_detection_eval_metrics_ap50(coco_eval, os.path.join(output_dir, 'instance_segmentation_results_iter' + str(test_iter) + '.txt'), test_iter)
        return mAP, AP_all


    def evaluate_masks_all(
            self,
            all_boxes,
            all_segms,
            output_dir,
            test_iter,
    ):
        # res_file = os.path.join(output_dir, 'segmentations_' + self.dataset.name + '_results.json')
        res_file = os.path.join(output_dir, 'segmentations_iter_' + str(test_iter)  + '.json')
        results = []
        for cls_ind, cls in enumerate(self.dataset.classes):
            if cls == '__background__':
                continue
            if cls_ind >= len(all_boxes):
                break
            cat_id = self.dataset.category_to_id_map[cls]
            results.extend(self.segms_results_one_category(all_boxes[cls_ind], all_segms[cls_ind], cat_id))
        if logger:
            logger.info(
            'Writing segmentation results json to: {}'.format(
                os.path.abspath(res_file)))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)
        coco_dt = self.dataset.COCO.loadRes(str(res_file))
        coco_eval = COCOeval(self.dataset.COCO, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self.log_detection_eval_metrics_ap50(coco_eval, os.path.join(output_dir, 'instance_segmentation_results.txt'), test_iter)
        return coco_eval.stats

    def evaluate_inst_ssegs_all(self, segmentation_results, test_iter):

        # confusion_matrix = np.zeros((config.dataset.num_seg_classes, config.dataset.num_seg_classes))
        confusion_matrix = np.zeros((config.dataset.num_classes, config.dataset.num_classes))
        for i, roidb in enumerate(self.roidb):

            seg_gt = np.array(Image.open(roidb['image'].replace('images', 'labels_16cls').replace('leftImg8bit.png', 'gtFine_labelTrainIds.png'))).astype('float32')

            segmentation_result = np.uint8(np.squeeze(np.copy(segmentation_results[i])))
            segmentation_result = Image.fromarray(segmentation_result)

            seg_pred = np.array(segmentation_result.resize((seg_gt.shape[1], seg_gt.shape[0]), Image.NEAREST))
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]

            # selct 6 classse (10 'person', 11 'rider', 12 'car', 13 'bus', 14 'motorcycle', 15 'bicycle')
            seg_gt = seg_gt - 9
            seg_gt[seg_gt<0] = 0

            seg_pred = seg_pred[ignore_index]

            confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, config.dataset.num_classes)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        # IU_array = (tp / np.maximum(1.0, pos + res - tp)).tolist()
        IU_array = (100* tp / np.maximum(1.0, pos + res - tp)).tolist()

        # selct 6 classse (delete 0 'background')
        IU_array_6 = IU_array[1:]
        mean_IU_6 = np.array(IU_array_6).mean()

        logger.info('Eval_seg_' + str(test_iter))
        logger.info('IoU_array:')
        for i in range(len(IU_array_6)):
            logger.info('%.1f' % IU_array_6[i])
        logger.info('mIoU:%.1f' % mean_IU_6)

    def get_pallete(self):

        pallete_raw = np.zeros((256, 3)).astype('uint8')
        pallete = np.zeros((256, 3)).astype('uint8')

        pallete_raw[5, :] =  [111,  74,   0]
        pallete_raw[6, :] =  [ 81,   0,  81]
        pallete_raw[7, :] =  [128,  64, 128]
        pallete_raw[8, :] =  [244,  35, 232]
        pallete_raw[9, :] =  [250, 170, 160]
        pallete_raw[10, :] = [230, 150, 140]
        pallete_raw[11, :] = [ 70,  70,  70]
        pallete_raw[12, :] = [102, 102, 156]
        pallete_raw[13, :] = [190, 153, 153]
        pallete_raw[14, :] = [180, 165, 180]
        pallete_raw[15, :] = [150, 100, 100]
        pallete_raw[16, :] = [150, 120,  90]
        pallete_raw[17, :] = [153, 153, 153]
        pallete_raw[18, :] = [153, 153, 153]
        pallete_raw[19, :] = [250, 170,  30]
        pallete_raw[20, :] = [220, 220,   0]
        pallete_raw[21, :] = [107, 142,  35]
        pallete_raw[22, :] = [152, 251, 152]
        pallete_raw[23, :] = [ 70, 130, 180]
        pallete_raw[24, :] = [220,  20,  60]
        pallete_raw[25, :] = [255,   0,   0]
        pallete_raw[26, :] = [  0,   0, 142]
        pallete_raw[27, :] = [  0,   0,  70]
        pallete_raw[28, :] = [  0,  60, 100]
        pallete_raw[29, :] = [  0,   0,  90]
        pallete_raw[30, :] = [  0,   0, 110]
        pallete_raw[31, :] = [  0,  80, 100]
        pallete_raw[32, :] = [  0,   0, 230]
        pallete_raw[33, :] = [119,  11,  32]

        # train2regular = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        # selct 16 classse (delete 9 'terrain', 14 'truck', 16 'train')
        train2regular = [7, 8, 11, 12, 13, 17, 19, 20, 21, 23, 24, 25, 26, 28, 32, 33]

        for i in range(len(train2regular)):
            pallete[i, :] = pallete_raw[train2regular[i], :]

        pallete = pallete.reshape(-1)

        # return pallete_raw
        return pallete

    def vis_all_seg(self, pred_segmentations, save_path=None):
        self.write_segmentation_result(pred_segmentations, save_path)

    def evaluate_ssegs(self, pred_segmentations, res_file_folder):
        self.write_segmentation_result(pred_segmentations, res_file_folder)

        confusion_matrix = np.zeros((config.dataset.num_seg_classes, config.dataset.num_seg_classes))
        for i, roidb in enumerate(self.roidb):

            seg_gt = np.array(Image.open(roidb['image'].replace('images', 'labels_16cls').replace('leftImg8bit.png', 'gtFine_labelTrainIds.png'))).astype('float32')

            seg_pathes = os.path.split(roidb['image'].replace('images', 'labels_16cls').replace('leftImg8bit.png', 'gtFine_labelTrainIds.png'))
            res_image_name = seg_pathes[-1][:-len('_gtFine_labelTrainIds.png')]
            res_save_path = os.path.join(res_file_folder, res_image_name + '.png')

            seg_pred = Image.open(res_save_path)

            seg_pred = np.array(seg_pred.resize((seg_gt.shape[1], seg_gt.shape[0]), Image.NEAREST))
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]

            confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, config.dataset.num_seg_classes)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()

        evaluation_results = {'meanIU':mean_IU, 'IU_array':IU_array, 'confusion_matrix': confusion_matrix}

        def convert_confusion_matrix(confusion_matrix):
            cls_sum = confusion_matrix.sum(axis=1)
            confusion_matrix = confusion_matrix / cls_sum.reshape((-1, 1))
            return confusion_matrix

        logger.info('evaluate segmentation:')
        meanIU = evaluation_results['meanIU']
        IU_array = evaluation_results['IU_array']
        confusion_matrix = convert_confusion_matrix(evaluation_results['confusion_matrix'])
        logger.info('IU_array:')
        for i in range(len(IU_array)):
            # logger.info('%.5f' % IU_array[i])
            logger.info('%.1f' % IU_array[i]*100)
        # logger.info('meanIU:%.5f' % meanIU)
        logger.info('meanIU:%.1f' % meanIU*100)
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        import re
        # confusion_matrix = re.sub('[\[\]]', '', np.array2string(confusion_matrix, separator='\t'))
        # logger.info('confusion_matrix:')
        # logger.info(confusion_matrix)

        return IU_array

    def evaluate_ssegs_folder(self, res_file_folder):
        all_ssegs = []
        confusion_matrix = np.zeros((config.dataset.num_seg_classes, config.dataset.num_seg_classes))
        for i, roidb in enumerate(self.roidb):

            seg_gt = np.array(Image.open(roidb['image'].replace('images', 'labels_16cls').replace('leftImg8bit.png', 'gtFine_labelTrainIds.png'))).astype('float32')

            seg_pathes = os.path.split(roidb['image'].replace('images', 'labels_16cls').replace('leftImg8bit.png', 'gtFine_labelTrainIds.png'))
            res_image_name = seg_pathes[-1][:-len('_gtFine_labelTrainIds.png')]
            res_save_path = os.path.join(res_file_folder, res_image_name + '.png')

            seg_pred = Image.open(res_save_path)

            seg_pred = np.array(seg_pred.resize((seg_gt.shape[1], seg_gt.shape[0]), Image.NEAREST))

            all_ssegs.append(seg_pred)

            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]

            confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, config.dataset.num_seg_classes)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()

        evaluation_results = {'meanIU':mean_IU, 'IU_array':IU_array, 'confusion_matrix': confusion_matrix}

        def convert_confusion_matrix(confusion_matrix):
            cls_sum = confusion_matrix.sum(axis=1)
            confusion_matrix = confusion_matrix / cls_sum.reshape((-1, 1))
            return confusion_matrix
        IU_array = evaluation_results['IU_array']
        return IU_array, all_ssegs


    def evaluate_ssegs_all(self, segmentation_results, test_iter):

        confusion_matrix = np.zeros((config.dataset.num_seg_classes, config.dataset.num_seg_classes))
        for i, roidb in enumerate(self.roidb):

            seg_gt = np.array(Image.open(roidb['image'].replace('images', 'labels_16cls').replace('leftImg8bit.png', 'gtFine_labelTrainIds.png'))).astype('float32')

            segmentation_result = np.uint8(np.squeeze(np.copy(segmentation_results[i])))
            segmentation_result = Image.fromarray(segmentation_result)

            seg_pred = np.array(segmentation_result.resize((seg_gt.shape[1], seg_gt.shape[0]), Image.NEAREST))
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]

            confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, config.dataset.num_seg_classes)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        # IU_array = (tp / np.maximum(1.0, pos + res - tp)).tolist()
        IU_array = (100* tp / np.maximum(1.0, pos + res - tp)).tolist()
        mean_IU = np.array(IU_array).mean()

        logger.info('Eval_seg_' + str(test_iter))
        logger.info('IoU_array:')
        for i in range(len(IU_array)):
            logger.info('%.1f' % IU_array[i])
        logger.info('meanIoU: %.1f' % mean_IU)

        return mean_IU

    def write_segmentation_result(self, segmentation_results, res_file_folder):
        """
        Write the segmentation result to result_file_folder
        :param segmentation_results: the prediction result
        :param result_file_folder: the saving folder
        :return: [None]
        """
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        pallete = self.get_pallete()
        for i, roidb in enumerate(self.roidb):

            seg_pathes = os.path.split(roidb['image'])
            res_image_name = seg_pathes[-1][:-len('_leftImg8bit.png')]
            res_save_path = os.path.join(res_file_folder, res_image_name + '.png')

            segmentation_result = np.uint8(np.squeeze(np.copy(segmentation_results[i])))
            segmentation_result = Image.fromarray(segmentation_result)
            segmentation_result.putpalette(pallete)
            segmentation_result.save(res_save_path)

