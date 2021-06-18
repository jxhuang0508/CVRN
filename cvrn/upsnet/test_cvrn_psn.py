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
    pprint.pprint(config)
    logger.info('test config:{}\n'.format(pprint.pformat(config)))

    # create models
    gpus = [int(_) for _ in config.gpus.split(',')]
    test_maskrcnn_branch = eval(config.symbol)().cuda(device=gpus[0])

    # create data loader
    test_dataset = eval(config.dataset.dataset)(image_sets=config.dataset.test_image_set.split('+'), flip=False,
                                                result_path=final_output_path, phase='test')

    if args.eval_only:
        test_dataset.evaluate_boxes_folder(os.path.join(final_output_path, 'results'))
        test_dataset.evaluate_masks_folder(os.path.join(final_output_path, 'results'))
        test_dataset.evaluate_ssegs_folder(os.path.join(final_output_path, 'results', 'ssegs'))
        test_dataset.evaluate_panoptic_folder(os.path.join(final_output_path, 'results', 'pans_combined'))
        sys.exit()

    test_maskrcnn_branch.load_state_dict(torch.load(args.weight_path), resume=True)
    for p in test_maskrcnn_branch.parameters():
        p.requires_grad = False
    test_maskrcnn_branch = DataParallel(test_maskrcnn_branch, device_ids=gpus, gather_output=False).to(gpus[0])
    test_maskrcnn_branch.eval()

    test_deeplab_branch = Res_Deeplab(num_classes=config.dataset.num_seg_classes).cuda(device=gpus[0])
    test_deeplab_branch.load_state_dict(torch.load('../pretrained_models/cvrn_psn_deeplab_branch.pth'))
    test_deeplab_branch = DataParallel(test_deeplab_branch, device_ids=gpus, gather_output=False).to(gpus[0])
    test_deeplab_branch.eval()

    from CRST.dataset.helpers.labels_cityscapes_synthia import id2label, trainId2label
    DATA_TGT_DIRECTORY = config.dataset.dataset_path
    DATA_TGT_TEST_LIST_PATH = 'CRST/dataset/list/cityscapes/val.lst'
    label_2_id = 255 * np.ones((256,))
    for l in id2label:
        if l in (-1, 255):
            continue
        label_2_id[l] = id2label[l].trainId
    id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
    valid_labels = sorted(set(id_2_label.ravel()))
    _, _, _, test_num = parse_split_list(DATA_TGT_TEST_LIST_PATH)
    scorer = ScoreUpdater(valid_labels, config.dataset.num_seg_classes, test_num, logger)
    scorer.reset()
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)  # BGR
    IMG_STD = np.array((1.0, 1.0, 1.0), dtype=np.float32)
    TEST_IMAGE_SIZE = [1024,2048]
    TEST_SCALE = [0.5,0.8,1.0]
    TEST_FLIPPING = True
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142,
               0, 60, 100, 0, 0, 230, 119, 11, 32]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    def colorize_mask(mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask
    all_ssegs = []
    os.makedirs(os.path.join(final_output_path, 'results', 'ssegs'), exist_ok=True)
    testloader_crst = torch.utils.data.DataLoader(GTA5TestDataSet(DATA_TGT_DIRECTORY, DATA_TGT_TEST_LIST_PATH, test_size=TEST_IMAGE_SIZE, test_scale=1.0, mean=IMG_MEAN, std=IMG_STD, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=False)
    interp = nn.Upsample(size=TEST_IMAGE_SIZE, mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(testloader_crst):
            image, label, _, name = batch
            img = image.clone()
            for scale_idx in range(len(TEST_SCALE)):
                image = torch.nn.functional.interpolate(img, scale_factor=TEST_SCALE[scale_idx], mode='bilinear', align_corners=True)
                output2 = test_deeplab_branch(image.cuda().unsqueeze(0))
                coutput = interp(output2).cpu().data[0].numpy()
                if TEST_FLIPPING:
                    output2 = test_deeplab_branch(torch.from_numpy(image.numpy()[:,:,:,::-1].copy()).cuda().unsqueeze(0))
                    coutput = 0.5 * ( coutput + interp(output2).cpu().data[0].numpy()[:,:,::-1] )
                if scale_idx == 0:
                    output = coutput.copy()
                else:
                    output = output+coutput
            output = output/len(TEST_SCALE)
            output = output.transpose(1,2,0)
            amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            all_ssegs.append(amax_output)
            pred_label = amax_output.copy()
            label = label_2_id[np.asarray(label.numpy(), dtype=np.uint8)]
            scorer.update(pred_label.flatten(), label.flatten(), index)
            amax_output_col = colorize_mask(amax_output)
            image_name = name[0].split('/')[-1].replace('_leftImg8bit.png','.png')
            amax_output_col.save(os.path.join(final_output_path, 'results', 'ssegs', image_name))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False,
                                              num_workers=2, drop_last=False, pin_memory=False,
                                              collate_fn=test_dataset.collate)

    i_iter = 0
    idx = 0
    test_iter = test_loader.__iter__()
    all_boxes = [[] for _ in range(test_dataset.num_classes)]
    all_masks = [[] for _ in range(test_dataset.num_classes)]

    data_timer = Timer()
    net_timer = Timer()
    post_timer = Timer()

    while i_iter < len(test_loader):
        data_timer.tic()
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

        data_time = data_timer.toc()
        net_timer.tic()
        with torch.no_grad():
            output = test_maskrcnn_branch(*batch)
            output = im_detect(output, batch, im_infos)

        torch.cuda.synchronize()
        net_time = net_timer.toc()

        post_timer.tic()
        for score, box, mask, cls_idx, im_info in zip(output['scores'], output['boxes'], output['masks'],
                                                      output['cls_inds'], im_infos):
            im_post(all_boxes, all_masks, score, box, mask, cls_idx, test_dataset.num_classes,
                    np.round(im_info[:2] / im_info[2]).astype(np.int32))
            idx += 1

        post_time = post_timer.toc()
        s = 'Batch %d/%d, data_time:%.3f, net_time:%.3f, post_time:%.3f' % (
        idx, len(test_dataset), data_time, net_time, post_time)
        logging.info(s)

    # trim redundant predictions
    for i in range(1, test_dataset.num_classes):
        all_boxes[i] = all_boxes[i][:len(test_loader)]
        if config.network.has_mask_head:
            all_masks[i] = all_masks[i][:len(test_loader)]

    os.makedirs(os.path.join(final_output_path, 'results'), exist_ok=True)

    results = {'all_boxes': all_boxes,
               'all_masks': all_masks if config.network.has_mask_head else None,
               }

    with open(os.path.join(final_output_path, 'results', 'results_list.pkl'), 'wb') as f:
        pickle.dump(results, f, protocol=2)

    if config.test.vis_mask:
        test_dataset.vis_all_mask(all_boxes, all_masks, os.path.join(final_output_path, 'results', 'vis'))
    else:
        test_dataset.evaluate_boxes(all_boxes, os.path.join(final_output_path, 'results'))
        test_dataset.evaluate_masks(all_boxes, all_masks, os.path.join(final_output_path, 'results'))
        test_dataset.evaluate_ssegs(all_ssegs, os.path.join(final_output_path, 'results', 'ssegs'))
        logging.info('combined pano result:')
        all_panos = test_dataset.get_combined_pan_result(all_ssegs, all_boxes, all_masks, stuff_area_limit=config.test.panoptic_stuff_area_limit)
        test_dataset.evaluate_panoptic(all_panos, os.path.join(final_output_path, 'results', 'pans_combined'))


def parse_split_list(list_name):
    image_list = []
    image_name_list = []
    label_list = []
    file_num = 0
    with open(list_name) as f:
        for item in f.readlines():
            fields = item.strip().split('\t')
            image_name = fields[0].split('/')[-1]
            image_list.append(fields[0])
            image_name_list.append(image_name)
            label_list.append(fields[1])
            file_num += 1
    return image_list, image_name_list, label_list, file_num

class ScoreUpdater(object):
    # only IoU are computed. accu, cls_accu, etc are ignored.
    def __init__(self, valid_labels, c_num, x_num, logger=None, label=None, info=None):
        self._valid_labels = valid_labels

        self._confs = np.zeros((c_num, c_num))
        self._per_cls_iou = np.zeros(c_num)
        self._logger = logger
        self._label = label
        self._info = info
        self._num_class = c_num
        self._num_sample = x_num

    @property
    def info(self):
        return self._info

    def reset(self):
        self._start = time.time()
        self._computed = np.zeros(self._num_sample) # one-dimension
        self._confs[:] = 0

    def fast_hist(self,label, pred_label, n):
        k = (label >= 0) & (label < n)
        return np.bincount(n * label[k].astype(int) + pred_label[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(self,hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def do_updates(self, conf, i, computed=True):
        if computed:
            self._computed[i] = 1
        self._per_cls_iou = self.per_class_iu(conf)

    def update(self, pred_label, label, i, computed=True):
        conf = self.fast_hist(label, pred_label, self._num_class)
        self._confs += conf
        self.do_updates(self._confs, i, computed)
        self.scores(i)

    def scores(self, i=None, logger=None):
        x_num = self._num_sample
        ious = np.nan_to_num( self._per_cls_iou )

        logger = self._logger if logger is None else logger
        if logger is not None:
            if i is not None:
                speed = 1. * self._computed.sum() / (time.time() - self._start)
                logger.info('Done {}/{} with speed: {:.2f}/s'.format(i + 1, x_num, speed))
            name = '' if self._label is None else '{}, '.format(self._label)
            logger.info('{}mean iou: {:.2f}%'. \
                        format(name, np.mean(ious) * 100))
            from CRST.util import np_print_options
            with np_print_options(formatter={'float': '{:5.2f}'.format}):
                logger.info('\n{}'.format(ious * 100))

        return ious

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
            pred_ssegs_all.append(output_all[i]['fcn_prob'])
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