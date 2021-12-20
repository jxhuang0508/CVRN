

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
import tensorboardX
import cv2
import torch.utils.data.distributed as distributed
from skimage.exposure import match_histograms
import torch.nn.functional as F

# torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join (os.path.dirname(__file__), '..'))

from collections import deque
from upsnet.config.config import config
from upsnet.config.parse_args import parse_args
from lib.utils.logging import create_logger
from upsnet.operators.functions.entropy_loss import entropy_loss, sigmoid_loss

args = parse_args()

if config.train.use_horovod:
    import horovod.torch as hvd
    from horovod.torch.mpi_ops import allreduce_async

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

is_master = (not config.train.use_horovod) or hvd.rank() == 0

if is_master:
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.source_dataset.image_set)
    writer = tensorboardX.SummaryWriter(log_dir=os.path.join(config.output_path, 'tensorboard',
                                                             os.path.basename(args.cfg).split('.')[0],
                                                             '_'.join(config.source_dataset.image_set.split('+')),
                                                             time.strftime('%Y-%m-%d-%H-%M')))
else:
    final_output_path = os.path.join(config.output_path, os.path.basename(args.cfg).split('.')[0], '{}'.format(
        '_'.join([iset for iset in config.source_dataset.image_set.split('+')])))

from upsnet.dataset import *
from upsnet.models import *
from lib.utils.callback import Speedometer
from lib.utils.data_parallel import DataParallel
from lib.utils.metric import AvgMetric
from lib.nn.optimizer import SGD, Adam, clip_grad


config.train.RANDOM_SEED = 1234
# INIT
import random
_init_fn = None
if True:
    torch.manual_seed(config.train.RANDOM_SEED)
    torch.cuda.manual_seed(config.train.RANDOM_SEED)
    np.random.seed(config.train.RANDOM_SEED)
    random.seed(config.train.RANDOM_SEED)

    def _init_fn(worker_id):
        np.random.seed(config.train.RANDOM_SEED + worker_id)

cv2.ocl.setUseOpenCL(False)
cudnn.enabled = True
cudnn.benchmark = False


def lr_poly(base_lr, iter, max_iter, warmup_iter=0):
    power = 0.9
    if iter < warmup_iter:
        alpha = iter / warmup_iter
        return min(base_lr * (1 / 10.0 * (1 - alpha) + alpha), base_lr * ((1 - float(iter) / max_iter) ** (power)))
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def get_step_index(iter, decay_iters):
    for idx, decay_iter in enumerate(decay_iters):
        if iter < decay_iter:
            return idx
    return len(decay_iters)


def lr_factor(base_lr, iter, decay_iter, warmup_iter=0):
    if iter < warmup_iter:
        alpha = iter / warmup_iter
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)
    return base_lr * (0.1 ** get_step_index(iter, decay_iter))


def adjust_learning_rate(optimizer, iter, config):
    assert config.train.lr_schedule in ['step', 'poly']
    if config.train.lr_schedule == 'step':
        return lr_factor(config.train.lr, iter, config.train.decay_iteration, config.train.warmup_iteration)
    if config.train.lr_schedule == 'poly':
        return lr_poly(config.train.lr, iter, config.train.max_iteration, config.train.warmup_iteration)

def hist_match(images, previous_images):
    im_src = np.asarray(images.squeeze(0).transpose(0,1).transpose(1,2), np.float32)
    im_trg = np.asarray(previous_images.squeeze(0).transpose(0,1).transpose(1,2), np.float32)
    images_aug = match_histograms(im_src, im_trg, multichannel=True)
    return torch.from_numpy(images_aug).transpose(1,2).transpose(0,1).unsqueeze(0)

def upsnet_train():
    # entropy minimizing
    if is_master:
        logger.info('training config:{}\n'.format(pprint.pformat(config)))
    gpus = [torch.device('cuda', int(_)) for _ in config.gpus.split(',')]
    num_replica = hvd.size() if config.train.use_horovod else len(gpus)
    num_gpus = 1 if config.train.use_horovod else len(gpus)

    # create models
    train_model = eval(config.symbol)().cuda()

    # create optimizer
    params_lr = train_model.get_params_lr()
    # we use custom optimizer and pass lr=1 to support different lr for different weights
    optimizer = SGD(params_lr, lr=1, momentum=config.train.momentum, weight_decay=config.train.wd)
    if config.train.use_horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=train_model.named_parameters())
    optimizer.zero_grad()

    # create data loader
    source_dataset = eval(config.source_dataset.dataset)(image_sets=config.source_dataset.image_set.split('+'), flip=config.train.flip, result_path=final_output_path)
    target_dataset = eval(config.dataset.dataset)(image_sets=config.dataset.image_set.split('+'),
                                                  flip=config.train.flip, result_path=final_output_path, phase='test')
    if config.train.use_horovod:
        train_sampler = distributed.DistributedSampler(source_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        target_sampler = distributed.DistributedSampler(target_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = torch.utils.data.DataLoader(source_dataset, batch_size=config.train.batch_size,
                                                   sampler=train_sampler, num_workers=num_gpus * 4, drop_last=False,
                                                   collate_fn=source_dataset.collate)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=config.train.batch_size,
                                                    sampler=target_sampler, num_workers=num_gpus * 4, drop_last=False,
                                                    collate_fn=target_dataset.collate)
    else:
        train_loader = torch.utils.data.DataLoader(source_dataset, batch_size=config.train.batch_size,
                                                   shuffle=config.train.shuffle,
                                                   num_workers=num_gpus * 4 if not config.debug_mode else num_gpus * 4,
                                                   drop_last=False, collate_fn=source_dataset.collate, worker_init_fn=_init_fn)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=config.train.batch_size,
                                                    shuffle=config.train.shuffle,
                                                    num_workers=num_gpus * 4 if not config.debug_mode else num_gpus * 4,
                                                    drop_last=False, collate_fn=target_dataset.collate, worker_init_fn=_init_fn)

    # preparing
    curr_iter = config.train.begin_iteration
    batch_end_callback = [Speedometer(num_replica * config.train.batch_size, config.train.display_iter)]
    metrics = []
    metrics_name = []

    if config.network.has_rpn:
        metrics.extend([AvgMetric(name='rpn_cls_loss'), AvgMetric(name='rpn_bbox_loss'), ])
        metrics_name.extend(['rpn_cls_loss', 'rpn_bbox_loss'])
    if config.network.has_rcnn:
        metrics.extend([AvgMetric(name='rcnn_accuracy'), AvgMetric(name='cls_loss'), AvgMetric(name='bbox_loss'), ])
        metrics_name.extend(['rcnn_accuracy', 'cls_loss', 'bbox_loss'])
    if config.network.has_mask_head:
        metrics.extend([AvgMetric(name='mask_loss'), ])
        metrics_name.extend(['mask_loss'])
    if config.network.has_fcn_head:
        metrics.extend([AvgMetric(name='fcn_loss'), ])
        metrics_name.extend(['fcn_loss'])
        if config.train.fcn_with_roi_loss:
            metrics.extend([AvgMetric(name='fcn_roi_loss'), ])
            metrics_name.extend(['fcn_roi_loss'])
    if config.network.has_panoptic_head:
        metrics.extend([AvgMetric(name='panoptic_accuracy'), AvgMetric(name='panoptic_loss'), ])
        metrics_name.extend(['panoptic_accuracy', 'panoptic_loss'])

    # preparing metrics for target (UDA)
    metrics_target = []
    metrics_name_target = []
    if config.train.fcn_roi_consistency_weight > 0:
        metrics_target.extend([AvgMetric(name='frc_loss'), ])
        metrics_name_target.extend(['frc_loss'])
    if config.train.rcnn_sida_weight > 0:
        metrics_target.extend([AvgMetric(name='rcnn_sida_loss'),])
        metrics_name_target.extend(['rcnn_sida_loss'])
    if config.train.mask_sida_weight > 0:
        metrics_target.extend([AvgMetric(name='mask_sida_loss'),])
        metrics_name_target.extend(['mask_sida_loss'])
    if config.train.fcn_sida_weight > 0:
        metrics_target.extend([AvgMetric(name='fcn_sida_loss'),])
        metrics_name_target.extend(['fcn_sida_loss'])

    if config.train.resume:
        train_model.load_state_dict(
            torch.load(os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.pth')), resume=True)
        optimizer.load_state_dict(
            torch.load(os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.state.pth')))
        if config.train.use_horovod:
            hvd.broadcast_parameters(train_model.state_dict(), root_rank=0)
    else:
        if is_master:
            train_model.load_state_dict(torch.load(config.network.pretrained))

        if config.train.use_horovod:
            hvd.broadcast_parameters(train_model.state_dict(), root_rank=0)

    if not config.train.use_horovod:
        train_model = DataParallel(train_model, device_ids=[int(_) for _ in config.gpus.split(',')]).to(gpus[0])

    if is_master:
        batch_end_callback[0](0, 0)

    train_model.eval()

    # start training ---------------------------------------------------------------------------------------------------
    target_iterator = target_loader.__iter__()
    inner_iter_target = 0

    print('len_source_loader:', len(train_loader))
    print('len_target_loader:', len(target_loader))

    while curr_iter < config.train.max_iteration:
        if True:
            inner_iter = 0
            train_iterator = train_loader.__iter__()

            while inner_iter + num_gpus <= len(train_loader):
                batch = []
                for gpu_id in gpus:
                    data, label, _ = train_iterator.next()
                    # source data F.interpolate(previous_images.clone(), size=feature.shape[2:4], mode='nearest')

                    if curr_iter > (1 + config.train.begin_iteration):
                        data['data'] = (hist_match(data['data'].detach().clone(), F.interpolate(previous_images.clone(), size=data['data'].shape[2:4], mode='bilinear', align_corners=True))).contiguous()

                    for k, v in data.items():
                        data[k] = v if not torch.is_tensor(v) else v.pin_memory().to(gpu_id, non_blocking=True)
                    for k, v in label.items():
                        label[k] = v if not torch.is_tensor(v) else v.pin_memory().to(gpu_id, non_blocking=True)

                    batch.append((data, label))
                    inner_iter += 1
                lr = adjust_learning_rate(optimizer, curr_iter, config)

                optimizer.zero_grad()
                if config.train.use_horovod:
                    output = train_model(data, label)
                else:
                    output = train_model(*batch)

                loss = 0
                if config.network.has_rpn:
                    loss = loss + output['rpn_cls_loss'] + output['rpn_bbox_loss']
                if config.network.has_rcnn:
                    loss = loss + output['cls_loss'] + output['bbox_loss']
                if config.network.has_mask_head:
                    loss = loss + output['mask_loss']
                if config.network.has_fcn_head:
                    loss = loss + output['fcn_loss'] * config.train.fcn_loss_weight
                    if config.train.fcn_with_roi_loss:
                        loss = loss + output['fcn_roi_loss'] * config.train.fcn_loss_weight * 0.2
                if config.network.has_panoptic_head:
                    loss = loss + output['panoptic_loss'] * config.train.panoptic_loss_weight
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 10)
                optimizer.step(lr)

                losses = []
                losses.append(loss.item())
                for l in metrics_name:
                    losses.append(output[l])

                loss = losses[0]
                if is_master:
                    writer.add_scalar('train_total_loss', loss, curr_iter)
                for i, (metric, l) in enumerate(zip(metrics, metrics_name)):
                    loss = losses[i + 1]
                    if is_master:
                        writer.add_scalar('train_' + l, loss, curr_iter)
                        metric.update(_, _, loss)
                curr_iter += 1

                if is_master:
                    if curr_iter % config.train.display_iter == 0:
                        for callback in batch_end_callback:
                            callback(curr_iter, metrics)

                if inner_iter_target + num_gpus > len(target_loader):
                    print(inner_iter_target)
                    print(len(target_loader))

                    while True:
                        try:
                            target_iterator.next()
                        except:
                            break
                    inner_iter_target = 0
                    target_iterator = target_loader.__iter__()

                batch_target = []
                for gpu_id in gpus:
                    data_target, _, _ = target_iterator.next()

                    current_images_target = data_target['data'].detach().clone()
                    if curr_iter > (1 + config.train.begin_iteration):
                        data_target_aug = (hist_match(data_target['data'].detach().clone(), previous_images)).contiguous()
                        data_target['data_cs_aug'] = data_target_aug.detach().clone()
                    else:
                        data_target['data_cs_aug'] = data_target['data'].detach().clone()
                    previous_images = current_images_target

                    for k, v in data_target.items():
                        data_target[k] = v if not torch.is_tensor(v) else v.pin_memory().to(gpu_id, non_blocking=True)
                    batch_target.append((data_target,))
                    inner_iter_target += 1

                optimizer.zero_grad()
                if config.train.use_horovod:
                    output = train_model(data_target)
                else:
                    output = train_model(*batch_target)

                # Consistency loss
                loss = 0
                output_loss = dict()
                if config.train.fcn_roi_consistency_weight > 0:
                    frc_loss = output['frc_loss']
                    loss = loss + frc_loss * config.train.fcn_roi_consistency_weight
                    output_loss['frc_loss'] = frc_loss
                if config.train.rcnn_sida_weight > 0:
                    rcnn_sida_loss = output['rcnn_sida_loss']
                    loss = loss + rcnn_sida_loss * config.train.rcnn_sida_weight
                    output_loss['rcnn_sida_loss'] = rcnn_sida_loss
                if config.train.mask_sida_weight > 0:
                    mask_sida_loss = output['mask_sida_loss']
                    loss = loss + mask_sida_loss * config.train.mask_sida_weight
                    # if mask_sida_loss > 0:
                    #     loss = loss + mask_sida_loss * config.train.mask_sida_weight
                    output_loss['mask_sida_loss'] = mask_sida_loss
                if config.train.fcn_sida_weight > 0:
                    fcn_sida_loss = output['fcn_sida_loss']
                    loss = loss + fcn_sida_loss * config.train.fcn_sida_weight
                    output_loss['fcn_sida_loss'] = fcn_sida_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 10)
                optimizer.step(lr)

                losses = []
                # losses.append(loss.item())
                for l in metrics_name_target: losses.append(output_loss[l])

                # loss = losses[0]
                if is_master:
                    writer.add_scalar('train_total_loss', loss, curr_iter)
                for i, (metric_target, l) in enumerate(zip(metrics_target, metrics_name_target)):
                    # loss = losses[i + 1]
                    loss = losses[i]
                    if is_master:
                        writer.add_scalar('train_' + l, loss, curr_iter)
                        metric_target.update(_, _, loss)

                if curr_iter in config.train.decay_iteration:
                    if is_master:
                        logger.info('decay momentum buffer')
                    for k in optimizer.state_dict()['state'].keys():
                        optimizer.state_dict()['state'][k]['momentum_buffer'].div_(10)

                if is_master:
                    if curr_iter % config.train.display_iter == 0:
                        for callback in batch_end_callback:
                            callback(curr_iter, metrics_target)

                if is_master:
                    if curr_iter % config.train.snapshot_step == 0:
                        logger.info('taking snapshot ...')
                        torch.save(train_model.module.state_dict(),
                                   os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.pth'))
                        torch.save(optimizer.state_dict(),
                                   os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.state.pth'))

            while True:
                try:
                    train_iterator.next()
                except:
                    break

        for metric in metrics:
            metric.reset()
        for metric in metrics_target:
            metric.reset()

    if is_master and config.train.use_horovod:
        logger.info('taking snapshot ...')
        torch.save(train_model.state_dict(),
                   os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.pth'))
        torch.save(optimizer.state_dict(),
                   os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.state.pth'))
    elif not config.train.use_horovod:
        logger.info('taking snapshot ...')
        torch.save(train_model.module.state_dict(),
                   os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.pth'))
        torch.save(optimizer.state_dict(),
                   os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.state.pth'))


if __name__ == '__main__':
    upsnet_train()
