
import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from skimage.exposure import match_histograms

from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc

def train_ss_aux_cross_style(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    print('cross-style regularization pretrain ----------------------------------------- CSR')
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        # hist match
        if i_iter > 1:
            images_source_aug = (hist_match(images_source.detach().clone(),
                                       F.interpolate(previous_images.detach().clone(), size=images_source.shape[2:4],
                                                     mode='bilinear', align_corners=True))).contiguous().detach().clone()
        else:
            images_source_aug = images_source.detach().clone()

        # pred_src_aux, pred_src_main = model(images_source.cuda(device))
        pred_src_aux, pred_src = model(images_source_aug.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        # pred_src_main = interp(pred_src_main)
        pred_src_main = interp(pred_src)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # cross style regularization over target
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        if i_iter == 0:
            previous_images = images.detach().clone()

        with torch.no_grad():
            pred_trg_aux_ori, pred_trg = model(images.cuda(device))
            if cfg.TRAIN.MULTI_LEVEL:
                pred_trg_aux = interp_target(pred_trg_aux_ori)
            else:
                loss_adv_trg_aux = 0
            pred_trg_main = interp_target(pred_trg)

        ### cross-style self-supervised
        scale_ratio = np.random.randint(80, 120) / 100.0
        scale_size_target = (
        round(input_size_target[1] * scale_ratio / 8) * 8, round(input_size_target[0] * scale_ratio / 8) * 8)
        interp_target_sc = nn.Upsample(size=scale_size_target, mode='bilinear', align_corners=True)
        # hist match
        images_aug = hist_match(images, previous_images)

        images_sc = interp_target_sc(images_aug).detach().clone()
        pred_trg_aux_sc, pred_trg_sc = model(images_sc.cuda(device))
        interp_target_sc2trg = nn.Upsample(size=(pred_trg.shape[-2], pred_trg.shape[-1]), mode='bilinear',
                                           align_corners=True)
        pred_trg_sc_ss = interp_target_sc2trg(pred_trg_sc)
        out_trg_sc_ss = F.softmax(pred_trg_sc_ss)
        out_trg = F.softmax(pred_trg).detach()
        criterionGc = torch.nn.L1Loss()
        loss_self_supervised = criterionGc(out_trg_sc_ss, out_trg)
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_sc_aux_ss = interp_target_sc2trg(pred_trg_aux_sc)
            out_trg_sc_aux_ss = F.softmax(pred_trg_sc_aux_ss)
            out_trg_aux = F.softmax(pred_trg_aux_ori).detach()
            loss_self_supervised_aux = criterionGc(out_trg_sc_aux_ss, out_trg_aux)
        else:
            loss_self_supervised_aux = 0
        loss = (cfg.TRAIN.LAMBDA_ADV_SCSS * loss_self_supervised +
                cfg.TRAIN.LAMBDA_ADV_SCSS * loss_self_supervised_aux)
        loss.backward()

        optimizer.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': 0,
                          'loss_adv_trg_main': 0,
                          'loss_d_aux': 0,
                          'loss_d_main': 0}
        print_losses(current_losses, i_iter)

        ### self-supervised
        current_losses_sc = {'loss_self_supervised': loss_self_supervised,
                             'loss_self_supervised_aux': loss_self_supervised_aux}
        print_losses(current_losses_sc, i_iter)

        previous_images = images.detach().clone()

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')

            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def hist_match(images, previous_images):
    im_src = np.asarray(images.squeeze(0).transpose(0,1).transpose(1,2), np.float32)
    im_trg = np.asarray(previous_images.squeeze(0).transpose(0,1).transpose(1,2), np.float32)
    images_aug = match_histograms(im_src, im_trg, multichannel=True)
    return torch.from_numpy(images_aug).transpose(1,2).transpose(0,1).unsqueeze(0)



def train_domain_adaptation(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'SSAUX_cross_style':
        train_ss_aux_cross_style(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
