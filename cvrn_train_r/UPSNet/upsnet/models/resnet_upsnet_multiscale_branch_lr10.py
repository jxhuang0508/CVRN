

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from upsnet.config.config import config
from upsnet.models.resnet import get_params, resnet_rcnn, ResNetBackbone
from upsnet.models.fpn import FPN
from upsnet.models.rpn import RPN, RPNLoss
from upsnet.models.rcnn import RCNN, MaskBranch, RCNNLoss, MaskRCNNLoss, RCNNLoss_ups
from upsnet.models.fcn import FCNHead
from upsnet.operators.modules.pyramid_proposal import PyramidProposal
from upsnet.operators.modules.proposal_mask_target import ProposalMaskTarget
from upsnet.operators.modules.mask_roi import MaskROI
from upsnet.operators.modules.unary_logits import MaskTerm, SegTerm
from upsnet.operators.modules.mask_removal import MaskRemoval
from upsnet.operators.modules.mask_matching import MaskMatching
from upsnet.operators.functions.entropy_loss import prob_2_entropy, sigmoid_2_entropy, entropy_loss
from skimage.exposure import match_histograms

if config.train.use_horovod and config.network.use_syncbn:
    from upsnet.operators.modules.distbatchnorm import BatchNorm2d


class resnet_upsnet_multiscale_branch_lr10(resnet_rcnn):

    def __init__(self, backbone_depth):
        super(resnet_upsnet_multiscale_branch_lr10, self).__init__()

        self.num_classes = config.dataset.num_classes
        self.num_seg_classes = config.dataset.num_seg_classes
        self.num_reg_classes = (2 if config.network.cls_agnostic_bbox_reg else config.dataset.num_classes)
        self.num_anchors = config.network.num_anchors

        # backbone net
        self.resnet_backbone = ResNetBackbone(backbone_depth)
        # FPN, RPN, Instance Head and Semantic Head
        self.fpn = FPN(feature_dim=config.network.fpn_feature_dim, with_norm=config.network.fpn_with_norm,
                       upsample_method=config.network.fpn_upsample_method)
        self.rpn = RPN(num_anchors=config.network.num_anchors, input_dim=config.network.fpn_feature_dim)
        self.rcnn = RCNN(self.num_classes, self.num_reg_classes, dim_in=config.network.fpn_feature_dim,
                         with_norm=config.network.rcnn_with_norm)
        self.mask_branch = MaskBranch(self.num_classes, dim_in=config.network.fpn_feature_dim,
                                      with_norm=config.network.rcnn_with_norm)
        self.fcn_head = eval(config.network.fcn_head)(config.network.fpn_feature_dim, self.num_seg_classes,
                                                      num_layers=config.network.fcn_num_layers,
                                                      with_norm=config.network.fcn_with_norm, upsample_rate=4,
                                                      with_roi_loss=config.train.fcn_with_roi_loss)
        self.mask_roi = MaskROI(clip_boxes=True, bbox_class_agnostic=False, top_n=config.test.max_det,
                                num_classes=self.num_classes, score_thresh=config.test.score_thresh)

        # Panoptic Head
        # param for training
        self.box_keep_fraction = config.train.panoptic_box_keep_fraction
        self.enable_void = config.train.panoptic_box_keep_fraction < 1

        self.mask_roi_panoptic = MaskROI(clip_boxes=True, bbox_class_agnostic=False, top_n=config.test.max_det,
                                         num_classes=self.num_classes, nms_thresh=0.5, class_agnostic=True,
                                         score_thresh=config.test.panoptic_score_thresh)
        self.mask_removal = MaskRemoval(fraction_threshold=0.3)
        self.seg_term = SegTerm(config.dataset.num_seg_classes)
        self.mask_term = MaskTerm(config.dataset.num_seg_classes, box_scale=1 / 4.0)
        self.mask_matching = MaskMatching(config.dataset.num_seg_classes, enable_void=self.enable_void)

        # # Loss layer
        self.rpn_loss = RPNLoss(config.train.rpn_batch_size * config.train.batch_size)
        self.mask_rcnn_loss = MaskRCNNLoss(config.train.batch_rois * config.train.batch_size)
        self.fcn_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.panoptic_loss = nn.CrossEntropyLoss(ignore_index=255, reduce=False)
        if config.train.fcn_with_roi_loss:
            self.fcn_roi_loss = nn.CrossEntropyLoss(ignore_index=255, reduce=False)
        self.initialize()

    def initialize(self):
        pass

    def forward(self, data, label=None):

        self.pyramid_proposal = PyramidProposal(feat_stride=config.network.rpn_feat_stride,
                                                scales=config.network.anchor_scales,
                                                ratios=config.network.anchor_ratios,
                                                rpn_pre_nms_top_n=config.train.rpn_pre_nms_top_n,
                                                rpn_post_nms_top_n=config.train.rpn_post_nms_top_n,
                                                threshold=config.train.rpn_nms_thresh,
                                                rpn_min_size=config.train.rpn_min_size,
                                                individual_proposals=config.train.rpn_individual_proposals)
        self.proposal_target = ProposalMaskTarget(num_classes=self.num_reg_classes,
                                                  batch_images=config.train.batch_size,
                                                  batch_rois=config.train.batch_rois,
                                                  fg_fraction=config.train.fg_fraction,
                                                  mask_size=config.network.mask_size,
                                                  binary_thresh=config.network.binary_thresh)
        self.pyramid_proposal_test = PyramidProposal(feat_stride=config.network.rpn_feat_stride,
                                                scales=config.network.anchor_scales,
                                                ratios=config.network.anchor_ratios,
                                                rpn_pre_nms_top_n=config.test.rpn_pre_nms_top_n,
                                                rpn_post_nms_top_n=config.test.rpn_post_nms_top_n,
                                                threshold=config.test.rpn_nms_thresh,
                                                rpn_min_size=config.test.rpn_min_size,
                                                individual_proposals=config.train.rpn_individual_proposals)

        if label is not None:
            res2, res3, res4, res5 = self.resnet_backbone(data['data'])
            fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.fpn(res2, res3, res4, res5)

            # RPN loss
            if config.network.has_rpn:
                rpn_cls_score, rpn_cls_prob, rpn_bbox_pred = [], [], []
                for feat in [fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6]:
                    rpn_cls_score_p, rpn_bbox_pred_p, rpn_cls_prob_p = self.rpn(feat)
                    rpn_cls_score.append(rpn_cls_score_p)
                    rpn_cls_prob.append(rpn_cls_prob_p)
                    rpn_bbox_pred.append(rpn_bbox_pred_p)
                rois, _ = self.pyramid_proposal(rpn_cls_prob, rpn_bbox_pred, data['im_info'])
                if config.network.has_mask_head:
                    rois, cls_label, bbox_target, bbox_inside_weight, bbox_outside_weight, mask_rois, mask_target, roi_has_mask, nongt_inds = self.proposal_target(
                            rois, label['roidb'], data['im_info'])
                else:
                    rois, cls_label, bbox_target, bbox_inside_weight, bbox_outside_weight = self.proposal_rcnn_target(
                            rois, label['roidb'], data['im_info'])
                rpn_cls_loss, rpn_bbox_loss = self.rpn_loss(rpn_cls_score, rpn_bbox_pred, label)

            if config.network.has_rcnn:
                rcnn_output = self.rcnn([fpn_p2, fpn_p3, fpn_p4, fpn_p5], rois)
                cls_score, bbox_pred = rcnn_output['cls_score'], rcnn_output['bbox_pred']
                if not config.network.has_mask_head:
                    # RCNN loss
                    cls_loss, bbox_loss, rcnn_acc = \
                        self.rcnn_loss_ups(cls_score, bbox_pred, cls_label, bbox_target, bbox_inside_weight, bbox_outside_weight)

            # Semantic head loss
            if config.network.has_fcn_head:
                if config.train.fcn_with_roi_loss:
                    fcn_rois, _ = self.get_gt_rois(label['roidb'], data['im_info'])
                    fcn_rois = fcn_rois.to(rois.device)
                    fcn_output = self.fcn_head(*[fpn_p2, fpn_p3, fpn_p4, fpn_p5, fcn_rois])
                else:
                    fcn_output = self.fcn_head(*[fpn_p2, fpn_p3, fpn_p4, fpn_p5])
                fcn_loss = self.fcn_loss(fcn_output['fcn_output'], label['seg_gt'])
                if config.train.fcn_with_roi_loss:
                    fcn_roi_loss = self.fcn_roi_loss(fcn_output['fcn_roi_score'], label['seg_roi_gt'])
                    fcn_roi_loss = fcn_roi_loss.mean()

            if config.network.has_mask_head:
                # Instance head loss
                mask_score = self.mask_branch([fpn_p2, fpn_p3, fpn_p4, fpn_p5], mask_rois)
                cls_loss, bbox_loss, mask_loss, rcnn_acc = \
                    self.mask_rcnn_loss(cls_score, bbox_pred, mask_score,
                                        cls_label, bbox_target, bbox_inside_weight, bbox_outside_weight, mask_target)

            # Panoptic head
            if config.network.has_panoptic_head:
                # extract gt rois for panoptic head
                gt_rois, cls_idx = self.get_gt_rois(label['roidb'], data['im_info'])
                if self.enable_void:
                    keep_inds = np.random.choice(gt_rois.shape[0], max(int(gt_rois.shape[0] * self.box_keep_fraction), 1),
                                                 replace=False)
                    gt_rois = gt_rois[keep_inds]
                    cls_idx = cls_idx[keep_inds]
                gt_rois, cls_idx = gt_rois.to(rois.device), cls_idx.to(rois.device)

                # Calc mask logits with gt rois
                mask_score = self.mask_branch([fpn_p2, fpn_p3, fpn_p4, fpn_p5], gt_rois)
                mask_score = mask_score.gather(1, cls_idx.view(-1, 1, 1, 1).expand(-1, -1, config.network.mask_size,
                                                                                   config.network.mask_size))

                # Calc panoptic logits
                seg_logits, seg_inst_logits = self.seg_term(cls_idx, fcn_output['fcn_score'], gt_rois)
                mask_logits = self.mask_term(mask_score, gt_rois, cls_idx, fcn_output['fcn_score'])

                if self.enable_void:
                    void_logits = torch.max(
                        fcn_output['fcn_score'][:, (config.dataset.num_seg_classes - config.dataset.num_classes + 1):, ...],
                        dim=1, keepdim=True)[0] - torch.max(seg_inst_logits, dim=1, keepdim=True)[0]
                    inst_logits = seg_inst_logits + mask_logits
                    panoptic_logits = torch.cat([seg_logits, inst_logits, void_logits], dim=1)
                else:
                    panoptic_logits = torch.cat([seg_logits, (seg_inst_logits + mask_logits)], dim=1)

                # generate gt for panoptic head
                with torch.no_grad():
                    if self.enable_void:
                        panoptic_gt = self.mask_matching(label['seg_gt_4x'], label['mask_gt'], keep_inds=keep_inds)
                    else:
                        panoptic_gt = self.mask_matching(label['seg_gt_4x'], label['mask_gt'])

                # Panoptic head loss
                panoptic_acc = self.calc_panoptic_acc(panoptic_logits, panoptic_gt)
                panoptic_loss = self.panoptic_loss(panoptic_logits, panoptic_gt)
                panoptic_loss = panoptic_loss.mean()

            output = dict()
            if config.network.has_rpn:
                output['rpn_cls_loss'] = rpn_cls_loss.unsqueeze(0).mean()
                output['rpn_bbox_loss'] = rpn_bbox_loss.unsqueeze(0).mean()
            if config.network.has_rcnn:
                output['cls_loss'] = cls_loss.unsqueeze(0).mean()
                output['bbox_loss'] = bbox_loss.unsqueeze(0).mean()
                output['rcnn_accuracy'] = rcnn_acc.unsqueeze(0).mean()
            if config.network.has_mask_head:
                output['mask_loss'] = mask_loss.unsqueeze(0).mean()
            if config.network.has_fcn_head:
                output['fcn_loss'] = fcn_loss.unsqueeze(0).mean()
            if config.network.has_panoptic_head:
                output['panoptic_loss'] = panoptic_loss.unsqueeze(0).mean()
                output['panoptic_accuracy'] = panoptic_acc.unsqueeze(0).mean()
            if config.train.fcn_with_roi_loss:
                output.update({'fcn_roi_loss': fcn_roi_loss})

            return output

        else:
            # Train on target, no label
            res2, res3, res4, res5 = self.resnet_backbone(data['data'])
            fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.fpn(res2, res3, res4, res5)

            if config.network.has_rpn:
                rpn_cls_score, rpn_cls_prob, rpn_bbox_pred = [], [], []
                for feat in [fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6]:
                    rpn_cls_score_p, rpn_bbox_pred_p, rpn_cls_prob_p = self.rpn(feat)
                    rpn_cls_score.append(rpn_cls_score_p)
                    rpn_cls_prob.append(rpn_cls_prob_p)
                    rpn_bbox_pred.append(rpn_bbox_pred_p)

            if config.network.has_rcnn:
                rois, _ = self.pyramid_proposal_test(rpn_cls_prob, rpn_bbox_pred, data['im_info'])
                rcnn_output = self.rcnn([fpn_p2, fpn_p3, fpn_p4, fpn_p5], rois)
                cls_score, bbox_pred = rcnn_output['cls_score'], rcnn_output['bbox_pred']
                cls_prob = F.softmax(cls_score, dim=1)
                cls_prob_all, mask_rois, cls_idx = self.mask_roi(rois, bbox_pred, cls_prob, data['im_info'])

            frc_loss = 0
            if config.train.fcn_roi_consistency_weight > 0:
                if config.network.has_mask_head:
                    mask_score = self.mask_branch([fpn_p2, fpn_p3, fpn_p4, fpn_p5], mask_rois)
                    # cls_prob_max, cls_idx = torch.max(cls_prob, )
                    mask_score = mask_score.gather(1, cls_idx.view(-1, 1, 1, 1).expand(-1, -1, config.network.mask_size, config.network.mask_size))
                    mask_prob = torch.sigmoid(mask_score)

                fcn_output = self.fcn_head(*[fpn_p2, fpn_p3, fpn_p4, fpn_p5, mask_rois])
                # fcn_prob = F.softmax(fcn_output['fcn_output'], dim=1)
                fcn_roi_score = fcn_output['fcn_roi_score']
                cls_idx = cls_idx.clone().detach().cpu().numpy()

                frc_loss_num = 0
                for ind in range(len(cls_idx)):
                    cls = cls_idx[ind]
                    if cls == 0:
                        continue
                    else:
                        fcn_roi_score_fg = fcn_roi_score[ind,][-self.num_classes+cls,:,:]
                        fcn_roi_score_bg_all = torch.cat((fcn_roi_score[ind,][:-self.num_classes+cls,:,:],fcn_roi_score[ind,][-self.num_classes+cls+1:,:,:]),dim=0)
                        fcn_roi_score_bg, _ = torch.max(fcn_roi_score_bg_all,dim=0)
                        fcn_mask_score = torch.cat((fcn_roi_score_bg.unsqueeze(0),fcn_roi_score_fg.unsqueeze(0)),dim=0)
                        fcn_mask_prob = torch.softmax(fcn_mask_score,dim=0)[1].unsqueeze(0)
                        fcn_mask_ent = (sigmoid_2_entropy(fcn_mask_prob.unsqueeze(0))).mean()
                        mask_prob_p = mask_prob[ind]
                        mask_prob_p_ent = (sigmoid_2_entropy(mask_prob_p.unsqueeze(0))).mean()
                        if fcn_mask_ent < mask_prob_p_ent:
                            frc_loss_p = torch.nn.MSELoss()(mask_prob_p, fcn_mask_prob.detach())
                            frc_loss += frc_loss_p
                        else:
                            frc_loss_p = torch.nn.MSELoss()(fcn_mask_prob, mask_prob_p.detach())
                            frc_loss += frc_loss_p
                        frc_loss_num += 1

                if frc_loss_num > 0:
                    frc_loss = frc_loss / frc_loss_num

            if config.train.rcnn_sida_weight > 0 or config.train.fcn_sida_weight > 0:
                image = data['data_cs_aug'].detach().clone()
                scale_ori = image.shape[-2]
                max_size_ori = image.shape[-1]


                scale_ratio = np.random.randint(80, 120) / 100.0
                scale_size = round(scale_ori * scale_ratio / 32) * 32
                interp_sc = nn.Upsample(size=(scale_size,int(scale_size*max_size_ori/scale_ori/32)*32), mode='bilinear', align_corners=True)


                image_sc = interp_sc(image)

                res2_sc, res3_sc, res4_sc, res5_sc = self.resnet_backbone(image_sc)
                fpn_p2_sc, fpn_p3_sc, fpn_p4_sc, fpn_p5_sc, fpn_p6_sc = self.fpn(res2_sc, res3_sc, res4_sc, res5_sc)
                if config.network.has_rcnn:
                    rois_sc = rois * scale_size / scale_ori

                if config.train.rcnn_sida_weight:
                    rcnn_output_sc = self.rcnn([fpn_p2_sc, fpn_p3_sc, fpn_p4_sc, fpn_p5_sc, fpn_p6_sc], rois_sc)
                    cls_score_sc, bbox_pred_sc = rcnn_output_sc['cls_score'], rcnn_output_sc['bbox_pred']
                    cls_prob_sc = F.softmax(cls_score_sc, dim=1)

                    if config.train.mask_sida_weight:
                        mask_sida_loss = 0
                        mask_score = self.mask_branch([fpn_p2, fpn_p3, fpn_p4, fpn_p5], rois)
                        mask_prob = torch.sigmoid(mask_score)
                        mask_score_sc = self.mask_branch([fpn_p2_sc, fpn_p3_sc, fpn_p4_sc, fpn_p5_sc], rois_sc)
                        mask_prob_sc = torch.sigmoid(mask_score_sc)

                    # ----------------------------------new ------------------------------------------------------------------------------------
                    rcnn_sida_loss = 0
                    rcnn_sida_loss_num = 0
                    # rcnn_sida_num = 0
                    # rcnn_sida_sc_num = 0
                    for ind in range(cls_prob_sc.shape[0]):
                        # print('revised dida ++++++')
                        cls_prob_p = cls_prob[ind,]
                        cls_prob_sc_p = cls_prob_sc[ind,]
                        rcnn_sida_loss_p = torch.nn.L1Loss()(cls_prob_sc_p, cls_prob_p.detach())

                        if cls_prob_p.max() < config.test.score_thresh:
                            rcnn_sida_loss += rcnn_sida_loss_p * 0
                        else:
                            rcnn_sida_loss += rcnn_sida_loss_p
                            rcnn_sida_loss_num += 1

                        if config.train.mask_sida_weight:
                            mask_prob_p = mask_prob[ind,]
                            mask_prob_sc_p = mask_prob_sc[ind,]
                            mask_sida_loss_p = torch.nn.L1Loss()(mask_prob_sc_p, mask_prob_p.detach())

                            if cls_prob_p.max() < config.test.score_thresh:
                                mask_sida_loss += mask_sida_loss_p * 0
                            else:
                                mask_sida_loss += mask_sida_loss_p
                    if rcnn_sida_loss_num > 0:
                        rcnn_sida_loss = rcnn_sida_loss / rcnn_sida_loss_num
                        if config.train.mask_sida_weight:
                            mask_sida_loss = mask_sida_loss / rcnn_sida_loss_num



                if config.train.fcn_sida_weight:
                    if config.train.fcn_with_roi_loss:
                        fcn_output = self.fcn_head(*[fpn_p2, fpn_p3, fpn_p4, fpn_p5, rois_sc])
                        fcn_output_sc = self.fcn_head(*[fpn_p2_sc, fpn_p3_sc, fpn_p4_sc, fpn_p5_sc, rois_sc])
                    else:
                        fcn_output = self.fcn_head(*[fpn_p2, fpn_p3, fpn_p4, fpn_p5])
                        fcn_output_sc = self.fcn_head(*[fpn_p2_sc, fpn_p3_sc, fpn_p4_sc, fpn_p5_sc])
                    fcn_prob = F.softmax(fcn_output['fcn_score'], dim=1)
                    fcn_prob_sc = F.softmax(fcn_output_sc['fcn_score'], dim=1)

                    interp_ori = nn.Upsample(size=(fcn_prob.shape[-2],fcn_prob.shape[-1]), mode='bilinear', align_corners=True)
                    fcn_prob_sc_ori = interp_ori(fcn_prob_sc)


                    fcn_prob_ent_pixel = torch.mean(prob_2_entropy(fcn_prob), dim=1).detach()
                    fcn_prob_sc_ori_ent_pixel = torch.mean(prob_2_entropy(fcn_prob_sc_ori), dim=1).detach()
                    fcn_prob_sc_ori_max_pixel, _ = torch.max(fcn_prob_sc_ori, dim=1)
                    loss_weight_ori = (fcn_prob_sc_ori_ent_pixel < fcn_prob_ent_pixel).float() * (fcn_prob_sc_ori_max_pixel > 0.5).float()
                    fcn_sida_loss = self.weighted_l1_loss(fcn_prob, fcn_prob_sc_ori.detach(), loss_weight_ori.float())
                    fcn_prob_sc_max_pixel, _ = torch.max(fcn_prob, dim=1)
                    loss_weight = (fcn_prob_ent_pixel < fcn_prob_sc_ori_ent_pixel).float() * (fcn_prob_sc_max_pixel > 0.5).float()
                    fcn_sida_loss = fcn_sida_loss + self.weighted_l1_loss(fcn_prob_sc_ori, fcn_prob.detach(), loss_weight.float())

                # import pdb
                # pdb.set_trace()

            results = dict()
            if config.train.fcn_roi_consistency_weight > 0:
                if frc_loss > 0:
                    results['frc_loss'] = frc_loss.unsqueeze(0).mean()
                else:
                    results['frc_loss'] = 0
            if config.train.rcnn_sida_weight > 0:
                results['rcnn_sida_loss'] = rcnn_sida_loss.unsqueeze(0).mean()
            if config.train.mask_sida_weight > 0:
                results['mask_sida_loss'] = mask_sida_loss.unsqueeze(0).mean()
                # if mask_sida_loss > 0:
                #     results['mask_sida_loss'] = mask_sida_loss.unsqueeze(0).mean()
                # else:
                #     results['mask_sida_loss'] = 0
            if config.train.fcn_sida_weight > 0:
                results['fcn_sida_loss'] = fcn_sida_loss.unsqueeze(0).mean()

            return results

    def weighted_l1_loss(self, input, target, weights):
        loss = weights * torch.abs(input - target)
        loss = torch.mean(loss)
        return loss

    def calc_panoptic_acc(self, panoptic_logits, gt):
        _, output_cls = torch.max(panoptic_logits.data, 1, keepdim=True)
        ignore = (gt == 255).long().sum()
        correct = (output_cls.view(-1) == gt.data.view(-1)).long().sum()
        total = (gt.view(-1).shape[0]) - ignore
        assert total != 0
        panoptic_acc = correct.float() / total.float()
        return panoptic_acc

    def get_params_lr(self):
        ret = []
        gn_params = []
        gn_params_name = []
        for n, m in self.named_modules():
            if isinstance(m, nn.GroupNorm) or (
                    config.train.use_horovod and config.network.use_syncbn and isinstance(m, BatchNorm2d)):
                gn_params.append(m.weight)
                gn_params.append(m.bias)
                gn_params_name.append(n + '.weight')
                gn_params_name.append(n + '.bias')

        ret.append({'params': gn_params, 'lr': 1, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['resnet_backbone.res3', 'resnet_backbone.res4',
                                                            'resnet_backbone.res5'], ['weight'])], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['resnet_backbone.res3', 'resnet_backbone.res4',
                                                            'resnet_backbone.res5'], ['bias'])], 'lr': 2,
                    'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['fpn'], ['weight'], exclude=gn_params_name)], 'lr': 10})
        ret.append({'params': [_ for _ in get_params(self, ['fpn'], ['bias'], exclude=gn_params_name)], 'lr': 20,
                    'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['rcnn'], ['weight'], exclude=gn_params_name)], 'lr': 10})
        ret.append({'params': [_ for _ in get_params(self, ['rcnn'], ['bias'], exclude=gn_params_name)], 'lr': 20,
                    'weight_decay': 0})
        ret.append(
            {'params': [_ for _ in get_params(self, ['mask_branch'], ['weight'], exclude=gn_params_name)], 'lr': 10})
        ret.append(
            {'params': [_ for _ in get_params(self, ['mask_branch'], ['bias'], exclude=gn_params_name)], 'lr': 20,
             'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['rpn'], ['weight'])], 'lr': 10})
        ret.append({'params': [_ for _ in get_params(self, ['rpn'], ['bias'])], 'lr': 20, 'weight_decay': 0})
        ret.append(
            {'params': [_ for _ in get_params(self, ['fcn_head'], ['weight'], exclude=gn_params_name)], 'lr': 10})
        ret.append({'params': [_ for _ in get_params(self, ['fcn_head'], ['bias'], exclude=gn_params_name)], 'lr': 20,
                    'weight_decay': 0})

        return ret

    def get_gt_rois(self, roidb, im_info):
        gt_inds = np.where((roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0))[0]
        rois = roidb['boxes'][gt_inds] * im_info[0, 2]
        cls_idx = roidb['gt_classes'][gt_inds]
        return torch.from_numpy(np.hstack((np.zeros((rois.shape[0], 1), dtype=np.float32), rois))), torch.from_numpy(
            cls_idx).long()

    def bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def hist_match(self, images, previous_images):
        GPU_id = images.device.index
        im_src = np.asarray(images.squeeze(0).transpose(0,1).transpose(1,2).cpu(), np.float32)
        im_trg = np.asarray(previous_images.squeeze(0).transpose(0,1).transpose(1,2).cpu(), np.float32)
        images_aug = match_histograms(im_src, im_trg, multichannel=True)
        return torch.from_numpy(images_aug).transpose(1,2).transpose(0,1).unsqueeze(0).cuda('cuda:' + str(GPU_id))

def resnet_101_upsnet_multiscale_branch_lr10():
    return resnet_upsnet_multiscale_branch_lr10([3, 4, 23, 3])


def resnet_50_upsnet_multiscale_branch_lr10():
    return resnet_upsnet_multiscale_branch_lr10([3, 4, 6, 3])
