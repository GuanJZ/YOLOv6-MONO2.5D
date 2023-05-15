#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy, box_iou
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner

class ComputeLoss:
    '''Loss computation func.'''
    def __init__(self, 
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 ori_img_size=640,
                 warmup_epoch=4,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                     'dims': 1.0,
                     'conf': 1.0,
                     'orint': 10.0
                 }
                 ):
        
        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size
        
        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.dim_loss = nn.MSELoss().cuda()
        self.conf_loss = nn.CrossEntropyLoss().cuda()

        self.loss_weight = loss_weight       
        
    def __call__(
        self,
        outputs,
        targets,
        epoch_num,
        step_num
    ):
        
        feats, pred_scores, pred_distri, pred_dim, pred_orient, pred_conf = outputs
        anchors, anchor_points, n_anchors_list, stride_tensor = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)
   
        assert pred_scores.type() == pred_distri.type() == pred_dim.type() == pred_orient.type() == pred_conf.type()
        gt_bboxes_scale = torch.full((1,4), self.ori_img_size).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # targets
        targets =self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:5] #xyxy
        gt_dims = targets[:, :, 5:8]
        gt_orients = targets[:, :, 12:16]
        gt_confs = targets[:, :, 16:18]
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        
        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri) #xyxy

        try:
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                        anchors,
                        n_anchors_list,
                        gt_labels,
                        gt_bboxes,
                        mask_gt,
                        pred_bboxes.detach() * stride_tensor)
            else:
                target_labels, target_bboxes, target_scores, target_dims, target_orients, target_confs, fg_mask = \
                    self.formal_assigner(
                        pred_scores.detach(),
                        pred_bboxes.detach() * stride_tensor,
                        pred_dim.detach(),
                        pred_orient.detach(),
                        pred_conf.detach(),
                        anchor_points,
                        gt_labels,
                        gt_bboxes,
                        gt_dims,
                        gt_orients,
                        gt_confs,
                        mask_gt)

        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            if epoch_num < self.warmup_epoch:
                _anchors = anchors.cpu().float()
                _n_anchors_list = n_anchors_list
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                        _anchors,
                        _n_anchors_list,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt,
                        _pred_bboxes * _stride_tensor)

            else:
                _pred_scores = pred_scores.detach().cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _anchor_points = anchor_points.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                        _pred_scores,
                        _pred_bboxes * _stride_tensor,
                        _anchor_points,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt)

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()
        #Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # rescale bbox
        target_bboxes /= stride_tensor

        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)
            
        target_scores_sum = target_scores.sum()
		# avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson 
        if target_scores_sum > 0:
        	loss_cls /= target_scores_sum
        
        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)

        loss_dim = self.dimloss(pred_dim, target_dims, fg_mask)
        loss_conf = self.confloss(pred_conf, target_confs, fg_mask)
        loss_orient = self.orientation_loss(pred_orient, target_orients, target_confs, fg_mask)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl + \
               self.loss_weight['dims'] * loss_dim + \
               self.loss_weight['conf']*loss_conf + \
               self.loss_weight['orint']*loss_orient

        return loss, \
            torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0), 
                         (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                         (self.loss_weight['class'] * loss_cls).unsqueeze(0),
                            loss_dim.unsqueeze(0),
                            loss_conf.unsqueeze(0),
                            loss_orient.unsqueeze(0))).detach()
     
    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 21)).tolist()
        # target_list 比 target 多两个全0行
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        # 这一步是为了补全batch_szie的样本数量一致,如果缺少了,就用-1, 0, ..., 0补全
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0] + [0]*16]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:5] = xywh2xyxy(batch_target)
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)

    def dimloss(self, pred_dim, target_dims, fg_mask):
        num_pos = fg_mask.sum()
        if num_pos > 0:
            dim_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 3])
            pred_dims_pos = torch.masked_select(pred_dim,
                                                  dim_mask).reshape([-1, 3])
            target_dims_pos = torch.masked_select(
                                                target_dims, dim_mask).reshape([-1, 3])
            loss_dim = self.dim_loss(pred_dims_pos.float(), target_dims_pos.float())
        else:
            loss_dim = pred_dim.sum() * 0

        return loss_dim

    def confloss(self, pred_conf, target_confs, fg_mask):
        num_pos = fg_mask.sum()
        if num_pos > 0:
            conf_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 2])
            pred_confs_pos = torch.masked_select(pred_conf,
                                                conf_mask).reshape([-1, 2])
            target_confs_pos = torch.masked_select(
                target_confs, conf_mask).reshape([-1, 2])
            loss_conf = self.conf_loss(pred_confs_pos, target_confs_pos)
        else:
            loss_conf = pred_conf.sum() * 0

        return loss_conf

    def orientation_loss(self, pred_orient, target_orients, target_confs, fg_mask):
        num_pos = fg_mask.sum()
        if num_pos > 0:
            orient_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_orient_pos = torch.masked_select(pred_orient,
                                                orient_mask).reshape([-1, 2, 2])
            target_orients_pos = torch.masked_select(
                                target_orients, orient_mask).reshape([-1, 2, 2])

            conf_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 2])
            target_confs_pos = torch.masked_select(
                target_confs, conf_mask).reshape([-1, 2])

            _, select_orint_idx = torch.max(target_confs_pos, 1)
            select_pred_orient = torch.zeros(pred_orient_pos.shape[0], 2).to(pred_orient_pos.device)
            select_target_orient = torch.zeros(pred_orient_pos.shape[0], 2).to(pred_orient_pos.device)
            for i in range(len(pred_orient_pos)):
                select_pred_orient[i, :] =  pred_orient_pos[i, select_orint_idx[i]]
                select_target_orient[i, :] = target_orients_pos[i, select_orint_idx[i]]

            pred_angle_diff = torch.atan2(select_pred_orient[:, 1], select_pred_orient[:, 0])
            target_angle_diff = torch.atan2(select_target_orient[:, 1], select_target_orient[:, 0])

            loss_orient = -1* torch.cos(target_angle_diff - pred_angle_diff).mean()

        else:
            loss_orient = pred_orient.sum() * 0

        return loss_orient.to(torch.float64)


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score,gt_score, label, alpha=0.75, gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):

        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum
               
            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                    target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                        target_ltrb_pos) * bbox_weight
                if target_scores_sum == 0:
                    loss_dfl = loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.

        else:
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)
