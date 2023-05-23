#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from tqdm import tqdm
import numpy as np
import cv2
import json
import torch
import yaml
import shutil
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolov6.data.data_load import create_dataloader
from yolov6.utils.events import LOGGER, NCOLS
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.checkpoint import load_checkpoint
from yolov6.utils.torch_utils import time_sync, get_model_info

'''
python tools/eval.py --task 'train'/'val'/'speed'
'''


class Evaler:
    def __init__(self,
                 data,
                 batch_size=32,
                 img_size=640,
                 conf_thres=0.03,
                 iou_thres=0.65,
                 device='',
                 half=True,
                 save_dir='',
                 test_load_size=640,
                 letterbox_return_int=False,
                 force_no_pad=False,
                 not_infer_on_rect=False,
                 scale_exact=False,
                 verbose=False,
                 do_coco_metric=True,
                 do_pr_metric=False,
                 plot_curve=True,
                 plot_confusion_matrix=False,
                 do_3d=False,
                 do_distance=False,
                 val_trt=False
                 ):
        assert do_pr_metric or do_coco_metric, 'ERROR: at least set one val metric'
        self.data = data
        self.batch_size = batch_size
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.half = half
        self.save_dir = save_dir
        self.test_load_size = test_load_size
        self.letterbox_return_int = letterbox_return_int
        self.force_no_pad = force_no_pad
        self.not_infer_on_rect = not_infer_on_rect
        self.scale_exact = scale_exact
        self.verbose = verbose
        self.do_coco_metric = do_coco_metric
        self.do_pr_metric = do_pr_metric
        self.plot_curve = plot_curve
        self.plot_confusion_matrix = plot_confusion_matrix
        self.do_3d = do_3d
        self.do_distance = do_distance
        self.val_trt = val_trt
        self.model_names = data["names"]

    def init_engine(self, engine):
        import tensorrt as trt
        from collections import namedtuple, OrderedDict
        LOGGER.info("init trt engine ...")
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()
        return context, bindings, binding_addrs, model.get_binding_shape(0)[0]

    def init_model(self, model, weights, task):
        LOGGER.info("init PyTorch model ...")
        if task != 'train':
            model = load_checkpoint(weights, map_location=self.device)
            self.stride = int(model.stride.max())
            if self.device.type != 'cpu':
                model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(model.parameters())))
            # switch to deploy
            from yolov6.layers.common import RepVGGBlock
            for layer in model.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
            LOGGER.info("Switch model to deploy modality.")
            LOGGER.info("Model Summary: {}".format(get_model_info(model, self.img_size)))
        model.half() if self.half else model.float()
        return model

    def init_data(self, dataloader, task):
        '''Initialize dataloader.
        Returns a dataloader for task val or speed.
        '''
        LOGGER.info("init data ...")
        self.is_coco = self.data.get("is_coco", False)
        self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        if task != 'train':
            pad = 0.0 if task == 'speed' else 0.5
            eval_hyp = {
                "test_load_size":self.test_load_size,
                "letterbox_return_int":self.letterbox_return_int,
            }
            if self.force_no_pad:
                pad = 0.0
            rect = not self.not_infer_on_rect
            self.stride = 32
            dataloader = create_dataloader(self.data[task if task in ('train', 'val', 'test') else 'val'],
                                           self.img_size, self.batch_size, self.stride, hyp=eval_hyp, check_labels=True, pad=pad, rect=rect,
                                           data_dict=self.data, task=task)[0]
        return dataloader

    def predict_model(self, model, dataloader, task):
        '''Model prediction
        Predicts the whole dataset and gets the prediced results and inference time.
        '''
        self.speed_result = torch.zeros(4, device=self.device)
        pred_results = []
        pbar = tqdm(dataloader, desc=f"Inferencing model in {task} datasets.", ncols=NCOLS)

        if self.val_trt:
            context, bindings, binding_addrs, trt_batch_size = model
            assert trt_batch_size >= self.batch_size, f'The batch size you set is {self.batch_size}, it must <= tensorrt binding batch size {trt_batch_size}.'
            tmp = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
            # warm up for 10 times
            for _ in range(10):
                binding_addrs['images'] = int(tmp.data_ptr())
                context.execute_v2(list(binding_addrs.values()))

        if self.do_3d:
            # 3d predicts
            preds_3d, labels_3d, img_paths = [], [], []
            save_pred_3d = True
            if save_pred_3d:
                pred_3d_save_dir = os.path.join(self.save_dir, "pred_results")
                if os.path.exists(pred_3d_save_dir):
                    shutil.rmtree(pred_3d_save_dir)
                os.makedirs(pred_3d_save_dir)
            if self.do_distance:
                stats_distance = []
                seen_distance = []
                distance = [[0, 30], [30, 60], [60, 90], [90, 120], [120, 150], [150, 10000]]
        # whether to compute metric and plot PR curve and P、R、F1 curve under iou50 match rule
        if self.do_pr_metric:
            stats_2d, ap = [], []
            seen = 0
            iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
            niou = iouv.numel()
            if self.plot_confusion_matrix:
                from yolov6.utils.metrics import ConfusionMatrix
                confusion_matrix = ConfusionMatrix(nc=model.nc)

        for i, (imgs, targets, paths, shapes) in enumerate(pbar):

            # pre-process
            t1 = time_sync()
            imgs = imgs.to(self.device, non_blocking=True)
            imgs = imgs.half() if self.half and not self.val_trt else imgs.float()
            imgs /= 255
            self.speed_result[1] += time_sync() - t1  # pre-process time

            # Inference
            t2 = time_sync()
            if self.val_trt:
                binding_addrs['images'] = int(imgs.data_ptr())
                context.execute_v2(list(binding_addrs.values()))
                outputs = bindings["outputs"].data
            else:
                outputs, _ = model(imgs)
            self.speed_result[2] += time_sync() - t2  # inference time

            # post-process
            t3 = time_sync()
            outputs = non_max_suppression(outputs, self.conf_thres, self.iou_thres, multi_label=True)
            self.speed_result[3] += time_sync() - t3  # post-process time
            self.speed_result[0] += len(outputs)

            if self.do_pr_metric:
                import copy
                eval_outputs = copy.deepcopy([x.detach().cpu() for x in outputs])

            # save result
            pred_results.extend(self.convert_to_coco_format(outputs, imgs, paths, shapes, self.ids))

            # for tensorboard visualization, maximum images to show: 8
            if i == 0:
                vis_num = min(len(imgs), 8)
                vis_outputs = outputs[:vis_num]
                vis_paths = paths[:vis_num]

            if not self.do_pr_metric:
                continue

            # Statistics per image
            # This code is based on
            # https://github.com/ultralytics/yolov5/blob/master/val.py
            for si, pred in enumerate(eval_outputs):
                # labels
                # (0: type_id,  1: xc, 2: yc, 3: w, 4: h, 5: H_diff, 6: W_diff, 7: L_diff,
                #  8: X, 9: Y, 10: Z, 11: ry, 12, 14: cos, 13, 15: sin, 16, 17: confidence,
                #  18:truncated, 19: occluded, 20: alpha,)
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats_2d.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                self.scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                show_nms_output = False
                if show_nms_output:
                    nms_save_dir = os.path.join(self.save_dir, "nms_output")
                    os.makedirs(nms_save_dir, exist_ok=True)
                    img_nms = cv2.imread(paths[si])
                    for pnms in predn:
                        if pnms[4] > 0.00001:
                            cv2.rectangle(img_nms, (int(pnms[0]), int(pnms[1])), (int(pnms[2]), int(pnms[3])), (0, 0, 255),
                                          1)

                    cv2.imwrite(os.path.join(nms_save_dir, os.path.basename(paths[si])), img_nms)

                # predn
                # from ([xyxy, conf, cls, (H_diff, W_diff, L_diff), ([cos, sin], [cos, sin]), (conf_cos, conf_sin)])
                # to
                # [xyxy, conf, cls, (H, W, L), Ry]

                # 1. predn += HWL_ave
                # from ([xyxy, conf, cls, (H_diff, W_diff, L_diff), ([cos, sin], [cos, sin]), (conf_cos, conf_sin)])
                # to
                # [xyxy, conf, cls, (H, W, L), ([cos, sin], [cos, sin]), (conf_cos, conf_sin)]

                # HWL_ave = torch.tensor(np.loadtxt(os.path.join(os.path.dirname(self.data.get(task)), f"{task}_ave_HWL.txt")))
                # for idx, hwl_ave in enumerate(HWL_ave):
                #     predn[predn[:, 5] == idx, 6:9] += hwl_ave[1:]
                predn[:, 6:9] = np.exp(predn[:, 6:9])

                # 2. theta_ray
                img_width = shapes[si][0][1]
                intrinsic_path = paths[si].replace("images", "calibs").replace("jpg", "txt")
                with open(intrinsic_path, 'r')as f:
                    parse_file = f.read().strip().splitlines()
                    for line in parse_file:
                        if line is not None and line.split()[0] == "P2:":
                            proj_matrix = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
                theta_ray = self.calc_theta_ray(img_width, predn[:, :4], proj_matrix)

                # 3. alpha
                orint, conf = predn[:, 9:13], predn[:, 13:]
                _, conf_idxs = torch.max(conf, dim=1)
                alpha = torch.zeros(conf.shape[0])
                for enum, orient_idx in enumerate(conf_idxs):
                    cos, sin = orint[enum, orient_idx*2], orint[enum, orient_idx*2+1]
                    # 因为数据预处理将alpha的区间从[-pi, pi]移动到[0, 2*pi], 所以这里还需要再减去pi
                    alpha[enum] = torch.atan2(sin, cos) + (orient_idx + 0.5 -1) * torch.pi
                # 4. Ry
                Ry = theta_ray + alpha

                # predn
                # from [xyxy, conf, cls, (H, W, L), ([cos, sin], [cos, sin]), (conf_cos, conf_sin)]
                # to
                # [xyxy, conf, cls, (H, W, L), Ry]
                predn_processed = torch.cat((predn[:, :9], Ry[:, None]), dim=1)

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                if nl:

                    from yolov6.utils.nms import xywh2xyxy

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    tbox[:, [0, 2]] *= imgs[si].shape[1:][1]
                    tbox[:, [1, 3]] *= imgs[si].shape[1:][0]

                    self.scale_coords(imgs[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                    # labelsn
                    # from (cls, xyxy, HWL_diff, XYZ, Ry)
                    # to
                    # (cls, xyxy, HWL, XYZ, Ry)
                    labelsn = torch.cat((labels[:, 0:1], tbox, labels[:, 5:12]), 1)  # native-space labels
                    # HWL_ave = torch.tensor(
                    #     np.loadtxt(os.path.join(os.path.dirname(self.data.get(task)), f"{task}_ave_HWL.txt")))
                    # for idx, hwl_ave in enumerate(HWL_ave):
                    #     labelsn[labelsn[:, 0] == idx, 5:8] += hwl_ave[1:]

                    from yolov6.utils.metrics import process_batch, compute_location
                    # get correct
                    correct = process_batch(predn_processed, labelsn, iouv)

                    if self.plot_confusion_matrix:
                        confusion_matrix.process_batch(predn_processed, labelsn)

                    if self.do_3d:
                        matches = compute_location(predn_processed, labelsn)
                        # predn3d
                        # from predn [xyxy, conf, cls, (H, W, L), Ry]
                        # to
                        # (ndarray)[cls, 0, 0, 0, x1, y1, x2, y2, H, W, L, X, Y, Z, Ry, conf]
                        predn_3d = np.zeros((predn_processed.shape[0], 16))
                        predn_arr = predn_processed.numpy()
                        labelsn_arr = labelsn.numpy()
                        for enum, m in enumerate(matches):
                            predn_3d[enum, 0] = predn_arr[int(m[0]), 5]
                            predn_3d[enum, 4:8] = predn_arr[int(m[0]), :4]
                            predn_3d[enum, 8:11] = predn_arr[int(m[0]), 6:9]
                            predn_3d[enum, 11:14] = labelsn_arr[int(m[1]), 8:11]
                            predn_3d[enum, 14] = predn_arr[int(m[0]), 9]
                            predn_3d[enum, 15] = predn_arr[int(m[0]), 4]
                        preds_3d.append(predn_3d)
                        save_pred_3d = True
                        if save_pred_3d:
                            pred_3d_save_path = os.path.join(pred_3d_save_dir, os.path.basename(paths[si]).replace("jpg", "txt"))
                            np.savetxt(pred_3d_save_path, predn_3d, delimiter=" ", fmt='%.08f')

                        # labelsn_3d
                        # from labelsn (cls, xyxy, HWL, XYZ, Ry)
                        # to
                        # (ndarray)[cls, 0, 0, 0, x1, y1, x2, y2, H, W, L, X, Y, Z, Ry]
                        labelsn_3d = np.zeros((labelsn.shape[0], 15))
                        labelsn_3d[:, 0] = labelsn[:, 0]
                        labelsn_3d[:, 4:8] = labelsn[:, 1:5]
                        labelsn_3d[:, 8:11] = labelsn[:, 5:8]
                        labelsn_3d[:, 11:] = labelsn[:, 8:]

                        labels_3d.append(labelsn_3d)
                        img_paths.append(paths[si])

                        # 按照不同的距离计算指标
                        if self.do_distance:

                            single_seen_dist = [0]*len(distance)
                            single_stats_dist = [None]*len(distance)
                            for dist_idx, dist in enumerate(distance):
                                single_seen_dist[dist_idx] += 1
                                iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
                                niou = iouv.numel()
                                filter_label_idx = np.all(
                                    (np.linalg.norm(labelsn_3d[:, 11:14], ord=2, axis=1) >= dist[0],
                                     np.linalg.norm(labelsn_3d[:, 11:14], ord=2, axis=1) < dist[1]),
                                    axis=0)
                                nl_dist = filter_label_idx.sum()
                                filter_pred_idx = np.all(
                                    (np.linalg.norm(predn_3d[:, 11:14], ord=2, axis=1) >=dist[0],
                                     np.linalg.norm(predn_3d[:, 11:14], ord=2, axis=1) <dist[1]),
                                    axis=0)
                                np_dist = filter_label_idx.sum()

                                pred_dist = torch.tensor(predn_3d[filter_pred_idx],device=self.device)
                                label_dist = torch.tensor(labelsn_3d[filter_label_idx],device=self.device)
                                tcls_dist = list(label_dist[:, 0].cpu())
                                if np_dist == 0:
                                    if nl_dist:
                                        single_stats_dist[dist_idx] = \
                                            (torch.zeros(0, niou, dtype=torch.bool),
                                             torch.Tensor(), torch.Tensor(), tcls_dist)
                                    continue

                                correct_dist = torch.zeros(np_dist, niou, dtype=torch.bool)
                                if nl_dist:
                                    from yolov6.utils.metrics import process_batch_3d
                                    correct_dist = process_batch_3d(pred_dist, label_dist, iouv)

                                single_stats_dist[dist_idx] = (correct_dist.cpu(), pred_dist[:, -1].cpu(), pred_dist[:, 0].cpu(), tcls_dist)

                            seen_distance.append(single_seen_dist)
                            stats_distance.append(single_stats_dist)

                # Append statistics (correct, conf, pcls, tcls)
                stats_2d.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if self.do_pr_metric:

            if self.do_distance:
                LOGGER.info("distance metric")
                stats_distance_spilt = []
                for x_dist in zip(*stats_distance):
                    x_dist = [i for i in x_dist if i is not None]
                    stats_distance_spilt.append([np.concatenate(x, 0) for x in zip(*x_dist)])
                for dist, stats in zip(distance, stats_distance_spilt):
                    LOGGER.info(f"\ndistance -- {dist[0]} ~ {dist[1]}:\n")
                    if len(stats) and stats[0].any():

                        from yolov6.utils.metrics import ap_per_class
                        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=self.plot_curve, save_dir=self.save_dir,
                                                              names=self.model_names)
                        AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() - 1
                        LOGGER.info(f"IOU 50 best mF1 thershold near {AP50_F1_max_idx / 1000.0}.")
                        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                        mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:,
                                                                           AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
                        nt = np.bincount(stats[3].astype(np.int64), minlength=len(self.model_names))  # number of targets per class

                        # Print results
                        s = ('%-16s' + '%12s' * 7) % (
                        'Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
                        LOGGER.info(s)
                        pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
                        LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, f1.mean(0)[AP50_F1_max_idx], map50, map))

                        self.pr_metric_result = (map50, map)

                        # Print results per class
                        if self.verbose and len(self.model_names) > 1:
                            for i, c in enumerate(ap_class):
                                LOGGER.info(
                                    pf % (self.model_names[c], seen, nt[c], p[i, AP50_F1_max_idx], r[i, AP50_F1_max_idx],
                                          f1[i, AP50_F1_max_idx], ap50[i], ap[i]))

                        if self.plot_confusion_matrix:
                            confusion_matrix.plot(save_dir=self.save_dir, names=list(self.model_names))
                    else:
                        LOGGER.info("Calculate metric failed, might check dataset.")
                        self.pr_metric_result = (0.0, 0.0)

            # Compute statistics
            LOGGER.info("2D metric")
            stats = [np.concatenate(x, 0) for x in zip(*stats_2d)]  # to numpy
            if len(stats) and stats[0].any():

                from yolov6.utils.metrics import ap_per_class
                p, r, ap, f1, ap_class = ap_per_class(*stats, plot=self.plot_curve, save_dir=self.save_dir, names=self.model_names)
                AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() -1
                LOGGER.info(f"IOU 50 best mF1 thershold near {AP50_F1_max_idx/1000.0}.")
                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
                nt = np.bincount(stats[3].astype(np.int64), minlength=len(self.model_names))  # number of targets per class

                # Print results
                s = ('%-16s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
                LOGGER.info(s)
                pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
                LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, f1.mean(0)[AP50_F1_max_idx], map50, map))

                self.pr_metric_result = (map50, map)

                # Print results per class
                if self.verbose and len(self.model_names) > 1:
                    for i, c in enumerate(ap_class):
                        LOGGER.info(pf % (self.model_names[c], seen, nt[c], p[i, AP50_F1_max_idx], r[i, AP50_F1_max_idx],
                                           f1[i, AP50_F1_max_idx], ap50[i], ap[i]))

                if self.plot_confusion_matrix:
                    confusion_matrix.plot(save_dir=self.save_dir, names=list(self.model_names))

                if self.do_3d:
                    from yolov6.utils.show_2d3d_box import show_2d3d_box
                    conf_thres = AP50_F1_max_idx/1000.0
                    # conf_thres = 0
                    final_preds_3d = [pred[pred[:, -1] >= conf_thres] for pred in preds_3d]

                    LOGGER.info("writing 3D BBoxes")
                    show_2d3d_box(final_preds_3d, labels_3d, img_paths, self.data["names"], self.save_dir)
            else:
                LOGGER.info("Calculate metric failed, might check dataset.")
                self.pr_metric_result = (0.0, 0.0)

        return pred_results, vis_outputs, vis_paths


    def eval_model(self, pred_results, model, dataloader, task):
        '''Evaluate models
        For task speed, this function only evaluates the speed of model and outputs inference time.
        For task val, this function evaluates the speed and mAP by pycocotools, and returns
        inference time and mAP value.
        '''
        LOGGER.info(f'\nEvaluating speed.')
        self.eval_speed(task)

        if not self.do_coco_metric and self.do_pr_metric:
            return self.pr_metric_result
        LOGGER.info(f'\nEvaluating mAP by pycocotools.')
        if task != 'speed' and len(pred_results):
            if 'anno_path' in self.data:
                anno_json = self.data['anno_path']
            else:
                # generated coco format labels in dataset initialization
                task = 'val' if task == 'train' else task
                dataset_root = os.path.dirname(os.path.dirname(self.data[task]))
                base_name = os.path.basename(self.data[task])
                anno_json = os.path.join(dataset_root, 'annotations', f'instances_{base_name}.json')
            pred_json = os.path.join(self.save_dir, "predictions.json")
            LOGGER.info(f'Saving {pred_json}...')
            with open(pred_json, 'w') as f:
                json.dump(pred_results, f)

            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            cocoEval = COCOeval(anno, pred, 'bbox')
            if self.is_coco:
                imgIds = [int(os.path.basename(x).split(".")[0])
                            for x in dataloader.dataset.img_paths]
                cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()

            #print each class ap from pycocotool result
            if self.verbose:

                import copy
                val_dataset_img_count = cocoEval.cocoGt.imgToAnns.__len__()
                val_dataset_anns_count = 0
                label_count_dict = {"images":set(), "anns":0}
                label_count_dicts = [copy.deepcopy(label_count_dict) for _ in range(len(self.model_names))]
                for _, ann_i in cocoEval.cocoGt.anns.items():
                    if ann_i["ignore"]:
                        continue
                    val_dataset_anns_count += 1
                    nc_i = self.coco80_to_coco91_class().index(ann_i['category_id']) if self.is_coco else ann_i['category_id']
                    label_count_dicts[nc_i]["images"].add(ann_i["image_id"])
                    label_count_dicts[nc_i]["anns"] += 1

                s = ('%-16s' + '%12s' * 7) % ('Class', 'Labeled_images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
                LOGGER.info(s)
                #IOU , all p, all cats, all gt, maxdet 100
                coco_p = cocoEval.eval['precision']
                coco_p_all = coco_p[:, :, :, 0, 2]
                map = np.mean(coco_p_all[coco_p_all>-1])

                coco_p_iou50 = coco_p[0, :, :, 0, 2]
                map50 = np.mean(coco_p_iou50[coco_p_iou50>-1])
                mp = np.array([np.mean(coco_p_iou50[ii][coco_p_iou50[ii]>-1]) for ii in range(coco_p_iou50.shape[0])])
                mr = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
                mf1 = 2 * mp * mr / (mp + mr + 1e-16)
                i = mf1.argmax()  # max F1 index

                pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
                LOGGER.info(pf % ('all', val_dataset_img_count, val_dataset_anns_count, mp[i], mr[i], mf1[i], map50, map))

                #compute each class best f1 and corresponding p and r
                for nc_i in range(len(self.model_names)):
                    coco_p_c = coco_p[:, :, nc_i, 0, 2]
                    map = np.mean(coco_p_c[coco_p_c>-1])

                    coco_p_c_iou50 = coco_p[0, :, nc_i, 0, 2]
                    map50 = np.mean(coco_p_c_iou50[coco_p_c_iou50>-1])
                    p = coco_p_c_iou50
                    r = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
                    f1 = 2 * p * r / (p + r + 1e-16)
                    i = f1.argmax()
                    LOGGER.info(pf % (self.model_names[nc_i], len(label_count_dicts[nc_i]["images"]), label_count_dicts[nc_i]["anns"], p[i], r[i], f1[i], map50, map))
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            # Return results
            if not self.val_trt:
                model.float()  # for training
            if task != 'train':
                LOGGER.info(f"Results saved to {self.save_dir}")
            return (map50, map)
        return (0.0, 0.0)

    def eval_speed(self, task):
        '''Evaluate model inference speed.'''
        if task != 'train':
            n_samples = self.speed_result[0].item()
            pre_time, inf_time, nms_time = 1000 * self.speed_result[1:].cpu().numpy() / n_samples
            for n, v in zip(["pre-process", "inference", "NMS"],[pre_time, inf_time, nms_time]):
                LOGGER.info("Average {} time: {:.2f} ms".format(n, v))
    def calc_theta_ray(self, width, box_2d, proj_matrix):
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0, 0]))
        center = (box_2d[:, 2] + box_2d[:, 0]) / 2
        dx = center - (width / 2)

        mult = np.ones(dx.shape)
        mult[dx < 0] = -1
        dx = np.abs(dx)
        angle = np.arctan((2 * dx * np.tan(fovx / 2)) / width)
        angle = angle * mult

        return angle

    def box_convert(self, x):
        '''Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right.'''
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        '''Rescale coords (xyxy) from img1_shape to img0_shape.'''
        if ratio_pad is None:  # calculate from img0_shape
            gain = [min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])]  # gain  = old / new
            if self.scale_exact:
                gain = [img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]]
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        if self.scale_exact:
            coords[:, [0, 2]] /= gain[1]  # x gain
        else:
            coords[:, [0, 2]] /= gain[0]  # raw x gain
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, [1, 3]] /= gain[0]  # y gain

        if isinstance(coords, torch.Tensor):  # faster individually
            coords[:, 0].clamp_(0, img0_shape[1])  # x1
            coords[:, 1].clamp_(0, img0_shape[0])  # y1
            coords[:, 2].clamp_(0, img0_shape[1])  # x2
            coords[:, 3].clamp_(0, img0_shape[0])  # y2
        else:  # np.array (faster grouped)
            coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
            coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
        return coords

    def convert_to_coco_format(self, outputs, imgs, paths, shapes, ids):
        pred_results = []
        for i, pred in enumerate(outputs):
            if len(pred) == 0:
                continue
            path, shape = Path(paths[i]), shapes[i][0]
            self.scale_coords(imgs[i].shape[1:], pred[:, :4], shape, shapes[i][1])
            image_id = int(path.stem) if self.is_coco else path.stem
            bboxes = self.box_convert(pred[:, 0:4])
            bboxes[:, :2] -= bboxes[:, 2:] / 2
            cls = pred[:, 5]
            scores = pred[:, 4]
            for ind in range(pred.shape[0]):
                category_id = ids[int(cls[ind])]
                bbox = [round(x, 3) for x in bboxes[ind].tolist()]
                score = round(scores[ind].item(), 5)
                pred_data = {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score
                }
                pred_results.append(pred_data)
        return pred_results

    @staticmethod
    def check_task(task):
        if task not in ['train', 'val', 'test', 'speed']:
            raise Exception("task argument error: only support 'train' / 'val' / 'test' / 'speed' task.")

    @staticmethod
    def check_thres(conf_thres, iou_thres, task):
        '''Check whether confidence and iou threshold are best for task val/speed'''
        if task != 'train':
            if task == 'val' or task == 'test':
                if conf_thres > 0.03:
                    LOGGER.warning(f'The best conf_thresh when evaluate the model is less than 0.03, while you set it to: {conf_thres}')
                if iou_thres != 0.65:
                    LOGGER.warning(f'The best iou_thresh when evaluate the model is 0.65, while you set it to: {iou_thres}')
            if task == 'speed' and conf_thres < 0.4:
                LOGGER.warning(f'The best conf_thresh when test the speed of the model is larger than 0.4, while you set it to: {conf_thres}')

    @staticmethod
    def reload_device(device, model, task):
        # device = 'cpu' or '0' or '0,1,2,3'
        if task == 'train':
            device = next(model.parameters()).device
        else:
            if device == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            elif device:
                os.environ['CUDA_VISIBLE_DEVICES'] = device
                assert torch.cuda.is_available()
            cuda = device != 'cpu' and torch.cuda.is_available()
            device = torch.device('cuda:0' if cuda else 'cpu')
        return device

    @staticmethod
    def reload_dataset(data, task='val'):
        with open(data, errors='ignore') as yaml_file:
            data = yaml.safe_load(yaml_file)
        task = 'test' if task == 'test' else 'val'
        path = data.get(task, 'val')
        if not os.path.exists(path):
            raise Exception('Dataset not found.')
        return data

    @staticmethod
    def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x

    def eval_trt(self, engine, stride=32):
        self.stride = stride
        def init_engine(engine):
            import tensorrt as trt
            from collections import namedtuple,OrderedDict
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.ERROR)
            trt.init_libnvinfer_plugins(logger, namespace="")
            with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            return context, bindings, binding_addrs, model.get_binding_shape(0)[0]

        def init_data(dataloader, task):
            self.is_coco = self.data.get("is_coco", False)
            self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range(1000))
            pad = 0.0 if task == 'speed' else 0.5
            dataloader = create_dataloader(self.data[task if task in ('train', 'val', 'test') else 'val'],
                                           self.img_size, self.batch_size, self.stride, check_labels=True, pad=pad, rect=False,
                                           data_dict=self.data, task=task)[0]
            return dataloader

        def convert_to_coco_format_trt(nums, boxes, scores, classes, paths, shapes, ids):
            pred_results = []
            for i, (num, detbox, detscore, detcls) in enumerate(zip(nums, boxes, scores, classes)):
                n = int(num[0])
                if n == 0:
                    continue
                path, shape = Path(paths[i]), shapes[i][0]
                gain = shapes[i][1][0][0]
                pad = torch.tensor(shapes[i][1][1]*2).to(self.device)
                detbox = detbox[:n, :]
                detbox -= pad
                detbox /= gain
                detbox[:, 0].clamp_(0, shape[1])
                detbox[:, 1].clamp_(0, shape[0])
                detbox[:, 2].clamp_(0, shape[1])
                detbox[:, 3].clamp_(0, shape[0])
                detbox[:,2:] = detbox[:,2:] - detbox[:,:2]
                detscore = detscore[:n]
                detcls = detcls[:n]

                image_id = int(path.stem) if path.stem.isnumeric() else path.stem

                for ind in range(n):
                    category_id = ids[int(detcls[ind])]
                    bbox = [round(x, 3) for x in detbox[ind].tolist()]
                    score = round(detscore[ind].item(), 5)
                    pred_data = {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": score
                    }
                    pred_results.append(pred_data)
            return pred_results

        context, bindings, binding_addrs, trt_batch_size = init_engine(engine)
        assert trt_batch_size >= self.batch_size, f'The batch size you set is {self.batch_size}, it must <= tensorrt binding batch size {trt_batch_size}.'
        tmp = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        # warm up for 10 times
        for _ in range(10):
            binding_addrs['images'] = int(tmp.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
        dataloader = init_data(None,'val')
        self.speed_result = torch.zeros(4, device=self.device)
        pred_results = []
        pbar = tqdm(dataloader, desc="Inferencing model in validation dataset.", ncols=NCOLS)
        for imgs, targets, paths, shapes in pbar:
            nb_img = imgs.shape[0]
            if nb_img != self.batch_size:
                # pad to tensorrt model setted batch size
                zeros = torch.zeros(self.batch_size - nb_img, 3, *imgs.shape[2:])
                imgs = torch.cat([imgs, zeros],0)
            t1 = time_sync()
            imgs = imgs.to(self.device, non_blocking=True)
            # preprocess
            imgs = imgs.float()
            imgs /= 255

            self.speed_result[1] += time_sync() - t1  # pre-process time

            # inference
            t2 = time_sync()
            binding_addrs['images'] = int(imgs.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
            # in the last batch, the nb_img may less than the batch size, so we need to fetch the valid detect results by [:nb_img]
            nums = bindings['num_dets'].data[:nb_img]
            boxes = bindings['det_boxes'].data[:nb_img]
            scores = bindings['det_scores'].data[:nb_img]
            classes = bindings['det_classes'].data[:nb_img]
            self.speed_result[2] += time_sync() - t2  # inference time

            self.speed_result[3] += 0
            pred_results.extend(convert_to_coco_format_trt(nums, boxes, scores, classes, paths, shapes, self.ids))
            self.speed_result[0] += self.batch_size
        return dataloader, pred_results
