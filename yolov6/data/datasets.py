#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
from io import UnsupportedOperation
import os
import os.path as osp
import random
import json
import time
import hashlib
from pathlib import Path

from multiprocessing.pool import Pool

import cv2
import numpy as np
import torch
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from .data_augment import (
    augment_hsv,
    letterbox,
    mixup,
    random_affine,
    mosaic_augmentation,
)
from yolov6.utils.events import LOGGER

# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])
VID_FORMATS.extend([f.upper() for f in VID_FORMATS])
# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break


class TrainValDataset(Dataset):
    '''YOLOv6 train_loader/val_loader, loads images and labels for training and validation.'''
    def __init__(
        self,
        img_dir,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        check_images=False,
        check_labels=False,
        stride=32,
        pad=0.0,
        rank=-1,
        data_dict=None,
        task="train",
        bins=2,
        overlap=0.1
    ):
        assert task.lower() in ("train", "val", "test", "speed"), f"Not supported task: {task}"
        t1 = time.time()
        self.__dict__.update(locals())
        self.main_process = self.rank in (-1, 0)
        self.task = self.task.capitalize()
        self.class_names = data_dict["names"]
        self.bins = bins
        self.overlap = overlap
        self.img_paths, self.labels = self.get_imgs_labels(self.img_dir)
        if self.rect:
            shapes = [self.img_info[p]["shape"] for p in self.img_paths]
            self.shapes = np.array(shapes, dtype=np.float64)
            self.batch_indices = np.floor(
                np.arange(len(shapes)) / self.batch_size
            ).astype(
                int
            )  # batch indices of each image
            self.sort_files_shapes()
        t2 = time.time()
        if self.main_process:
            LOGGER.info(f"%.1fs for dataset initialization." % (t2 - t1))

    def __len__(self):
        """Get the length of dataset"""
        return len(self.img_paths)

    def __getitem__(self, index):
        """Fetching a data sample for a given key.
        This function applies mosaic and mixup augments during training.
        During validation, letterbox augment is applied.
        """
        # Mosaic Augmentation
        if self.augment and random.random() < self.hyp["mosaic"]:
            img, labels = self.get_mosaic(index)
            shapes = None
            # MixUp augmentation
            if random.random() < self.hyp["mixup"]:
                img_other, labels_other = self.get_mosaic(
                    random.randint(0, len(self.img_paths) - 1)
                )
                img, labels = mixup(img, labels, img_other, labels_other)

        else:
            # Load image
            if self.hyp and "test_load_size" in self.hyp:
                img, (h0, w0), (h, w) = self.load_image(index, self.hyp["test_load_size"])
            else:
                img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = (
                self.batch_shapes[self.batch_indices[index]]
                if self.rect
                else self.img_size
            )  # final letterboxed shape
            if self.hyp and "letterbox_return_int" in self.hyp:
                img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, return_int=self.hyp["letterbox_return_int"])
            else:
                img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

            shapes = (h0, w0), ((h * ratio / h0, w * ratio / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:
                w *= ratio
                h *= ratio
                # new boxes
                boxes = np.copy(labels[:, 1:])
                boxes[:, 0] = (
                    w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]
                )  # top left x
                boxes[:, 1] = (
                    h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]
                )  # top left y
                boxes[:, 2] = (
                    w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
                )  # bottom right x
                boxes[:, 3] = (
                    h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
                )  # bottom right y
                labels[:, 1:] = boxes

                labels[:, 18] = labels[:, 18] * w + pad[0]
                labels[:, 19] = labels[:, 19] * h + pad[1]

            if self.augment:
                img, labels = random_affine(
                    img,
                    labels,
                    degrees=self.hyp["degrees"],
                    translate=self.hyp["translate"],
                    scale=self.hyp["scale"],
                    shear=self.hyp["shear"],
                    new_shape=(self.img_size, self.img_size),
                )

        if len(labels):
            h, w = img.shape[:2]

            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2

            # labels[:, 18] = labels[:, 18].clip(0, w - 1e-3) # keypoint_x
            # labels[:, 19] = labels[:, 19].clip(0, h - 1e-3) # keypoint_y

            boxes = np.copy(labels[:, 1:5])
            boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) / w  # x center
            boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) / h  # y center
            boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w  # width
            boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h  # height
            labels[:, 1:5] = boxes
            labels[:, 18] = labels[:, 18] / w
            labels[:, 19] = labels[:, 19] / h

        if self.augment:
            img, labels = self.general_augment(img, labels)

        labels_out = torch.zeros((len(labels), 24))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_paths[index], shapes

    def load_image(self, index, force_load_size=None):
        """Load image.
        This function loads image by cv2, resize original image to target shape(img_size) with keeping ratio.

        Returns:
            Image, original shape of image, resized image shape
        """
        path = self.img_paths[index]
        try:
            im = cv2.imread(path)
            assert im is not None, f"opencv cannot read image correctly or {path} not exists"
        except:
            im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
            assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

        h0, w0 = im.shape[:2]  # origin shape
        if force_load_size:
            r = force_load_size / max(h0, w0)
        else:
            r = self.img_size / max(h0, w0)
        if r != 1:
            im = cv2.resize(
                im,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA
                if r < 1 and not self.augment
                else cv2.INTER_LINEAR,
            )
        return im, (h0, w0), im.shape[:2]

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def get_imgs_labels(self, img_dir):

        assert osp.exists(img_dir), f"{img_dir} is an invalid directory path!"
        valid_img_record = osp.join(
            osp.dirname(img_dir), "." + osp.basename(img_dir) + ".json"
        )
        NUM_THREADS = min(16, os.cpu_count())

        img_paths = glob.glob(osp.join(img_dir, "**/*"), recursive=True)
        img_paths = sorted(
            p for p in img_paths if p.split(".")[-1].lower() in IMG_FORMATS and os.path.isfile(p)
        )
        assert img_paths, f"No images found in {img_dir}."

        img_hash = self.get_hash(img_paths)
        if osp.exists(valid_img_record):
            with open(valid_img_record, "r") as f:
                cache_info = json.load(f)
                if "image_hash" in cache_info and cache_info["image_hash"] == img_hash:
                    img_info = cache_info["information"]
                else:
                    self.check_images = True
        else:
            self.check_images = True

        # check images
        if self.check_images and self.main_process:
            img_info = {}
            nc, msgs = 0, []  # number corrupt, messages
            LOGGER.info(
                f"{self.task}: Checking formats of images with {NUM_THREADS} process(es): "
            )
            with Pool(NUM_THREADS) as pool:
                pbar = tqdm(
                    pool.imap(TrainValDataset.check_image, img_paths),
                    total=len(img_paths),
                )
                for img_path, shape_per_img, nc_per_img, msg in pbar:
                    if nc_per_img == 0:  # not corrupted
                        img_info[img_path] = {"shape": shape_per_img}
                    nc += nc_per_img
                    if msg:
                        msgs.append(msg)
                    pbar.desc = f"{nc} image(s) corrupted"
            pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))

            cache_info = {"information": img_info, "image_hash": img_hash}
            # save valid image paths.
            with open(valid_img_record, "w") as f:
                json.dump(cache_info, f)

        # check and load anns
        base_dir = osp.basename(img_dir)
        if base_dir != "":
            label_dir = osp.join(
            osp.dirname(osp.dirname(img_dir)), "labels_yolo_MONO_2.5D_KEYPOINT", osp.basename(img_dir)
            )
            assert osp.exists(label_dir), f"{label_dir} is an invalid directory path!"
        else:
            sub_dirs= []
            label_dir = img_dir
            for rootdir, dirs, files in os.walk(label_dir):
                for subdir in dirs:
                    sub_dirs.append(subdir)
            assert "labels_yolo_MONO_2.5D_KEYPOINT" in sub_dirs, f"Could not find a labels directory!"


        # Look for labels in the save relative dir that the images are in
        def _new_rel_path_with_ext(base_path: str, full_path: str, new_ext: str):
            rel_path = osp.relpath(full_path, base_path)
            return osp.join(osp.dirname(rel_path), osp.splitext(osp.basename(rel_path))[0] + new_ext)


        img_paths = list(img_info.keys())
        label_paths = sorted(
            osp.join(label_dir, _new_rel_path_with_ext(img_dir, p, ".txt"))
            for p in img_paths
        )
        assert label_paths, f"No labels found in {label_dir}."
        label_hash = self.get_hash(label_paths)
        if "label_hash" not in cache_info or cache_info["label_hash"] != label_hash:
            self.check_labels = True

        if self.check_labels:
            cache_info["label_hash"] = label_hash
            nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number corrupt, messages
            LOGGER.info(
                f"{self.task}: Checking formats of labels with {NUM_THREADS} process(es): "
            )
            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(
                    TrainValDataset.check_label_files, zip(img_paths, label_paths)
                )
                pbar = tqdm(pbar, total=len(label_paths)) if self.main_process else pbar
                for (
                    img_path,
                    labels_per_file,
                    nc_per_file,
                    nm_per_file,
                    nf_per_file,
                    ne_per_file,
                    msg,
                ) in pbar:
                    if nc_per_file == 0:
                        if ne_per_file == 1:
                            img_info.pop(img_path)
                        else:
                            img_info[img_path]["labels"] = labels_per_file
                    else:
                        img_info.pop(img_path)
                    nc += nc_per_file
                    nm += nm_per_file
                    nf += nf_per_file
                    ne += ne_per_file
                    if msg:
                        msgs.append(msg)
                    if self.main_process:
                        pbar.desc = f"{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files"
            if self.main_process:
                pbar.close()
                with open(valid_img_record, "w") as f:
                    json.dump(cache_info, f)
            if msgs:
                LOGGER.info("\n".join(msgs))
            if nf == 0:
                LOGGER.warning(
                    f"WARNING: No labels found in {osp.dirname(img_paths[0])}. "
                )

        if self.task.lower() == "val":
            if self.data_dict.get("is_coco", False): # use original json file when evaluating on coco dataset.
                assert osp.exists(self.data_dict["anno_path"]), "Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml"
            else:
                assert (
                    self.class_names
                ), "Class names is required when converting labels to coco format for evaluating."
                save_dir = osp.join(osp.dirname(osp.dirname(img_dir)), "annotations")
                if not osp.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = osp.join(
                    save_dir, "instances_" + osp.basename(img_dir) + ".json"
                )
                TrainValDataset.generate_coco_format_labels(
                    img_info, self.class_names, save_path
                )

        # 现在labels返回的和以前完全不一样了,返回的是Rope3d的labels, 将type转换成了type_id
        # 所以后面有两件事情需要做: 1. generate_coco_format_labels生成instance_val.json的时候,不需要将xc, yc, w, h转换为x1, y1, x2, y2;
        #                      2. attributes_3d_preprocess在最后的时候需要将 x1, y1, x2, y2 转换成 xc, yc, w, h, 不然后面的代码全都要修改,容易出bug;
        img_paths, labels = list(
            zip(
                *[
                    (
                        img_path,
                        np.array(info["labels"], dtype=np.float32)
                        if info["labels"]
                        else np.zeros((0, 15), dtype=np.float32),
                    )
                    for img_path, info in img_info.items()
                ]
            )
        )
        self.img_info = img_info
        LOGGER.info(
            f"{self.task}: Final numbers of valid images: {len(img_paths)}/ labels: {len(labels)}. "
        )


        return img_paths, labels

    def get_mosaic(self, index):
        """Gets images and labels after mosaic augments"""
        indices = [index] + random.choices(
            range(0, len(self.img_paths)), k=3
        )  # 3 additional image indices
        random.shuffle(indices)
        imgs, hs, ws, labels = [], [], [], []
        for index in indices:
            img, _, (h, w) = self.load_image(index)
            # print(self.img_paths[index])
            labels_per_img = self.labels[index]
            imgs.append(img)
            hs.append(h)
            ws.append(w)
            labels.append(labels_per_img)
        img, labels = mosaic_augmentation(self.img_size, imgs, hs, ws, labels, self.hyp, self.img_paths, indices)
        return img, labels

    def general_augment(self, img, labels):
        """Gets images and labels after general augment
        This function applies hsv, random ud-flip and random lr-flips augments.
        """
        nl = len(labels)

        # HSV color-space
        augment_hsv(
            img,
            hgain=self.hyp["hsv_h"],
            sgain=self.hyp["hsv_s"],
            vgain=self.hyp["hsv_v"],
        )

        # Flip up-down
        if random.random() < self.hyp["flipud"]:
            img = np.flipud(img)
            if nl:
                labels[:, 2] = 1 - labels[:, 2]
                labels[:, 19] = 1 - labels[:, 19]


        # Flip left-right
        if random.random() < self.hyp["fliplr"]:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]
                labels[:, 18] = 1 - labels[:, 18]

        return img, labels

    def sort_files_shapes(self):
        '''Sort by aspect ratio.'''
        batch_num = self.batch_indices[-1] + 1
        s = self.shapes  # wh
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        self.img_paths = [self.img_paths[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * batch_num
        for i in range(batch_num):
            ari = ar[self.batch_indices == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]
        self.batch_shapes = (
            np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(
                int
            )
            * self.stride
        )

    @staticmethod
    def check_image(im_file):
        '''Verify an image.'''
        nc, msg = 0, ""
        try:
            im = Image.open(im_file)
            im.verify()  # PIL verify
            im = Image.open(im_file)  # need to reload the image after using verify()
            shape = im.size  # (width, height)
            try:
                im_exif = im._getexif()
                if im_exif and ORIENTATION in im_exif:
                    rotation = im_exif[ORIENTATION]
                    if rotation in (6, 8):
                        shape = (shape[1], shape[0])
            except:
                im_exif = None
            if im_exif and ORIENTATION in im_exif:
                rotation = im_exif[ORIENTATION]
                if rotation in (6, 8):
                    shape = (shape[1], shape[0])

            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(
                            im_file, "JPEG", subsampling=0, quality=100
                        )
                        msg += f"WARNING: {im_file}: corrupt JPEG restored and saved"
            return im_file, shape, nc, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {im_file}: ignoring corrupt image: {e}"
            return im_file, None, nc, msg

    @staticmethod
    def check_label_files(args):
        img_path, lb_path = args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        try:
            if osp.exists(lb_path):
                nf = 1  # label found
                with open(lb_path, "r") as f:
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]
                    labels = [
                        [float(lb) for lb in lbs] for lbs in labels
                    ]

                    labels = [
                       lbs for lbs in labels if abs(lbs[8]) >= 1e-6 and abs(lbs[9]) >= 1e-6 and abs(lbs[10]) >= 1e-6
                    ]

                    labels = np.array(labels, dtype=np.float32)

                if len(labels):
                    assert all(
                        len(l) == 23 for l in labels
                    ), f"{lb_path}: wrong label format."
                    assert (
                        labels[:, [0, 1, 2, 3, 4]] >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                    assert (
                        labels[:, 1:5] <= 1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):  # duplicate row check
                        labels = labels[indices]  # remove duplicates
                        msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                    labels = labels.tolist()
                else:
                    ne = 1  # label empty
                    labels = []
            else:
                nm = 1  # label missing
                labels = []

            return img_path, labels, nc, nm, nf, ne, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"
            return img_path, None, nc, nm, nf, ne, msg

    @staticmethod
    def generate_coco_format_labels(img_info, class_names, save_path):
        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        LOGGER.info(f"Convert to COCO format")
        for i, (img_path, info) in enumerate(tqdm(img_info.items())):
            labels = info["labels"] if info["labels"] else []
            img_id = osp.splitext(osp.basename(img_path))[0]
            img_w, img_h = info["shape"]
            dataset["images"].append(
                {
                    "file_name": os.path.basename(img_path),
                    "id": img_id,
                    "width": img_w,
                    "height": img_h,
                }
            )
            if labels:
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    x2 = (x + w / 2) * img_w
                    y2 = (y + h / 2) * img_h
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": img_id,
                            "iscrowd": 0,
                            # mask
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        with open(save_path, "w") as f:
            json.dump(dataset, f)
            LOGGER.info(
                f"Convert to COCO format finished. Resutls saved in {save_path}"
            )

    @staticmethod
    def get_hash(paths):
        """Get the hash value of paths"""
        assert isinstance(paths, list), "Only support list currently."
        h = hashlib.md5("".join(paths).encode())
        return h.hexdigest()

    def attributes_3d_preprocess(self, img_paths, labels):
        """
        Args:
            img_paths (tuple(str)): all img paths
            labels (tuple(ndarray)): all labels (type_id, truncated, occluded, alpha, x1, y1, x2, y2, H, W, L, X, Y, Z, ry)
        Return:
            labels (tuple(ndarray)): propcessed all labels (type_id, truncated, occluded, alpha, cx, cy, w, h, H_diff, W_diff, L_diff, X, Y, Z, ry, bin_conf, bin_cos_sin)
        """
        # read camera calib matrix
        # self.proj_matrix tuple(np.matrix)
        calib_paths = tuple([osp.join(osp.dirname(im).replace("images", "calibs"), osp.basename(im).replace("jpg", "txt")) \
                             for im in img_paths])
        def get_calib(paths):
            calib_list = []
            for p in paths:
                with open(p, 'r')as f:
                    parse_file = f.read().strip().splitlines()
                    for line in parse_file:
                        if line is not None and line.split()[0] == "P2:":
                            calib_matrix = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
                            calib_list.append(calib_matrix)
            return tuple(calib_list)

        self.proj_matrix = get_calib(calib_paths)

        # ave HWL
        self.average_dims = {}
        for i in range(len(self.class_names)):
            if i not in self.average_dims.keys():
                self.average_dims[i] = {}
            self.average_dims[i]["total"] = np.zeros(3)
            self.average_dims[i]["count"] = 0
            self.average_dims[i]["hwl"] = []

        for label in labels:
            for i in range(len(self.class_names)):
                self.average_dims[i]["total"] += label[label[:, 0] == i][:, 8:11].sum(axis=0)
                self.average_dims[i]["count"] += (label[:, 0] == i).sum()
                self.average_dims[i]["hwl"].extend(label[label[:, 0] == i][:, 8:11])

        for i in range(len(self.class_names)):
            self.average_dims[i]["ave"] = self.average_dims[i]["total"] / (self.average_dims[i]["count"] + 1e-6)
            self.average_dims[i]["hwl"] = np.array(self.average_dims[i]["hwl"])

        labels = list(labels)
        # (type_id, truncated, occluded, alpha, x1, y1, x2, y2, H, W, L, X, Y, Z, ry)
        # to
        # (type_id, truncated, occluded, alpha, x1, y1, x2, y2, H_diff, W_diff, L_diff, X, Y, Z, ry, H_ave, W_ave, L_ave)
        for idx, label in enumerate(labels):
            ave_dims = np.zeros((label.shape[0], 3))
            for i in range(len(label)):
                label[i, 8:11] -= self.average_dims[label[i, 0]]["ave"]
                ave_dims[i, :] = self.average_dims[label[i, 0]]["ave"]
            labels[idx] = np.concatenate((label, ave_dims), axis=1)


        # angle_bins
        interval = 2 * np.pi / self.bins
        self.angle_bins = np.zeros(self.bins)
        for i in range(1, self.bins):
            self.angle_bins[i] = i * interval
        self.angle_bins += interval / 2 # center of the bin


        # bin_ranges for confidence
        # [(min angle in bin, max angle in bin), ...]
        self.bin_ranges = np.zeros((self.bins, 2))
        for i in range(0, self.bins):
            self.bin_ranges[i, 0] = (i * interval - self.overlap) % (2 * np.pi)
            self.bin_ranges[i, 1] = (i * interval + interval + self.overlap) % (2 * np.pi)


        def get_bin(angle, bin_ranges):
            bin_idxs = []

            def is_between(min, max, angle):
                max = (max - min) if (max - min) > 0 else (max - min) + 2 * np.pi
                angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2 * np.pi
                return angle < max

            for bin_idx, bin_range in enumerate(bin_ranges):
                if is_between(bin_range[0], bin_range[1], angle):
                    bin_idxs.append(bin_idx)

            return bin_idxs

        # orientations, confidences
        orientations = []
        confidences = []
        for i in range(len(labels)):
            nl = labels[i].shape[0]
            orientation = np.zeros((nl, self.bins, 2))
            confidence  = np.zeros((nl, self.bins))
            angles = (labels[i][:, 3] + np.pi).reshape(nl, 1)
            for an_id, angle in enumerate(angles):
                bin_idxs = get_bin(angle[0], self.bin_ranges)
                for bin_idx in bin_idxs:
                    angle_diff = angle - self.angle_bins[bin_idx]
                    orientation[an_id, bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)]).squeeze(axis=1)
                    confidence[an_id, bin_idx] = 1

            orientations.append(orientation)
            confidences.append(confidence)

        # compute theta_ray
        def calc_theta_ray(width, box_2d, proj_matrix):
            fovx = 2 * np.arctan(width / (2 * proj_matrix[0, 0]))
            center = (box_2d[2] + box_2d[0]) / 2
            dx = center - (width / 2)

            mult = 1
            if dx < 0:
                mult = -1
            dx = abs(dx)
            angle = np.arctan((2 * dx * np.tan(fovx / 2)) / width)
            angle = angle * mult

            return angle

        theta_rays = []
        for i, label in enumerate(labels):
            theta = np.zeros((label.shape[0], 1))
            H, W = cv2.imread(img_paths[i]).shape[:2]
            for idx in range(len(label)):
                theta[idx, 0] = calc_theta_ray(W, label[idx, 4:8], self.proj_matrix[i])
            theta_rays.append(theta)

        # (type_id, truncated, occluded, alpha, x1, y1, x2, y2, H_diff, W_diff, L_diff, X, Y, Z, ry, H_ave, W_ave, L_ave)
        # to
        # (type_id, truncated, occluded, alpha, xc, yc, w, h, H_diff, W_diff, L_diff, X, Y, Z, ry, H_ave, W_ave, L_ave)
        for label in labels:
            # xyxy-> cxcywh
            label[:, 6] = (label[:, 6] - label[:, 4]) / W
            label[:, 7] = (label[:, 7] - label[:, 5]) / H
            label[:, 4] = label[:, 4] / W + label[:, 6] / 2
            label[:, 5] = label[:, 5] / H + label[:, 7] / 2

        # (type_id, truncated, occluded, alpha,xc, yc, w, h, H_diff, W_diff, L_diff, X, Y, Z, ry, H_ave, W_ave, L_ave)
        # to
        # (type_id, truncated, occluded, alpha, xc, yc, w, h, H_diff, W_diff, L_diff, X, Y, Z, ry, H_ave, W_ave, L_ave, theta_ray, cos, sin, confidence)
        labels_tmp = []
        for label, theta, orientation, confidence in zip(labels, theta_rays, orientations, confidences):
            labels_tmp.append(
                np.concatenate((label, theta, orientation.reshape(-1, self.bins*2), confidence), axis=1)
            )

        labels_final = []
        for idx, label in enumerate(labels_tmp):
            tmp = label[:, 1:4]
            label = np.delete(label, [1, 2, 3], axis=1)
            labels_final.append(np.concatenate((label, tmp), axis=1))


        return tuple(labels_final)


        # return labels tuple(ndarray)


class LoadData:
    def __init__(self, path):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '**/*.*'), recursive=True))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise FileNotFoundError(f'Invalid path {p}')
        imgp = [i for i in files if i.split('.')[-1] in IMG_FORMATS]
        vidp = [v for v in files if v.split('.')[-1] in VID_FORMATS]
        self.files = imgp + vidp
        self.nf = len(self.files)
        self.type = 'image'
        if any(vidp):
            self.add_video(vidp[0])  # new video
        else:
            self.cap = None
    @staticmethod
    def checkext(path):
        file_type = 'image' if path.split('.')[-1].lower() in IMG_FORMATS else 'video'
        return file_type
    def __iter__(self):
        self.count = 0
        return self
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.checkext(path) == 'video':
            self.type = 'video'
            ret_val, img = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.add_video(path)
                ret_val, img = self.cap.read()
        else:
            # Read image
            self.count += 1
            img = cv2.imread(path)  # BGR
        return img, path, self.cap
    def add_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def __len__(self):
        return self.nf  # number of files
