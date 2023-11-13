# 读取labels的路径
# 将label.txt逐个打开
# 转换
# 保存为新的txt,在新路径下

import argparse
from tqdm import tqdm
import os
import shutil
import numpy as np
import os.path as osp
import cv2
import traceback
from multiprocessing.pool import Pool

from yolov6.utils.show_2d3d_box import Data, detect_data, \
    read_kitti_ext, read_kitti_cal, get_camera_3d_8points_g2c, \
    project_3d_world, color_list

# class_names = ["Pedestrian", "Truck", "Car", "Cyclist", "Misc"]
class_names = ['pedestrian', 'cyclist', 'car', 'big_vehicle']
class_ids = {}
for class_id, class_name in enumerate(class_names):
    class_ids[class_name] = float(class_id)

NUM_THREADS = min(48, os.cpu_count())

bins, overlap = 2, 0.1

def get_keypoint(args):
    label, img_path, img_shape = args
    import copy
    label_cpoy = copy.deepcopy(label)
    # img = cv2.imread(img_path)
    result = detect_data(label_cpoy, class_names)
    name = os.path.basename(img_path)
    calib_file = img_path.replace("images", "calibs").replace("jpg", "txt")
    p2 = read_kitti_cal(calib_file)

    extrinsic_file = img_path.replace("images", "extrinsics").replace("jpg", "yaml")
    world2camera = read_kitti_ext(extrinsic_file).reshape((4, 4))
    camera2world = np.linalg.inv(world2camera).reshape(4, 4)

    keypoints = []
    for result_index in range(len(result)):
        t = result[result_index]
        cam_bottom_center = [t.X, t.Y, t.Z]  # bottom center in Camera coordinate

        bottom_center_in_world = camera2world * np.matrix(cam_bottom_center + [1.0]).T
        verts3d = project_3d_world(p2, bottom_center_in_world, t.w, t.h, t.l, t.yaw, camera2world)

        if verts3d is None:
            continue
        verts3d = verts3d.astype(int)
        keypoint = (verts3d[3] + verts3d[2]) / 2
        keypoint = keypoint / img_shape[::-1]
        keypoints.append(keypoint)

    return np.asarray(keypoints)


def convert_label2yolo(raw_label_path, new_labels_path, raw_label, imgs_path):
    with open(raw_label_path, 'r') as f:
        labels = [
            x.split() for x in f.read().strip().splitlines() if len(x)
        ]

        import copy

        labels_copy = copy.deepcopy(labels)
        for i, lb in enumerate(labels_copy):
            labels[i][0] = class_ids[lb[0]]

        labels = np.array(labels, dtype=np.float32)
        # 在这里转换kitti数据集的格式 [label, xmin, ymin, xmax, ymax] -> [label, x_center, y_center, w, h] * (W, H)
        # 将类别名称转换成类别标签
        # 提取[label[0], label[4], label[5], label[6], label[7]]
        labels = labels[:, [0, 4, 5, 6, 7]]
        import cv2
        img_path = os.path.join(imgs_path, raw_label.replace("txt", "jpg"))
        H, W = cv2.imread(img_path).shape[:2]

        labels[:, 3] = (labels[:, 3] - labels[:, 1]) / W
        labels[:, 4] = (labels[:, 4] - labels[:, 2]) / H
        labels[:, 1] = labels[:, 1] / W + labels[:, 3] / 2
        labels[:, 2] = labels[:, 2] / H + labels[:, 4] / 2

        np.savetxt(os.path.join(new_labels_path, raw_label), np.around(labels, 6), delimiter=" ")

def get_image_shape(image_path):
    return cv2.imread(image_path).shape[:2]

def attributes_3d_keypoint_preprocess(img_paths, labels, save_keypoint):
    """
    Args:
        img_paths (tuple(str)): all img paths
        labels (tuple(ndarray)): all labels (type_id, truncated, occluded, alpha, x1, y1, x2, y2, H, W, L, X, Y, Z, ry)
    Return:
        labels (tuple(ndarray)): propcessed all labels (type_id, cx, cy, w, h, H, W, L, X, Y, Z, ry, bin_conf, bin_cos_sin, keypoint_x, keypoint_y, truncated, occluded, alpha, )
    """

    # angle_bins
    interval = 2 * np.pi / bins
    angle_bins = np.zeros(bins)
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    # bin_ranges for confidence
    # [(min angle in bin, max angle in bin), ...]
    bin_ranges = np.zeros((bins, 2))
    for i in range(0, bins):
        bin_ranges[i, 0] = (i * interval - overlap) % (2 * np.pi)
        bin_ranges[i, 1] = (i * interval + interval + overlap) % (2 * np.pi)

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
    print("orient and conf ...")
    for i in tqdm(range(len(labels))):
        nl = labels[i].shape[0]
        orientation = np.zeros((nl, bins, 2))
        confidence  = np.zeros((nl, bins))
        angles = (labels[i][:, 3] + np.pi).reshape(nl, 1)
        for an_id, angle in enumerate(angles):
            bin_idxs = get_bin(angle[0], bin_ranges)
            for bin_idx in bin_idxs:
                angle_diff = angle - angle_bins[bin_idx]
                orientation[an_id, bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)]).squeeze(axis=1)
                confidence[an_id, bin_idx] = 1

        orientations.append(orientation)
        confidences.append(confidence)
    print("image shape ...")
    imgs_shape = []
    with Pool(NUM_THREADS) as pool:
        pbar = pool.imap(get_image_shape, img_paths)
        pbar = tqdm(pbar, total=len(img_paths))
        for img_shape in pbar:
            imgs_shape.append(img_shape)
    pbar.close()

    # bbcp keypoint
    print("get bbcp keypoints ...")
    keypoints = []
    with Pool(NUM_THREADS) as pool:
        pbar = pool.imap(get_keypoint, zip(labels, img_paths, imgs_shape))
        pbar = tqdm(pbar, total=len(labels))
        for keypoint in pbar:
            keypoints.append(keypoint)
    pbar.close()

    # (type_id, truncated, occluded, alpha, x1, y1, x2, y2, H, W, L, X, Y, Z, ry)
    # to
    # (type_id, truncated, occluded, alpha, xc, yc, w, h, H, W, L, X, Y, Z, ry)

    print("xyxy -> cxcywh ...")
    for i, label in tqdm(enumerate(labels)):
        # xyxy-> cxcywh
        H, W = imgs_shape[i]
        label[:, 6] = (label[:, 6] - label[:, 4]) / W
        label[:, 7] = (label[:, 7] - label[:, 5]) / H
        label[:, 4] = label[:, 4] / W + label[:, 6] / 2
        label[:, 5] = label[:, 5] / H + label[:, 7] / 2

    # (type_id, truncated, occluded, alpha,xc, yc, w, h, H, W, L, X, Y, Z, ry)
    # to
    # (type_id, truncated, occluded, alpha, xc, yc, w, h, H, W, L, X, Y, Z, ry, cos, sin, confidence, bbcp_x, bbcp_y)
    labels_tmp = []
    if save_keypoint:
        for label, orientation, confidence, keypoint in zip(labels, orientations, confidences, keypoints):
            labels_tmp.append(
                np.concatenate((label, orientation.reshape(-1, bins*2), confidence, keypoint), axis=1)
            )
    else:
        for label, orientation, confidence in zip(labels, orientations, confidences):
            labels_tmp.append(
                np.concatenate((label, orientation.reshape(-1, bins*2), confidence), axis=1)
            )

    # (type_id, truncated, occluded, alpha, xc, yc, w, h, H, W, L, X, Y, Z, ry, cos, sin, confidence, bbcp_x, bbcp_y)
    # to
    # # (type_id, xc, yc, w, h, H, W, L, X, Y, Z, ry, cos, sin, confidence, bbcp_x, bbcp_y, truncated, occluded, alpha)
    labels_final = []
    for idx, label in enumerate(labels_tmp):
        tmp = label[:, 1:4]
        label = np.delete(label, [1, 2, 3], axis=1)
        labels_final.append(np.concatenate((label, tmp), axis=1))


    return labels_final

fine2coarse = {}
fine2coarse['van'] = 'car'
fine2coarse['car'] = 'car'
fine2coarse['bus'] = 'big_vehicle'
fine2coarse['truck'] = 'big_vehicle'
fine2coarse['cyclist'] = 'cyclist'
fine2coarse['motorcyclist'] = 'cyclist'
fine2coarse['tricyclist'] = 'cyclist'
fine2coarse['pedestrian'] = 'pedestrian'
fine2coarse['barrow'] = 'pedestrian'

def read_labels(args):
    raw_path, raw_label = args
    raw_label_path = os.path.join(raw_path, raw_label)
    with open(raw_label_path, 'r') as f:
        label = [
            x.split() for x in f.read().strip().splitlines() if len(x)
        ]

        label_new = []
        for i, lb in enumerate(label):
            if lb[0] in fine2coarse.keys():
                lb[0] = class_ids[fine2coarse[lb[0]]]
                label_new.append(lb)

    return np.array(label_new, dtype=np.float32), os.path.basename(raw_label_path).replace("txt", "jpg")

def main(args):
    TASK = args.task
    root = args.rope3d_path
    convert_type = args.convert_type

    for task in TASK:
        print(f"{task} ...")
        raw_labels_dir = f"{root}/{task}"
        imgs_dir = raw_labels_dir.replace("labels_raw", "images")
        new_labels_path = raw_labels_dir.replace("labels_raw", f"labels_yolo_{convert_type}")
        if os.path.exists(new_labels_path):
            shutil.rmtree(new_labels_path)
        os.makedirs(new_labels_path)

        raw_labels_list = sorted(os.listdir(raw_labels_dir))
        # raw_labels_list = [os.path.join(raw_labels_dir, i) for i in raw_labels_list]
        raw_labels_dir = [raw_labels_dir]*len(raw_labels_list)
        if convert_type == "MONO_2D":
            print(f"{convert_type}")
            for raw_label in tqdm(raw_labels_list):
                raw_label_path = os.path.join(raw_labels_dir, raw_label)
                convert_label2yolo(raw_label_path, new_labels_path, raw_label, imgs_dir)


        if convert_type == "MONO_2.5D_KEYPOINT":
            # 1. 将所有图像和labels保存在list中
            # 2. 使用数据处理代码处理 list(image_paths)和list(labels)
            print(f"convert type {convert_type} ...")
            labels = []
            image_paths = []
            print("read labels ...")
            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(read_labels, zip(raw_labels_dir, raw_labels_list))
                pbar = tqdm(pbar, total=len(raw_labels_list))
                for label, image_path in pbar:
                    labels.append(label)
                    image_paths.append(os.path.join(imgs_dir, image_path))
            pbar.close()

            print(f"{convert_type} ...")
            save_keypoint = True
            labels = attributes_3d_keypoint_preprocess(image_paths, labels, save_keypoint)

            print("saving ...")
            for im_path, lb in tqdm(zip(image_paths, labels)):
                lb_name = osp.basename(im_path).replace("jpg", "txt")
                np.savetxt(osp.join(new_labels_path, lb_name), lb, delimiter=" ", fmt='%.08f')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rope3d_path', default='data/Rope3D/labels_raw')
    # parser.add_argument('--rope3d_path',default='data/mini_rope3d/labels_raw')
    parser.add_argument('--convert_type', default="MONO_2.5D_KEYPOINT")
    parser.add_argument('--task', default=["train", "val"])
    args = parser.parse_args()
    print(args)

    main(args)