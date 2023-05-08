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

# class_names = ["Pedestrian", "Truck", "Car", "Cyclist", "Misc"]
class_names = ['trafficcone', 'tricyclist', 'van', 'cyclist', 'unknowns_movable', 'car', 'pedestrian',
               'unknown_unmovable', 'bus', 'truck', 'barrow', 'motorcyclist']
class_ids = {}
for class_id, class_name in enumerate(class_names):
    class_ids[class_name] = float(class_id)

convert_type = "MONO_2D"

bins, overlap = 2, 0.1

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

def attributes_3d_preprocess(img_paths, labels, task, imgs_path):
    """
    Args:
        img_paths (tuple(str)): all img paths
        labels (tuple(ndarray)): all labels (type_id, truncated, occluded, alpha, x1, y1, x2, y2, H, W, L, X, Y, Z, ry)
    Return:
        labels (tuple(ndarray)): propcessed all labels (type_id, truncated, occluded, alpha, cx, cy, w, h, H_diff, W_diff, L_diff, X, Y, Z, ry, bin_conf, bin_cos_sin)
    """
    # read camera calib matrix
    # self.proj_matrix tuple(np.matrix)

    # calib_paths = tuple([osp.join(osp.dirname(im).replace("images", "calibs"), osp.basename(im).replace("jpg", "txt")) \
    #                      for im in img_paths])
    # def get_calib(paths):
    #     calib_list = []
    #     print("calib matrix ...")
    #     for p in tqdm(paths):
    #         with open(p, 'r')as f:
    #             parse_file = f.read().strip().splitlines()
    #             for line in parse_file:
    #                 if line is not None and line.split()[0] == "P2:":
    #                     calib_matrix = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
    #                     calib_list.append(calib_matrix)
    #     return tuple(calib_list)
    #
    # proj_matrix = get_calib(calib_paths)

    # ave HWL
    average_dims = {}
    for i in range(len(class_names)):
        if i not in average_dims.keys():
            average_dims[i] = {}
        average_dims[i]["total"] = np.zeros(3)
        average_dims[i]["count"] = 0
        average_dims[i]["hwl"] = []

    print("average_dims ...")
    for label in tqdm(labels):
        for i in range(len(class_names)):
            average_dims[i]["total"] += label[label[:, 0] == i][:, 8:11].sum(axis=0)
            average_dims[i]["count"] += (label[:, 0] == i).sum()
            average_dims[i]["hwl"].extend(label[label[:, 0] == i][:, 8:11])

    for i in range(len(class_names)):
        average_dims[i]["ave"] = average_dims[i]["total"] / (average_dims[i]["count"] + 1e-6)
        average_dims[i]["hwl"] = np.array(average_dims[i]["hwl"])

    dump_ave_HWL = np.zeros((len(class_names), 4))
    for i in range(len(class_names)):
        dump_ave_HWL[i, 0] = i
        dump_ave_HWL[i, 1:] = average_dims[i]["ave"]

    np.savetxt(osp.join(osp.dirname(imgs_path), f"{task}_ave_HWL.txt"), dump_ave_HWL, delimiter=" ", fmt='%.08f')


    labels = list(labels)
    # (type_id, truncated, occluded, alpha, x1, y1, x2, y2, H, W, L, X, Y, Z, ry)
    # to
    # (type_id, truncated, occluded, alpha, x1, y1, x2, y2, H_diff, W_diff, L_diff, X, Y, Z, ry)
    for idx, label in enumerate(labels):
        ave_dims = np.zeros((label.shape[0], 3))
        for i in range(len(label)):
            label[i, 8:11] -= average_dims[label[i, 0]]["ave"]
            # ave_dims[i, :] = average_dims[label[i, 0]]["ave"]
        # labels[idx] = np.concatenate((label, ave_dims), axis=1)


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

    # theta_rays = []
    # print("theta ray ...")
    # for i, label in tqdm(enumerate(labels)):
    #     theta = np.zeros((label.shape[0], 1))
    #     H, W = cv2.imread(img_paths[i]).shape[:2]
    #     for idx in range(len(label)):
    #         theta[idx, 0] = calc_theta_ray(W, label[idx, 4:8], proj_matrix[i])
    #     theta_rays.append(theta)

    # (type_id, truncated, occluded, alpha, x1, y1, x2, y2, H_diff, W_diff, L_diff, X, Y, Z, ry)
    # to
    # (type_id, truncated, occluded, alpha, xc, yc, w, h, H_diff, W_diff, L_diff, X, Y, Z, ry)
    print("xyxy -> cxcywh ...")
    for i, label in tqdm(enumerate(labels)):
        # xyxy-> cxcywh
        H, W = cv2.imread(img_paths[i]).shape[:2]
        label[:, 6] = (label[:, 6] - label[:, 4]) / W
        label[:, 7] = (label[:, 7] - label[:, 5]) / H
        label[:, 4] = label[:, 4] / W + label[:, 6] / 2
        label[:, 5] = label[:, 5] / H + label[:, 7] / 2

    # (type_id, truncated, occluded, alpha,xc, yc, w, h, H_diff, W_diff, L_diff, X, Y, Z, ry)
    # to
    # (type_id, truncated, occluded, alpha, xc, yc, w, h, H_diff, W_diff, L_diff, X, Y, Z, ry, cos, sin, confidence)
    labels_tmp = []
    for label, orientation, confidence in zip(labels, orientations, confidences):
        labels_tmp.append(
            np.concatenate((label, orientation.reshape(-1, bins*2), confidence), axis=1)
        )

    # (type_id, truncated, occluded, alpha, xc, yc, w, h, H_diff, W_diff, L_diff, X, Y, Z, ry, cos, sin, confidence)
    # to
    # # (type_id, xc, yc, w, h, H_diff, W_diff, L_diff, X, Y, Z, ry, cos, sin, confidence, truncated, occluded, alpha)
    labels_final = []
    for idx, label in enumerate(labels_tmp):
        tmp = label[:, 1:4]
        label = np.delete(label, [1, 2, 3], axis=1)
        labels_final.append(np.concatenate((label, tmp), axis=1))


    return labels_final


def main(args):
    TASK = ["train", "val"]
    root = args.rope3d_path

    for task in TASK:
        raw_labels_path = f"{root}/{task}"
        imgs_path = raw_labels_path.replace("labels_raw", "images")
        new_labels_path = raw_labels_path.replace("labels_raw", f"labels_yolo_{convert_type}")
        if os.path.exists(new_labels_path):
            shutil.rmtree(new_labels_path)
        os.makedirs(new_labels_path)

        raw_labels_list = sorted(os.listdir(raw_labels_path))
        if convert_type == "MONO_2D":
            print(f"{convert_type}")
            for raw_label in tqdm(raw_labels_list):
                raw_label_path = os.path.join(raw_labels_path, raw_label)
                convert_label2yolo(raw_label_path, new_labels_path, raw_label, imgs_path)

        if convert_type == "MONO_3D":
            # 1. 将所有图像和labels保存在list中
            # 2. 使用数据处理代码处理 list(image_paths)和list(labels)
            labels = []
            image_paths = []
            print("read labels ...")
            for raw_label in tqdm(raw_labels_list):
                raw_label_path = os.path.join(raw_labels_path, raw_label)
                with open(raw_label_path, 'r') as f:
                    label = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]

                    for i, lb in enumerate(label):
                        label[i][0] = class_ids[lb[0]]

                    labels.append(np.array(label, dtype=np.float32))
                    image_paths.append(os.path.join(imgs_path, raw_label.replace("txt", "jpg")))
            labels = attributes_3d_preprocess(image_paths, labels, task, imgs_path)
            print("saving ...")
            for im_path, lb in tqdm(zip(image_paths, labels)):
                lb_name = osp.basename(im_path).replace("jpg", "txt")
                np.savetxt(osp.join(new_labels_path, lb_name), lb, delimiter=" ", fmt='%.08f')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rope3d_path', default='/share/wuweiguan_dataset/Rope3D/labels_raw')
    args = parser.parse_args()
    print(args)

    main(args)