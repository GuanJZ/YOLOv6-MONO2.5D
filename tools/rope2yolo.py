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


# class_names = ["Pedestrian", "Truck", "Car", "Cyclist", "Misc"]
class_names = ['pedestrian', 'cyclist', 'car', 'big_vehicle']
class_ids = {}
for class_id, class_name in enumerate(class_names):
    class_ids[class_name] = float(class_id)

NUM_THREADS = min(48, os.cpu_count())

bins, overlap = 2, 0.1


def convert_label2yolo(args):
    label, image_path, shape, new_labels_path = args
    H, W = shape[0], shape[1]
    label = label[:, [0, 4, 5, 6, 7]]
    label[:, 3] = (label[:, 3] - label[:, 1]) / W
    label[:, 4] = (label[:, 4] - label[:, 2]) / H
    label[:, 1] = label[:, 1] / W + label[:, 3] / 2
    label[:, 2] = label[:, 2] / H + label[:, 4] / 2

    label_file_path = os.path.basename(image_path).replace("jpg", "txt")
    np.savetxt(os.path.join(new_labels_path, label_file_path), np.around(label, 6), delimiter=" ")

    return True

def get_image_shape(image_path):
    return cv2.imread(image_path).shape[:2]


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

    img_path = raw_label_path.replace("labels_raw", "images").replace("txt", "jpg")
    H, W = cv2.imread(img_path).shape[:2]

    return np.array(label_new, dtype=np.float32), os.path.basename(raw_label_path).replace("txt", "jpg"), (H, W)

def main(args):
    TASK = args.task
    root = args.rope3d_path
    convert_type = args.convert_type

    for task in TASK:
        print(f"task: {task} ...")
        raw_labels_dir = f"{root}/{task}"
        imgs_dir = raw_labels_dir.replace("labels_raw", "images")
        new_labels_path = raw_labels_dir.replace("labels_raw", f"labels_yolo_{convert_type}")
        if os.path.exists(new_labels_path):
            shutil.rmtree(new_labels_path)
        os.makedirs(new_labels_path)

        print("list labels ...")
        raw_labels_list = sorted(os.listdir(raw_labels_dir))
        # raw_labels_list = [os.path.join(raw_labels_dir, i) for i in raw_labels_list]
        raw_labels_dir = [raw_labels_dir]*len(raw_labels_list)
        if convert_type == "MONO_2D":
            labels = []
            image_paths = []
            shapes = []
            print("read labels ...")
            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(read_labels, zip(raw_labels_dir, raw_labels_list))
                pbar = tqdm(pbar, total=len(raw_labels_list))
                for label, image_path, shape in pbar:
                    labels.append(label)
                    image_paths.append(os.path.join(imgs_dir, image_path))
                    shapes.append(shape)
            pbar.close()
            print(f"{convert_type}")
            new_labels_paths = [new_labels_path]*len(labels)
            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(convert_label2yolo, zip(labels, image_paths, shapes, new_labels_paths))
                pbar = tqdm(pbar, total=len(labels))
                for is_convert in pbar:
                    # np.savetxt(os.path.join(new_labels_path, label_file_path), np.around(label, 6), delimiter=" ")
                    if not is_convert:
                        print("failed")
            pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rope3d_path', default='data/mini_rope3d/labels_raw')
    # parser.add_argument('--rope3d_path',default='data/Rope3D/labels_raw')
    parser.add_argument('--convert_type', default="MONO_2D")
    parser.add_argument('--task', default=["train", "val"])
    args = parser.parse_args()
    print(args)

    main(args)