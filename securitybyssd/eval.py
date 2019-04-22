import torch
import os
import numpy as np
from pkg.utils import box_utils
from pkg.dataloader import openimagesloader
from pkg.predictor import *
from pkg.vggssd import *
from pkg.config import vgg_ssd_config as config

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
DATASET_DIRECTORY = 'data/open_images'
LABELS_DIRECTORY = 'checkpoint/open-images-model-labels.txt'
EVAL_DIRECTORY = 'evalresults'
MODEL = 'checkpoint/v2-Epoch-39-Loss-7.86013218334743.pth'
NMS = 'hard'
IOU_THRESHOLD = 0.5
USE_VOC_METRICS = False

def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = all_gt_boxes[class_index][image_id].clone().detach() 
    return true_case_stat, all_gt_boxes, all_difficult_cases

def compute_average_precision(precision, recall):
    """
    It computes average precision based on the definition of Pascal Competition. It computes the under curve area
    of precision and recall. Recall follows the normal definition. Precision is a variant.
    pascal_precision[i] = typical_precision[i:].max()
    """
    # identical but faster version of new_precision[i] = old_precision[i:].max()
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute under curve area
    areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]
    return areas.sum()


def compute_voc2007_average_precision(precision, recall):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap

def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_voc_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_voc_metric:
        return compute_voc2007_average_precision(precision, recall)
    else:
        return compute_average_precision(precision, recall)

if __name__ == '__main__':
    class_names = [name.strip() for name in open(LABELS_DIRECTORY).readlines()]

    dataset = openimagesloader.OpenImageData(DATASET_DIRECTORY, train_val_test="test")

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    net = VGGSSD(len(class_names), device, config=config, is_test=True)

    net.load(MODEL)
    net = net.to(device)
    print('Model Loaded')
    predictor = Predictor(net, device, config.image_size, config.image_mean, nms_method=NMS)

    results = []
    for i in range(len(dataset)):
        print("process image", i)
        image = dataset.get_image(i)
        print("Load Image")
        boxes, labels, probs = predictor.predict(image)
        print("Prediction")
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = '{}/det_test_{}.txt'.format(EVAL_DIRECTORY, class_name)
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = '{}/det_test_{}.txt'.format(EVAL_DIRECTORY, class_name)
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            IOU_THRESHOLD,
            USE_VOC_METRICS
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")

    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")