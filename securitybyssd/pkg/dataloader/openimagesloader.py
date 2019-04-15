from torch.utils.data import Dataset
import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
import cv2

class OpenImageData(Dataset):
    def __init__(self, filepath, train_val_test= 'train', transform=None, target_transform=None):
        self.filepath = filepath
        self.train_val_test = train_val_test.lower()
        assert self.train_val_test == 'train' or self.train_val_test == 'test' or self.train_val_test == 'validation'
        self.transform = transform
        self.target_transform = target_transform

        annotation_file = f"{self.filepath}/sub-{self.train_val_test}-annotations-bbox.csv"
        annotations = pd.read_csv(annotation_file)
        self.class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.data = []
        for image_id, group in annotations.groupby("ImageID"):
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            labels = np.array([self.class_dict[name] for name in group["ClassName"]])
            self.data.append({
                'image_id': image_id,
                'boxes': boxes,
                'labels': labels
            })
        self.ids = [info['image_id'] for info in self.data]
        self.class_stat = None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)
    
    def getitem(self, idx):
        image_info = self.data[idx]
        image_file = '{}/{}/{}.jpg'.format(self.filepath, self.train_val_test, image_info['image_id'])
        img = cv2.imread(image_file)
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = image_info['boxes']
        boxes[:, 0] *= img.shape[1]
        boxes[:, 1] *= img.shape[0]
        boxes[:, 2] *= img.shape[1]
        boxes[:, 3] *= img.shape[0]
        labels = image_info['labels']
        
        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], img, boxes, labels

    def __getitem__(self, idx):
        _, img, boxes, labels = self.getitem(idx)
        return img, boxes, labels

    def get_annotation(self, idx):
        image_id, image, boxes, labels = self.getitem(idx)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, idx):
        image_info = self.data[idx]
        image_file = '{}/{}/{}.jpg'.format(self.filepath, self.train_val_test, image_info['image_id'])
        img = cv2.imread(image_file)
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img, _ = self.transform(img)
        return img


##### MODULE TESTING #####
# oid = OpenImageData('../../data/open_images')
# print(oid.__getitem__(1))

##### TRANSFORM TESTING #####
# from transformations import *
# train_transform = TrainAugmentation(300)
# test_transform = TestTransform(300)
# train_oid = OpenImageData('../data/open_images', 'train', train_transform)
# test_oid = OpenImageData('../data/open_images', 'train', test_transform)
# print(train_oid.__getitem__(1))
# print(test_oid.__getitem__(1))