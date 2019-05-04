import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels

class ToPIL(object):
    def __init__(self):
        self.topil = transforms.ToPILImage()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im = self.topil(im)
        return im, boxes, labels

class ToTensor(object):
    def __init__(self):
        self.tt = transforms.ToTensor()
    
    def __call__(self, cvimage, boxes=None, labels=None):
        return self.tt(cvimage), boxes, labels

class Normalize:
    def __init__(self):
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    def __call__(self, image, boxes, labels):
        return self.norm(image), boxes, labels

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.cj = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, boxes, labels):
        return self.cj(image), boxes, labels

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        return cv2.resize(image,(self.size, self.size)), boxes, labels

class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, 0] += int(left)
        boxes[:, 1] += int(top)
        boxes[:, 2] += int(left)
        boxes[:, 3] += int(top)

        return image, boxes, labels

class ExpandImage(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels
        height, width, depth = image.shape
        max_left = min(width-self.size, min(boxes[:,0]))
        max_bottom = min(height-self.size, min(boxes[:,2]))
        left = int(random.uniform(0, max_left))
        bottom = int(random.uniform(0, max_bottom))
        expand_image = image[int(bottom):int(height), int(left):int(width)]
        boxes = boxes.copy()
        boxes[:,0] -= left
        boxes[:,1] -= bottom
        boxes[:,2] -= left
        boxes[:,3] -= bottom
        image = expand_image
        return image, boxes, labels

class TrainAugmentation(object):
    def __init__(self, size, mean):
        self.augment = Compose([
            # ExpandImage(size), #commented out due to exploding gradient. trying to debug under experimentation
            # Expand(mean),      #commented out due to exploding gradient. trying to debug under experimentation
            # RandomSampleCrop(),#commented out due to exploding gradient. trying to debug under experimentation
            # RandomMirror(),    #commented out due to exploding gradient. trying to debug under experimentation
            ToPercentCoords(),
            Resize(size),
            ToPIL(),
            ColorJitter(2,2,2,0.5),
            ToTensor(),
            Normalize()
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

class TestTransform(object):
    def __init__(self, size):
        self.augment = Compose([
            ToPercentCoords(),
            Resize(size),
            ToPIL(),
            ToTensor(),
            Normalize()
        ])

    def __call__(self, image, boxes, labels):
        return self.augment(image, boxes, labels)

class PredictionTransform(object):
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            # Simulating under dark conditions
            # ToPIL(),
            # ColorJitter(1,0,0,0),
            ToTensor(),
            Normalize()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image
