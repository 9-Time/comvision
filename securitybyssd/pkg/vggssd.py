import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Sequential, ModuleList, LeakyReLU, BatchNorm2d, ReLU
from .config import vgg_ssd_config as config
from .utils import box_utils
import numpy as np

from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #


class VGGSSD(nn.Module):
    def __init__(self, num_classes, device, is_test=False, config=None):
        super(VGGSSD, self).__init__()

        vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                  512, 512, 512]

        self.num_classes = num_classes
        self.device = device
        
        self.base_net = ModuleList(self.vgg(vgg_config))

        self.source_layer_indexes = [
            (23, BatchNorm2d(512)),
            len(self.base_net),
        ]
        self.extras = ModuleList([
            Sequential(
                Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                ReLU()
            )
        ])

        self.classification_headers = ModuleList([
            Conv2d(in_channels=512, out_channels=4 * num_classes, kernel_size=3, padding=1),
            Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
            Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
            Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=1, padding=0), 
        ])

        self.regression_headers = ModuleList([
            Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1),
            Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
            Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
            Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=1, padding=0), 
        ])

        self.is_test = is_test
        self.config = config
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in self.source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        self.priors = config.priors.to(self.device)

    def forward(self, x):
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations
    
    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    # referenced https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
    def vgg(self,cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers

class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)