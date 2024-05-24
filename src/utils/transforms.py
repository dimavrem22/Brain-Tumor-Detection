import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.ops import box_iou
import numpy as np
import torch
from src.utils.bbox import (
    center_to_corners_bbox,
    coco_to_corners_bbox,
    coco_to_center_bbox,
)


class ImgOnlyTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, label):
        return self.transform(img), label


class BBoxCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox):
        for t in self.transforms:
            image, bbox = t(image, bbox)
        return image, bbox


class BBoxBaseTransform:

    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, image, bbox):
        image = self.transform(image)
        return image, bbox


class BBoxResize:

    def __init__(self, size, interpolation=TF.InterpolationMode.BILINEAR):
        self.size = size
        self.width, self.height = self.size
        self.interpolation = interpolation

    def __call__(self, image, bbox):
        original_width = image.shape[2]
        original_height = image.shape[1]

        image = TF.resize(image, self.size, self.interpolation)

        bbox = bbox.copy()

        bbox[[0, 2]] = bbox[[0, 2]] * (self.width / original_width)
        bbox[[1, 3]] = bbox[[1, 3]] * (self.height / original_height)

        return image, bbox


class BBoxCocoToCenterNotation:

    def __call__(self, image, bbox):

        # [x_min, y_min, width, height] -> [x_center, y_center, width, height]
        return image, [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]]


class BBoxCocoToCornerNotation:

    def __call__(self, image, bbox):

        # [x_min, y_min, width, height] -> [x_min, y_min, x_max, y_max]
        return image, [bbox[0], bbox[1], bbox[0] + bbox[2],  bbox[1] + bbox[3]]


class BBoxAnchorEncode:

    def __init__(self, anchors, positive_iou_threshold=0.5, min_positive_iou=0.3):

        self.positive_iou_threshold = positive_iou_threshold
        self.min_positive_iou = min_positive_iou

        # assuming anchors are in format (center_x, center_y, width, height) normalized to 1
        self.anchors = torch.tensor(np.array(anchors), requires_grad=False)
        self.anchor_corners = center_to_corners_bbox(self.anchors)

    def __call__(self, image, bbox):

        bbox = np.round(bbox, 1)
        bbox_corners = coco_to_corners_bbox(
            torch.tensor(np.array([bbox]), requires_grad=False)
        )
        bbox_center = coco_to_center_bbox(
            torch.tensor(np.array([bbox]), requires_grad=False)
        )

        IoUs = box_iou(bbox_corners, self.anchor_corners).view(-1)

        labels = IoUs >= self.positive_iou_threshold

        if labels.sum() == 0:
            max_iou_anchor_idx = torch.argmax(IoUs)
            if IoUs[max_iou_anchor_idx] >= self.min_positive_iou:
                labels[max_iou_anchor_idx] = True

        # targets: adjustments to center point and width, height
        targets = torch.full((self.anchors.shape), torch.nan, dtype=torch.float)
        adjustment = (-1 * (self.anchors - bbox_center)).to(torch.float)
        targets[labels] = adjustment[labels]

        # This is the class of the anchor (tumor or no tumor)
        labels = labels.to(torch.float32)

        return image, (labels, targets, bbox_center[0])
