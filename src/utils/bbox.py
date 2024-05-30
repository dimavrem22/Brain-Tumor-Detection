import math
import torch
import numpy as np
from torchvision.ops import box_iou


def center_to_corners_bbox(bboxes):
    # Assuming bboxes is a torch tensor of shape [N, 4]
    corners_bboxes = torch.zeros_like(bboxes)
    corners_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x_min
    corners_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # x_max
    corners_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y_min
    corners_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # y_max
    return corners_bboxes


def coco_to_corners_bbox(bboxes):
    # Assuming bboxes is a torch tensor of shape [N, 4]
    corners_bboxes = torch.zeros_like(bboxes)
    corners_bboxes[:, 0] = bboxes[:, 0]  # x_min stays the same
    corners_bboxes[:, 1] = bboxes[:, 1]  # y_min stays the same
    corners_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x_max = x_min + width
    corners_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y_max = y_min + height
    return corners_bboxes


def coco_to_center_bbox(bboxes):
    # Assuming bboxes is a torch tensor of shape [N, 4]
    center_bboxes = torch.zeros_like(bboxes)
    center_bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2  # center_x=x_min+width/2
    center_bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2  # center_y =y_min+height/2
    center_bboxes[:, 2] = bboxes[:, 2]  # width stays the same
    center_bboxes[:, 3] = bboxes[:, 3]  # height stays the same
    return center_bboxes


def scale_anchors(unit_anchors, width, height):
    """
    Change the size of a unit anchor to specified dimentions.
    """
    anchors = unit_anchors.copy()
    anchors[:[0, 2]] *= width
    anchors[:[1, 3]] *= height
    return anchors


def generate_anchors(image_size, scales, aspect_ratios, feature_map_sizes):
    """
    Generate anchor boxes necessary for object detection.
    """

    anchor_boxes_centers = []  # For center coordinates
    anchor_boxes_corners = []  # For corner coordinates

    for feature_map_size in feature_map_sizes:

        step_size = image_size / feature_map_size  # Size of one grid cell

        for x in range(feature_map_size):
            for y in range(feature_map_size):
                center_x = (x + 0.5) * step_size
                center_y = (y + 0.5) * step_size

                for scale in scales:
                    for aspect_ratio in aspect_ratios:
                        box_height = image_size * scale / np.sqrt(aspect_ratio)
                        box_width = image_size * scale * np.sqrt(aspect_ratio)

                        # Center coordinates with width and height
                        anchor_boxes_centers.append(
                            [center_x, center_y, box_width, box_height]
                        )

                        # Convert to corner coordinates
                        x_min = center_x - box_width / 2
                        y_min = center_y - box_height / 2
                        x_max = center_x + box_width / 2
                        y_max = center_y + box_height / 2
                        anchor_boxes_corners.append([x_min, y_min, x_max, y_max])

    return np.array(anchor_boxes_centers), np.array(anchor_boxes_corners)


def non_maximum_supression(preds, bbox_centers, threshold, min_prob):
    """
    Method used to condense overlapping bounding boxes into those with greatest confidence.
    """

    # bbox: center to corner representation
    bbox_corners = center_to_corners_bbox(bbox_centers)

    # iou calculation
    ious = box_iou(bbox_corners)

    # highest to lowest confidence
    indices = sorted(range(len(preds)), key=lambda i: preds[i], reverse=True)

    # removing boxes below min threshold
    indices = [idx for idx in indices if preds[idx] >= min_prob]

    # non_max_supression
    selected_indices = []
    while indices:
        # Select the box with the highest score and remove it from the list
        current_index = indices.pop(0)
        selected_indices.append(current_index)

        boxes_to_keep = []
        for index in indices:
            # Calculate IoU between the current box and other boxes
            iou = ious(ious[current_index, index])
            # Keep the box if IoU is below the threshold
            if iou < threshold:
                boxes_to_keep.append(index)

        # Update the list of remaining indices
        indices = boxes_to_keep

    return selected_indices


def calculate_anchor_based_dataset_iou(bboxes, predictions, scores):
    """
    Calculates the average IoU for the list of samples between most confident prediction and ground truth.

    bboxes: actual boxes in format center, width, height
    predictions: predicted bounding boxes
    scores: class_pred_bit of each predicted box
    """

    ious = []

    for (
        bbox,
        score,
        preds,
    ) in zip(bboxes, scores, predictions):

        # get most confident box
        best_box_idx = np.argmax(score)

        best_pred_bbox = torch.tensor(preds[best_box_idx]).view((1, 4))
        bbox = torch.tensor(bbox).view((1, 4))

        best_pred_bbox = center_to_corners_bbox(best_pred_bbox)
        bbox = center_to_corners_bbox(bbox)

        iou = box_iou(best_pred_bbox, bbox)[0, 0].item()

        ious.append(iou)

    return np.mean(ious)


def calculate_single_bbox_iou_values(true_bboxes, pred_bboxes, bbox_format="center"):

    iou_values = []

    for true_bbox, pred_bbox in zip(true_bboxes, pred_bboxes):

        if bbox_format == "center":
            true_bbox = convert_to_corners(true_bbox)
            pred_bbox = convert_to_corners(pred_bbox)

        # Determine the coordinates of the intersection rectangle
        x_left = max(true_bbox[0], pred_bbox[0])
        y_top = max(true_bbox[1], pred_bbox[1])
        x_right = min(true_bbox[2], pred_bbox[2])
        y_bottom = min(true_bbox[3], pred_bbox[3])

        # Check if there is an intersection
        if x_right < x_left or y_bottom < y_top:
            iou_values.append(0)
            continue

        # Calculate the area of the intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate the area of both bounding boxes
        true_bbox_area = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
        pred_bbox_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])

        # Calculate the union area
        union_area = true_bbox_area + pred_bbox_area - intersection_area

        # Calculate the IoU
        iou = intersection_area / union_area

        iou_values.append(iou)

    return iou_values


def convert_to_corners(center_bbox):

    center_x = center_bbox[0]
    center_y = center_bbox[1]
    width = center_bbox[2]
    height = center_bbox[3]

    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    return [x1, y1, x2, y2]


def rotate_point(x, y, x_c, y_c, angle_degrees):

    # Convert angle from degrees to radians
    angle_radians = math.radians(angle_degrees)

    # Step 1: Translate point to the origin
    px_prime = x - x_c
    py_prime = y - y_c

    # Step 2: Rotate point
    px_double_prime = px_prime * math.cos(angle_radians) - py_prime * math.sin(angle_radians)
    py_double_prime = px_prime * math.sin(angle_radians) + py_prime * math.cos(angle_radians)

    # Step 3: Translate point back
    xn = px_double_prime + x_c
    yn = py_double_prime + y_c

    return xn, yn