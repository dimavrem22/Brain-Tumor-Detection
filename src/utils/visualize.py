
import random
import os
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.bbox import calculate_single_bbox_iou_values

def plot_images_and_bboxes(images: list, true_bboxes: list, pred_bboxes:list = None) -> None:

    if not pred_bboxes:
        pred_bboxes = [None] * len(images)

    assert len(images) == len(true_bboxes), "Number of images and true bboxes should be the same"
    assert len(images) == len(pred_bboxes), "Number of images and prediction bboxes should be the same"

    n_rows = 1
    n_samples = len(images)

    fig, axs = plt.subplots(n_rows, n_samples, figsize=(n_samples * 5, 5 * n_rows))

    for idx, (image, true_bbox, pred_bbox) in enumerate(zip(images, true_bboxes, pred_bboxes)):
        image_ax = axs[idx]
        # Assuming images and masks are Tensors of shape [C, H, W]
        image_ax.imshow(image.permute(1, 2, 0).numpy(), cmap="gray")
        image_ax.set_title("Image", fontsize=12, fontweight="bold")
        image_ax.axis("off")

        x_min, y_min, width, height = true_bbox
        # Draw the bounding box
        rect = mpatches.Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        image_ax.add_patch(rect)

        if pred_bbox is not None:
            x_min, y_min, width, height = true_bbox
            # Draw the bounding box
            rect = mpatches.Rectangle(
                (x_min, y_min), width, height, linewidth=2, edgecolor="b", facecolor="none"
            )
            image_ax.add_patch(rect)

    fig.suptitle("Images and Bounding Boxes", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()



def plot_sample_model_prediction(imgs, true_bboxes, pred_bboxes, save_dir, n_samples=5):

    indices = list(range(len(true_bboxes)))
    ious = calculate_single_bbox_iou_values(true_bboxes, pred_bboxes)

    if n_samples == 'all':
        sampled_indices = indices
    else:
        sampled_indices = random.sample(indices, n_samples)

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # iterate throgh sampled indices
    for i, idx in enumerate(sampled_indices):

        img = imgs[idx]
        true_bbox = true_bboxes[idx]
        pred_bbox = pred_bboxes[idx]
        iou = ious[idx]

        fig, ax = plt.subplots(1)
        ax.set_title(f'Sample {i+1} (IoU = {float(iou):.3})')

        # Show the image
        np_img = np.transpose(np.array(img), (1, 2, 0))
        ax.imshow(np_img)

        # Convert center coordinates to top-left coordinates
        true_bbox_top_left = [true_bbox[0] - true_bbox[2] / 2, true_bbox[1] - true_bbox[3] / 2, true_bbox[2], true_bbox[3]]
        pred_bbox_top_left = [pred_bbox[0] - pred_bbox[2] / 2, pred_bbox[1] - pred_bbox[3] / 2, pred_bbox[2], pred_bbox[3]]

        # Plot the true bounding box
        true_rect = plt.Rectangle((true_bbox_top_left[0], true_bbox_top_left[1]), true_bbox_top_left[2], true_bbox_top_left[3],
                                  linewidth=2, edgecolor='g', facecolor='none', label='True')
        ax.add_patch(true_rect)

        # Plot the predicted bounding box
        pred_rect = plt.Rectangle((pred_bbox_top_left[0], pred_bbox_top_left[1]), pred_bbox_top_left[2], pred_bbox_top_left[3],
                                  linewidth=2, edgecolor='r', facecolor='none', label='Predicted')
        ax.add_patch(pred_rect)

        # Add legend
        ax.legend()

        # Save the plot
        plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'))
        plt.close()
