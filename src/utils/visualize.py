import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
