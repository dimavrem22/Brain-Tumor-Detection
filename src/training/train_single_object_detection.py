from typing import List
from src.data.bbox import BoundingBoxDetectionDataset
import torch
from torch.utils.data import DataLoader
import pathlib
from dataclasses import dataclass, field
from src.models.model_utils import save_checkpoint
from src.models.object_detection.efficientnet import EfficientNet
from src.training.training_utils import get_device
from src.utils.loss_functions import BBoxLoss
from tqdm import tqdm
import numpy as np
import os
import pprint as pp
from sklearn.metrics import roc_auc_score

from src.data.classification import DataSplit
from src.utils.transforms import (
    BBoxBaseTransform,
    BBoxCompose,
    BBoxResize,
    BBoxAnchorEncode,
)

DATASET_BASE_DIR = pathlib.Path(__file__).parent.parent.parent / "datasets"


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 5
    device: torch.device = torch.device("cpu")
    dataset_root_dir: str = DATASET_BASE_DIR
    optimizer: str = "Adam"
    image_size: int = 256
    pretrained_backbone: bool = True
    efficient_net_version: str = "b0"
    predictor_hidden_dims: list = [64, 16]
    save_dir_path: str = None

def main_train_loop(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    anchors: List[np.array],
    loss_fn: torch.nn.Module,
    device: torch.device,
    n_epochs: int,
    configs: TrainingConfig,
):
    model.to(device)

    # keeping track of best model
    tr_log = {
        "tr_loss": [],
        "val_loss": [],
        "val_IoU": [],
        "configs": configs.__dict__,
    }

    for epoch in range(n_epochs):
        with tqdm(
            total=len(train_dataloader) + len(val_dataloader),
            desc=f"Epoch {epoch+1}/{n_epochs}",
            unit="batch",
        ) as pbar:

            # run an epoch of training
            train_loss = train_step(
                model, train_dataloader, loss_fn, optimizer, device, pbar
            )

            # evaluating performance on validation set
            val_loss, val_iou = evaluate(
                model, val_dataloader, loss_fn, device, anchors, pbar
            )

            # keeping track of metrics
            tr_log["tr_loss"].append(float(train_loss))
            tr_log["val_loss"].append(float(val_loss))
            tr_log["val_IoU"].append(float(val_iou))

            # checking if model needs to be saved
            if max(tr_log["val_IoU"]) == tr_log["val_IoU"][-1]:

                print(f"New best val iou.")

                # saving best model if save path is provided
                if configs.save_dir_path:
                    save_checkpoint(model, configs.save_dir_path + "/best_model.pt")

            # saving current model if save path is provided
            if configs.save_dir_path:
                save_checkpoint(model, configs.save_dir_path + "/last_model.pt")

    return tr_log


def train_step(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pbar: tqdm,
):
    """
    Run an epoch of training.
    """

    model.train()
    cumalative_loss = 0
    optimizer.zero_grad()

    for imgs, targets in train_dataloader:

        imgs, labels, targets = imgs.to(device), labels.to(device), targets.to(device)
        model.zero_grad()

        # shape: [batch_size, 4]
        y_hat_targets = model(imgs)

        optimizer.zero_grad()
        loss = loss_fn(y_hat_targets, targets)
        loss.backward()
        optimizer.step()

        cumalative_loss += loss.item()

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phase": "Train"})
        pbar.update()

    return cumalative_loss / len(train_dataloader)


def evaluate(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    anchors: torch.tensor,
    pbar: tqdm,
    use_regression_output=False,
):
    """
    Evaluate model performance on evaluation data.
    """

    scores = []
    true_labels = []
    adjustments = []
    bboxes = []

    model.eval()
    with torch.no_grad():
        cumalative_loss = 0
        for imgs, (labels, targets, true_bboxes) in eval_dataloader:
            imgs, labels, targets = (
                imgs.to(device),
                labels.to(device),
                targets.to(device),
            )
            y_hat_labels_batch, y_hat_targets_batch = model(imgs)
            y_hat_labels_batch = y_hat_labels_batch.view(
                y_hat_labels_batch.shape[0], -1
            )
            loss = loss_fn(y_hat_labels_batch, y_hat_targets_batch, labels, targets)
            cumalative_loss += loss.item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phase": "Evaluate"})
            pbar.update()

            # keeping track of predictions
            bboxes += true_bboxes.to("cpu").tolist()

            # using regression outputs to get final bbox prediction
            if use_regression_output:
                adjustments += (
                    np.repeat(np.expand_dims(anchors, axis=0), repeats=4, axis=0)
                ).tolist()

            # anchor is final bbox prediction (used while regressor is not bing trained)
            else:
                adjustments += (
                    torch.zeros(y_hat_targets_batch.shape) + anchors
                ).tolist()

            scores += y_hat_labels_batch.to("cpu").tolist()
            true_labels += labels.to("cpu").tolist()

    # calculating metrics (IoU and AUC)
    avg_iou = calculate_dataset_iou(
        bboxes=bboxes,
        predictions=adjustments,
        scores=scores,
    )

    print("AVG IOU: ", avg_iou)

    return cumalative_loss / len(eval_dataloader), avg_iou


def main():

    # TRAINING CONFIGURATIONS:
    # NOTE: EDIT THE TrainingConfig below to run your experiment
    training_config = TrainingConfig(
        batch_size = 16,
        learning_rate = 0.001. 
        num_epochs = 10,
        device = get_device(),
        dataset_root_dir = DATASET_BASE_DIR,
        optimizer = "Adam", 
        image_size = 256, 
        pretrained_backbone = True, 
        efficient_net_version = "b0",
        predictor_hidden_dims = [64, 16],
        save_dir_path =  None,
    )

    if training_config.save_dir_path is not None and not os.path.exists(
        training_config.save_dir_path
    ):
        os.makedirs(training_config.save_dir_path)
        print(f"Directory '{training_config.save_dir_path}' was created.")

    pp.pprint(training_config.__dict__)

    device = training_config.device

    # input transforms
    transform = BBoxCompose(
        [
            BBoxBaseTransform(),
            BBoxResize((training_config.image_size, training_config.image_size)),
            BBoxAnchorEncode(anchors=[], positive_iou_threshold=0, min_positive_iou=0)
        ]
    )

    # initializing datasets
    tr_dataset = BoundingBoxDetectionDataset(
        root_dir=DATASET_BASE_DIR, split=DataSplit.TRAIN, transform=transform
    )
    val_dataset = BoundingBoxDetectionDataset(
        root_dir=DATASET_BASE_DIR, split=DataSplit.VALIDATION, transform=transform
    )

    # initializing data loaders
    tr_data_loader = DataLoader(
        tr_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=0
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=0
    )

    # initializing model
    model = EfficientNet(
        efficient_net_v=training_config.efficient_net_version,
        pretrained=training_config.pretrained_backbone,
        predictor_hidden_dims=training_config.predictor_hidden_dims,
        output_dim=4
    )

    # initializing loss function
    loss = torch.nn.MSELoss()

    # initializing optimizer
    if training_config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config.learning_rate,
        )
    elif training_config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
        )
    else:
        raise ValueError(f"Invalid optimizer: {training_config.optimizer}")

    # Train Model
    tr_log = main_train_loop(
        model=model,
        train_dataloader=tr_data_loader,
        val_dataloader=val_data_loader,
        optimizer=optimizer,
        loss_fn=loss,
        device=device,
        n_epochs=training_config.num_epochs,
        configs=training_config,
    )

    print(tr_log)


if __name__ == "__main__":
    main()
