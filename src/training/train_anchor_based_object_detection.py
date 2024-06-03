from typing import List
from src.datasets.bbox import BoundingBoxDetectionDataset
import torch
from torch.utils.data import DataLoader
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from src.models.model_utils import load_checkpoint
from src.models.object_detection.efficientdet import EfficientDet, save_checkpoint
from src.utils.bbox import calculate_anchor_based_dataset_iou, generate_anchors
from src.utils.loss_functions import BBoxLoss
from src.utils.scipt_utils import ExperimentLog, get_device, save_configs_dict
from tqdm import tqdm
import numpy as np
import json
import pprint as pp
from sklearn.metrics import roc_auc_score

from src.enums import DataSplit
from src.utils.transforms import (
    BBoxAnchorEncode,
    BBoxBaseTransform,
    BBoxCompose,
    BBoxResize,
)

warnings.filterwarnings("ignore")


DATASET_BASE_DIR = pathlib.Path(__file__).parent.parent.parent / "datasets"
EXPERIMENTS_BASE_DIR = pathlib.Path(__file__).parent.parent.parent / "experiments"


@dataclass
class ExperimentLog:
    tr_loss: list = field(default_factory=lambda:[])
    val_loss: list = field(default_factory=lambda:[])
    val_iou: list = field(default_factory=lambda:[])
    val_auc: list = field(default_factory=lambda:[])
    test_loss: float = None
    test_iou: float = None
    test_auc: float = None


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 5
    device: torch.device = torch.device("cpu")
    dataset_root_dir: str = DATASET_BASE_DIR
    optimizer: str = "Adam"
    anchor_aspect_ratios: List[float] = field(default_factory=lambda: [1.0])
    anchor_scales: List[float] = field(default_factory=lambda: [0.1, 0.175, 0.25])
    anchor_feature_map_sizes: List[int] = field(default_factory=lambda: [32, 16, 8, 4, 2])
    pretrained_backbone: bool = True
    image_size: int = 256
    clasification_loss: str = "weighted_bce"
    clasification_loss_weight: int = 1
    regression_loss: str = "smooth_l1"
    regression_loss_weight: int = 1
    save_dir_path: str = EXPERIMENTS_BASE_DIR / datetime.now().strftime('%y-%m-%d-%H-%M')


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
    test_dataloader: DataLoader | None = None,
):
    model.to(device)

    # creating paths
    best_model_path = configs.save_dir_path / "best_model.pt"
    last_model_path = configs.save_dir_path / "last_model.pt"

    # keeping track of best model
    experiment_log = ExperimentLog()

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
            val_loss, val_iou, val_auc = evaluate(
                model, val_dataloader, loss_fn, device, anchors, pbar
            )

            # keeping track of metrics
            experiment_log.tr_loss.append(float(train_loss))
            experiment_log.val_loss.append(float(val_loss))
            experiment_log.val_iou.append(float(val_iou))
            experiment_log.val_auc.append(float(val_auc))
            print(f"Validation metric: IoU = {val_iou}\tAUC = {val_auc}")

            # checking if model needs to be saved
            if max(experiment_log.val_iou) == val_iou:
                print(f"New best val iou.")
                # saving best model if save path is provided
                if configs.save_dir_path:
                    save_checkpoint(model, best_model_path)

            # saving current model if save path is provided
            if configs.save_dir_path:
                save_checkpoint(model, last_model_path)

    if test_dataloader:
        
        print("Running Test: ")

        # loading the best model
        load_checkpoint(model, best_model_path)

        # evaluating performance on test set
        test_loss, test_iou, test_auc, test_images, test_true_targets, test_predicted_targets = \
            evaluate(model, test_dataloader, loss_fn, device)
        
        print(f"Validation metric: IoU = {test_iou}\tAUC = {test_auc}")
        experiment_log.test_loss = test_loss
        experiment_log.test_iou = test_iou
        experiment_log.test_auc = test_auc


    return experiment_log


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

    for imgs_batch, (labels, targets, bboxes) in train_dataloader:

        imgs_batch, labels, targets = imgs_batch.to(device), labels.to(device), targets.to(device)
        model.zero_grad()
        y_hat_labels, y_hat_targets = model(imgs_batch)
        y_hat_labels = y_hat_labels.view(y_hat_labels.shape[0], -1)

        optimizer.zero_grad()
        loss = loss_fn(y_hat_labels, y_hat_targets, labels, targets)
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

    images = []
    scores = []
    true_labels = []
    pred_bboxes = []
    bboxes = []

    model.eval()
    with torch.no_grad():
        cumalative_loss = 0
        for imgs_batch, (labels_batch, targets_batch, true_bboxes) in eval_dataloader:
            imgs_batch, labels_batch, targets_batch = (
                imgs_batch.to(device),
                labels_batch.to(device),
                targets_batch.to(device),
            )
            pred_labels_batch, pred_targets_batch = model(imgs_batch)
            pred_labels_batch = pred_labels_batch.view(
                pred_labels_batch.shape[0], -1
            )
            loss = loss_fn(pred_labels_batch, pred_targets_batch, labels_batch, targets_batch)
            cumalative_loss += loss.item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phase": "Evaluate"})
            pbar.update()

            # using regression outputs to get final bbox prediction
            if use_regression_output:
                adjustments += (
                    np.repeat(np.expand_dims(anchors, axis=0), repeats=4, axis=0)
                ).tolist()

            # anchor is final bbox prediction (used while regressor is not bing trained)
            else:
                adjustments += (
                    torch.zeros(pred_targets_batch.shape) + anchors
                ).tolist()

            # keeping track of ground truths and predictions
            bboxes += true_bboxes.to("cpu").tolist()
            images += imgs_batch.to("cpu").tolist()
            pred_bboxes += pred_targets_batch.to("cpu").tolist()
            scores += pred_labels_batch.to("cpu").tolist()
            true_labels += labels_batch.to("cpu").tolist()

    # calculating metrics (IoU and AUC)
    avg_iou = calculate_anchor_based_dataset_iou(
        bboxes=bboxes,
        predictions=adjustments,
        scores=scores,
    )
    auc = roc_auc_score(
        np.reshape(np.array(true_labels), -1), np.reshape(np.array(scores), -1)
    )

    return cumalative_loss / len(eval_dataloader), avg_iou, auc, 


def main():

    # TRAINING CONFIGURATIONS:
    # NOTE: EDIT THE TrainingConfig below to run your experiment
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=0.01,
        num_epochs=5,
        device=get_device(),
        dataset_root_dir=DATASET_BASE_DIR,
        optimizer="SGD",
        clasification_loss="weighted_bce",
        clasification_loss_weight=1,
        regression_loss="smooth_l1",
        regression_loss_weight=0,  # freezing regression
        anchor_aspect_ratios=[1],
        anchor_scales=[0.1, 0.175, 0.25, 0.5, 0.35],
        anchor_feature_map_sizes=[32],
        pretrained_backbone=True,
        image_size=256,
        save_dir_path=None,
    )

    # creating experiment save directory
    training_config.save_dir_path.mkdir(parents=True, exist_ok=True)

    pp.pprint(training_config.__dict__)
    save_configs_dict(training_config.__dict__, save_path=training_config.save_dir_path / "configs.json")

    device = training_config.device

    # Defining Anchors
    anchors_centers, anchors_corners = generate_anchors(
        training_config.image_size,
        scales=training_config.anchor_scales,
        aspect_ratios=training_config.anchor_aspect_ratios,
        feature_map_sizes=training_config.anchor_feature_map_sizes,
    )

    # input transforms
    train_transform = BBoxCompose(
        [
            BBoxBaseTransform(),
            BBoxResize((training_config.image_size, training_config.image_size)),
            BBoxAnchorEncode(
                anchors_centers, positive_iou_threshold=0.5, min_positive_iou=0.3
            ),
        ]
    )
    eval_transform = BBoxCompose(
        [
            BBoxBaseTransform(),
            BBoxResize((training_config.image_size, training_config.image_size)),
            BBoxAnchorEncode(
                anchors_centers, positive_iou_threshold=0.5, min_positive_iou=0.3
            ),
        ]
    )

    # initializing datasets
    tr_dataset = BoundingBoxDetectionDataset(
        root_dir=DATASET_BASE_DIR, split=DataSplit.TRAIN, transform=train_transform
    )
    val_dataset = BoundingBoxDetectionDataset(
        root_dir=DATASET_BASE_DIR, split=DataSplit.VALIDATION, transform=eval_transform
    )
    test_dataset = BoundingBoxDetectionDataset(
        root_dir=DATASET_BASE_DIR, split=DataSplit.TEST, transform=eval_transform
    )

    # initializing data loaders
    tr_data_loader = DataLoader(
        tr_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=0
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=0
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=0
    )
    

    # initializing model
    model = EfficientDet(
        pretrained_backbone=training_config.pretrained_backbone,
        n_classes=1,
        n_anchors=len(training_config.anchor_aspect_ratios)
        * len(training_config.anchor_scales),
        bifpn_layers=3,
        n_channels=64,
    )

    # initializing loss function
    loss = BBoxLoss(
        class_loss=training_config.clasification_loss,
        class_loss_weight=training_config.clasification_loss_weight,
        reg_loss=training_config.regression_loss,
        reg_loss_weight=training_config.regression_loss_weight,
    )

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
    experiment_log = main_train_loop(
        model=model,
        train_dataloader=tr_data_loader,
        val_dataloader=val_data_loader,
        test_dataloader=test_data_loader,
        optimizer=optimizer,
        loss_fn=loss,
        device=device,
        anchors=anchors_centers,
        n_epochs=training_config.num_epochs,
        configs=training_config,
    )

    # saving experiment logs
    pp.pprint(experiment_log.__dict__)
    with open(training_config.save_dir_path / "experiment_log.json", "w") as f:
        json.dump(experiment_log.__dict__, f, indent=4)


if __name__ == "__main__":
    main()
