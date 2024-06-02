import torch
from torch.utils.data import DataLoader
import pathlib
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import os
import pprint as pp
import warnings 
import json
from datetime import datetime
from src.data.bbox import BoundingBoxDetectionDataset
from src.utils.bbox import calculate_single_bbox_iou_values
from src.data.classification import DataSplit
from src.models.model_utils import load_checkpoint, save_checkpoint
from src.models.object_detection.efficientnet import EfficientNet
from src.training.training_utils import get_device
from src.utils.transforms import (
    BBoxBaseTransform,
    BBoxCompose,
    BBoxResize,
    BBoxCocoToCenterFormat,
    DATA_AUGMENTATION_MAP
)
from src.utils.scipt_utils import save_configs_dict
from src.utils.visualize import plot_sample_model_prediction


warnings.filterwarnings("ignore")

DATASET_BASE_DIR = pathlib.Path(__file__).parent.parent.parent / "datasets"
EXPERIMENTS_BASE_DIR = pathlib.Path(__file__).parent.parent.parent / "experiments"


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 5
    device: torch.device = torch.device("cpu")
    dataset_root_dir: str = DATASET_BASE_DIR
    augmentations: list = field(default_factory=lambda:[])
    optimizer: str = "Adam"
    image_size: int = 256
    pretrained_backbone: bool = True
    efficient_net_version: str = "b0"
    predictor_hidden_dims: list = field(default_factory=lambda:[])
    save_dir_path: str = EXPERIMENTS_BASE_DIR / datetime.now().strftime('%y-%m-%d-%H-%M')

@dataclass
class ExperimentLog:
    tr_loss: list = field(default_factory=lambda:[])
    val_loss: list = field(default_factory=lambda:[])
    val_iou: list = field(default_factory=lambda:[])
    test_loss: float = None
    test_iou: float = None

def main_train_loop(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
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
            val_loss, val_iou, val_images, val_true_targets, val_predicted_targets = \
                evaluate(model, val_dataloader, loss_fn, device, pbar)

            # plottting visuals of predictions
            sample_pred_save_dir = configs.save_dir_path / f"epoch_{epoch+1}_samples"
            plot_sample_model_prediction(
                imgs=val_images,
                true_bboxes=val_true_targets,
                pred_bboxes=val_predicted_targets,
                save_dir=sample_pred_save_dir,
                n_samples=10,
            )

            # keeping track of metrics
            experiment_log.tr_loss.append(float(train_loss))
            experiment_log.val_loss.append(float(val_loss))
            experiment_log.val_iou.append(float(val_iou))
            print(f"Validation IoU: {val_iou}")

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
        test_loss, test_iou, test_images, test_true_targets, test_predicted_targets = \
            evaluate(model, test_dataloader, loss_fn, device)
        print(f"Test IoU: {test_iou}")
        experiment_log.test_loss = test_loss
        experiment_log.test_iou = test_iou

        # plottting visuals of predictions
        sample_pred_save_dir = configs.save_dir_path / "test_samples"
        plot_sample_model_prediction(
            imgs=test_images,
            true_bboxes=test_true_targets,
            pred_bboxes=test_predicted_targets,
            save_dir=sample_pred_save_dir,
            n_samples='all',
        )

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

    for imgs_batch, targets_batch in train_dataloader:
        
        # formatting targets and inputs
        imgs_batch = imgs_batch.type(torch.float32).to(device)
        targets_batch = targets_batch.type(torch.float32).to(device)

        # resetting gradients
        model.zero_grad()

        # forward pass
        pred_targets_batch = model(imgs_batch)

        # updating model and tracking loss
        optimizer.zero_grad()
        loss = loss_fn(targets_batch, pred_targets_batch)
        loss.backward()
        optimizer.step()
        cumalative_loss += loss.item()
        
        # updating progress bar
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phase": "Train"})
        pbar.update()

    return cumalative_loss / len(train_dataloader)


def evaluate(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    pbar: tqdm | None = None,
):
    """
    Evaluate model performance on evaluation data.
    """
    images = []
    true_targets = []
    predicted_targets = []

    model.eval()
    with torch.no_grad():
        cumalative_loss = 0
        for imgs_batch, targets_batch in eval_dataloader:

            # formatting inputs and targets
            imgs_batch = imgs_batch.type(torch.float32).to(device)
            targets_batch = targets_batch.type(torch.float32).to(device)

            # model prediction
            pred_targets_batch = model(imgs_batch)

            # calculating batch loss adding to cumulative
            loss = loss_fn(targets_batch, pred_targets_batch)
            cumalative_loss += loss.item()

            # updating progress bar
            if pbar:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phase": "Evaluate"})
                pbar.update()

            # keeping track of inputs, targets, and predictions
            images += imgs_batch.to("cpu").tolist()
            true_targets += targets_batch.to("cpu").tolist()
            predicted_targets += pred_targets_batch.to("cpu").tolist()

    # calculating IoU
    iou_values = calculate_single_bbox_iou_values(true_bboxes=true_targets, pred_bboxes=predicted_targets)
    avg_iou = np.mean(iou_values)

    return cumalative_loss / len(eval_dataloader), avg_iou, images, true_targets, predicted_targets


def main():

    # TRAINING CONFIGURATIONS:
    # NOTE: Edit the TrainingConfig below to run your experiment
    training_config = TrainingConfig(
        batch_size = 16,
        learning_rate = 0.001,
        num_epochs = 100,
        device = get_device(),
        dataset_root_dir = DATASET_BASE_DIR,
        augmentations=['rotation', 'reflection', 'crop'],
        optimizer = "Adam", 
        image_size = 256, 
        pretrained_backbone = True, 
        efficient_net_version = "b4",
        predictor_hidden_dims = [64, 16],
        # save_dir_path = EXPERIMENTS_BASE_DIR / 'exp_1',
    )
    
    # creating experiment save directory
    training_config.save_dir_path.mkdir(parents=True, exist_ok=True)

    # saving configs
    pp.pprint(training_config.__dict__)
    save_configs_dict(training_config.__dict__, save_path=training_config.save_dir_path / "configs.json")

    device = training_config.device

    # Data augmentaion only applied to the training set
    augmentation_transforms = [DATA_AUGMENTATION_MAP[a] for a in training_config.augmentations]

    # input transforms
    train_transform = BBoxCompose(
        [
            BBoxBaseTransform(),
            BBoxResize((training_config.image_size, training_config.image_size)),
            BBoxCocoToCenterFormat(),
            *augmentation_transforms,
        ]
    )
    eval_transform = BBoxCompose(
        [
            BBoxBaseTransform(),
            BBoxResize((training_config.image_size, training_config.image_size)),
            BBoxCocoToCenterFormat(),
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
    model = EfficientNet(
        efficient_net_v=training_config.efficient_net_version,
        pretrained=training_config.pretrained_backbone,
        predictor_hidden_dims=training_config.predictor_hidden_dims,
        output_dim=4
    )

    # initializing loss function
    loss = torch.nn.SmoothL1Loss()

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
        n_epochs=training_config.num_epochs,
        configs=training_config,
    )   

    # saving experiment logs
    pp.pprint(experiment_log.__dict__)
    with open(training_config.save_dir_path / "experiment_log.json", "w") as f:
        json.dump(experiment_log.__dict__, f, indent=4)


if __name__ == "__main__":
    main()
