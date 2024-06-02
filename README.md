# Brain Tumor Detection with Computer Vision

This work aims to identify the location of brain tumors in MRI images through the use of object-detection.

These are personal contributions and continuations of a group project for CS4100. You can find the project report [here](./resources/Project_Report.pdf).
 
## Datasets

The datasets used in this project are available on Kaggle:

- [Brain Tumor Image Dataset: Semantic Segmentation](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation)

![Example Samples](./resources/bbox_dataset.png)

## Problem Formulation

### Single Object Detection

Directly predict the coordinates of the ground truth bounding box (center of the box and its width, height).

Benefit: simple and naive approach

Drawbacks: relies on the assumption that there is exactly 1 tumor bounding box per image

![single-object-formulation](./resources/single-bbox-prediction.png)

### Anchor-based Object Detection

Pre-define anchors boxes of various sizes and aspect ratios. For each anchor box, make a classification prediction (is there an object in this anchor?) and a regression prediction (how to make the anchor box overlay the true object bounding box).

Benefits: can handle cases when there are no tumors or many tumors in a single image

Drawbacks: more difficult to train due to spacity of the labels; more complicated model architecture.

![anchor-based-object-formulation](./resources/anchor-based-prediction.png)

## Setup Instructions

### Step 1: Install Dependencies

Install the necessary Python packages using the requirements.txt file:

```bash
pip install -r requirements.txt
```

Alternatively, you can create a conda environment using the following commands:

```bash
conda env create -f environment.yaml
conda activate brain_tumor_detection_env
```

### Step 2: Setup Environment Variables

To create a base configuration for the project, run the following command:

```bash
cp config/env_local.env .env
```

This will create a `.env` file in the root dir of the project. However, to actually run training and testing scripts, you will need to fill in the values in the `.env` file.

### Step 3: Kaggle API Authentication

Follow the instructions to set up your Kaggle API credentials. You can find the Kaggle API authentication instructions in the [Kaggle API Documentation](https://www.kaggle.com/docs/api).

### Step 4: Download Datasets

Refer to the `notebooks/downloading_datasets.ipynb` notebook for step-by-step instructions on using the Kaggle API to download the datasets required for this project. The datasets will be downloaded to the `./datasets` folder, which is configured to be ignored by git.

## Running Experiments:

### Single Object Detection

Navigate to the `main` function in `src.scripts.train_object_detection.py` and edit the `training_configs`.
To run the specified experiment you can use...

```bash
python -m src.scripts.train_object_detection
```

### Anchor-Based Object Detection

## Some Experimental Results
