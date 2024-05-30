# Brain Cancer Classifier

This work is a personal continuation of CS4100 Final Project.

## Datasets

The datasets used in this project are available on Kaggle:

- [Brain Tumor Image Dataset: Semantic Segmentation](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation)
- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Brain MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

## Setup Instructions

### Step 1: Install Dependencies

Install the necessary Python packages using the requirements.txt file:

```bash
pip install -r requirements.txt
```

Alternatively, you can create a conda environment using the following commands:

```bash
conda env create -f environment.yaml
conda activate ai_project_env
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

### Object Detection

Navigate to the `main` function in `src.scripts.train_object_detection.py` and edit the `training_configs`.
To run the specified experiment you can use...

```bash
python -m src.scripts.train_object_detection
```
