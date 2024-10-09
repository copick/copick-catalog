###album catalog: copick

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - zarr
  - numpy
  - pandas
  - scikit-learn==1.3.2
  - joblib
  - h5py
  - torch
  - optuna
  - pip:
    - album
    - copick
"""

def run():
    import os
    import numpy as np
    import torch
    import optuna
    from torch.utils.data import DataLoader
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss, TverskyLoss
    from monai.transforms import Compose, LoadImage, AddChannel, ScaleIntensity, ToTensor
    from monai.metrics import DiceMetric
    import mlflow
    from tqdm import tqdm
    import random

    args = get_args()
    copick_config_path = args.copick_config_path
    voxel_spacing = float(args.voxel_spacing)
    tomo_type = args.tomo_type
    epochs = int(args.epochs)
    random_seed = int(args.random_seed)

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Set up experiment
    mlflow.set_experiment('UNet-Optuna-Copick')

    def load_data(copick_config_path):
        """Load the dataset and create training and validation sets."""
        root = copick.from_file(copick_config_path)
        data_dicts = []
        for run in tqdm(root.runs):
            tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).numpy()
            segmentation = run.get_segmentations(name='paintedPicks', user_id='user0', voxel_size=voxel_spacing, is_multilabel=True)[0].numpy()
            data_dicts.append({"image": tomogram, "label": segmentation})
        return data_dicts[:len(data_dicts)//2], data_dicts[len(data_dicts)//2:]

    def objective(trial):
        """Objective function to optimize the UNet model."""
        params = {
            "channels": trial.suggest_categorical("channels", [(32, 64, 128), (48, 64, 80, 80)]),
            "strides": trial.suggest_categorical("strides", [(2, 2, 1), (2, 2)]),
            "num_res_units": trial.suggest_int("num_res_units", 1, 3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3),
        }

        train_files, val_files = load_data(copick_config_path)
        transform = Compose([LoadImage(image_only=True), AddChannel(), ScaleIntensity(), ToTensor()])
        train_loader = DataLoader(train_files, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_files, batch_size=2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=len(train_files[0]["label"].unique()) + 1,
            channels=params["channels"],
            strides=params["strides"],
            num_res_units=params["num_res_units"],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)
        dice_metric = DiceMetric(include_background=False)

        # Train the model for the specified number of epochs
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_data in train_loader:
                optimizer.zero_grad()
                images = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch_data in val_loader:
                images = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                outputs = model(images)
                dice_metric(outputs, labels)

        avg_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        return avg_dice

    # Optimize using Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Log best trial
    print(f"Best trial: {study.best_trial.params}")
    mlflow.log_params(study.best_trial.params)
    mlflow.log_metric("best_dice", study.best_trial.value)

setup(
    group="model-search",
    name="unet-model-search",
    version="0.0.1",
    title="Optimize UNet with Optuna on Copick Data",
    description="A solution that optimizes a 3D UNet model using Optuna, with data from Copick.",
    solution_creators=["Kyle Harrington and Zhuowen Zhao"],
    tags=["unet", "copick", "optuna", "optimization", "segmentation"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Tomogram type to use for each tomogram."},
        {"name": "epochs", "type": "integer", "required": True, "description": "Number of training epochs."},
        {"name": "random_seed", "type": "integer", "required": False, "default": 17171, "description": "Random seed for reproducibility."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
