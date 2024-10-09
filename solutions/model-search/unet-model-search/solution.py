###album catalog: copick

from album.runner.api import setup, get_args

env_file = """
channels:
- pytorch
- nvidia
- conda-forge
dependencies:
- python=3.10
- pip
- zarr
- numpy<2
- pandas
- scikit-learn==1.3.2
- joblib
- h5py
- pytorch
- torchvision
- torchaudio
- cudatoolkit
- pytorch-cuda
- einops
- monai
- optuna
- mlflow
- nibabel
- pytorch-lightning
- pip:
    - album
    - copick
"""

def run():
    import os
    import torch
    import copick
    import numpy as np
    from tqdm import tqdm
    from monai.data import DataLoader, Dataset, CacheDataset
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        AsDiscrete,
        RandRotate90d,
        RandFlipd,
        RandCropByLabelClassesd,
        RandGaussianNoised,
        RandShiftIntensityd,
        RandZoomd,
        NormalizeIntensityd,
        Orientationd
    )
    import mlflow
    import optuna
    from monai.networks.nets import UNet
    from monai.losses import GeneralizedDiceFocalLoss
    from monai.metrics import DiceMetric, ConfusionMatrixMetric
    from monai.transforms import AsDiscrete
    from monai.data import decollate_batch    
    import torch.nn.functional as F

    def compute_class_weights(train_loader, out_channels, device):
        class_counts = torch.zeros(out_channels).to(device)

        for batch_data in train_loader:
            labels = batch_data["label"].to(device)
            for c in range(out_channels):
                class_counts[c] += (labels == c).sum().item()

        # Invert the frequency to get the weights: more frequent classes get lower weights
        total_count = class_counts.sum().item()
        class_weights = total_count / (out_channels * class_counts)
        
        # Normalize the weights to sum to 1, or adjust according to your need
        class_weights = class_weights / class_weights.sum()
        
        return class_weights

    def objective(trial, train_loader, val_loader, device, random_seed, out_channels, epochs):
        with mlflow.start_run(nested=True):
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

            spatial_dims = 3
            in_channels = 1
            channels = trial.suggest_categorical("channels", [(8, 16, 32), (16, 32, 64)])
            strides = trial.suggest_categorical("strides", [(1, 2, 2), (2, 2, 2)])
            num_res_units = trial.suggest_int("num_res_units", 1, 2)

            model = UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
                num_res_units=num_res_units
            ).to(device)

            # Dynamically compute class weights from the data
            class_weights = compute_class_weights(train_loader, out_channels, device)

            optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_loguniform("lr", 1e-5, 1e-2))
            
            # Use GeneralizedDiceFocalLoss
            loss_function = GeneralizedDiceFocalLoss(
                include_background=True,
                softmax=True,
                weight=class_weights,
                lambda_gdl=1.0,
                lambda_focal=1.0,
                gamma=2.0
            )
            
            dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
            confusion_metric = ConfusionMatrixMetric(include_background=False, reduction="mean")

            best_metric = -1
            best_metric_epoch = -1
            early_stopping_patience = 5  # Stop if no improvement after 5 epochs
            epochs_without_improvement = 0

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                print(f"Epoch {epoch + 1}/{epochs}...")

                # Training loop
                for batch_data in train_loader:
                    inputs = batch_data["image"].to(device)
                    labels = batch_data["label"].to(device)
                    
                    # Manually apply one-hot encoding here
                    labels_one_hot = F.one_hot(labels.long(), num_classes=out_channels).permute(0, 4, 1, 2, 3).float().to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels_one_hot)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                print(f"Training loss for epoch {epoch + 1}: {epoch_loss:.4f}")
                mlflow.log_metric(f"epoch_{epoch + 1}_loss", epoch_loss)

                # Validation loop
                model.eval()
                dice_metric.reset()
                confusion_metric.reset()
                val_loss = 0
                with torch.no_grad():
                    for batch_data in val_loader:
                        val_inputs = batch_data["image"].to(device)
                        val_labels = batch_data["label"].to(device)
                        
                        # Manually apply one-hot encoding in validation as well
                        val_labels_one_hot = F.one_hot(val_labels.long(), num_classes=out_channels).permute(0, 4, 1, 2, 3).float().to(device)

                        val_outputs = model(val_inputs)
                        val_loss += loss_function(val_outputs, val_labels_one_hot).item()
                        dice_metric(y_pred=val_outputs, y=val_labels_one_hot)
                        confusion_metric(y_pred=val_outputs, y=val_labels_one_hot)

                metric = dice_metric.aggregate().item()
                confusion_matrix = confusion_metric.aggregate()

                dice_metric.reset()
                confusion_metric.reset()

                print(f"Validation Dice score for epoch {epoch + 1}: {metric:.4f}")
                mlflow.log_metric(f"epoch_{epoch + 1}_val_dice", metric)
                mlflow.log_metric(f"epoch_{epoch + 1}_val_loss", val_loss)
                mlflow.log_metric(f"epoch_{epoch + 1}_confusion_matrix", confusion_matrix)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                        break

            mlflow.log_metric("best_val_dice", best_metric)
            mlflow.log_param("best_metric_epoch", best_metric_epoch)

            print(f"Best validation Dice score: {best_metric:.4f} at epoch {best_metric_epoch}")
            
            return best_metric


    # TODO temporary for the cropping label errors
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()
    copick_config_path = args.copick_config_path
    voxel_spacing = float(args.voxel_spacing)
    tomo_type = args.tomo_type
    epochs = int(args.epochs)
    random_seed = int(args.random_seed)
    num_classes = int(args.num_classes)
    out_channels = num_classes + 1  # Background class included
    num_trials = int(args.num_trials)
    batch_size = int(args.batch_size)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    def load_data():
        root = copick.from_file(copick_config_path)
        data_dicts = []
        for run in tqdm(root.runs[:2]):
            tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram('wbp').numpy().astype('float32')
            segmentation = run.get_segmentations(name='paintedPicks', user_id='user0', voxel_size=10, is_multilabel=True)[0].numpy().astype('int64')
            membrane_seg = run.get_segmentations(name='membrane', user_id="data-portal")[0].numpy().astype('int64')
            segmentation[membrane_seg == 1] = 1
            data_dicts.append({"image": tomogram, "label": segmentation})
        return data_dicts[:len(data_dicts)//2], data_dicts[len(data_dicts)//2:]

    non_random_transforms = Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS")
    ])

    random_transforms = Compose([
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[96, 96, 96],
            num_classes=num_classes,  # Updated to use num_classes argument
            num_samples=16
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandGaussianNoised(keys=["image"], prob=0.2),
        RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.9, max_zoom=1.1),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2)
    ])

    train_files, val_files = load_data()
    train_ds = CacheDataset(data=train_files, transform=non_random_transforms, cache_rate=1.0)
    train_ds = Dataset(data=train_ds, transform=random_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    val_ds = CacheDataset(data=val_files, transform=non_random_transforms, cache_rate=1.0)
    val_ds = Dataset(data=val_ds, transform=random_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mlflow.set_experiment('unet-model-search')
    with mlflow.start_run():
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, train_loader, val_loader, device, random_seed, out_channels, epochs), n_trials=num_trials)

        print(f"Best trial: {study.best_trial.value}")
        print(f"Best params: {study.best_params}")

        # Log the best trial in MLflow
        mlflow.log_metric("best_trial_value", study.best_trial.value)
        mlflow.log_params(study.best_trial.params)

setup(
    group="model-search",
    name="unet-model-search",
    version="0.0.19",
    title="UNet with Optuna optimization",
    description="Optimization of UNet using Optuna with Copick data.",
    solution_creators=["Kyle Harrington and Zhuowen Zhao"],
    tags=["unet", "copick", "optuna", "segmentation"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Tomogram type to use for each tomogram."},
        {"name": "epochs", "type": "integer", "required": True, "description": "Number of training epochs."},
        {"name": "num_classes", "type": "integer", "required": True, "description": "Number of classes for segmentation, excluding the background class."},
        {"name": "num_trials", "type": "integer", "required": True, "description": "Number of trials for Optuna optimization."},
        {"name": "batch_size", "type": "integer", "required": True, "description": "Batch size for the DataLoader."},
        {"name": "random_seed", "type": "integer", "required": False, "default": 17171, "description": "Random seed for reproducibility."}
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
