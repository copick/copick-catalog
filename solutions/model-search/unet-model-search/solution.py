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
        Orientationd,
        NormalizeIntensityd,
        RandRotate90d,
        RandFlipd,
        RandCropByLabelClassesd
    )
    import mlflow
    import optuna
    from monai.networks.nets import UNet
    from monai.losses import TverskyLoss
    from monai.metrics import DiceMetric, ConfusionMatrixMetric
    from monai.transforms import AsDiscrete
    from monai.data import decollate_batch    
    from monai.handlers import EarlyStopping

    def objective(trial, train_loader, val_loader, device, random_seed, out_channels, epochs):
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        spatial_dims = 3
        in_channels = 1
        channels = trial.suggest_categorical("channels", [(16, 32, 64), (32, 64, 128, 256), (64, 128, 256), (48, 64, 80, 80)])
        strides = trial.suggest_categorical("strides", [(1, 2, 2), (2, 2, 2), (2, 2, 3)])
        num_res_units = trial.suggest_int("num_res_units", 1, 3)

        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_loguniform("lr", 1e-4, 1e-2))
        loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)
        dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

        early_stopping = EarlyStopping(patience=5, min_delta=1e-4, score_function=lambda val_metric: val_metric, trainer=None)

        best_metric = -1
        best_metric_epoch = -1
        for epoch in range(epochs):  # Use the epochs argument here
            model.train()
            for batch_data in train_loader:
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_outputs = model(val_inputs)
                    val_outputs = [AsDiscrete(argmax=True, to_onehot=out_channels)(i) for i in decollate_batch(val_outputs)]
                    val_labels = [AsDiscrete(to_onehot=out_channels)(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            
            if early_stopping.step(metric):
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

            # Track the performance in MLflow for each epoch
            mlflow.log_metric("epoch", epoch)
            mlflow.log_metric("val_dice", metric)
        
        mlflow.log_metric("best_val_dice", best_metric)
        mlflow.log_param("best_metric_epoch", best_metric_epoch)
        
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
            tomogram = run.get_voxel_spacing(10).get_tomogram('wbp').numpy().astype('float32')
            segmentation = run.get_segmentations(name='paintedPicks', user_id='user0', voxel_size=10, is_multilabel=True)[0].numpy().astype('int64')
            membrane_seg = run.get_segmentations(name='membrane', user_id="data-portal")[0].numpy().astype('int64')
            segmentation[membrane_seg == 1] = 1
            data_dicts.append({"image": tomogram, "label": segmentation})
        return data_dicts[:len(data_dicts)//2], data_dicts[len(data_dicts)//2:]

    non_random_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
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
    version="0.0.10",
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
