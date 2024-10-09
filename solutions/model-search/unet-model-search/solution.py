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
    from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        Orientationd,
        NormalizeIntensityd,
        RandRotate90d,
        RandFlipd,
        RandCropByLabelClassesd,
        AsDiscrete
    )
    from monai.networks.nets import UNet
    from monai.losses import TverskyLoss
    from monai.metrics import DiceMetric, ConfusionMatrixMetric
    import mlflow

    args = get_args()
    copick_config_path = args.copick_config_path
    voxel_spacing = float(args.voxel_spacing)
    tomo_type = args.tomo_type
    epochs = int(args.epochs)
    random_seed = int(args.random_seed)
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    def load_data():
        """Load the dataset and create training and validation sets."""
        root = copick.from_file(copick_config_path)
        data_dicts = []
        for run in tqdm(root.runs[:2]):
            tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).numpy()
            segmentation = run.get_segmentations(name='paintedPicks', user_id='user0', voxel_size=voxel_spacing, is_multilabel=True)[0].numpy()
            membrane_seg = run.get_segmentations(name='membrane', user_id="data-portal")[0].numpy()
            segmentation[membrane_seg == 1] = 1  
            data_dicts.append({"image": tomogram, "label": segmentation})
        return data_dicts[:len(data_dicts)//2], data_dicts[len(data_dicts)//2:]

    # Non-random transforms to be cached
    non_random_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS")
    ])

    # Random transforms to be applied during training
    random_transforms = Compose([
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[96, 96, 96],
            num_classes=8,
            num_samples=16
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),    
    ])

    train_files, val_files = load_data()
    train_ds = CacheDataset(data=train_files, transform=non_random_transforms, cache_rate=1.0)
    train_ds = Dataset(data=train_ds, transform=random_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    
    val_ds = CacheDataset(data=val_files, transform=non_random_transforms, cache_rate=1.0)
    val_ds = Dataset(data=val_ds, transform=random_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=torch.cuda.is_available())

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=9,  # 8 classes + background
        channels=(48, 64, 80, 80),
        strides=(2, 2, 1),
        num_res_units=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
    recall_metric = ConfusionMatrixMetric(include_background=False, metric_name="recall", reduction="None")

    def train(train_loader, model, optimizer, loss_function, dice_metric, epochs):
        best_metric = -1
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            model.train()
            for batch_data in train_loader:
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 2 == 0:
                model.eval()
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_outputs = model(val_inputs)
                    metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=9)(i) for i in decollate_batch(val_outputs)]
                    metric_val_labels = [AsDiscrete(to_onehot=9)(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=metric_val_outputs, y=metric_val_labels)
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                if metric > best_metric:
                    best_metric = metric
                    torch.save(model.state_dict(), "best_metric_model.pth")
                print(f"Validation Dice: {metric:.4f}, Best: {best_metric:.4f}")

    mlflow.set_experiment('unet-model-search')
    with mlflow.start_run():
        train(train_loader, model, optimizer, loss_function, dice_metric, epochs)

setup(
    group="model-search",
    name="unet-model-search",
    version="0.0.6",
    title="UNet with Optuna optimization",
    description="Optimization of UNet using updated Monai transforms and Copick data.",
    solution_creators=["Kyle Harrington and Zhuowen Zhao"],
    tags=["unet", "copick", "optuna", "segmentation"],
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
    dependencies={"environment_file": env_file},
)