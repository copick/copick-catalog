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
  - pytorch
  - cudatoolkit
  - pytorch-cuda
  - torchvision
  - torchaudio
  - zarr
  - "numpy<2"
  - scipy
  - joblib
  - pip:
    - album
    - copick
"""

def run():
    import torch
    import numpy as np
    import copick
    import zarr
    from torchvision import transforms
    from numcodecs import Blosc
    import os
    import itertools
    from tqdm import tqdm

    def load_dinov2_model():
        """Load a smaller pretrained DINOv2 model (ViT-S/14)"""
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.eval()  # Set the model to evaluation mode
        return model

    def extract_features_from_model_batch(slice_batch, model):
        """Extract features from a batch of 2D slices using the pretrained model."""
        # Ensure all slices are 2D and replicate to 3 channels (RGB)
        for i in range(len(slice_batch)):
            if slice_batch[i].ndim == 2:  # If the slice is 2D, add a channel dimension
                slice_batch[i] = slice_batch[i].unsqueeze(0)  # Shape: (1, height, width)

        # Stack the batch into a single tensor
        slice_batch = torch.stack(slice_batch)

        # Replicate the grayscale slices to 3 channels (RGB)
        batch_3ch = slice_batch.repeat(1, 3, 1, 1)  # Shape: (batch_size, 3, height, width)

        # Resize and normalize the batch
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transformation
        batch_transformed = torch.stack([transform(slice_3ch) for slice_3ch in batch_3ch])

        # Move the batch to the GPU
        batch_transformed = batch_transformed.to(slice_batch.device)

        # Extract features in batch
        with torch.no_grad():
            features = model(batch_transformed)  # Shape: (batch_size, 384)

        return features  # Shape: (batch_size, 384)

    # Fetch arguments
    args = get_args()
    copick_config_path = args.copick_config_path
    run_name = args.run_name
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type
    feature_type = args.feature_type
    batch_size = args.batch_size
    stride_x = args.stride_x
    stride_y = args.stride_y
    stride_z = args.stride_z

    # Load Copick configuration
    root = copick.from_file(copick_config_path)
    run = root.get_run(run_name)
    if run is None:
        raise ValueError(f"Run with name '{run_name}' not found.")

    voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
    if voxel_spacing_obj is None:
        raise ValueError(f"Voxel spacing '{voxel_spacing}' not found in run '{run_name}'.")

    tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
    if tomogram is None:
        raise ValueError(f"Tomogram type '{tomo_type}' not found for voxel spacing '{voxel_spacing}'.")

    # Open highest resolution
    image = zarr.open(tomogram.zarr(), mode='r')['0']
    image = image[:]  # Load image fully into memory

    # Set the patch size for 224x224 patches
    patch_size = 224

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device {device}")
    model = load_dinov2_model()
    model.to(device)

    # Prepare output Zarr array directly in the tomogram store
    copick_features = tomogram.get_features(feature_type)
    if not copick_features:
        copick_features = tomogram.new_features(feature_type)
    feature_store = copick_features.zarr()

    # Create the Zarr array for features (3D spatial dimensions + 384 embedding vector)
    num_features = 384  # DINOv2_vits14 produces 384-dimensional features
    out_array = zarr.create(
        shape=(num_features, *image.shape),
        chunks=(num_features, 32, 32, 32),  # Chunk size set to (32, 32, 32)
        dtype='float32',
        compressor=Blosc(cname='zstd', clevel=3, shuffle=2),
        store=feature_store,
        overwrite=True
    )

    # Calculate total number of batches
    total_voxels_z = (image.shape[0] + stride_z - 1) // stride_z
    total_voxels_y = (image.shape[1] + stride_y - 1) // stride_y
    total_voxels_x = (image.shape[2] + stride_x - 1) // stride_x
    total_voxels = total_voxels_z * total_voxels_y * total_voxels_x

    total_batches = (total_voxels + batch_size - 1) // batch_size  # Total batches now correctly calculated

    # Initialize an empty batch
    slice_batch = []
    voxel_indices = []
    batch_count = 0

    print(f"Processing image of size {image.shape}")

    # Loop over each voxel in x, y, z with tqdm progress bar
    voxel_iterator = itertools.product(range(0, image.shape[0], stride_z),
                                       range(0, image.shape[1], stride_y),
                                       range(0, image.shape[2], stride_x))

    with tqdm(total=total_batches, desc="Batches Processed") as pbar:
        for z, y, x in voxel_iterator:
            # Extract a 224x224 patch centered around the voxel
            z_start = max(z - 112, 0)
            z_end = min(z + 112, image.shape[0])
            y_start = max(y - 112, 0)
            y_end = min(y + 112, image.shape[1])
            x_start = max(x - 112, 0)
            x_end = min(x + 112, image.shape[2])

            slice_tensor = torch.tensor(image[z, y_start:y_end, x_start:x_end], dtype=torch.float32, device=device).unsqueeze(0)

            # Pad the slice if it is smaller than 224x224
            if slice_tensor.shape[1] != patch_size or slice_tensor.shape[2] != patch_size:
                pad_y = patch_size - slice_tensor.shape[1]
                pad_x = patch_size - slice_tensor.shape[2]
                slice_tensor = torch.nn.functional.pad(slice_tensor, (0, pad_x, 0, pad_y))

            # Accumulate slice in the batch
            slice_batch.append(slice_tensor)
            voxel_indices.append((z, y, x))

            # If the batch is full, process it
            if len(slice_batch) == batch_size:
                batch_count += 1

                voxel_features_batch = extract_features_from_model_batch(slice_batch, model)

                # Store the features for each voxel in the batch
                for i, voxel_features in enumerate(voxel_features_batch):
                    z_idx, y_idx, x_idx = voxel_indices[i]
                    out_array[:, z_idx, y_idx, x_idx] = voxel_features.cpu().numpy().squeeze()

                # Clear the batch and voxel indices for the next iteration
                slice_batch = []
                voxel_indices = []

                pbar.update(1)

        # Process any remaining patches that didn't fill the final batch
        if slice_batch:
            voxel_features_batch = extract_features_from_model_batch(slice_batch, model)

            # Store the features for each voxel in the batch
            for i, voxel_features in enumerate(voxel_features_batch):
                z_idx, y_idx, x_idx = voxel_indices[i]
                out_array[:, z_idx, y_idx, x_idx] = voxel_features.cpu().numpy().squeeze()

            pbar.update(1)

    print(f"Feature extraction complete. Processed {batch_count} batches.")
    print(f"Features saved under feature type '{feature_type}'")

setup(
    group="copick",
    name="generate-dino-features",
    version="0.0.6",
    title="Generate DINOv2 Features from a Copick Run",
    description="Extract multiscale features from a tomogram using DINOv2 (ViT) and save them using Copick's API.",
    solution_creators=["Kyle Harrington"],
    tags=["feature extraction", "pretrained model", "image processing", "cryoet", "tomogram"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the Copick run to process."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing to be used."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Type of tomogram to process."},
        {"name": "feature_type", "type": "string", "required": True, "description": "Name for the feature type to be saved."},
        {"name": "batch_size", "type": "integer", "required": False, "default": 1024, "description": "Batch size for processing."},
        {"name": "stride_x", "type": "integer", "required": False, "default": 1, "description": "Stride along the x-axis."},
        {"name": "stride_y", "type": "integer", "required": False, "default": 1, "description": "Stride along the y-axis."},
        {"name": "stride_z", "type": "integer", "required": False, "default": 1, "description": "Stride along the z-axis."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)