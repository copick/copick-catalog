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
    from torchvision import models, transforms
    from numcodecs import Blosc
    import os

    def load_dinov2_model():
        """Load a smaller pretrained DINOv2 model (ViT-S/14)"""
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.eval()  # Set the model to evaluation mode
        return model

    def extract_features_from_model(chunk_tensor, model):
        """Extract features from a given 3D chunk using the pretrained model."""
        # Convert 3D tensor to 2D slices (DINOv2 handles 2D images)
        slices = [chunk_tensor[:, :, i].unsqueeze(0) for i in range(chunk_tensor.shape[2])]

        # Apply model to each 2D slice
        features = []
        for slice in slices:
            # Replicate the single channel to three channels
            slice_3ch = slice.repeat(3, 1, 1)  # Replicate grayscale to RGB (3-channel)

            # Resize to 224x224, the input size for DINOv2
            transformed_slice = transforms.Resize((224, 224))(slice_3ch)

            # Normalize using the ImageNet mean and standard deviation
            transformed_slice = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225])(transformed_slice)
            
            transformed_slice = transformed_slice.unsqueeze(0).to(chunk_tensor.device)  # Add batch dimension

            # Extract features using the model
            with torch.no_grad():
                feature = model(transformed_slice)
                features.append(feature)

        # Stack features along the depth axis
        return torch.stack(features, dim=2).squeeze(0)  # Concatenating features along the depth axis

    # Fetch arguments
    args = get_args()
    copick_config_path = args.copick_config_path
    run_name = args.run_name
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type
    feature_type = args.feature_type

    # Load Copick configuration
    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = copick.from_file(copick_config_path)
    print("Copick root loaded successfully")

    # Get run and voxel spacing
    run = root.get_run(run_name)
    if run is None:
        raise ValueError(f"Run with name '{run_name}' not found.")

    voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
    if voxel_spacing_obj is None:
        raise ValueError(f"Voxel spacing '{voxel_spacing}' not found in run '{run_name}'.")

    # Get tomogram
    tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
    if tomogram is None:
        raise ValueError(f"Tomogram type '{tomo_type}' not found for voxel spacing '{voxel_spacing}'.")

    # Open highest resolution
    image = zarr.open(tomogram.zarr(), mode='r')['0']

    # Determine chunk size from input Zarr
    input_chunk_size = image.chunks
    chunk_size = input_chunk_size if len(input_chunk_size) == 3 else input_chunk_size[1:]

    print(f"Processing image from run {run_name} with shape {image.shape} at voxel spacing {voxel_spacing}")
    print(f"Using chunk size: {chunk_size}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained DINOv2 model
    model = load_dinov2_model()
    model.to(device)

    # Prepare output Zarr array directly in the tomogram store
    print(f"Creating new feature store...")
    copick_features = tomogram.get_features(feature_type)
    if not copick_features:
        copick_features = tomogram.new_features(feature_type)
    feature_store = copick_features.zarr()

    # Create the Zarr array for features
    num_features = 768  # DINOv2 typically produces 768-dimensional features
    out_array = zarr.create(
        shape=(num_features, *image.shape),
        chunks=(num_features, *chunk_size),
        dtype='float32',
        compressor=Blosc(cname='zstd', clevel=3, shuffle=2),
        store=feature_store,
        overwrite=True
    )

    # Process each chunk
    for z in range(0, image.shape[0], chunk_size[0]):
        for y in range(0, image.shape[1], chunk_size[1]):
            for x in range(0, image.shape[2], chunk_size[2]):
                z_start = max(z, 0)
                z_end = min(z + chunk_size[0], image.shape[0])
                y_start = max(y, 0)
                y_end = min(y + chunk_size[1], image.shape[1])
                x_start = max(x, 0)
                x_end = min(x + chunk_size[2], image.shape[2])

                print(f"Processing {z_start}:{z_end}, {y_start}:{y_end}, {x_start}:{x_end}")

                chunk = image[z_start:z_end, y_start:y_end, x_start:x_end]

                # Convert to PyTorch tensor and move to device
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=device)

                # Extract features using DINOv2
                chunk_features = extract_features_from_model(chunk_tensor, model)

                # Store features
                out_array[:, z:z + chunk_size[0], y:y + chunk_size[1], x:x + chunk_size[2]] = chunk_features.cpu().numpy()

    print(f"Features saved under feature type '{feature_type}'")

setup(
    group="copick",
    name="generate-dino-features",
    version="0.0.3",
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
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
