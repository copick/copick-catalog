###album catalog: copick

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - zarr
  - ome-zarr
  - numpy<2
  - scikit-image
  - mrcfile
  - pip:
    - album
    - "copick[all]>=0.5.2"
"""


def run():
    # Imports
    import copick
    from copick.impl.filesystem import CopickRootFSSpec
    import copy
    import glob
    import mrcfile
    import numpy as np
    import os
    from skimage.transform import downscale_local_mean
    from typing import Any, Dict, List, Sequence, Tuple
    import zarr
    import ome_zarr.writer

    # Function definitions
    def ome_zarr_axes() -> List[Dict[str, str]]:
        return [
            {
                "name": "z",
                "type": "space",
                "unit": "angstrom",
            },
            {
                "name": "y",
                "type": "space",
                "unit": "angstrom",
            },
            {
                "name": "x",
                "type": "space",
                "unit": "angstrom",
            },
        ]

    def ome_zarr_transforms(voxel_size: float) -> List[List[Dict[str, Any]]]:
        return [
            [
                {
                    "scale": [voxel_size, voxel_size, voxel_size],
                    "type": "scale",
                }
            ],
            [
                {
                    "scale": [voxel_size * 2, voxel_size * 2, voxel_size * 2],
                    "type": "scale",
                }
            ],
            [
                {
                    "scale": [voxel_size * 4, voxel_size * 4, voxel_size * 4],
                    "type": "scale",
                }
            ],
        ]

    def pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
        pyramid = [image]
        for _ in range(levels - 1):
            image = downscale_local_mean(image, (2, 2, 2)).astype(np.int8)
            pyramid.append(image)
        return pyramid

    # Parsing Input
    args = get_args()
    copick_config_path = args.copick_config_path
    tomo_dir = args.tomo_dir
    tomo_type = args.tomo_type
    as_int8 = args.as_int8

    # Create Copick project
    root = copick.from_file(copick_config_path)

    # Write to static root
    if root.config.static_root is not None:
        os.makedirs(root.config.static_root.replace("local://", ""), exist_ok=True)
        os.makedirs(root.config.overlay_root.replace("local://", ""), exist_ok=True)

        config = copy.copy(root.config)
        config.overlay_root = config.static_root
        config.overlay_fs_args = config.static_fs_args

        root = CopickRootFSSpec(config)

    # Find tomograms
    tomo_paths = glob.glob(f"{tomo_dir}/*.mrc")

    for tp in tomo_paths:
        print(f"Importing {tp}")
        with mrcfile.open(tp, "r") as mrc:
            voxel_size = float(mrc.voxel_size.x)
            tomo = mrc.data

        if as_int8:
            min = np.percentile(tomo, 1)
            max = np.percentile(tomo, 99)
            tomo = np.clip(tomo, min, max)
            tomo = (tomo - min) / (max - min) * 255
            tomo = tomo - 128
            tomo = tomo.astype(np.int8)

        tomo_pyr = pyramid(tomo, 3)
        name = os.path.basename(tp).split(".")[0]

        # Create Run
        run = root.get_run(name)

        if run is None:
            run = root.new_run(name)

        # Create VoxelSpacing
        vs = run.get_voxel_spacing(voxel_size)

        if vs is None:
            vs = run.new_voxel_spacing(voxel_size)

        # Create Tomogram
        cptomo = vs.get_tomogram(tomo_type)

        if cptomo is None:
            cptomo = vs.new_tomogram(tomo_type)

        loc = cptomo.zarr()

        root_group = zarr.group(loc, overwrite=True)

        ome_zarr.writer.write_multiscale(
            pyramid=tomo_pyr,
            group=root_group,
            axes=ome_zarr_axes(),
            coordinate_transformations=ome_zarr_transforms(voxel_size),
            storage_options=dict(chunks=(256, 256, 256), overwrite=True),
            compute=True,
        )


setup(
    group="copick",
    name="setup_local_project",
    version="0.15.0",
    title="Set up a copick project.",
    description="Create a copick project. Optionally import tomograms.",
    solution_creators=["Utz H. Ermel"],
    tags=["copick", "setup", "tomogram", "import"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "copick_config_path",
            "type": "string",
            "required": True,
            "description": "Path to the Copick project root. Write access is expected for both overlay and static root.",
        },
        {
            "name": "tomo_dir",
            "type": "string",
            "required": True,
            "description": "Path to the directory containing the tomograms.",
        },
        {
            "name": "tomo_type",
            "type": "string",
            "required": True,
            "description": "Type of tomogram.",
        },
        {
            "name": "as_int8",
            "type": "boolean",
            "required": False,
            "default": True,
            "description": "Whether to write the segmentation as int8.",
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
