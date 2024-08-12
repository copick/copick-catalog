###album catalog: copick

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
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
    mask_dir = args.mask_dir
    mask_dir = mask_dir if mask_dir.endswith("/") else mask_dir + "/"
    voxel_size = args.voxel_size
    name_format = args.name_format
    as_int8 = args.as_int8
    out_object = args.out_object
    out_user = args.out_user
    out_session = args.out_session

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

    # Import Masks

    for run in root.runs:
        print(f"Importing mask for {run.name}")

        file = f"{mask_dir}{name_format.format(run_name=run.name)}"

        with mrcfile.open(file, "r") as mrc:
            tomo = mrc.data

        if as_int8:
            tomo = tomo.astype(np.int8)

        # Create Segmentation
        seg = run.get_segmentations(
            name=out_object,
            user_id=out_user,
            session_id=out_session,
            is_multilabel=False,
            voxel_size=voxel_size,
        )

        if len(seg) == 0:
            seg = run.new_segmentation(
                name=out_object,
                user_id=out_user,
                session_id=out_session,
                is_multilabel=False,
                voxel_size=voxel_size,
            )
        else:
            seg = seg[0]

        loc = seg.zarr()

        root_group = zarr.group(loc, overwrite=True)

        ome_zarr.writer.write_multiscale(
            pyramid=[tomo],
            group=root_group,
            axes=ome_zarr_axes(),
            coordinate_transformations=ome_zarr_transforms(voxel_size),
            storage_options=dict(chunks=(256, 256, 256), overwrite=True),
            compute=True,
        )


setup(
    group="copick",
    name="import_segmentations",
    version="0.7.1",
    title="Import segmentations.",
    description="Import segmentations into a copick project.",
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
            "name": "mask_dir",
            "type": "string",
            "required": True,
            "description": "Path to the directory containing the segmentation masks.",
        },
        {
            "name": "voxel_size",
            "type": "float",
            "required": True,
            "description": "Voxel size of the segmentation masks.",
        },
        {
            "name": "name_format",
            "type": "string",
            "required": True,
            "description": "Format string for the mask names. Use {run_name} as placeholder for the run name.",
        },
        {
            "name": "as_int8",
            "type": "boolean",
            "required": False,
            "default": True,
            "description": "Whether to write the segmentation as int8.",
        },
        {
            "name": "out_object",
            "description": "Name of the output pickable object.",
            "type": "string",
            "required": False,
            "default": "segmentation",
        },
        {
            "name": "out_user",
            "description": "User/Tool name for output points.",
            "type": "string",
            "required": False,
            "default": "seg",
        },
        {
            "name": "out_session",
            "description": "Output session, indicating this set was generated by a tool.",
            "type": "string",
            "required": False,
            "default": "0",
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
