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
  - numpy
  - trimesh
  - pip:
    - album
    - "copick[all] @ git+https://github.com/uermel/copick.git@2d29f48"
    - tqdm
"""


def run():
    # Imports
    import numpy as np
    import tqdm
    import trimesh as tm
    from typing import Sequence, Tuple
    import zarr

    import copick
    from copick.models import CopickPicks, CopickRun, CopickPoint, CopickLocation

    # Function definitions
    def shift_3D(shift):
        return np.array(
            [
                [1, 0, 0, shift[0]],
                [0, 1, 0, shift[1]],
                [0, 0, 1, shift[2]],
                [0, 0, 0, 1],
            ]
        )

    def rotation_3DX(angle):
        phi = np.radians(angle)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        )

    def rotation_3DY(angle):
        phi = np.radians(angle)
        return np.array(
            [
                [np.cos(phi), 0, np.sin(phi), 0],
                [0, 1, 0, 0],
                [-np.sin(phi), 0, np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        )

    def rotation_3DZ(angle):
        phi = np.radians(angle)
        return np.array(
            [
                [np.cos(phi), -np.sin(phi), 0, 0],
                [np.sin(phi), np.cos(phi), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def rotation_center(rot_matrix, center):
        s1 = shift_3D(-center)
        s2 = shift_3D(center)
        return s2 @ rot_matrix @ s1

    # Parsing Input
    args = get_args()
    copick_config_path = args.copick_config_path
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type

    angle = args.angle

    output_object = args.output_object
    output_user = args.output_user
    output_session = args.output_session

    # Code
    root = copick.from_file(copick_config_path)

    for run in root.runs:
        print(run.name)

        vs = run.get_voxel_spacing(voxel_spacing)
        tomo = vs.get_tomogram(tomo_type)
        pixel_max_dim = zarr.open(tomo.zarr())["0"].shape[::-1]
        pixel_center = np.floor(np.array(pixel_max_dim) / 2) + 1
        max_dim = np.array([d * voxel_spacing for d in pixel_max_dim])
        center = np.array([c * voxel_spacing for c in pixel_center])

        r = rotation_3DZ(angle)
        transform = rotation_center(r, center)
        box = tm.creation.box(
            extents=max_dim, transform=transform @ shift_3D(max_dim / 2)
        )

        nm = run.get_meshes(
            object_name=output_object, user_id=output_user, session_id=output_session
        )

        if len(nm) == 0:
            nm = run.new_mesh(
                object_name=output_object,
                user_id=output_user,
                session_id=output_session,
            )
        else:
            nm = nm[0]

        nm.mesh = box
        nm.store()


setup(
    group="copick",
    name="create_rec_limits",
    version="0.3.0",
    title="Create Reconstruction Limits",
    description="Create a mesh defining the valid reconstructed area.",
    solution_creators=["Utz H. Ermel"],
    tags=["copick", "plane", "fitting", "surface", "segmentation"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "copick_config_path",
            "type": "string",
            "required": True,
            "description": "Path to the Copick configuration JSON file.",
        },
        {
            "name": "voxel_spacing",
            "type": "float",
            "required": True,
            "description": "Voxel spacing.",
        },
        {
            "name": "tomo_type",
            "type": "string",
            "required": True,
            "description": "Type of tomogram.",
        },
        {
            "name": "angle",
            "type": "float",
            "required": True,
            "description": "Angle of the plane.",
        },
        {
            "name": "output_object",
            "type": "string",
            "required": False,
            "default": "reconstruction",
            "description": "Name of the output object.",
        },
        {
            "name": "output_user",
            "type": "string",
            "required": False,
            "default": "fit",
            "description": "Name of the output user.",
        },
        {
            "name": "output_session",
            "type": "string",
            "required": False,
            "default": "0",
            "description": "Name of the output session.",
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
