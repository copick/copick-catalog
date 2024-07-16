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
  - scipy
  - pip:
    - album
    - "copick[all]>=0.5.2"
    - tqdm
    - manifold3d
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
    def ensure_mesh(trimesh_object):
        if isinstance(trimesh_object, tm.Scene):
            if len(trimesh_object.geometry) == 0:
                return None
            else:
                return tm.util.concatenate(
                    [g for g in trimesh_object.geometry.values()]
                )
        elif isinstance(trimesh_object, tm.Trimesh):
            return trimesh_object
        else:
            raise ValueError("Input must be a Trimesh or Scene object")

    # Parsing Input
    args = get_args()
    copick_config_path = args.copick_config_path
    run_names = args.run_names
    object_a = args.object_a
    input_user_a = args.user_a
    input_session_a = args.session_a
    object_b = args.object_b
    input_user_b = args.user_b
    input_session_b = args.session_b

    output_object = args.output_object
    output_user = args.output_user
    output_session = args.output_session

    # Code
    root = copick.from_file(copick_config_path)

    if run_names == "":
        run_names = [r.name for r in root.runs]
    else:
        run_names = args.run_names.split(",")

    for rname in run_names:
        run = root.get_run(rname)
        print(run.name)

        mesh_a = run.get_meshes(
            object_name=object_a, user_id=input_user_a, session_id=input_session_a
        )[0]
        mesh_b = run.get_meshes(
            object_name=object_b, user_id=input_user_b, session_id=input_session_b
        )[0]

        ma = ensure_mesh(mesh_a.mesh)
        ma.fix_normals()
        mb = ensure_mesh(mesh_b.mesh)
        mb.fix_normals()

        intersect = tm.boolean.intersection([ma, mb])

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

        nm.mesh = intersect
        nm.store()


setup(
    group="copick",
    name="intersect_mesh",
    version="0.5.0",
    title="Intersect two meshes",
    description="Compute the intersection of two meshes.",
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
            "name": "run_names",
            "type": "string",
            "required": False,
            "default": "",
            "description": "Comma-separated list of run names.",
        },
        {
            "name": "object_a",
            "type": "string",
            "required": True,
            "description": "Name of the first object.",
        },
        {
            "name": "user_a",
            "type": "string",
            "required": True,
            "description": "Name of the first user.",
        },
        {
            "name": "session_a",
            "type": "string",
            "required": True,
            "description": "ID of the first session.",
        },
        {
            "name": "object_b",
            "type": "string",
            "required": True,
            "description": "Name of the second object.",
        },
        {
            "name": "user_b",
            "type": "string",
            "required": True,
            "description": "Name of the second user.",
        },
        {
            "name": "session_b",
            "type": "string",
            "required": True,
            "description": "ID of the second session.",
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
