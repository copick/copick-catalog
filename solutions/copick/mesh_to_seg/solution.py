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
  - scipy
  - trimesh[recommended,easy]
  - ome-zarr
  - pip:
    - album
    - "copick[all]>=0.6.0"
    - tqdm
    - rtree
    - manifold3d
"""


def run():
    # Imports
    import trimesh as tm
    import numpy as np
    import trimesh as tm
    from trimesh.ray.ray_triangle import RayMeshIntersector
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import tqdm
    from typing import Any, Dict, List, Sequence, Tuple
    import zarr
    import ome_zarr.writer

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

    def onesmask_z(mesh, voxel_dims, voxel_spacing):

        intersector = RayMeshIntersector(mesh)

        # XY
        # Create a grid of rays and intersect them with the mesh
        grid_x, grid_y = np.mgrid[0 : voxel_dims[0], 0 : voxel_dims[1]]
        ray_grid = (
            np.vstack([grid_x.ravel(), grid_y.ravel(), -np.ones((grid_x.size,))]).T
            * voxel_spacing
        )
        ray_dir = np.zeros((ray_grid.shape[0], 3))
        ray_dir[:, 2] = 1
        print("Intersecting Z")
        loc, _, _ = intersector.intersects_location(ray_grid, ray_dir)
        print("Done intersecting Z")

        # Sort by z-coordinate
        int_loc = np.round(loc / 7.84).astype("int")
        sort_idx = int_loc[:, 2].argsort()
        int_loc = int_loc[sort_idx, :]

        # Step through the z-coordinates and count the number of intersections
        img = np.zeros((voxel_dims[1], voxel_dims[0]), dtype="bool")
        vol = np.zeros((voxel_dims[2], voxel_dims[1], voxel_dims[0]), dtype="bool")

        for z in tqdm.trange(voxel_dims[2]):
            idx = int_loc[:, 2] == z
            img[int_loc[idx, 1], int_loc[idx, 0]] = np.logical_not(
                img[int_loc[idx, 1], int_loc[idx, 0]]
            )
            vol[z, :, :] = img

        return vol

    def onesmask_x(mesh, voxel_dims, voxel_spacing):

        intersector = RayMeshIntersector(mesh)

        # YZ
        # Create a grid of rays and intersect them with the mesh
        grid_y, grid_z = np.mgrid[0 : voxel_dims[1], 0 : voxel_dims[2]]
        ray_grid = (
            np.vstack([-np.ones((grid_y.size,)), grid_y.ravel(), grid_z.ravel()]).T
            * voxel_spacing
        )
        ray_dir = np.zeros((ray_grid.shape[0], 3))
        ray_dir[:, 0] = 1
        print("Intersecting X")
        loc, _, _ = intersector.intersects_location(ray_grid, ray_dir)
        print("Done intersecting X")

        # Sort by x-coordinate
        int_loc = np.round(loc / 7.84).astype("int")
        sort_idx = int_loc[:, 0].argsort()
        int_loc = int_loc[sort_idx, :]

        # Step through the x-coordinates and count the number of intersections
        img = np.zeros((voxel_dims[2], voxel_dims[1]), dtype="bool")
        vol = np.zeros((voxel_dims[2], voxel_dims[1], voxel_dims[0]), dtype="bool")

        for x in tqdm.trange(voxel_dims[0]):
            idx = int_loc[:, 0] == x
            img[int_loc[idx, 2], int_loc[idx, 1]] = np.logical_not(
                img[int_loc[idx, 2], int_loc[idx, 1]]
            )
            vol[:, :, x] = img

        return vol

    def onesmask(mesh, voxel_dims, voxel_spacing):
        vols = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futs = [
                executor.submit(onesmask_x, mesh.copy(), voxel_dims, voxel_spacing),
                executor.submit(onesmask_z, mesh.copy(), voxel_dims, voxel_spacing),
            ]

            for f in as_completed(futs):
                vols.append(f.result())

        return np.logical_and(vols[0], vols[1])

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

    def ome_zarr_transforms(voxel_size: float) -> List[Dict[str, Any]]:
        return [{"scale": [voxel_size, voxel_size, voxel_size], "type": "scale"}]

    # Parsing Input
    args = get_args()
    copick_config_path = args.copick_config_path
    run_names = args.run_names
    input_object = args.input_object
    input_user = args.input_user
    input_session = args.input_session

    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type

    output_object = input_object if args.output_object == "" else args.output_object
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

        mesh = run.get_meshes(
            object_name=input_object, user_id=input_user, session_id=input_session
        )[0].mesh
        mesh = ensure_mesh(mesh)

        vs = run.get_voxel_spacing(voxel_spacing)
        tomo = vs.get_tomogram(tomo_type)
        vox_dim = zarr.open(tomo.zarr())["0"].shape[::-1]

        vol = onesmask(mesh, vox_dim, voxel_spacing)

        nseg = run.get_segmentations(
            name=output_object,
            user_id=output_user,
            session_id=output_session,
            is_multilabel=False,
            voxel_size=voxel_spacing,
        )

        if len(nseg) == 0:
            nseg = run.new_segmentation(
                name=output_object,
                user_id=output_user,
                session_id=output_session,
                is_multilabel=False,
                voxel_size=voxel_spacing,
            )
        else:
            nseg = nseg[0]

        # Write the zarr file
        loc = nseg.zarr()
        root_group = zarr.group(loc, overwrite=True)

        ome_zarr.writer.write_multiscale(
            [vol.astype("uint8")],
            group=root_group,
            axes=ome_zarr_axes(),
            coordinate_transformations=[ome_zarr_transforms(voxel_spacing)],
            storage_options=dict(chunks=(256, 256, 256), overwrite=True),
            compute=True,
        )


setup(
    group="copick",
    name="mesh_to_seg",
    version="0.8.0",
    title="Convert Mesh to Segmentation",
    description="Convert a watertight mesh to a dense voxel segmentation.",
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
            "name": "input_object",
            "type": "string",
            "required": True,
            "description": "Name of the input object.",
        },
        {
            "name": "input_user",
            "type": "string",
            "required": True,
            "description": "Name of the input user.",
        },
        {
            "name": "input_session",
            "type": "string",
            "required": True,
            "description": "ID of the input session.",
        },
        {
            "name": "output_object",
            "type": "string",
            "required": False,
            "default": "",
            "description": "Name of the output object.",
        },
        {
            "name": "output_user",
            "type": "string",
            "required": False,
            "default": "from-mesh",
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
