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
    import numpy as np
    from scipy.stats.qmc import PoissonDisk
    import trimesh as tm
    from typing import Any, Dict, List, Sequence, Tuple
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

    def poisson_disk_in_out(
        n_in: int,
        n_out: int,
        mesh: tm.Trimesh,
        max_dim: Sequence[float],
        min_dist: float,
        edge_dist: float,
        input_points: np.ndarray,
        seed: int = 1234,
    ):
        max_max = np.max(max_dim)
        min_dist = min_dist / max_max

        engine = PoissonDisk(d=3, radius=min_dist, seed=seed)

        # Fill space
        points = engine.fill_space() * max_max

        # Reject points outside the volume
        lb = np.array([edge_dist, edge_dist, edge_dist])
        ub = max_dim - np.array([edge_dist, edge_dist, edge_dist])
        points = points[np.all(np.logical_and(points > lb, points < ub), axis=1), :]

        # Reject points that are too close to the input points
        for pt in input_points:
            dist = np.linalg.norm(points - pt, axis=1)
            points = points[dist > min_dist]

        # Check if points are inside/outside the mesh
        mask = mesh.contains(points)
        inv_mask = np.logical_not(mask)

        points_in = points[mask, :]
        points_out = points[inv_mask, :]

        # Shuffle output
        np.random.default_rng(seed).shuffle(points_in)
        np.random.default_rng(seed).shuffle(points_out)

        # Limit number of points to n_in and n_out
        if n_in > points_in.shape[0]:
            print(
                f"Warning: Not enough points inside the mesh. Requested {n_in}, found {points_in.shape[0]}"
            )
        n_in = min(n_in, points_in.shape[0])
        final_points_in = points_in[:n_in, :]

        if n_out > points_out.shape[0]:
            print(
                f"Warning: Not enough points outside the mesh. Requested {n_out}, found {points_out.shape[0]}"
            )
        n_out = min(n_out, points_out.shape[0])
        final_points_out = points_out[:n_out, :]

        return final_points_in, final_points_out

    # Parsing Input
    args = get_args()
    copick_config_path = args.copick_config_path
    run_names = args.run_names
    input_object = args.input_object
    input_user = args.input_user
    input_session = args.input_session

    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type

    num_surf = args.num_surf
    num_internal = args.num_internal
    num_random = args.num_random
    edge_dist = args.edge_dist
    min_dist = args.min_dist
    ang_dist = edge_dist * voxel_spacing
    seed = args.seed if args.seed != 0 else None

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

        vs = run.get_voxel_spacing(voxel_spacing)
        tomo = vs.get_tomogram(tomo_type)
        pixel_max_dim = zarr.open(tomo.zarr())["0"].shape[::-1]
        max_dim = np.array([d * voxel_spacing for d in pixel_max_dim])

        mesh = run.get_meshes(
            object_name=input_object, user_id=input_user, session_id=input_session
        )[0].mesh
        mesh = ensure_mesh(mesh)

        points_surf, _ = tm.sample.sample_surface_even(
            mesh, num_surf, radius=min_dist, seed=seed
        )
        points_vol, points_rand = poisson_disk_in_out(
            num_internal, num_random, mesh, max_dim, min_dist, ang_dist, points_surf
        )
        points_full = np.concatenate([points_surf, points_vol, points_rand], axis=0)

        copick_points = []

        for idx in range(points_full.shape[0]):
            pt = points_full[idx, :]
            if (
                pt[0] < ang_dist
                or pt[0] > max_dim[0] - ang_dist
                or pt[1] < ang_dist
                or pt[1] > max_dim[1] - ang_dist
                or pt[2] < ang_dist
                or pt[2] > max_dim[2] - ang_dist
            ):
                continue

            copick_points.append(
                CopickPoint(location=CopickLocation(x=pt[0], y=pt[1], z=pt[2]))
            )

        nep = run.get_picks(
            object_name=output_object, user_id=output_user, session_id=output_session
        )

        if len(nep) == 0:
            nep = run.new_picks(
                object_name=output_object,
                user_id=output_user,
                session_id=output_session,
            )
        else:
            nep = nep[0]

        nep.points = copick_points
        nep.store()


setup(
    group="copick",
    name="sample_mesh",
    version="0.6.0",
    title="Sample points in/on/outside a mesh.",
    description="Sample random points in/on/outside a mesh.",
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
            "name": "num_surf",
            "type": "integer",
            "required": False,
            "default": 500,
            "description": "Approx. number of points on the mesh surface.",
        },
        {
            "name": "num_internal",
            "type": "integer",
            "required": False,
            "default": 500,
            "description": "Approx. number of points inside the mesh.",
        },
        {
            "name": "num_random",
            "type": "integer",
            "required": False,
            "default": 100,
            "description": "Approx. number of random points outside the mesh (negative examples).",
        },
        {
            "name": "min_dist",
            "type": "float",
            "required": False,
            "default": 500,
            "description": "Minimum distance (in angstrom) between points.",
        },
        {
            "name": "edge_dist",
            "type": "float",
            "required": False,
            "default": 32,
            "description": "Minimum distance (in voxels) from the volume bounds.",
        },
        {
            "name": "seed",
            "type": "integer",
            "required": False,
            "default": 0,
            "description": "Random seed.",
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
            "default": "sampled",
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
