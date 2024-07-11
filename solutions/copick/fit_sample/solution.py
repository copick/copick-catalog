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
  - pytorch
  - trimesh
  - scipy
  - pip:
    - album
    - "copick[all] @ git+https://github.com/uermel/copick.git@2d29f48"
    - tqdm
    - torch-cubic-spline-grids
"""


def run():
    # Imports
    import numpy as np
    import torch
    import tqdm
    from typing import Sequence, Tuple
    import trimesh as tm
    import zarr

    from torch_cubic_spline_grids import CubicBSplineGrid2d
    import copick
    from copick.models import CopickPicks, CopickRun, CopickPoint, CopickLocation

    # Function definitions
    def fit_plane(
        picks: CopickPicks, resolution: Sequence[int], max_dim: Sequence[int]
    ) -> CubicBSplineGrid2d:
        pickarr = np.ndarray((len(picks.points), 3))
        for i, p in enumerate(picks.points):
            pickarr[i, :] = [p.location.x, p.location.y, p.location.z]

        x = pickarr[:, 0:2]
        x[:, 0] = x[:, 0] / max_dim[0]
        x[:, 1] = x[:, 1] / max_dim[1]

        y = pickarr[:, 2] / max_dim[2]

        grid = CubicBSplineGrid2d(resolution=tuple(resolution), n_channels=1)
        optimizer = torch.optim.Adam(grid.parameters(), lr=0.1)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        t = tqdm.trange(500, desc="RMS: ", leave=True)

        for i in t:
            pred = grid(x).squeeze()

            optimizer.zero_grad()
            loss = torch.sum((pred - y) ** 2) ** 0.5
            loss.backward()
            optimizer.step()
            t.set_description(f"RMS: {loss.item()}")
            t.refresh()

        return grid

    def grid_to_picks(
        grid: CubicBSplineGrid2d,
        resolution: Sequence[int],
        max_dim: Sequence[int],
        run: CopickRun,
        out_name: str,
        out_session: str = "0",
        out_user: str = "fit",
    ):
        x_test, y_test = torch.meshgrid(
            torch.linspace(0, 1, resolution[0]),
            torch.linspace(0, 1, resolution[1]),
            indexing="xy",
        )
        x_test = torch.stack([x_test, y_test], dim=2).view(-1, 2)

        testpoints = x_test.detach().numpy()
        npickarr = np.ndarray((testpoints.shape[0], 3))
        npickarr[:, 0] = x_test.detach().numpy()[:, 0] * max_dim[0]
        npickarr[:, 1] = x_test.detach().numpy()[:, 1] * max_dim[1]
        npickarr[:, 2] = grid(x_test).squeeze().detach().numpy() * max_dim[2]

        points = []
        for i in range(npickarr.shape[0]):
            points.append(
                CopickPoint(
                    location=CopickLocation(
                        x=npickarr[i, 0], y=npickarr[i, 1], z=npickarr[i, 2]
                    )
                )
            )

        pp = run.get_picks(
            object_name=out_name, user_id=out_user, session_id=out_session
        )

        if len(pp) == 0:
            pp = run.new_picks(
                object_name=out_name, session_id=out_session, user_id=out_user
            )
        else:
            pp = pp[0]

        pp.points = points
        pp.store()

        return pp, npickarr

    def triangulate_rect_grid(array, dim) -> Tuple[np.ndarray, np.ndarray]:
        npickarr = array.reshape(dim[0], dim[1], 3)

        vertices = []
        tris = []

        for i in range(dim[0] - 1):
            for j in range(dim[1] - 1):
                v1 = npickarr[i, j, :]
                v2 = npickarr[i + 1, j, :]
                v3 = npickarr[i + 1, j + 1, :]
                v4 = npickarr[i, j + 1, :]

                vertices.extend([v1, v2, v3, v4])
                lmax = len(vertices)
                tris.append([lmax - 2, lmax - 3, lmax - 4])
                tris.append([lmax - 4, lmax - 1, lmax - 2])

        vertices = np.array(vertices)
        tris = np.array(tris)

        return vertices, tris

    def fill_side(array1, array2, dim) -> Tuple[np.ndarray, np.ndarray]:
        arr1 = array1.reshape(dim[0], dim[1], 3)
        arr2 = array2.reshape(dim[0], dim[1], 3)

        vertices = []
        tris = []

        for i in range(49):
            for j in [0, dim[0] - 1]:
                v1 = arr1[j, i]
                v2 = arr2[j, i]
                v3 = arr2[j, i + 1]
                v4 = arr1[j, i + 1]

                vertices.extend([v1, v2, v3, v4])
                lmax = len(vertices)
                tris.append([lmax - 2, lmax - 3, lmax - 4])
                tris.append([lmax - 4, lmax - 1, lmax - 2])

            for j in [0, dim[1] - 1]:
                v1 = arr1[i, j]
                v2 = arr2[i, j]
                v3 = arr2[i + 1, j]
                v4 = arr1[i + 1, j]

                vertices.extend([v1, v2, v3, v4])
                lmax = len(vertices)
                tris.append([lmax - 2, lmax - 3, lmax - 4])
                tris.append([lmax - 4, lmax - 1, lmax - 2])

        vertices = np.array(vertices)
        tris = np.array(tris)

        return vertices, tris

    def triangulate_box(arr1, arr2, dim) -> tm.parent.Geometry:
        m1v, m1f = triangulate_rect_grid(arr1, dim)
        m2v, m2f = triangulate_rect_grid(arr2, dim)
        sv, sf = fill_side(arr1, arr2, dim)
        full = tm.util.append_faces(
            vertices_seq=[m1v, m2v, sv], faces_seq=[m1f, m2f, sf]
        )
        mesh = tm.Trimesh(vertices=full[0], faces=full[1])
        mesh.fix_normals()
        return mesh

    # Parsing Input
    args = get_args()
    copick_config_path = args.copick_config_path
    run_names = args.run_names
    top_object = args.top_object
    bottom_object = args.bottom_object
    input_user = args.input_user
    input_session = args.input_session
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type

    grid_resolution = [int(a) for a in args.grid_resolution.split(",")]
    fit_resolution = [int(a) for a in args.fit_resolution.split(",")]
    assert (
        len(grid_resolution) == 2
    ), "Grid Resolution must be 2 comma-separated integers."
    assert (
        len(fit_resolution) == 2
    ), "Fit Resolution must be 2 comma-separated integers."

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
        bottom = run.get_picks(
            object_name=bottom_object, user_id=input_user, session_id=input_session
        )[0]
        top = run.get_picks(
            object_name=top_object, user_id=input_user, session_id=input_session
        )[0]

        vs = run.get_voxel_spacing(voxel_spacing)
        tomo = vs.get_tomogram(tomo_type)
        max_dim = [d * voxel_spacing for d in zarr.open(tomo.zarr())["0"].shape[::-1]]

        bottom_grid = fit_plane(bottom, grid_resolution, max_dim)
        top_grid = fit_plane(top, grid_resolution, max_dim)

        picks_bot, points_bot = grid_to_picks(
            bottom_grid,
            fit_resolution,
            max_dim,
            run,
            out_name=bottom_object,
            out_user=output_user,
            out_session=output_session,
        )
        picks_top, points_top = grid_to_picks(
            top_grid,
            fit_resolution,
            max_dim,
            run,
            out_name=top_object,
            out_user=output_user,
            out_session=output_session,
        )

        full = triangulate_box(points_bot, points_top, fit_resolution)

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

        nm.mesh = full
        nm.store()


setup(
    group="copick",
    name="fit_sample",
    version="0.6.0",
    title="Fit Sample Volume",
    description="fit a mesh describing the sample from two sets of points defining upper and lower boundary.",
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
            "name": "top_object",
            "type": "string",
            "required": False,
            "default": "top-layer",
            "description": "Name of the object defining top layer.",
        },
        {
            "name": "bottom_object",
            "type": "string",
            "required": False,
            "default": "bottom-layer",
            "description": "Name of the object defining bottom layer.",
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
            "name": "grid_resolution",
            "type": "string",
            "required": False,
            "default": "5,5",
            "description": "Resolution of the grid (2 comma-separated ints).",
        },
        {
            "name": "fit_resolution",
            "type": "string",
            "required": False,
            "default": "50,50",
            "description": "Resolution of the fit (2 comma-separated ints).",
        },
        {
            "name": "output_object",
            "type": "string",
            "required": False,
            "default": "sample",
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
