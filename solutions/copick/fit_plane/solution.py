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

        return pp

    # Parsing Input
    args = get_args()
    copick_config_path = args.copick_config_path
    run_names = args.run_names
    input_object = args.input_object
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
        points = run.get_picks(
            object_name=input_object, user_id=input_user, session_id=input_session
        )[0]

        vs = run.get_voxel_spacing(voxel_spacing)
        tomo = vs.get_tomogram(tomo_type)
        max_dim = [d * voxel_spacing for d in zarr.open(tomo.zarr())["0"].shape[::-1]]

        bottom_grid = fit_plane(points, grid_resolution, max_dim)

        grid_to_picks(
            bottom_grid,
            fit_resolution,
            max_dim,
            run,
            out_name=output_object,
            out_user=output_user,
            out_session=output_session,
        )


setup(
    group="copick",
    name="fit_plane",
    version="0.5.0",
    title="Fit Plane",
    description="fit a plane to a set of copick points.",
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
            "default": "",
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
