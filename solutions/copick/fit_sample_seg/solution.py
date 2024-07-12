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
  - numpy<=2
  - pytorch
  - trimesh
  - scipy
  - scikit-image
  - pip:
    - album
    - "copick[all] @ git+https://github.com/uermel/copick.git@2d29f48"
    - tqdm
    - torch-cubic-spline-grids
"""


def run():
    # Imports
    import numpy as np
    from skimage import measure
    import torch
    import tqdm
    from typing import Sequence, Tuple
    import trimesh as tm
    import zarr

    from torch_cubic_spline_grids import CubicBSplineGrid2d
    import copick
    from copick.models import CopickPicks, CopickRun, CopickPoint, CopickLocation

    # Function definitions
    def xy_to_z(xy, plane_normal, plane_offset):
        plane_normal = plane_normal / torch.norm(plane_normal)
        d = torch.matmul(xy, plane_normal[[2, 1]]) + plane_offset
        return -d / plane_normal[0]

    def get_largest_component(invol: np.ndarray) -> np.ndarray:
        out = measure.label(invol)
        props = measure.regionprops(out)
        largest = max(props, key=lambda x: x.area)
        out[out != largest.label] = 0
        out[out == largest.label] = 1
        return out

    def fit_plane_to_vol(
        invol: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vol = torch.tensor(invol, dtype=torch.float32).to(device)
        voldim = torch.tensor(vol.shape, dtype=torch.int32).to(device)

        # For evaluation of the b-spline function
        yy2, xx2 = torch.meshgrid(
            torch.arange(0, vol.shape[1], 1),
            torch.arange(0, vol.shape[2], 1),
            indexing="ij",
        )

        # For computing the mask
        zz, _, _ = torch.meshgrid(
            torch.arange(0, vol.shape[0], 1),
            torch.arange(0, vol.shape[1], 1),
            torch.arange(0, vol.shape[2], 1),
            indexing="ij",
        )

        # Norm to [0, 1]
        xx2 = xx2 / voldim[2]
        yy2 = yy2 / voldim[1]
        zz = zz / voldim[0]
        xy2 = torch.stack([yy2, xx2], dim=2).view(-1, 2).to(device)
        zz.to(device)

        plane_normal = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0])).to(device)
        top_offset = torch.nn.Parameter(torch.tensor([-0.8])).to(device)
        bot_offset = torch.nn.Parameter(torch.tensor([-0.2])).to(device)

        optimizer = torch.optim.Adam([plane_normal, top_offset, bot_offset], lr=0.1)

        t = tqdm.trange(20, desc="1 - IoU: ", leave=True)

        for _ in t:
            # Eval the function
            optimizer.zero_grad()
            zz_top = (
                xy_to_z(xy2, plane_normal, top_offset)
                .squeeze()
                .reshape((voldim[1], voldim[2]))
            ).to(device)
            zz_bot = (
                xy_to_z(xy2, plane_normal, bot_offset)
                .squeeze()
                .reshape((voldim[1], voldim[2]))
            ).to(device)

            valid = torch.sigmoid(1000 * (zz_top - zz)) * torch.sigmoid(
                1000 * (zz - zz_bot)
            )

            intersection = torch.sum(valid * vol)
            union = torch.sum(valid) + torch.sum(vol) - intersection
            loss = 1 - intersection / union

            loss.backward()
            optimizer.step()
            t.set_description(f"1 - IoU: {loss.item()}")
            t.refresh()

        return plane_normal, top_offset, bot_offset

    def plane_to_picks(
        plane_normal: torch.Tensor,
        plane_offset: torch.Tensor,
        resolution: Sequence[int],
        max_dim: Sequence[int],
        run: CopickRun,
        out_name: str,
        out_session: str = "0",
        out_user: str = "fit",
    ):
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, resolution[1]),
            torch.linspace(0, 1, resolution[0]),
            indexing="ij",
        )
        x_test = torch.stack([yy, xx], dim=2).view(-1, 2)

        testpoints = x_test.detach().numpy()
        npickarr = np.ndarray((testpoints.shape[0], 3))
        npickarr[:, 0] = x_test.detach().numpy()[:, 1] * max_dim[0]
        npickarr[:, 1] = x_test.detach().numpy()[:, 0] * max_dim[1]
        npickarr[:, 2] = (
            xy_to_z(x_test, plane_normal, plane_offset).squeeze().detach().numpy()
            * max_dim[2]
        )

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
    seg_name = args.seg_name
    top_object = args.top_object
    bottom_object = args.bottom_object
    seg_label_name = args.seg_label_name
    input_user = args.input_user
    input_session = args.input_session
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type

    fit_resolution = [int(a) for a in args.fit_resolution.split(",")]

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

    label = root.get_object(name=seg_label_name).label

    for rname in run_names:
        run = root.get_run(rname)
        print(run.name)

        vs = run.get_voxel_spacing(voxel_spacing)
        tomo = vs.get_tomogram(tomo_type)
        max_dim = [d * voxel_spacing for d in zarr.open(tomo.zarr())["0"].shape[::-1]]

        print("\t Getting Segmentation")
        seg = run.get_segmentations(
            user_id=input_user,
            session_id=input_session,
            name=seg_name,
            voxel_size=voxel_spacing,
            is_multilabel=True,
        )[0]
        vol = np.array(zarr.open(seg.zarr())["0"])
        vol[vol != label] = 0
        vol[vol == label] = 1
        print("\t Getting Largest Component")
        vol = get_largest_component(vol)

        print("\t Fitting Box")
        plane_normal, top_offset, bot_offset = fit_plane_to_vol(vol)

        print("\t Generate Picks")
        picks_bot, points_bot = plane_to_picks(
            plane_normal,
            bot_offset,
            fit_resolution,
            max_dim,
            run,
            out_name=bottom_object,
            out_user=output_user,
            out_session=output_session,
        )
        picks_top, points_top = plane_to_picks(
            plane_normal,
            top_offset,
            fit_resolution,
            max_dim,
            run,
            out_name=top_object,
            out_user=output_user,
            out_session=output_session,
        )

        print("\t Generate Mesh")
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
    name="fit_sample_seg",
    version="0.4.0",
    title="Fit Sample Volume from segmentation",
    description="fit a mesh describing the sample from a binary segmentation.",
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
            "name": "seg_name",
            "type": "string",
            "required": False,
            "default": "segmentation",
            "description": "Name of the object defining bottom layer.",
        },
        {
            "name": "seg_label_name",
            "type": "string",
            "required": False,
            "default": "valid-sample",
            "description": "Name of the label defining the sample.",
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
