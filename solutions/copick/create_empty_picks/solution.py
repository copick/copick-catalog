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
  - pip:
    - album
    - "copick[all]>=0.5.2"
"""


def run():
    # Imports
    import copick
    from copick.impl.filesystem import CopickRootFSSpec
    import copy
    import os

    # Function definitions

    # Parsing Input
    args = get_args()
    copick_config_path = args.copick_config_path
    out_object = args.out_object
    out_user = args.out_user
    out_session = args.out_session
    overwrite = args.overwrite
    run_names = args.run_names

    # Create Copick project
    root = copick.from_file(copick_config_path)

    # Write to static root
    if root.config.config_type == "filesystem":
        if root.config.static_root is not None:
            os.makedirs(root.config.static_root.replace("local://", ""), exist_ok=True)
            os.makedirs(root.config.overlay_root.replace("local://", ""), exist_ok=True)

            config = copy.copy(root.config)
            config.overlay_root = config.static_root
            config.overlay_fs_args = config.static_fs_args

            root = CopickRootFSSpec(config)

    if run_names == "":
        run_names = [r.name for r in root.runs]
    else:
        run_names = args.run_names.split(",")

    # Create picks
    for name in run_names:
        run = root.get_run(name)
        print(f"Creating picks for {run.name}")

        picks = run.get_picks(
            object_name=out_object, user_id=out_user, session_id=out_session
        )

        if len(picks) == 0:
            picks = run.new_picks(
                object_name=out_object, user_id=out_user, session_id=out_session
            )
        else:
            if overwrite:
                picks = picks[0]
            else:
                raise ValueError(
                    f"Picks already exist for {run.name}. Set overwrite to True to overwrite."
                )

        picks.points = []
        picks.store()


setup(
    group="copick",
    name="create_empty_picks",
    version="0.3.1",
    title="Create empty picks.",
    description="Create empty picks inside a copick project.",
    solution_creators=["Utz H. Ermel"],
    tags=["copick", "setup", "picks", "creation"],
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
            "name": "out_object",
            "description": "Name of the output pickable object.",
            "type": "string",
            "required": True,
        },
        {
            "name": "out_user",
            "description": "User/Tool name for output points.",
            "type": "string",
            "required": True,
        },
        {
            "name": "out_session",
            "description": "Output session, indicating this set was generated by a tool.",
            "type": "string",
            "required": True,
        },
        {
            "name": "overwrite",
            "description": "Whether to overwrite existing picks.",
            "type": "boolean",
            "required": False,
            "default": False,
        },
        {
            "name": "run_names",
            "type": "string",
            "required": False,
            "default": "",
            "description": "Comma-separated list of run names.",
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
