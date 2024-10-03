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
  - pip:
    - album
    - copick
"""

def run():
    import copick
    import numpy as np
    import zarr

    # Get command-line arguments
    args = get_args()
    copick_config_path = args.copick_config_path
    voxel_spacing = float(args.voxel_spacing)
    input_user_id = args.input_user_id
    input_session_id = args.input_session_id
    run_name = args.run_name
    output_user_id = args.output_user_id
    multilabel_name = args.multilabel_name

    # Load the Copick project
    project = copick.from_file(copick_config_path)

    # Determine which runs to process
    runs_to_process = []
    if run_name:
        run = project.get_run(run_name)
        if run is None:
            raise ValueError(f"Run '{run_name}' not found in the project.")
        runs_to_process = [run]
    else:
        runs_to_process = project.runs

    # Get all pickable objects (including particles)
    pickable_objects = project.pickable_objects

    # Process each run
    for run in runs_to_process:
        print(f"Processing run: {run.name}")
        
        # Get the voxel spacing
        voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
        if voxel_spacing_obj is None:
            print(f"Voxel spacing {voxel_spacing} not found in run '{run.name}'. Skipping this run.")
            continue

        # Initialize an empty multilabel segmentation
        multilabel_segmentation = None
        shape = None

        # Process each pickable object
        for obj in pickable_objects:
            # Get the segmentation for this object (particle or non-particle)
            segmentations = run.get_segmentations(
                user_id=input_user_id,
                session_id=input_session_id,
                name=obj.name,
                voxel_size=voxel_spacing,
                is_multilabel=False
            )
            
            if segmentations:
                seg = segmentations[0]  # Assume there's only one matching segmentation
                mask = seg.numpy()
                
                if multilabel_segmentation is None:
                    shape = mask.shape
                    multilabel_segmentation = np.zeros(shape, dtype=np.uint16)
                
                # Add this mask to the multilabel segmentation using the object's label
                multilabel_segmentation[mask > 0] = obj.label

        if multilabel_segmentation is None:
            print(f"No valid segmentations found to create a multilabel segmentation for run '{run.name}'. Skipping this run.")
            continue

        # Create a new multilabel segmentation
        new_segmentation = run.new_segmentation(
            voxel_size=voxel_spacing,
            name=multilabel_name,
            session_id="0",
            is_multilabel=True,
            user_id=output_user_id
        )

        # Save the multilabel segmentation
        new_segmentation.from_numpy(multilabel_segmentation)

        print(f"Multilabel segmentation '{multilabel_name}' created successfully for run '{run.name}' at voxel spacing {voxel_spacing}.")

    print("Processing completed for all specified runs.")

setup(
    group="copick",
    name="binary-to-multilabel-segmentation",
    version="0.0.2",
    title="Create Multilabel Segmentation from Individual Segmentations",
    description="Creates a multilabel segmentation by combining individual segmentations for all pickable objects, including particles, in a Copick project. Can process a single run or all runs in the project.",
    solution_creators=["Kyle Harrington"],
    tags=["copick", "segmentation", "multilabel"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration file."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing to use for the segmentation."},
        {"name": "input_user_id", "type": "string", "required": True, "description": "User ID for input segmentations."},
        {"name": "input_session_id", "type": "string", "required": True, "description": "Session ID for input segmentations."},
        {"name": "run_name", "type": "string", "required": False, "description": "Name of the run to process. If not provided, all runs will be processed."},
        {"name": "output_user_id", "type": "string", "required": True, "description": "User ID for the output multilabel segmentation."},
        {"name": "multilabel_name", "type": "string", "required": True, "description": "Name for the new multilabel segmentation."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)