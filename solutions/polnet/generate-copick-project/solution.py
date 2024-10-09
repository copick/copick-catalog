###album catalog: copick

from album.runner.api import get_args, setup

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python==3.9
  - pip
  - mrcfile
  - numpy 
  - pandas
  - pip:
    - album
    - copick
    - "git+https://github.com/jtschwar/polnet"
    - "git+https://github.com/copick/copick-utils"
"""

def run():
    from scipy.spatial.transform import Rotation as R
    from gui.core.all_features2 import all_features2 
    from copick_utils.writers import write
    import mrcfile, copick, os
    import pandas as pd
    import numpy as np

    def add_points(copick_run, csvFile, in_user_id, in_session_id):

        unique_labels = np.unique(csvFile['Label'])
        for ii in range(1,len(unique_labels)+1):

            try:
                proteinName = csvFile[csvFile['Label'] == ii]['Code'].iloc[0].split('_')[0]

                points = csvFile[csvFile['Label'] == ii][['X', 'Y', 'Z']]
                
                qOrientations =  csvFile[csvFile['Label'] == ii][['Q2', 'Q3', 'Q4', 'Q1']]

                orientations = np.zeros([points.shape[0], 4, 4])
                for jj in range(points.shape[0]):

                    v = qOrientations.iloc[jj].to_numpy()

                    # # Create a rotation object from quaternions, reshape needed to maintain the correct input shape
                    r = R.from_quat(v)         

                    # # Convert to Euler angles using the 'zyx' sequence, output angles in degrees
                    orientations[jj,:3,:3] = r.as_matrix()  

                # Translation in Final Column Should Be 1
                orientations[:,3,3] = 1

                # Create New Picks and Save In Run
                picks = copick_run.new_picks(object_name=proteinName, 
                                            user_id=in_user_id, session_id=in_session_id) 
                picks.from_numpy(points.to_numpy(), orientations)               
            except Exception as e: 
                pass

    def extract_membrane_segmentation(segVol, csvFile, pickable_objects):

        membranes = np.zeros(segVol.shape, dtype=np.uint8)
        unique_labels = np.unique(csvFile['Label'])
        for ii in range(1,len(unique_labels)+1):

            proteinName = csvFile[csvFile['Label'] == ii]['Code'].iloc[0].split('_')[0]

            if proteinName not in pickable_objects:
                membranes[segVol == ii] = 1

        return membranes            

    # Splitting Arguments
    def split_args(arg):
        return arg.split(',') if arg else []

    # Splitting 
    def split_float_args(arg):
        return [float(x) for x in arg.split(',')] if arg else []
    
    # Splitting 
    def split_int_args(arg):
        return [int(x) for x in arg.split(',')] if arg else []   

    # Fetch arguments
    args = get_args()

    # Setup paths and configurations
    COPICK_CONFIG_PATH = args.copick_config_path  # Path to the Copick configuration file
    PROTEINS_LIST = split_args(args.proteins_list)
    MB_PROTEINS_LIST = split_args(args.mb_proteins_list)
    MEMBRANES_LIST = split_args(args.membranes_list)

    SESSION_ID = '0'
    USER_ID = 'polnet'
    SEGMENTATION_NAME = 'membrane'
    TOMO_TYPE = 'wbp'

    # Initialize Copick root
    root = copick.from_file(COPICK_CONFIG_PATH)

    # Determine Objects Are Particles
    objects = root.pickable_objects
    pickable_objects = [o.name for o in objects if o.is_particle]    

    # Define tomography and feature extraction parameters
    NTOMOS = 1
    SURF_DEC = 0.9
    MMER_TRIES = 20
    PMER_TRIES = 100
    VOI_SHAPE = split_int_args(args.tomo_shape)
    x = VOI_SHAPE[0]; y = VOI_SHAPE[1]; z = VOI_SHAPE[2]
    VOI_OFFS = ((int(x * 0.025), int(x * 0.975)), 
                (int(y * 0.025), int(y * 0.975)), 
                (int(z * 0.025), int(z * 0.975)) )    
    
    voxel_spacing = args.voxel_size
    NUM_TOMOS_PER_SNR = args.num_tomos_per_snr
    DETECTOR_SNR = split_float_args(args.snr)
    minAng, maxAng, angIncr = split_float_args(args.tilt_range)

    # Generate a list of angles
    TILT_ANGS = [angle for angle in 
                (minAng + i * angIncr for i in range(int((maxAng - minAng) / angIncr) + 1))
                if angle <= maxAng]

    # Do we Always want to Hard Code this ?
    MALIGN_MIN, MALIGN_MAX, MALIGN_SIGMA = 1, 5, 0.5

    print(f'\n[Polnet - Protein Inputs]\nConfig_Path: {COPICK_CONFIG_PATH}\nProtein List: {PROTEINS_LIST}\nMembrane-Bound Protein List: {MB_PROTEINS_LIST}\nMembranes List: {MEMBRANES_LIST}')
    print(f'\n[Polnet - Copick Write Query]\nUserID: {USER_ID}\nSessionID: {SESSION_ID}\nSegmentation Name: {SEGMENTATION_NAME}\nTomogram Name: {TOMO_TYPE}')
    print(f'\n[Polnet - Tomogram Simulation Parameters]\nVoxel Size: {voxel_spacing}\nTomo Dimensions (Voxels): {VOI_SHAPE}\nSNR: {DETECTOR_SNR}\n'
         f'Number of Tomos Per SNR: {NUM_TOMOS_PER_SNR}\nTilt Series Range (Min,Max,Delta): {minAng}, {maxAng}, {angIncr}\nTilt Angles: {TILT_ANGS}\n')    

    # Iterate Per SNR
    run_ids = [run.name for run in root.runs] 
    if len(run_ids) == 0:   currTSind = 1
    else:   currTSind = int(max(run_ids, key=lambda x: int(x.split('_')[1])).split('_')[1]) + 1
    for SNR in DETECTOR_SNR:

        # Per SNR, Produce A Certain Number of Tomograms
        for ii in range(NUM_TOMOS_PER_SNR):          

            # Create a permanent directory in ./tmp_polnet_output
            RUN_NAME = f'TS_{currTSind}'
            permanent_dir = f"./tmp_polnet_output/{RUN_NAME}"

            # Ensure Copick run exists
            copick_run = root.get_run(RUN_NAME)
            if not copick_run:
                copick_run = root.new_run(RUN_NAME)

            TEM_DIR = os.path.join(permanent_dir, 'tem')
            TOMOS_DIR = os.path.join(permanent_dir, 'tomos')

            # Call the function to generate features
            all_features2(NTOMOS, VOI_SHAPE, permanent_dir, VOI_OFFS, voxel_spacing, MMER_TRIES, PMER_TRIES,
                        MEMBRANES_LIST, [], PROTEINS_LIST, MB_PROTEINS_LIST, SURF_DEC,
                        TILT_ANGS, SNR, MALIGN_MIN, MALIGN_MAX, MALIGN_SIGMA ) 
        
            # Read CSV and Write Coordinates to StarFile
            points_csv_path = os.path.join(permanent_dir, 'tomos_motif_list.csv')        
            csvFile = pd.read_csv( points_csv_path, delimiter='\t' )
            add_points(copick_run, csvFile, USER_ID, SESSION_ID)

            vol = mrcfile.read (os.path.join(TEM_DIR, 'out_rec3d.mrc') )
            write.tomogram(copick_run, vol, voxelSize = voxel_spacing)

            ground_truth = mrcfile.read( os.path.join(TOMOS_DIR, 'tomo_lbls_0.mrc'))
            # write.segmentation(copick_run, ground_truth, USER_ID, voxelSize=voxel_spacing)                         

            # Extract Membranes and Save as Binary Segmentation
            membranes = extract_membrane_segmentation(ground_truth, csvFile, pickable_objects)
            write.segmentation(copick_run, membranes, USER_ID, 
                            segmentationName='membrane', voxelSize=voxel_spacing, 
                            multilabel_seg = False)            

setup(
    group="polnet",
    name="generate-copick-project",
    version="0.1.0",
    title="Generate a Copick Project with polnet",
    description="Generate a Copick Project Composed with polnet simulations.",
    solution_creators=["Jonathan Schwartz and Kyle Harrington"],
    cite=[{"text": "Martinez-Sanchez A.*, Jasnin M., Phelippeau H. and Lamm L. Simulating the cellular context in synthetic datasets for cryo-electron tomography, bioRxiv.", "url": "https://github.com/anmartinezs/polnet"}],
    tags=["synthetic data", "deep learning", "cryoet", "tomogram"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration file"},
        {"name": "num_tomos_per_snr", "type": "integer", "required": False, "default": 1, "description": "Number of Tomograms to Produce Per Specified SNR"},
        {"name": "snr", "type": "string", "required": False, "default": "0.5", "description": "Comma-separated list of SNRs to Apply to Tomograms"}, 
        {"name": "tilt_range", "type":"string", "required": False, "default": "-60,60,3", "description": "Comma-separated List of Min,Max and Increment for the Tilt Range"},
        {"name": "tomo_shape", "type":"string", "required": False, "default": "630,630,200", "description": "Comma-separated List of Tomogram Dimensions (in Pixels)" },
        {"name": "voxel_size", "type": "float", "required": False, "default": 10, "description": "Voxel Size for Simulated Tomograms"},     
        {"name": "proteins_list", "type": "string", "required": True, "description": "Comma-separated list of protein file paths"},
        {"name": "mb_proteins_list", "type": "string", "required": False,  "default": "", "description": "Comma-separated list of membrane protein file paths"},
        {"name": "membranes_list", "type": "string", "required": False, "default": "","description": "Comma-separated list of membrane file paths"}
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
