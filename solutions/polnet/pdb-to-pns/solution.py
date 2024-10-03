###album catalog: copick

from album.runner.api import get_args, setup

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python>3.10
  - pip
  - mrcfile
  - numpy
  - pip:
    - album
    - "git+https://github.com/jtschwar/polnet"
    - biotite
"""

args = [
    {
        "name": "pdb_id",
        "description":"PDB ID for the Macromolecule to Model",
        "type": "string",
        "required": True, 
    },  
    {
        "name": "protein_name",
        "description":"Name of the Protein for the Associated PDB ID",
        "type": "string",
        "required": True, 
    },  
    {
        "name": "pdb_write_path",
        "description":"Folder Path for Where to Save the Downloaded PDB File",
        "type": "string",
        "required": True, 
    },   
    {
        "name": "mrc_write_path",
        "description":"Folder Path for Where to Save the Converted PDB file to MRC",
        "type": "string",
        "required": True, 
    },    
    {
        "name": "polnet_write_path",
        "description":"Folder Path for Where to Save the Converted MRC file for Polnet",
        "type": "string",
        "required": True, 
    },              
    {
        "name": "voxel_size",
        "description":"Voxel size per Ångström, scale of the tomogram resolution",
        "type": "float",
        "required": False, 
        "default": 10,
    }, 
    {
        "name": "resolution",
        "description":"Tomogram resolution in Ångströms.",
        "type": "float",
        "required": False, 
        "default": 30,
    },
    {
        "name": "mmer_iso",
        "description": "Isocontour",
        "type": "float",
        "required": False,
        "default": 0,
    },
    {
        "name": "pmer_l", 
        "description": "Flag to save segmentation scores.",
        "type": "float",
        "required": False,
        "default": 1.1
    },
    {
        "name": "pmer_occ",
        "description": "Flag to Save Inverted Segmentation Mask",
        "type": "float",
        "required": False,
        "default": 0.2
    },    
    {
        "name": "pmer_over_tol",
        "description": "Overlappoing Tolerance (Percentage)",
        "type": "float",
        "required": False,
        "default": 0.05,
    },  
    {
        "name": "is_membrane",
        "description": "Indicate Protein Type",
        "type": "boolean",
        "required": False,
        "default": False
    },
    {
        "name": "pmer_reverse_normals",
        "description": "Reverse membrane normal (For Membrane-Bound Protein)",
        "type": "boolean",
        "required": False,
        "default": False
    }            
]

def run():
    # Imports 
    from gui.core.utilities import write_mmolecules, pdb_2_mrc
    from gui.core.vtk_utilities import select_isosurface
    import biotite.database.rcsb as rcsb    
    import urllib.request
    import os

    # Parse Arguments
    args = get_args()
    
    # Fetch the PDB file and save it locally
    pdb_id = args.pdb_id
    pdb_write_path = args.pdb_write_path
    print(f'\n[Pdb Import Parameters]:\nPDB ID: {pdb_id}\nPDB download path: {pdb_write_path}')
    assembly_id = 1  # The first biological assembly
    url = f"https://files.rcsb.org/download/{pdb_id}-assembly{assembly_id}.cif"
    pdb_path = f"{pdb_write_path}/{pdb_id}_assembly{assembly_id}.cif"

    # Step 2: Download the biological assembly CIF file
    urllib.request.urlretrieve(url, pdb_path)

    offset = 20
    het = True
    chains = None
    m = None

    apix = args.voxel_size
    res = args.resolution
    protein_name = args.protein_name
    mmer_id = f'{protein_name}_{pdb_id}'
    mrc_write_path = args.mrc_write_path
    os.makedirs(mrc_write_path, exist_ok=True)
    mrc_output_path = os.path.join(mrc_write_path, f'{mmer_id}.mrc')
    print(f'\n[Pdb to MRC Convert Parameters]:\nMRC Write Path: {mrc_output_path}\nVoxel-Size: {apix}\nResolution: {res}\n')
    pdb_2_mrc(pdb_path, mrc_output_path, apix, res, offset, het, chains, m)
    print(f'MRC File Saved to {mrc_output_path}')

    mmer_iso = args.mmer_iso
    pmer_l = args.pmer_l
    pmer_occ = args.pmer_occ
    pmer_over_tol = args.pmer_over_tol
    pmer_reverse_normals = args.pmer_reverse_normals
    is_membrane = args.is_membrane

    # Create GUI to Select Iso-Contour
    if mmer_iso == 0:
        mmer_iso = select_isosurface(mrc_output_path, 800, 600)

    # Write Macromolecule Protein File
    output_path = args.polnet_write_path
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, mmer_id)
    print(f'\n[MRC to PNS Convert] Writing PNS or PMS File to {output_path}')
    write_mmolecules(mmer_id, mrc_output_path, output_path, mmer_iso , pmer_l, 
                    3000, pmer_occ, pmer_over_tol, pmer_reverse_normals, is_membrane)

setup(
    group="polnet",
    name="pdb-to-pns",
    version="0.2.0",
    title="Generate Polnet PNS file from PDB",
    description="This solution downloads PDBs and uses Polnet to convert into MRCs, and simulation-compatible textfile.",
    solution_creators=["Jonathan Schwartz"],
    cite=[{"text": "Polnet team.", "url": "https://github.com/anmartinezs/polnet/tree/main"}],
    tags=["PDB", "grid", "surface", "structural biology", "pns"],
    license="MIT",
    album_api_version="0.5.1",
    args=args,
    run=run,
    dependencies={"environment_file": env_file},
)