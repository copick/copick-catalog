###album catalog: copick

from io import StringIO

from album.runner.api import setup

env_file = StringIO(
    """channels:
  - conda-forge
dependencies:
  - python>=3.10
  - pip
  - numpy
  - scipy
  - napari
  - dask
  - zarr
  - qtpy
  - pyside6
  - pip:
    - copick
    - git+https://github.com/copick/napari-copick
    - paramiko
    - trimesh
    - s3fs
    - smbprotocol
    - "sshfs>=2024.6.0"
    - pooch
"""
)


def run():
    from album.runner.api import get_args
    import napari
    import sys
    from qtpy.QtWidgets import QWidget, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QLabel, QFileDialog, QLineEdit, QMenu, QAction, QFormLayout, QComboBox, QSpinBox
    from qtpy.QtCore import Qt, QPoint
    from napari.utils import DirectLabelColormap
    from napari_copick import CopickPlugin

    config_path = get_args().config_path
    
    viewer = napari.Viewer()
    copick_plugin = CopickPlugin(viewer, config_path=config_path)
    viewer.window.add_dock_widget(copick_plugin, area="right")
    napari.run()


setup(
    group="visualization",
    name="napari-copick",
    version="0.0.2",
    title="Run napari-copick.",
    description="Run the napari-copick",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "copick team.", "url": "https://copick.github.io/copick"}],
    tags=["imaging", "cryoet", "Python", "napari", "copick"],
    license="MIT",
    covers=[
        {
            "description": "Copick Plugin screenshot.",
            "source": "cover.png",
        }
    ],
    album_api_version="0.5.1",
    args=[
        {
            "name": "config_path",
            "description": "Path to the copick config file",
            "type": "string",
            "required": True,
        }
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
