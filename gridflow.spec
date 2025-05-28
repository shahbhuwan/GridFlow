# -*- mode: python ; coding: utf-8 -*-
import os
import glob
import fiona
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None

# Collect vocab JSON files
vocab_src_dir = os.path.join('gui', 'vocab')
vocab_datas = [(src, 'gui/vocab') for src in glob.glob(os.path.join(vocab_src_dir, '*.json'))]

# Collect Iowa shapefile files
shapefile_dir = os.path.join('gridflow', 'iowa_border')
shapefile_datas = [(src, 'gridflow/iowa_border') for src in glob.glob(os.path.join(shapefile_dir, '*'))]

# Collect dynamic libraries and data for dependencies
hiddenimports = [
    'gridflow',
    'gridflow.commands',
    'gridflow.entry',
    'gridflow.cmip5_downloader',
    'gridflow.cmip6_downloader',
    'gridflow.prism_downloader',
    'gridflow.crop_netcdf',
    'gridflow.clip_netcdf',
    'gridflow.catalog_generator',
    'gridflow.logging_utils',
    'gridflow.__main__',
    'gui',
    'gui.main',
    'geopandas',
    'PyQt5',
    'PyQt5.QtWidgets',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.sip',  # Required for PyQt5
    'PyQt5.Qt',    # Ensure Qt enums are included
    'requests',
    'netCDF4',
    'numpy',
    'dateutil',  # Fixed: Changed from 'python_dateutil' to 'dateutil'
    'shapely',
    'pyproj',
    'fiona',
    'fiona._shim',  # Added to address missing fiona._shim
    'jinja2',  # Added to address missing jinja2 (optional if not using styled DataFrames)
]
hiddenimports += collect_submodules('fiona')
hiddenimports += collect_submodules('geopandas', filter=lambda name: not name.startswith('geopandas.tests'))
hiddenimports += collect_submodules('shapely', filter=lambda name: not name.startswith('shapely.tests'))
hiddenimports += collect_submodules('netCDF4')
hiddenimports += collect_submodules('PyQt5', filter=lambda name: not name.startswith('PyQt5.uic.port_v2'))

# Collect data files
fiona_gdal_data = os.path.join(os.path.dirname(fiona.__file__), 'gdal_data')
fiona_datas = [(os.path.join(fiona_gdal_data, f), 'fiona/gdal_data') for f in os.listdir(fiona_gdal_data) if os.path.isfile(os.path.join(fiona_gdal_data, f))]

# Manually specify PROJ data path if not found automatically
proj_data_dir = r"D:\MyLibrary\GridFlow-v1\gridflow\test-env\Library\share\proj"
if os.path.exists(proj_data_dir):
    proj_datas = [(os.path.join(proj_data_dir, f), 'pyproj/proj') for f in os.listdir(proj_data_dir) if os.path.isfile(os.path.join(proj_data_dir, f))]
else:
    proj_datas = []

datas = [
    ('gridflow_logo.png', '.'),
    ('gridflow_logo.ico', '.'),
] + vocab_datas + shapefile_datas + fiona_datas + proj_datas
datas += collect_data_files('pyproj')
datas += collect_data_files('netCDF4')
datas += collect_data_files('shapely')

binaries = (
    collect_dynamic_libs('shapely') +
    collect_dynamic_libs('netCDF4') +
    collect_dynamic_libs('geopandas') +
    collect_dynamic_libs('PyQt5')
)

a = Analysis(
    ['gridflow_wrapper.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hook-dpi.py', 'hook-pyproj-prefix.py'],
    excludes=[
        'matplotlib',
        'sip',
        'tensorflow',
        'torch',
        'scipy',
        # Exclude unused Qt modules to reduce warnings and bundle size
        'PyQt5.Qt3DCore',
        'PyQt5.Qt3DRender',
        'PyQt5.Qt3DExtras',
        'PyQt5.Qt3DAnimation',
        'PyQt5.Qt3DInput',
        'PyQt5.Qt3DLogic',
        'PyQt5.Qt3DQuickScene2D',
        'PyQt5.QtWebEngine',
        'PyQt5.QtMultimediaQuick',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='gridflow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='gridflow_logo.ico',
    manifest='dpi_aware.exe.manifest',
)