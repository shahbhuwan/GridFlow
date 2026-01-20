# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gridflow\\gui.py'],
    pathex=[],
    binaries=[],
    datas=[('gridflow\\vocab\\cmip5_ensemble.json', 'vocab'), ('gridflow\\vocab\\cmip5_experiment.json', 'vocab'), ('gridflow\\vocab\\cmip5_institute.json', 'vocab'), ('gridflow\\vocab\\cmip5_model.json', 'vocab'), ('gridflow\\vocab\\cmip5_time_frequency.json', 'vocab'), ('gridflow\\vocab\\cmip5_variable.json', 'vocab'), ('gridflow\\vocab\\cmip6_activity_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_experiment_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_frequency.json', 'vocab'), ('gridflow\\vocab\\cmip6_grid_label.json', 'vocab'), ('gridflow\\vocab\\cmip6_institution_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_member_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_resolution.json', 'vocab'), ('gridflow\\vocab\\cmip6_source_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_source_type.json', 'vocab'), ('gridflow\\vocab\\cmip6_table_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_variable_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_variant_label.json', 'vocab'), ('conus_border\\conus.cpg', 'conus_border'), ('conus_border\\conus.dbf', 'conus_border'), ('conus_border\\conus.prj', 'conus_border'), ('conus_border\\conus.qmd', 'conus_border'), ('conus_border\\conus.shp', 'conus_border'), ('conus_border\\conus.shx', 'conus_border'), ('gridflow_logo.svg', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='GridFlow',
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
    icon=['gridflow_logo.ico'],
)
