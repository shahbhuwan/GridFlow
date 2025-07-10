# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['gridflow\\gui.py'],
    pathex=[],
    binaries=[],
    datas=[('gridflow\\vocab\\cmip5_ensemble.json', 'vocab'), ('gridflow\\vocab\\cmip5_experiment.json', 'vocab'), ('gridflow\\vocab\\cmip5_institute.json', 'vocab'), ('gridflow\\vocab\\cmip5_model.json', 'vocab'), ('gridflow\\vocab\\cmip5_time_frequency.json', 'vocab'), ('gridflow\\vocab\\cmip5_variable.json', 'vocab'), ('gridflow\\vocab\\cmip6_activity_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_experiment_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_frequency.json', 'vocab'), ('gridflow\\vocab\\cmip6_grid_label.json', 'vocab'), ('gridflow\\vocab\\cmip6_institution_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_member_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_resolution.json', 'vocab'), ('gridflow\\vocab\\cmip6_source_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_source_type.json', 'vocab'), ('gridflow\\vocab\\cmip6_table_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_variable_id.json', 'vocab'), ('gridflow\\vocab\\cmip6_variant_label.json', 'vocab'), ('iowa_border\\iowa_border.CPG', 'iowa_border'), ('iowa_border\\iowa_border.dbf', 'iowa_border'), ('iowa_border\\iowa_border.prj', 'iowa_border'), ('iowa_border\\iowa_border.sbn', 'iowa_border'), ('iowa_border\\iowa_border.sbx', 'iowa_border'), ('iowa_border\\iowa_border.shp', 'iowa_border'), ('iowa_border\\iowa_border.shp.xml', 'iowa_border'), ('iowa_border\\iowa_border.shx', 'iowa_border'), ('gridflow_logo.svg', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
