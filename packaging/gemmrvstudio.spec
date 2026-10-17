# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None
repo_dir = os.path.abspath(os.getcwd())

datas = [
    (os.path.join(repo_dir, 'resources'), 'resources'),
    (os.path.join(repo_dir, 'example'), 'example'),
]

a = Analysis(
    [os.path.join(repo_dir, 'gemm_dashboard.py')],
    pathex=[repo_dir],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'serial',
        'serial.tools',
        'serial.tools.list_ports',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.QtSvg',
        'PyQt5.Qsci',
        'numpy',
        'gemm_engine',
        'gemm_tui',
        'gemm_quantizer',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='gemm_dashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='gemm_dashboard',
)
