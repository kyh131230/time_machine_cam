from PyInstaller.utils.hooks import collect_submodules, copy_metadata

# -*- mode: python ; coding: utf-8 -*-

datas = [('ui/*', 'ui/'), ('style/*', 'style/'), ('img/*', 'img/'), ('style/cursor/*', 'style/cursor'), ('style/font/*', 'style/font'), ('clickable_label.py', '.'), ('qr.py', '.'), ('replicate_tasks.py', '.'),('frame_boxes.json', '.'),
('setting.py', '.'),('senior(male).png', '.'), ]

hiddenimports=[]

datas = datas + copy_metadata('replicate')
hiddenimports = (hiddenimports if 'hiddenimports' in globals() else []) + collect_submodules('replicate')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    [],
    exclude_binaries=True,
    name='AI Life Photo Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
