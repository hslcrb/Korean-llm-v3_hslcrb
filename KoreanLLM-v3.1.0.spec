# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# transformers, datasets, accelerate, bitsandbytes, tokenizers, pyarrow, matplotlib의 리소스 및 서브모듈 수집
for pkg in ['transformers', 'datasets', 'accelerate', 'bitsandbytes', 'tokenizers', 'pyarrow', 'matplotlib']:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports += ['tkinter', 'matplotlib.backends.backend_tkagg']

# torchvision 제외 (LLM에 불필요하며 C++ torchvision::nms 누락 충돌 방지)
excludes = ['torchvision']

a = Analysis(
    ['korean_llm_advanced_v3.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KoreanLLM-v3.1.0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='KoreanLLM-v3.1.0',
)
