# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

spec_file = globals().get('__file__')
project_root = os.path.abspath(os.path.dirname(spec_file)) if spec_file else os.path.abspath(os.getcwd())

datas = [
    ('neuttsair', 'neuttsair'),
    ('index.html', '.'),
    ('chat.html', '.'),
    ('migrations', 'migrations'),
    ('/Users/akashdeepdeb/.pyenv/versions/3.11.11/lib/python3.11/site-packages/opuslib', 'opuslib'),
]
binaries = []
hiddenimports = [
    'db_service',
    'models',
    'tts_service',
    'llm_service',
    'stt_service',
    'utils',
    'path_utils',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'engineio.async_drivers.asgi',
    'socketio.async_drivers.asgi',
    'mlx.core',
    'mlx.nn',
    'moshi_mlx',
    'rustymimi',
    'sentencepiece',
    'numpy',
    'huggingface_hub',
    'websockets.legacy',
    'websockets.legacy.server',
    'soundfile',
    'opuslib',
    'onnxruntime',
]

# Collect all MLX and other complex dependencies
tmp_ret = collect_all('mlx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('opuslib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('moshi_mlx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('rustymimi')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('llama_cpp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('language_tags')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

block_cipher = None

a = Analysis(
    ['openai.py'],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torch.distributed', 'torch.testing', 'torch.utils.tensorboard', 'torchaudio', 'torchvision',
        'pandas', 'pyarrow', 'scipy', 'sklearn', 'numba', 'llvmlite', 'matplotlib', 'PIL',
        'librosa', 'perth', 'pooch', 'lazy_loader',
        'transformers', 'tokenizers', 'safetensors', 'accelerate', 'sympy', 'networkx',
        'neucodec'
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
    name='api',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
