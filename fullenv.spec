# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import sysconfig
from pathlib import Path
from PyInstaller.utils.hooks import collect_all
import flash_attn
import pathlib
import deepspeed

current_dir = Path(os.path.abspath('.'))

# 收集关键数据资源（若存在）
datas = []
# 注入当前 Python 头文件，供 Triton 运行时使用 (提供 Python.h 等)
try:
    py_include = sysconfig.get_paths().get('include') or sysconfig.get_config_var('INCLUDEPY')
    if py_include and Path(py_include).exists():
        pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        # 注意这里：_internal/include/python3.x 与报错中的 -I 路径一致
        datas.append((py_include, f"_internal/include/{pyver}"))
except Exception:
    pass

assets_prompt = current_dir / 'assets' / 'prompt_audio'
if assets_prompt.exists():
    datas.append((str(assets_prompt), 'assets/prompt_audio'))

ttsfrd_res = current_dir / 'pretrained_models' / 'CosyVoice-ttsfrd' / 'resource'
if ttsfrd_res.exists():
    datas.append((str(ttsfrd_res), 'pretrained_models/CosyVoice-ttsfrd/resource'))

lightllm_src = current_dir / 'light_tts'
if lightllm_src.exists():
    datas.append((str(lightllm_src), 'light_tts'))
cosyvoice_src = current_dir / 'cosyvoice'
if cosyvoice_src.exists():
    datas.append((str(cosyvoice_src), 'cosyvoice'))

third_party_dir = current_dir / 'third_party'
if third_party_dir.exists():
    datas.append((str(third_party_dir), 'third_party'))

fa_path = pathlib.Path(flash_attn.__file__).parent
for py_file in fa_path.rglob("*.py"):
    rel_path = py_file.relative_to(fa_path.parent)  # 保留 flash_attn/xxx/yyy.py 层级
    datas.append((str(py_file), str(rel_path.parent)))

ds_path = pathlib.Path(deepspeed.__file__).parent
for py_file in ds_path.rglob("*.py"):
    rel_path = py_file.relative_to(ds_path.parent)  # 保留 deepspeed层级
    datas.append((str(py_file), str(rel_path.parent)))
try:
    import transformers
    tf_path = pathlib.Path(transformers.__file__).parent
    # 只收集必要的文件，避免包过大
    for pattern in ["*.py", "*.json", "*.txt"]:
        for py_file in tf_path.rglob(pattern):
            if any(skip in str(py_file) for skip in ["__pycache__", ".git", "tests"]):
                continue
            rel_path = py_file.relative_to(tf_path.parent)
            datas.append((str(py_file), str(rel_path.parent)))
except ImportError:
    pass

# 收集 xformers 的源码文件，避免冻结后仅有 .pyc 导致 inspect 失败
try:
    import xformers
    xf_path = pathlib.Path(xformers.__file__).parent
    for pattern in ["*.py", "*.json", "*.txt"]:
        for src_file in xf_path.rglob(pattern):
            if any(skip in str(src_file) for skip in ["__pycache__", ".git", "tests"]):
                continue
            rel_path = src_file.relative_to(xf_path.parent)
            datas.append((str(src_file), str(rel_path.parent)))
except ImportError:
    pass

try:
    import Cython
    cython_path = pathlib.Path(Cython.__file__).parent / "Utility"
    if cython_path.exists():
        for f in cython_path.iterdir():
            # 收集 Utility 目录下所有文件（.c/.cpp/.pxd/.h/.pyx/.py 等）
            if f.is_file():
                datas.append((str(f), "Cython/Utility"))
except ImportError:
    pass

try:
    import conformer
    conformer_path = pathlib.Path(conformer.__file__).parent
    for py_file in conformer_path.rglob("*.py"):
        rel_path = py_file.relative_to(conformer_path.parent)  # 保留 conformer层级
        datas.append((str(py_file), str(rel_path.parent)))
except ImportError:
    pass

try:
    import rich
    rich_path = pathlib.Path(rich.__file__).parent
    for py_file in rich_path.rglob("*.py"):
        rel_path = py_file.relative_to(rich_path.parent)  
        datas.append((str(py_file), str(rel_path.parent)))
except ImportError:
    pass

# 添加 whisper 包的数据文件
try:
    import whisper
    whisper_path = pathlib.Path(whisper.__file__).parent
    # 添加 whisper 的 assets 目录
    whisper_assets = whisper_path / "assets"
    if whisper_assets.exists():
        datas.append((str(whisper_assets), "whisper/assets"))
    # 添加所有 .npz 文件
    for npz_file in whisper_path.rglob("*.npz"):
        rel_path = npz_file.relative_to(whisper_path.parent)
        datas.append((str(npz_file), str(rel_path.parent)))
    # 添加所有 .json 文件
    for json_file in whisper_path.rglob("*.json"):
        rel_path = json_file.relative_to(whisper_path.parent)
        datas.append((str(json_file), str(rel_path.parent)))
except ImportError:
    pass

include_path = pathlib.Path(sysconfig.get_paths()["include"])
if include_path.exists():
    datas.append((str(include_path), str(include_path.relative_to(sys.prefix))))


print("在 .spec 文件里添加下面内容：\n")
print("datas = [")
for d in datas:
    print(f"    {d},")
print("]")


# 让 PyInstaller 自动收集依赖；不做激进的 excludes
hiddenimports = [
    'light_tts',
    'light_tts.server.api_server',
    'flash_attn.ops.triton',
    'flash_attn',
    'flash_attn.layers',
    'transformers',
    'transformers.models',
    'xformers',
    "deepspeed",
    "deepspeed.ops.op_builder",
    "deepspeed.ops.adam",
    "deepspeed.ops.lamb",
    "deepspeed.ops.sparse_attn",
    "deepspeed.ops.transformer",
    "deepspeed.ops.adam_cpu",
    "torch.utils.cpp_extension",
    "setuptools.command.build_ext",
    "Cython.Compiler.Main",
    "conformer",
    "wget",
    "whisper",
    "whisper.audio",
    "whisper.model",
    "whisper.decoding",
    "whisper.tokenizer",
    "openai_whisper",
    "rich",
]
excludes = [
]

# 收集 onnxruntime 的全部二进制和资源
try:
    _data, _bins, _hidden = collect_all('onnxruntime')
    datas += _data
    hiddenimports += _hidden
    # 注意: binaries 需在 Analysis 调用中注入
    _onnx_bins = _bins
except Exception:
    _onnx_bins = []

# 收集 triton 的全部二进制和资源，确保 JIT 可用
try:
    _tri_data, _tri_bins, _tri_hidden = collect_all('triton')
    datas += _tri_data
    hiddenimports += _tri_hidden
    _triton_bins = _tri_bins
except Exception:
    _triton_bins = []

# 收集 tiktoken 的二进制扩展，解决 tiktoken_ext 缺失
try:
    _tk_data, _tk_bins, _tk_hidden = collect_all('tiktoken')
    datas += _tk_data
    hiddenimports += _tk_hidden + [
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        'tiktoken_ext.openai_private',
        'tiktoken_ext._tiktoken',
    ]
    _tiktoken_bins = _tk_bins
except Exception:
    _tiktoken_bins = []

a = Analysis(
    ['run_fullenv.py'],
    pathex=[str(current_dir)],
    binaries=_onnx_bins + _triton_bins + _tiktoken_bins,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        str(current_dir / 'pyi_runtime_disable_typeguard.py'),
        str(current_dir / 'pyi_runtime_triton_cache.py'),
    ],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    # 重要: 让模块以文件形式存在，保证 importlib 找到 .origin，避免 Triton 在冻结环境下 origin=None
    noarchive=True,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='cosyvoice_fullenv',
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
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='cosyvoice_fullenv',
)


