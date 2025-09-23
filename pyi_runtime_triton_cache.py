import os
import sys
from pathlib import Path
import os, sysconfig

inc = sysconfig.get_paths().get("include")
if inc and os.path.exists(inc):
    os.environ["CPATH"] = inc + (":" + os.environ["CPATH"] if "CPATH" in os.environ else "")
# Base directory for frozen app (onedir/onefile)
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    base_dir = Path(sys._MEIPASS)
elif getattr(sys, 'frozen', False):
    base_dir = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
else:
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# Prepare writable cache directories for Triton, TorchInductor, etc.
cache_root = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'cosyvoice_fullenv'
triton_cache = cache_root / 'triton'
torchinductor_cache = cache_root / 'torchinductor'
triton_cache.mkdir(parents=True, exist_ok=True)
torchinductor_cache.mkdir(parents=True, exist_ok=True)

# Point Triton and Torch to writable caches
os.environ.setdefault('TRITON_CACHE_DIR', str(triton_cache))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(torchinductor_cache))
# Avoid attempting to write inside bundled stdlib
os.environ.setdefault('PYTORCH_KERNEL_CACHE_PATH', str(torchinductor_cache))

# Ensure temp dir exists and is writable (for onefile extractions)
os.environ.setdefault('TMPDIR', str(cache_root / 'tmp'))
Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)

# Optional: verbose to help diagnose in user logs
os.environ.setdefault('TRITON_DEBUG', '0')
