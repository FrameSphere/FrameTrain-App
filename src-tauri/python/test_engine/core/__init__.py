# test_engine/core/__init__.py
import sys as _sys
from pathlib import Path as _Path

# Gemeinsame Dataset-Logik (python/ft_data) fuer beide Engines importierbar machen.
_PY_ROOT = str(_Path(__file__).resolve().parents[2])
if _PY_ROOT not in _sys.path:
    _sys.path.append(_PY_ROOT)

