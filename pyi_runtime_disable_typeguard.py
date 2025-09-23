"""
PyInstaller runtime hook: disable typeguard to avoid 'could not get source code'.
This runs before any user code and patches typeguard decorators to no-ops.
"""
import os

os.environ.setdefault("TYPEGUARD_DISABLE", "true")

try:
    import typeguard  # type: ignore

    def _no_typechecked(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator

    # Patch top-level symbol
    try:
        typeguard.typechecked = _no_typechecked  # type: ignore[attr-defined]
    except Exception:
        pass

    # Patch v4 decorators module if present
    try:
        from typeguard import decorators as _tg_decorators  # type: ignore
        _tg_decorators.typechecked = _no_typechecked  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:
    # typeguard not installed or failed to import; nothing to do
    pass


