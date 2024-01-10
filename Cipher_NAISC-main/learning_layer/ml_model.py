"""Shim: loads learning-layer/ml_model.py via importlib (hyphen folder → dot package)."""
import importlib.util
import pathlib
import sys

_layer_dir = str(pathlib.Path(__file__).parent.parent / "learning-layer")
if _layer_dir not in sys.path:
    sys.path.insert(0, _layer_dir)

_src = pathlib.Path(_layer_dir) / "ml_model.py"
_spec = importlib.util.spec_from_file_location("_ml_model_impl", _src)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["_ml_model_impl"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

from _ml_model_impl import *          # noqa: F401, F403
from _ml_model_impl import CipherMLModel  # noqa: F401
