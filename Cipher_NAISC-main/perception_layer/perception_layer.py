"""
Shim that loads perception-layer/perception_layer.py via importlib so callers
can do `from perception_layer.perception_layer import PerceptionLayer` with
sys.path.insert(0, '.') at the project root, despite the source folder using
a hyphen in its name (which Python cannot import directly).
"""
import importlib.util
import pathlib
import sys

_layer_dir = str(pathlib.Path(__file__).parent.parent / "perception-layer")
if _layer_dir not in sys.path:
    sys.path.insert(0, _layer_dir)

_src = pathlib.Path(_layer_dir) / "perception_layer.py"
_spec = importlib.util.spec_from_file_location("_perception_layer_impl", _src)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["_perception_layer_impl"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

from _perception_layer_impl import *  # noqa: F401, F403, E402
from _perception_layer_impl import PerceptionLayer, PerceptionResult  # noqa: E402
