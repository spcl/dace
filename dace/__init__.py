# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import sys
from .version import __version__
from .dtypes import *

# Import built-in hooks
from .builtin_hooks import *

# Python frontend
from .frontend.python.interface import *
from .frontend.python.wrappers import *
from .frontend.python.ndloop import ndrange
from .frontend.operations import reduce, elementwise

from . import data, hooks, subsets
from .codegen.compiled_sdfg import CompiledSDFG
from .config import Config
from .sdfg import SDFG, SDFGState, InterstateEdge, nodes, ControlFlowRegion
from .sdfg.propagation import propagate_memlets_sdfg, propagate_memlet
from .memlet import Memlet
from .symbolic import symbol

# Run Jupyter notebook code
from .jupyter import *

# Import hooks from config last (as it may load classes from within dace)
hooks._install_hooks_from_config()

from pathlib import Path
import sys
from dace import config

import os

def _setup_external_transformations_path():
    raw_path = config.Config.get("external_transformations_path")

    if not raw_path:
        raise ValueError("external_transformations_path value does not exist in the configuration.")

    expanded_path = os.path.expanduser(raw_path)
    expanded_path = os.path.expandvars(expanded_path)

    path_obj = Path(expanded_path).resolve()

    if not path_obj.exists():
        raise FileNotFoundError(f"External transformations path does not exist: {path_obj}")

    if not path_obj.is_dir():
        raise NotADirectoryError(f"External transformations path is not a directory: {path_obj}")

    external_transformations_path = str(path_obj)
    if external_transformations_path not in sys.path:
        sys.path.insert(0, external_transformations_path)

    return external_transformations_path

# Usage
try:
    _setup_external_transformations_path()
except (ValueError, FileNotFoundError, NotADirectoryError) as e:
    print(f"Failed to setup external transformations path: {e}")

# Hack that enables using @dace as a decorator
# See https://stackoverflow.com/a/48100440/6489142
class DaceModule(sys.modules[__name__].__class__):

    def __call__(self, *args, **kwargs):
        return function(*args, **kwargs)


sys.modules[__name__].__class__ = DaceModule
