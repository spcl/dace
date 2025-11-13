# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import sys
from .version import __version__
from .dtypes import *

# Import built-in hooks
from .builtin_hooks import *

from . import data, hooks, subsets
from .codegen.compiled_sdfg import CompiledSDFG
from .config import Config
from .sdfg import SDFG, SDFGState, InterstateEdge, nodes, ControlFlowRegion
from .sdfg.propagation import propagate_memlets_sdfg, propagate_memlet
from .memlet import Memlet
from .symbolic import symbol

# Python frontend
from .frontend.python.interface import *
from .frontend.python.wrappers import *
from .frontend.python.ndloop import ndrange
from .frontend.operations import reduce, elementwise

# Run Jupyter notebook code
from .jupyter import *

# Import hooks from config last (as it may load classes from within dace)
hooks._install_hooks_from_config()

import sys
import os

raw_path = Config.get("external_transformations_path")
if raw_path is not None:
    __external_transformations_path__ = os.path.expanduser(os.path.expandvars(raw_path))
    sys.path.insert(0, __external_transformations_path__)


# Lazy loading for ml module to avoid eager TensorFlow/PyTorch imports
def __getattr__(name):
    if name == 'ml':
        import importlib
        ml_module = importlib.import_module('.ml', package='dace')
        # Cache the module to avoid re-importing
        globals()['ml'] = ml_module
        return ml_module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Hack that enables using @dace as a decorator
# See https://stackoverflow.com/a/48100440/6489142
class DaceModule(sys.modules[__name__].__class__):

    def __call__(self, *args, **kwargs):
        return function(*args, **kwargs)


sys.modules[__name__].__class__ = DaceModule
