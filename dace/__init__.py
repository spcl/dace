# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import sys
from .version import __version__
from .dtypes import *

# Python frontend
from .frontend.python.interface import *
from .frontend.python.wrappers import *
from .frontend.python.ndloop import ndrange
from .frontend.operations import *

from . import data, subsets
from .config import Config
from .hooks import *
from .sdfg import SDFG, SDFGState, InterstateEdge, nodes
from .sdfg.propagation import propagate_memlets_sdfg, propagate_memlet
from .memlet import Memlet
from .symbolic import symbol

# Run Jupyter notebook code
from .jupyter import *


# Hack that enables using @dace as a decorator
# See https://stackoverflow.com/a/48100440/6489142
class DaceModule(sys.modules[__name__].__class__):
    def __call__(self, *args, **kwargs):
        return function(*args, **kwargs)


sys.modules[__name__].__class__ = DaceModule
