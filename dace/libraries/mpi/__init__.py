# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Deprecated alias of :mod:`dace.libraries.comm`, which superseded the MPI library."""
import sys
import warnings

import dace.library
from dace.libraries import comm
from dace.libraries.comm import *

warnings.warn("dace.libraries.mpi is deprecated, use dace.libraries.comm", DeprecationWarning, stacklevel=2)

# Serve the submodules of the old paths (e.g., ``dace.libraries.mpi.nodes.send``) and the environment
# names that expanded SDFGs saved with the old paths refer to.
for _name, _module in list(sys.modules.items()):
    if _name.startswith(comm.__name__ + "."):
        sys.modules[__name__ + _name[len(comm.__name__) :]] = _module
for _path, _env in list(dace.library._DACE_REGISTERED_ENVIRONMENTS.items()):
    if _path.startswith(comm.__name__ + "."):
        dace.library._DACE_REGISTERED_ENVIRONMENTS.setdefault(__name__ + _path[len(comm.__name__) :], _env)
