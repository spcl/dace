# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The tile DSLs the Python backend writes tile kernels in, by their ``compiler.python.tile_dialect`` name."""

from dace.codegen.python.dialects.cutile import CuTile
from dace.codegen.python.dialects.triton import Triton

DIALECTS = {"triton": Triton, "cutile": CuTile}
