# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Communication library: point-to-point and collective communication nodes with MPI and NCCL (RCCL
on the HIP backend) implementations. ``library.comm.default_implementation`` selects the backend of
the nodes that have both."""

from dace.library import register_library
from .nodes import *
from .environments import *
from .utils import *
from . import communicators, schedule

register_library(__name__, "comm")
