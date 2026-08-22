# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library exposing integer-sort primitives.

Currently provides :class:`~dace.libraries.sort.nodes.integer_sort.IntegerSort` --
a 1-D integer sort with CPU (`ska_sort`), CUDA (`cub::DeviceRadixSort`), and a
portable `std::sort` fallback. Used by passes that need to sort integer indices,
notably the scatter-conflict guard.
"""
from dace.library import register_library
from .environments import *
from .nodes import *

register_library(__name__, "sort")
