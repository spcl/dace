# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library exposing sort primitives.

:class:`~dace.libraries.sort.nodes.integer_sort.IntegerSort` is a 1-D integer sort with CPU
(`ska_sort`), CUDA (`gpucub::DeviceRadixSort`), and a portable `std::sort` fallback, used by
passes that need to sort integer indices, notably the scatter-conflict guard.
:class:`~dace.libraries.sort.nodes.sort.Sort` is ``numpy.sort`` / ``numpy.argsort`` along an
axis of any arithmetic array, stable, one ``std::stable_sort`` per row.
"""

from dace.library import register_library
from .environments import *
from .nodes import *

register_library(__name__, "sort")
