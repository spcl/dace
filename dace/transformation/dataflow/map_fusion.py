# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Make all map fusion transformations available."""

from .map_fusion_serial import SerialMapFusion
from .map_fusion_parallel import ParallelMapFusion

# Compatibility with previous versions of DaCe and clients.
MapFusion = SerialMapFusion
