# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Make all map fusion transformations available."""

from .map_fusion_serial import MapFusionSerial
from .map_fusion_parallel import MapFusionParallel

# Compatibility with previous versions of DaCe and clients.
MapFusion = MapFusionSerial
