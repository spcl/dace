# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Toolchain environment shims for tile-op expansions.

Per-backend environments pull in the K=1 tile-op header for the chosen ISA
(``dace/tile_ops/<backend>.h``): scalar / avx512 / avx2 / arm_neon / arm_sve.
"""
from .tile_backends import TileOpsScalar, TileOpsAVX512, TileOpsAVX2, TileOpsNeon, TileOpsSVE
