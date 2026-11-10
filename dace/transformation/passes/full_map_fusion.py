# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Main's module and name for :class:`FuseMaps`, so code written against main (dace-fortran) imports it here too."""

from dace.transformation.passes.fuse_maps import FuseMaps

FullMapFusion = FuseMaps
