# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The name dace-fortran imports the fusion pipeline under."""

from dace.transformation.passes.fuse_maps import FuseMaps as FullMapFusion

__all__ = ["FullMapFusion"]
