# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import warnings
from dace.transformation import dataflow as dftrans

try:
    from warnings import deprecated
except ImportError:

    def deprecated(msg):
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return lambda cls: cls


@deprecated('MapFusion is deprecated please use MapFusionVertical instead.')
class MapFusion(dftrans.MapFusionVertical):
    """Compatibility layer for deprecated `MapFusion` name.

    Before there was `MapFusionHorizontal` there was only `MapFusion`, which performed the
    same operations as `MapFusionVertical`. To enable a smooth transition, this deprecated
    alias is provided.
    """
    pass
