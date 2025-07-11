# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from warnings import deprecated

import dace
from dace.transformation import dataflow as dftrans


@deprecated
class MapFusion(dftrans.MapFusionVertical):
    """Compatibility layer for deprecated `MapFusion` name.

    Before there was `MapFusionHorizontal` there was only `MapFusion`, which performed the
    same operations as `MapFusionVertical`. To enable a smooth transition, this deprecated
    alias is provided.
    """
    pass
