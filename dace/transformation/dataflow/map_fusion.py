# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import warnings
from typing import Any

from dace.transformation.dataflow import MapFusionVertical


class MapFusion(MapFusionVertical):
    """Compatibility layer for deprecated `MapFusion` name.

    Before there was `MapFusionHorizontal` there was only `MapFusion`, which performed the
    same operations as `MapFusionVertical`. To enable a smooth transition, this deprecated
    alias is provided.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Deprecated: use `MapFusionVertical` instead."""
        super().__init__(*args, **kwargs)

        warnings.warn('MapFusion is deprecated please use MapFusionVertical instead.', DeprecationWarning, stacklevel=2)
