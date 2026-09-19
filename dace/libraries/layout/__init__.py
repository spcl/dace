# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``layout`` library: the layout-change algebra (DSL + optimizer) and the LayoutChange node."""
from dace.library import register_library

from dace.libraries.layout.algebra import (
    Digit,
    LayoutMap,
    Permute,
    Block,
    Unblock,
    Pad,
    Shuffle,
    Zip,
    Unzip,
    identity_map,
    compose_ops,
    simplify_ops,
    is_identity,
    physical_index_exprs,
    ops_to_list,
    ops_from_list,
)
from dace.libraries.layout.shuffle import (
    ShuffleFunction,
    register_shuffle,
    get_shuffle,
    is_registered,
    emit_shuffle_globals,
)
from dace.libraries.layout.lowering import build_relayout, build_relayout_sdfg, relayout_map
from dace.libraries.layout.layout_change import (
    LayoutChange,
    add_layout_change,
    fold_layout_changes,
)

register_library(__name__, "layout")
