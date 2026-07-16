# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Device-aware lowering selection for layout-inserted library nodes.

The layout transformations (``PermuteDimensions``, ``GemmToTensorDot``, ``RewriteCopyForLayout``)
and the ``LayoutChange`` node describe a relayout SEMANTICALLY; they must not bake in a
device-specific library lowering. Choosing the expansion is a separate, device-driven step that
runs just before compilation -- this module.

On CPU the ``pure`` (map) expansion is selected; on GPU the ``cuTENSOR`` expansion is selected (its
expansion self-falls-back to the pure GPU map for dtypes cuTENSOR does not support). Only nodes
whose implementation is still unset are touched -- an explicitly chosen implementation (e.g. an
``HPTT`` transpose the caller pinned) is preserved.
"""
from typing import Set, Type

import dace
from dace.sdfg import nodes as nd

_CPU_IMPL = "pure"
_GPU_IMPL = "cuTENSOR"


def _layout_node_types() -> Set[Type[nd.LibraryNode]]:
    """The library-node types the layout passes insert (imported lazily to avoid import loops)."""
    from dace.libraries.linalg import TensorTranspose, TensorDot
    from dace.libraries.layout.layout_change import LayoutChange
    return {TensorTranspose, TensorDot, LayoutChange}


def select_layout_lowering(sdfg: dace.SDFG, device: str = "cpu") -> int:
    """Assign a device-appropriate implementation to every layout library node whose
    implementation is still unset.

    :param sdfg: the SDFG to patch in place (recurses into nested SDFGs).
    :param device: ``"cpu"`` (``pure`` expansion) or ``"gpu"`` (``cuTENSOR`` expansion).
    :returns: the number of nodes whose implementation was set.
    """
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    impl = _GPU_IMPL if device == "gpu" else _CPU_IMPL
    types = tuple(_layout_node_types())
    count = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, types) and node.implementation is None:
            node.implementation = impl
            count += 1
    return count
