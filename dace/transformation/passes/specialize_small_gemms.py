# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower a small GEMM that a map or loop runs once per iteration inside that iteration.

Such a product (scattering_self_energies' ``Norb x Norb`` products, Norb = 10) is one iteration's work. As a
library call it costs more than it computes: on the CPU a BLAS call (or a parallel region) per iteration, on the
GPU a host loop that launches one vendor GEMM per iteration, because the offload keeps a map on the host when its
body issues a device-wide library call. Lowered as a loop nest it runs in the thread that runs the iteration.
"""

from typing import Any

from dace import SDFG, dtypes, properties
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.matmul import MatMul, _get_matmul_operands, compile_time_extent, is_gemm_shape
from dace.sdfg import nodes
from dace.sdfg.narrowing import as_range
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf

#: Per-dimension matmul extent at or below which canonicalization picks an inlined expansion over
#: a BLAS call. MEASURED against OpenBLAS at 64/128/256 cubed: OpenBLAS wins at every one of them
#: (0.1/13/25 ms against 36/48/75 ms for plain ``'pure'``), because the plain nest emits one
#: ``reduce_atomic`` per multiply-add. So the override is worth taking only where the BLAS call
#: overhead really dominates, and only in the ``'rowwise'`` form, which carries no atomic.
SMALL_MATMUL_DIM = 32


def matmul_extents_small(state: SDFGState, node: nodes.LibraryNode, limit: int) -> bool:
    """True iff every operand/output extent of a matmul library ``node`` is a compile-time constant
    (a literal, or a symbol ``SDFG.specialize`` fixed) at most ``limit`` (the matmul is known-small).
    Any other symbolic extent -> not known-small."""
    saw = False
    for e in list(state.in_edges(node)) + list(state.out_edges(node)):
        if e.data is None or e.data.subset is None:
            continue
        for ext in as_range(e.data.subset).size():
            saw = True
            value = compile_time_extent(ext, state.sdfg)
            if value is None or value > limit:
                return False
    return saw


def small_gemm_in_repeated_scope(node: nodes.Node, state: SDFGState) -> bool:
    """A small matrix-matrix product that an enclosing map or loop runs once per iteration.

    A ``Gemm``, or a ``MatMul`` that specializes to one, whose every extent is a compile-time constant at most
    :data:`SMALL_MATMUL_DIM`, under a map or a loop (across nested-SDFG boundaries).

    :param node: Any node.
    :param state: The state holding ``node``.
    """
    if isinstance(node, MatMul):
        try:
            if not is_gemm_shape(*_get_matmul_operands(node, state, state.sdfg)):
                return False
        except ValueError:
            return False
    elif not isinstance(node, Gemm):
        return False
    if not matmul_extents_small(state, node, SMALL_MATMUL_DIM):
        return False
    return any(
        isinstance(scope, (nodes.MapEntry, LoopRegion))
        for scope in xfh.get_parent_map_and_loop_scopes(state.sdfg, node, state)
    )


def under_parallel_map(node: nodes.Node, state: SDFGState) -> bool:
    """Some map enclosing ``node`` (across nested-SDFG boundaries) is parallel, so it can be the kernel."""
    return any(
        isinstance(scope, nodes.MapEntry) and scope.map.schedule != dtypes.ScheduleType.Sequential
        for scope in xfh.get_parent_map_and_loop_scopes(state.sdfg, node, state)
    )


@properties.make_properties
@xf.explicit_cf_compatible
class SpecializeSmallGemms(ppl.Pass):
    """Pin each small GEMM a map or loop repeats (:func:`small_gemm_in_repeated_scope`) to a loop nest.

    - CPU: the row-wise (ikj) expansion, single-threaded -- no BLAS call and no parallel region per iteration.
    - GPU, before the offload: under a parallel map, the in-thread ``pure`` loop. The offload then reads the map
      as a kernel instead of a host loop of vendor calls, and each thread runs its iteration's products. A product under loops alone has no map to be its
      kernel and keeps the device call.

    A ``MatMul`` becomes the ``Gemm`` it stands for first on both targets. The choice is pinned (``auto_select_implementation = False``), so the implementation selection after the
    specialization stages keeps it.
    """

    CATEGORY: str = "Specialization"

    device = properties.EnumProperty(dtype=dtypes.DeviceType, default=dtypes.DeviceType.CPU, desc="Target device")

    def __init__(self, device: dtypes.DeviceType = dtypes.DeviceType.CPU):
        super().__init__()
        self.device = device

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Nodes

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """:returns: How many GEMMs were pinned to a loop nest, or ``None`` if none was."""
        # Only the ``Gemm`` a ``MatMul`` stands for has the loop-nest expansions.
        on_gpu = self.device == dtypes.DeviceType.GPU
        for node, state in list(sdfg.all_nodes_recursive()):
            if (
                isinstance(node, MatMul)
                and small_gemm_in_repeated_scope(node, state)
                and (not on_gpu or under_parallel_map(node, state))
            ):
                node.expand(state)
        pinned = 0
        for node, state in sdfg.all_nodes_recursive():
            if not small_gemm_in_repeated_scope(node, state):
                continue
            if self.device == dtypes.DeviceType.CPU and "rowwise" in type(node).implementations:
                node.implementation = "rowwise"
                node.schedule = dtypes.ScheduleType.Sequential
            elif self.device == dtypes.DeviceType.GPU and isinstance(node, Gemm) and under_parallel_map(node, state):
                node.implementation = "pure"
            else:
                continue
            node.auto_select_implementation = False
            pinned += 1
        return pinned or None
