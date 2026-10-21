# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Range-privatize the data-dependent scatter reductions of parallel CPU maps.

A parallel map accumulating ``A[B[f(i)], ...] (op)= v`` would otherwise lower to an OpenMP
array-section ``reduction(op:A[0:n])``: every thread zeroes and combines a private copy of all of
``A``. Range privatization gives each thread a copy of only the rows ``[lo, hi]`` of ``A`` its static
chunk writes, found by a pre-pass over the chunk's index values ``B[f(i)]``; the body accumulates
there and the copy is combined into ``A``, plainly where no other thread's rows reach and atomically
elsewhere. The pass owns the analysis and sets ``Map.omp_range_privatize``; ``CPUCodeGen`` renders it.
"""

import ast
from dataclasses import dataclass
from typing import Any

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace.frontend import operations
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: Operators the lowering combines: each has an identity and a ``wcr_fixed`` atomic or critical update.
RANGE_PRIVATIZABLE_OPS = frozenset(
    {
        dtypes.ReductionType.Sum,
        dtypes.ReductionType.Product,
        dtypes.ReductionType.Min,
        dtypes.ReductionType.Max,
        dtypes.ReductionType.Logical_And,
        dtypes.ReductionType.Logical_Or,
        dtypes.ReductionType.Bitwise_And,
        dtypes.ReductionType.Bitwise_Or,
        dtypes.ReductionType.Bitwise_Xor,
    }
)


@dataclass(frozen=True)
class ScatterIndexSources:
    """The rows one accumulator is written at, in the scope of the parallel map.

    :ivar loops: Sequential maps between the parallel map and the body, outermost first.
    :ivar reads: ``(array, index)`` pairs; every row written is ``array[index]`` for one pair.
    """

    loops: tuple
    reads: tuple


def _contiguous_rows(desc: data.Data) -> bool:
    """Whether ``desc`` is a plain C-contiguous array, so its rows are consecutive ``strides[0]``-long spans."""
    if type(desc) is not data.Array or desc.start_offset != 0:
        return False
    expected = 1
    for size, stride in reversed(list(zip(desc.shape, desc.strides))):
        if symbolic.simplify(stride - expected) != 0:
            return False
        expected = expected * size
    return symbolic.simplify(desc.total_size - expected) == 0


def _same_layout(inner: data.Data, outer: data.Data) -> bool:
    """Whether ``inner`` indexes the memory of ``outer`` element for element."""
    if len(inner.shape) != len(outer.shape) or inner.start_offset != outer.start_offset:
        return False
    pairs = list(zip(inner.shape, outer.shape)) + list(zip(inner.strides, outer.strides))
    return all(symbolic.simplify(a - b) == 0 for a, b in pairs)


def scatter_index_sources(
    state: SDFGState, map_entry: nodes.MapEntry, name: str, redtype: dtypes.ReductionType
) -> ScatterIndexSources | None:
    """Index sources of the ``redtype`` WCR accumulator ``name`` of ``map_entry``, or ``None``.

    Requires: one WCR edge chain from a nested SDFG through sequential maps to the exit; inside the
    nest the accumulator is only written, with ``redtype``, at a first index that is a symbol assigned
    once from an element ``B[e]`` of a nested input array, ``e`` over the nest's mapped symbols; the
    accumulator and ``B`` laid out inside as outside, and the outer ``B`` not written in the map.
    """
    if not _contiguous_rows(state.sdfg.arrays[name]):
        return None
    map_exit = state.exit_node(map_entry)
    into_exit = [e for e in state.in_edges(map_exit) if e.data.data == name]
    if len(into_exit) != 1:
        return None
    path = state.memlet_path(into_exit[0])
    body = path[0].src
    if not isinstance(body, nodes.NestedSDFG) or not isinstance(path[-1].dst, nodes.AccessNode):
        return None
    loops = []
    for edge in path:
        if edge.data.data != name or edge.data.wcr is None:
            return None
        if operations.detect_reduction_type(edge.data.wcr) != redtype:
            return None
        if edge.dst is map_exit:
            break
        if not isinstance(edge.dst, nodes.MapExit) or edge.dst.map.schedule != dtypes.ScheduleType.Sequential:
            return None
        entry = state.entry_node(edge.dst)
        if any(not c.startswith("IN_") for c in entry.in_connectors):
            return None
        loops.append(entry)

    # A nest receives each array at its base pointer, so an inner index addresses the outer array
    # wherever the inner descriptor lays the array out the same.
    inner = body.sdfg
    acc = path[0].src_conn
    if not _same_layout(inner.arrays[acc], state.sdfg.arrays[name]):
        return None
    shape = state.sdfg.arrays[name].shape

    symbols = set()
    for istate in inner.states():
        for node in istate.data_nodes():
            if node.data != acc:
                continue
            if any(not e.data.is_empty() for e in istate.out_edges(node)):
                return None
            for e in istate.in_edges(node):
                if e.data.wcr is None or operations.detect_reduction_type(e.data.wcr) != redtype:
                    return None
                written = e.data.get_dst_subset(e, istate)
                if written is None or written.dims() != len(shape):
                    return None
                row, row_end, _ = written[0]
                if row != row_end or not str(row).isidentifier():
                    return None
                symbols.add(str(row))
        for node in istate.nodes():
            if isinstance(node, nodes.MapEntry) and symbols & set(node.map.params):
                return None
    if not symbols or symbols & set(body.symbol_mapping):
        return None
    if symbols & {r.loop_variable for r in inner.all_control_flow_regions() if isinstance(r, LoopRegion)}:
        return None

    assigned: dict[str, list[str]] = {s: [] for s in symbols}
    for region in inner.all_control_flow_regions():
        for edge in region.edges():
            if acc in edge.data.read_symbols():
                return None
            for sym, rhs in edge.data.assignments.items():
                if sym in assigned:
                    assigned[sym].append(rhs)
    scope = state.scope_subgraph(map_entry)
    mapping = {k: symbolic.pystr_to_symbolic(v) for k, v in body.symbol_mapping.items()}
    reads = []
    for sym in sorted(symbols):
        if len(assigned[sym]) != 1:
            return None
        rhs = ast.parse(assigned[sym][0], mode="eval").body
        if not isinstance(rhs, ast.Subscript) or not isinstance(rhs.value, ast.Name):
            return None
        if any(isinstance(n, (ast.Subscript, ast.Call)) for n in ast.walk(rhs.slice)):
            return None
        src = rhs.value.id
        if src not in body.in_connectors or src in body.out_connectors or src == acc:
            return None
        elements = rhs.slice.elts if isinstance(rhs.slice, ast.Tuple) else [rhs.slice]
        index = subsets.Range([(e, e, 1) for e in (symbolic.pystr_to_symbolic(ast.unparse(x)) for x in elements)])
        if not set(map(str, index.free_symbols)) <= set(mapping):
            return None
        index.replace(mapping)
        feeds = [e for e in state.in_edges(body) if e.dst_conn == src]
        if len(feeds) != 1:
            return None
        source = state.memlet_path(feeds[0])[0].src
        outer = feeds[0].data
        if not isinstance(source, nodes.AccessNode) or source in scope.nodes() or outer.data != source.data:
            return None
        if outer.data == name or not isinstance(state.sdfg.arrays[outer.data], data.Array):
            return None
        if not _same_layout(inner.arrays[src], state.sdfg.arrays[outer.data]):
            return None
        if any(
            isinstance(n, nodes.AccessNode) and n.data == outer.data and scope.in_degree(n) > 0 for n in scope.nodes()
        ):
            return None
        reads.append((outer.data, tuple(index.min_element())))
    return ScatterIndexSources(tuple(loops), tuple(reads))


def range_private_accumulators(state: SDFGState, map_entry: nodes.MapEntry) -> dict[str, tuple]:
    """``{accumulator: (reduction type, ScatterIndexSources)}`` of the scatters ``map_entry`` can range-privatize.

    The map must be a statically scheduled ``CPU_Multicore`` map outside any persistent region, and an
    accumulator a transient-or-argument array the map does not also read.
    """
    m = map_entry.map
    if m.schedule != dtypes.ScheduleType.CPU_Multicore or m.omp_schedule not in (
        dtypes.OMPScheduleType.Default,
        dtypes.OMPScheduleType.Static,
    ):
        return {}
    if symbolic.issymbolic(m.omp_chunk_size) or int(m.omp_chunk_size) > 0:
        return {}
    scope = state.entry_node(map_entry)
    while scope is not None:
        if isinstance(scope, nodes.MapEntry) and scope.map.schedule == dtypes.ScheduleType.CPU_Persistent:
            return {}
        scope = state.entry_node(scope)
    read = {
        e.src.data for e in state.in_edges(map_entry) if isinstance(e.src, nodes.AccessNode) and not e.data.is_empty()
    }
    out = {}
    for edge in state.out_edges(state.exit_node(map_entry)):
        name = edge.data.data
        if edge.data.wcr is None or name is None or name in out or name in read:
            continue
        desc = state.sdfg.arrays[name]
        if desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            continue
        redtype = operations.detect_reduction_type(edge.data.wcr)
        if redtype not in RANGE_PRIVATIZABLE_OPS:
            continue
        sources = scatter_index_sources(state, map_entry, name, redtype)
        if sources is not None:
            out[name] = (redtype, sources)
    return out


@properties.make_properties
@transformation.explicit_cf_compatible
class RangePrivatizeScatterReductions(ppl.Pass):
    """Sets ``Map.omp_range_privatize`` on every parallel CPU map with a range-privatizable array scatter."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Memlets))

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> set[tuple[int, str]] | None:
        """Marks every qualifying map in ``sdfg`` and its nested SDFGs.

        :return: The marked maps as ``(cfg id, map label)``, or ``None`` if none qualified.
        """
        marked = set()
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for node in state.nodes():
                    if isinstance(node, nodes.MapEntry) and not node.map.omp_range_privatize:
                        if range_private_accumulators(state, node):
                            node.map.omp_range_privatize = True
                            marked.add((nsdfg.cfg_id, node.map.label))
        return marked or None
