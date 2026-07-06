# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a triangular matrix-symmetrization loop nest to a ``Symmetrize`` node.

A hand-written symmetrization::

    for i in range(0, M - 1):
        for j in range(i + 1, M):
            X[j, i] = X[i, j]

is a perfect two-level nest whose inner body copies one triangle of a square
matrix onto the other across the diagonal. It is embarrassingly parallel, but
in-place it reads and writes the SAME array at symmetric data-dependent indices
(``X[i, j]`` vs ``X[j, i]``), so ``LoopToMap`` conservatively refuses and leaves
it sequential. This pass recognises the nest -- an outer loop whose sole body is
an inner triangular loop (lower bound ``outer + offset``) whose sole body copies
``X[transpose] = X[src]`` on one 2-D array -- and replaces it with a
:class:`~dace.libraries.standard.nodes.symmetrize.Symmetrize` node, whose
expansion emits the parallel triangular copy directly.

The match is conservative: both loops unit-stride, a perfect one-child nest
(empty connective states tolerated), a constant nonnegative inner offset, a
single 2-D array read and written at transposed single-point subscripts using
exactly the two loop variables, and no other body effect.
"""
from typing import Optional

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes.analysis import loop_analysis


def _const_nonneg_int(value) -> Optional[int]:
    """``value`` as a nonnegative Python ``int`` if constant, else ``None``."""
    try:
        s = symbolic.pystr_to_symbolic(str(value))
    except Exception:
        return None
    return int(s) if s.is_Integer and int(s) >= 0 else None


def _unit_stride(loop: LoopRegion) -> bool:
    stride = loop_analysis.get_loop_stride(loop)
    try:
        return stride is not None and symbolic.simplify(stride) == 1
    except Exception:
        return False


def _single_child_loop(region: ControlFlowRegion) -> Optional[LoopRegion]:
    """The region's one child ``LoopRegion`` if every other block is an empty
    state (perfect nest with connective tissue tolerated), else ``None``."""
    loop = None
    for b in region.nodes():
        if isinstance(b, LoopRegion):
            if loop is not None:
                return None
            loop = b
        elif isinstance(b, SDFGState):
            if b.nodes():
                return None
        else:
            return None
    return loop


def _single_body_state(loop: LoopRegion) -> Optional[SDFGState]:
    """The loop's one non-empty body state, or ``None`` if not a single compute
    state (empty connective states tolerated)."""
    blocks = list(loop.nodes())
    if not all(isinstance(b, SDFGState) for b in blocks):
        return None
    non_empty = [b for b in blocks if b.nodes()]
    return non_empty[0] if len(non_empty) == 1 else None


def _is_copy_tasklet(node) -> bool:
    """Whether ``node`` is a single-input pure-copy tasklet ``__out = __inp``."""
    if not isinstance(node, nodes.Tasklet):
        return False
    code = node.code.as_string.strip()
    if code.count("=") != 1:
        return False
    lhs, rhs = (s.strip() for s in code.split("=", 1))
    return len(node.in_connectors) == 1 and len(node.out_connectors) == 1 and rhs in node.in_connectors and \
        lhs in node.out_connectors


def _node_side_subset(edge, node, array: str):
    """The subset of ``edge`` on ``node``'s side, where ``node`` is an
    AccessNode of ``array``.

    For a self-copy edge (both ends ``array``) DaCe puts the source region in
    ``subset`` and the destination region in ``other_subset``; otherwise the
    ``array`` side is whichever of ``subset`` / ``other_subset`` matches
    ``memlet.data``.
    """
    mem = edge.data
    other = edge.dst if node is edge.src else edge.src
    if isinstance(other, nodes.AccessNode) and other.data == array:
        return mem.subset if node is edge.src else mem.other_subset
    return mem.subset if mem.data == array else mem.other_subset


def _point_indices(subset, outer: str, inner: str) -> Optional[list]:
    """If ``subset`` is a 2-D single point over exactly ``{outer, inner}``,
    return the ordered list of which variable each axis is (``[outer, inner]``
    or ``[inner, outer]``); else ``None``."""
    if subset is None or len(subset) != 2:
        return None
    order = []
    for rb, re_, _ in subset.ndrange():
        if rb != re_:
            return None
        s = str(rb).strip()
        if s == outer:
            order.append(outer)
        elif s == inner:
            order.append(inner)
        else:
            return None
    return order if set(order) == {outer, inner} else None


@explicit_cf_compatible
class LoopToSymmetrize(ppl.Pass):
    """Lift a triangular in-place symmetrization loop nest to a ``Symmetrize`` node."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Lift every matching symmetrization nest in ``sdfg`` and its nested SDFGs."""
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions(recursive=True)):
                for outer in list(cfg.nodes()):
                    if isinstance(outer, LoopRegion) and self._try_lift(cfg, outer, sd):
                        count += 1
        return count or None

    def _try_lift(self, cfg: ControlFlowRegion, outer: LoopRegion, sdfg: dace.SDFG) -> bool:
        if not outer.loop_variable or not _unit_stride(outer):
            return False
        inner = _single_child_loop(outer)
        if inner is None or not inner.loop_variable or not _unit_stride(inner):
            return False
        outer_var, inner_var = outer.loop_variable, inner.loop_variable

        # Inner lower bound must be ``outer_var + const_offset`` (triangular).
        inner_init = loop_analysis.get_init_assignment(inner)
        if inner_init is None:
            return False
        try:
            offset = symbolic.simplify(symbolic.pystr_to_symbolic(inner_init) - symbolic.pystr_to_symbolic(outer_var))
        except Exception:
            return False
        col_offset = _const_nonneg_int(offset)
        if col_offset is None:
            return False

        # Inner body: a single state doing an in-place transposed copy of ONE
        # 2-D array -- either a direct ``X -> X`` copy edge or a
        # ``X(read) -> copy-tasklet -> X(write)`` chain.
        state = _single_body_state(inner)
        if state is None:
            return False
        extracted = self._extract_symmetric_copy(state, outer_var, inner_var)
        if extracted is None:
            return False
        array, read_order, write_order = extracted
        desc = sdfg.arrays.get(array)
        if desc is None or len(desc.shape) != 2:
            return False

        # source_upper: the READ (source, preserved) triangle is the upper one iff
        # the read subset is [outer, inner] (inner >= outer + offset >= outer).
        source_upper = (read_order == [outer_var, inner_var])

        row_lo = str(loop_analysis.get_init_assignment(outer))
        row_hi = str(symbolic.simplify(symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(outer)) + 1))
        col_hi = str(symbolic.simplify(symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(inner)) + 1))

        self._replace(cfg, outer, array, desc, row_lo, row_hi, col_offset, col_hi, source_upper)
        return True

    def _extract_symmetric_copy(self, state: SDFGState, outer_var: str, inner_var: str):
        """Match an in-place transposed copy of one 2-D array in ``state``.

        The body must read one square array ``X`` at a single point, pass the
        value through only copy-passthrough nodes (transient scratch AccessNodes
        and/or ``__out = __inp`` copy tasklets), and write ``X`` at the
        transposed point -- i.e. ``X[write] = X[read]`` with ``read`` and
        ``write`` transposes. Handles the direct ``X -> X`` copy edge, the
        ``X -> tasklet -> X`` form, and the ``X -> scratch -> X`` form uniformly.

        :returns: ``(array, read_order, write_order)`` where each order lists
                  which of ``{outer_var, inner_var}`` each axis is, or ``None``.
        """
        sdfg = state.sdfg
        access = [n for n in state.nodes() if isinstance(n, nodes.AccessNode)]
        others = [n for n in state.nodes() if not isinstance(n, nodes.AccessNode)]
        # Every non-access node must be a pure copy tasklet.
        if any(not _is_copy_tasklet(n) for n in others):
            return None
        # Exactly one source (read) and one sink (write) access node, same array X.
        sources = [n for n in access if state.in_degree(n) == 0 and state.out_degree(n) >= 1]
        sinks = [n for n in access if state.out_degree(n) == 0 and state.in_degree(n) >= 1]
        if len(sources) != 1 or len(sinks) != 1:
            return None
        src, sink = sources[0], sinks[0]
        if src.data != sink.data:
            return None
        array = src.data
        # Any intermediate access node must be a transient scratch (not the target).
        for n in access:
            if n is src or n is sink:
                continue
            desc = sdfg.arrays.get(n.data)
            if desc is None or not desc.transient:
                return None
        src_oes = [e for e in state.out_edges(src) if e.data is not None and not e.data.is_empty()]
        sink_ies = [e for e in state.in_edges(sink) if e.data is not None and not e.data.is_empty()]
        if len(src_oes) != 1 or len(sink_ies) != 1:
            return None
        read_subset = _node_side_subset(src_oes[0], src, array)
        write_subset = _node_side_subset(sink_ies[0], sink, array)
        read_order = _point_indices(read_subset, outer_var, inner_var)
        write_order = _point_indices(write_subset, outer_var, inner_var)
        if read_order is None or write_order is None or read_order != list(reversed(write_order)):
            return None
        return array, read_order, write_order

    def _replace(self, cfg: ControlFlowRegion, outer: LoopRegion, array: str, desc, row_lo: str, row_hi: str,
                 col_offset: int, col_hi: str, source_upper: bool) -> None:
        """Replace the ``outer`` loop nest with a state holding a ``Symmetrize`` node."""
        from dace.libraries.standard.nodes import Symmetrize
        was_start = cfg.start_block is outer
        in_edges = list(cfg.in_edges(outer))
        out_edges = list(cfg.out_edges(outer))

        sym_state = cfg.add_state(outer.label + "_symmetrize", is_start_block=was_start)
        node = Symmetrize(outer.label + "_sym",
                          row_lo=row_lo,
                          row_hi=row_hi,
                          col_offset=col_offset,
                          col_hi=col_hi,
                          source_upper=source_upper)
        sym_state.add_node(node)

        def _full() -> dace.Memlet:
            # Fresh subset object per edge -- DaCe forbids two memlets sharing
            # one subset instance.
            return dace.Memlet(data=array, subset=dace.subsets.Range([(0, s - 1, 1) for s in desc.shape]))

        sym_state.add_edge(sym_state.add_read(array), None, node, "_in", _full())
        sym_state.add_edge(node, "_out", sym_state.add_write(array), None, _full())

        for e in in_edges:
            cfg.add_edge(e.src, sym_state, e.data)
        for e in out_edges:
            cfg.add_edge(sym_state, e.dst, e.data)
        cfg.remove_node(outer)


__all__ = ["LoopToSymmetrize"]
