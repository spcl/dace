# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a pure tensor-permutation loop nest to a transpose library node.

A hand-written transpose is a perfectly nested loop whose innermost body copies
one array element to another with the axes permuted::

    for i in range(N):            for i in range(M):
      for j in range(M):            for j in range(N):
        B[i, j] = A[j, i]             for k in range(K):
                                        B[i, j, k] = A[k, i, j]

The body is a PURE copy (``out[perm(idx)] = in[idx]``, no arithmetic, no
reduction, no carry). This pass recognises such a nest -- ``d`` perfectly nested
unit-or-strided loops whose innermost single body state copies one ``d``-D array
to a DISTINCT ``d``-D array at point subscripts that are a bijective permutation
of the loop variables -- and replaces it with a single library node:

- ``d == 2``  -> :class:`~dace.libraries.linalg.nodes.transpose.Transpose`
  (a BLAS ``omatcopy`` / ``cublasgeam`` matrix transpose).
- ``d >= 3``  -> :class:`~dace.libraries.linalg.nodes.ttranspose.TensorTranspose`
  (an HPTT / cuTENSOR N-D axis permutation), with ``axes`` set to the permutation.

Matching is by MEMLET SUBSET, not AST: each of the read and write memlets must be
a point subset whose per-dimension index is affine in exactly ONE loop variable
(``coeff*v + off``, ``coeff`` a positive integer constant), and the loop-variable
-> dimension map must be a bijection on each side. The permutation ``axes[b]`` is
the input dimension indexed by the same loop variable as output dimension ``b``.

The match is conservative -- it REFUSES on any deviation:

- The identity permutation (``B[i, j] = A[i, j]``): a plain copy, not a transpose.
- A repeated axis / mixed-variable subscript (``A[i + j]``, ``A[i, i]``): not a
  clean bijection.
- Any arithmetic in the body (``B[i, j] = A[j, i] + 1``): not a pure copy.
- ``in`` and ``out`` the same array (in-place): LoopToSymmetrize's domain.
- A non-constant / non-positive loop stride, or a subscript coefficient that is
  not a positive integer constant.

Strided / offset loops (``for i in range(lo, hi, inc)``, ``inc > 0``) are handled:
the emitted memlets carry the exact ``lo:hi:inc`` sub-region, and when the access
is not the full array the operands are routed through strided Views (whose strides
encode the per-axis step) so the library node still sees a dense operand. Only when
every axis covers its whole array are plain full-array memlets emitted.
"""
from typing import Dict, List, Optional, Tuple

import sympy

import dace
from dace import subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes.analysis import loop_analysis


def _const_pos_int(value) -> Optional[int]:
    """``value`` as a positive Python ``int`` if constant, else ``None``."""
    try:
        s = symbolic.simplify(symbolic.pystr_to_symbolic(str(value)))
    except Exception:
        return None
    return int(s) if s.is_Integer and int(s) > 0 else None


def _single_child_loop(region: ControlFlowRegion) -> Optional[LoopRegion]:
    """The region's one child ``LoopRegion`` if every other block is an empty
    state (perfect nest with connective tissue tolerated), else ``None`` (also
    ``None`` when the region holds a non-empty compute state -- i.e. is the
    innermost level)."""
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


def _descend_perfect_nest(outer: LoopRegion) -> Tuple[Optional[List[LoopRegion]], Optional[SDFGState]]:
    """Descend a chain of perfectly nested ``LoopRegion`` s from ``outer`` to the
    innermost compute loop. Returns ``(loops, body_state)`` where ``loops`` is the
    outer-to-inner list and ``body_state`` is the innermost loop's single body
    state, or ``(None, None)`` on any structural deviation."""
    loops: List[LoopRegion] = []
    cur = outer
    while True:
        loops.append(cur)
        child = _single_child_loop(cur)
        if child is None:
            break
        cur = child
    body = _single_body_state(loops[-1])
    if body is None:
        return None, None
    return loops, body


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
    """The subset of ``edge`` on ``node``'s side, where ``node`` is an AccessNode
    of ``array``. For a same-array self-copy DaCe puts the source region in
    ``subset`` and the destination in ``other_subset``; otherwise the ``array``
    side is whichever of ``subset`` / ``other_subset`` matches ``memlet.data``."""
    mem = edge.data
    other = edge.dst if node is edge.src else edge.src
    if isinstance(other, nodes.AccessNode) and other.data == array:
        return mem.subset if node is edge.src else mem.other_subset
    return mem.subset if mem.data == array else mem.other_subset


def _extract_permutation_copy(state: SDFGState):
    """Match a pure cross-array copy of one ``d``-D array to a DISTINCT ``d``-D
    array in ``state``: read one array at a single point, pass the value through
    only copy-passthrough nodes (transient scratch AccessNodes and/or
    ``__out = __inp`` copy tasklets), and write a DIFFERENT array at a single
    point.

    :returns: ``(in_array, out_array, read_subset, write_subset)`` or ``None``.
    """
    sdfg = state.sdfg
    access = [n for n in state.nodes() if isinstance(n, nodes.AccessNode)]
    others = [n for n in state.nodes() if not isinstance(n, nodes.AccessNode)]
    # Every non-access node must be a pure copy tasklet (rejects arithmetic bodies).
    if any(not _is_copy_tasklet(n) for n in others):
        return None
    sources = [n for n in access if state.in_degree(n) == 0 and state.out_degree(n) >= 1]
    sinks = [n for n in access if state.out_degree(n) == 0 and state.in_degree(n) >= 1]
    if len(sources) != 1 or len(sinks) != 1:
        return None
    src, sink = sources[0], sinks[0]
    if src.data == sink.data:
        return None  # in-place -- LoopToSymmetrize's domain
    in_array, out_array = src.data, sink.data
    # Any intermediate access node must be a transient scratch (not an operand).
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
    read_subset = _node_side_subset(src_oes[0], src, in_array)
    write_subset = _node_side_subset(sink_ies[0], sink, out_array)
    if read_subset is None or write_subset is None:
        return None
    return in_array, out_array, read_subset, write_subset


def _axis_affine(idx, loop_var_syms):
    """Classify a single-point index expression ``idx``. Returns
    ``(loop_var_sym, coeff, off)`` if ``idx`` is affine in EXACTLY one loop
    variable (``coeff*v + off``, ``coeff`` a positive integer constant, ``off``
    loop-variable-free), else ``None``."""
    used = [v for v in loop_var_syms if v in idx.free_symbols]
    if len(used) != 1:
        return None
    v = used[0]
    try:
        coeff = symbolic.simplify(sympy.diff(idx, v))
        off = symbolic.simplify(idx - coeff * v)
    except Exception:
        return None
    if not (coeff.is_Integer and int(coeff) > 0):
        return None
    if any(lv in off.free_symbols for lv in loop_var_syms):
        return None
    return v, int(coeff), off


def _classify_side(subset, loop_var_syms, ndims: int):
    """Match every axis of ``subset`` (a point subset of ``ndims`` axes) to one
    loop variable. Returns ``(axis_of_var, coeff_of_var, off_of_var)`` -- dicts
    keyed by loop-variable symbol -- forming a bijection loop-var <-> axis, or
    ``None`` on any deviation (non-point axis, repeated / missing variable,
    mixed subscript)."""
    ndr = list(subset.ndrange())
    if len(ndr) != ndims:
        return None
    axis_of_var: Dict[object, int] = {}
    coeff_of_var: Dict[object, int] = {}
    off_of_var: Dict[object, object] = {}
    for axis, (lo, hi, st) in enumerate(ndr):
        if symbolic.simplify(lo - hi) != 0 or symbolic.simplify(st - 1) != 0:
            return None  # not a single point
        res = _axis_affine(symbolic.pystr_to_symbolic(str(lo)), loop_var_syms)
        if res is None:
            return None
        v, coeff, off = res
        if v in axis_of_var:
            return None  # variable used on two axes -- not a bijection
        axis_of_var[v] = axis
        coeff_of_var[v] = coeff
        off_of_var[v] = off
    if set(axis_of_var.keys()) != set(loop_var_syms):
        return None
    return axis_of_var, coeff_of_var, off_of_var


class _Plan:
    """Everything needed to emit the transpose libnode for a matched nest."""

    def __init__(self, in_array, out_array, axes, read_range, write_range, in_full, out_full):
        self.in_array = in_array
        self.out_array = out_array
        self.axes = axes
        self.read_range = read_range
        self.write_range = write_range
        self.full = in_full and out_full


@explicit_cf_compatible
class LoopToTranspose(ppl.Pass):
    """Lift a pure tensor-permutation loop nest to a ``Transpose`` (2-D) or
    ``TensorTranspose`` (N-D) library node."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions(recursive=True)):
                for outer in list(cfg.nodes()):
                    if isinstance(outer, LoopRegion) and self._try_lift(cfg, outer, sd):
                        count += 1
        return count or None

    def _try_lift(self, cfg: ControlFlowRegion, outer: LoopRegion, sdfg: dace.SDFG) -> bool:
        # Only start from an OUTERMOST loop: an inner loop is picked up by descending
        # from its outer parent, so a non-outermost start would double-match.
        p = outer.parent_graph
        while p is not None and not isinstance(p, dace.SDFG):
            if isinstance(p, LoopRegion):
                return False
            p = p.parent_graph

        loops, body = _descend_perfect_nest(outer)
        if loops is None or len(loops) < 2:
            return False

        # Per-loop (init, last-inclusive, stride); stride must be a positive int constant.
        ranges: Dict[object, Tuple[object, object, int]] = {}
        loop_var_syms: List[object] = []
        for loop in loops:
            if not loop.loop_variable:
                return False
            lo = loop_analysis.get_init_assignment(loop)
            last = loop_analysis.get_loop_end(loop)
            inc = loop_analysis.get_loop_stride(loop)
            if lo is None or last is None or inc is None:
                return False
            inc_i = _const_pos_int(inc)
            if inc_i is None:
                return False
            v = symbolic.pystr_to_symbolic(loop.loop_variable)
            if v in ranges:
                return False  # duplicated iterator name -- not a clean nest
            ranges[v] = (symbolic.pystr_to_symbolic(str(lo)), symbolic.pystr_to_symbolic(str(last)), inc_i)
            loop_var_syms.append(v)

        d = len(loops)
        extracted = _extract_permutation_copy(body)
        if extracted is None:
            return False
        in_array, out_array, read_subset, write_subset = extracted

        in_desc = sdfg.arrays.get(in_array)
        out_desc = sdfg.arrays.get(out_array)
        if in_desc is None or out_desc is None:
            return False
        if len(in_desc.shape) != d or len(out_desc.shape) != d:
            return False
        if in_desc.dtype != out_desc.dtype or in_desc.storage != out_desc.storage:
            return False

        read_side = _classify_side(read_subset, loop_var_syms, d)
        write_side = _classify_side(write_subset, loop_var_syms, d)
        if read_side is None or write_side is None:
            return False
        in_axis, in_coeff, in_off = read_side
        out_axis, out_coeff, out_off = write_side

        # Permutation: output axis b is indexed by some loop var w; axes[b] is the
        # INPUT axis that same w indexes. TensorTranspose(axes) == np.transpose(in, axes).
        var_at_out = {b: v for v, b in out_axis.items()}
        axes = [in_axis[var_at_out[b]] for b in range(d)]
        if axes == list(range(d)):
            return False  # identity permutation -- a plain copy, not a transpose

        # Per-axis accessed ranges (the exact lo:hi:inc sub-grid, incl. affine coeff/off).
        def _side_ranges(axis_of_var, coeff_of_var, off_of_var, ndims):
            var_at_axis = {a: v for v, a in axis_of_var.items()}
            out = []
            for a in range(ndims):
                v = var_at_axis[a]
                c, off = coeff_of_var[v], off_of_var[v]
                lo_v, last_v, inc_v = ranges[v]
                out.append((symbolic.simplify(c * lo_v + off), symbolic.simplify(c * last_v + off),
                            symbolic.simplify(c * inc_v)))
            return out

        in_ranges = _side_ranges(in_axis, in_coeff, in_off, d)
        out_ranges = _side_ranges(out_axis, out_coeff, out_off, d)

        def _is_full(rng_list, desc) -> bool:
            for (lo, hi, st), sz in zip(rng_list, desc.shape):
                if symbolic.simplify(lo) != 0 or symbolic.simplify(st - 1) != 0 or \
                        symbolic.simplify(hi - (sz - 1)) != 0:
                    return False
            return True

        plan = _Plan(in_array, out_array, axes, subsets.Range(in_ranges), subsets.Range(out_ranges),
                     _is_full(in_ranges, in_desc), _is_full(out_ranges, out_desc))
        self._replace(cfg, outer, sdfg, plan, in_desc, out_desc, d)
        return True

    def _replace(self, cfg: ControlFlowRegion, outer: LoopRegion, sdfg: dace.SDFG, plan: _Plan, in_desc, out_desc,
                 d: int) -> None:
        """Splice ``outer`` out, replacing the nest with a state holding the
        transpose library node wired to the operand arrays (directly for a
        full-array access, via strided Views otherwise)."""
        from dace.libraries.linalg.nodes.transpose import Transpose
        from dace.libraries.linalg.nodes.ttranspose import TensorTranspose

        was_start = cfg.start_block is outer
        in_edges = list(cfg.in_edges(outer))
        out_edges = list(cfg.out_edges(outer))
        state = cfg.add_state(outer.label + "_transpose", is_start_block=was_start)

        if d == 2:
            node = Transpose(outer.label + "_transpose", dtype=in_desc.dtype)
            in_conn, out_conn = "_inp", "_out"
        else:
            node = TensorTranspose(outer.label + "_transpose", axes=list(plan.axes))
            in_conn, out_conn = "_inp_tensor", "_out_tensor"
        state.add_node(node)

        if plan.full:
            # Whole-array access on every axis: plain full-array memlets.
            state.add_edge(state.add_read(plan.in_array), None, node, in_conn,
                           Memlet(data=plan.in_array, subset=subsets.Range(list(plan.read_range.ndrange()))))
            state.add_edge(node, out_conn, state.add_write(plan.out_array), None,
                           Memlet(data=plan.out_array, subset=subsets.Range(list(plan.write_range.ndrange()))))
        else:
            # Offset / strided sub-grid: route each operand through a strided View whose
            # per-axis stride is the array stride times the access step, so the library
            # node (whose pure expansion assumes a dense operand) still reads/writes the
            # correct elements. The connecting memlet carries the lo:hi:inc origin+extent.
            in_steps = [r[2] for r in plan.read_range.ndrange()]
            out_steps = [r[2] for r in plan.write_range.ndrange()]
            iv_name, _ = sdfg.add_view(plan.in_array + "_tview",
                                       list(plan.read_range.size()),
                                       in_desc.dtype,
                                       storage=in_desc.storage,
                                       strides=[in_desc.strides[a] * in_steps[a] for a in range(d)],
                                       find_new_name=True)
            ov_name, _ = sdfg.add_view(plan.out_array + "_tview",
                                       list(plan.write_range.size()),
                                       out_desc.dtype,
                                       storage=out_desc.storage,
                                       strides=[out_desc.strides[b] * out_steps[b] for b in range(d)],
                                       find_new_name=True)
            iv = state.add_access(iv_name)
            ov = state.add_access(ov_name)
            in_dense = subsets.Range([(0, s - 1, 1) for s in plan.read_range.size()])
            out_dense = subsets.Range([(0, s - 1, 1) for s in plan.write_range.size()])
            state.add_edge(state.add_read(plan.in_array), None, iv, "views",
                           Memlet(data=plan.in_array, subset=subsets.Range(list(plan.read_range.ndrange()))))
            state.add_edge(iv, None, node, in_conn, Memlet(data=iv_name, subset=in_dense))
            state.add_edge(node, out_conn, ov, None, Memlet(data=ov_name, subset=out_dense))
            state.add_edge(ov, "views", state.add_write(plan.out_array), None,
                           Memlet(data=plan.out_array, subset=subsets.Range(list(plan.write_range.ndrange()))))

        for e in in_edges:
            cfg.add_edge(e.src, state, e.data)
        for e in out_edges:
            cfg.add_edge(state, e.dst, e.data)
        cfg.remove_node(outer)


__all__ = ["LoopToTranspose"]
