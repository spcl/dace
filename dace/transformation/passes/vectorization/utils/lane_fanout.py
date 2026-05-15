# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared lane-fanout detection used by the gather / scatter / strided-load / strided-store passes.

All passes look for an access node tagged ``_packed`` whose neighbours
are a contiguous fan of ``assign_<i>`` tasklets reading or writing the
same indirect array, and collapse that fan into a single intrinsic call
(``gather`` / ``scatter`` for arbitrary indices, ``strided_*`` for a
fixed-increment progression).
"""
import re
from typing import List, Optional, Set, Tuple

import sympy as sp

import dace
from dace import SDFG
from dace.sdfg import nodes

from dace.transformation.passes.vectorization.utils.name_schemes import PackedNameScheme

_ASSIGN_LABEL_RE = re.compile(r"^(?:assign|a)_(\d+)$")

_ITER_MASK_PREFIX = "_iter_mask"


def _find_iter_mask(sdfg: dace.SDFG) -> Optional[str]:
    """
    Return the name of an ``_iter_mask`` array in ``sdfg.arrays``, or ``None``.

    When the iteration mask is in scope, the collapsed tasklet must
    consume it via a ``_mask`` connector and the ``_masked`` template
    variant.

    :param sdfg: The SDFG whose arrays to inspect.
    :returns: The mask array name, or ``None`` if absent.
    """
    for name in sdfg.arrays:
        if name == _ITER_MASK_PREFIX or name.startswith(_ITER_MASK_PREFIX + "_"):
            return name
    return None


def sort_tasklets_by_number(tasklets):
    """
    Sort tasklets with labels of the form ``assign_<number>`` by the numeric part.

    :param tasklets: Tasklets to sort.
    :returns: The tasklets ordered by their numeric label suffix.
    :raises ValueError: If a tasklet label does not match ``assign_<number>``.
    """

    def get_number(tasklet):
        m = _ASSIGN_LABEL_RE.match(tasklet.label)
        if m is None:
            raise ValueError(f"Tasklet label {tasklet.label} does not match pattern 'assign_<number>'")
        return int(m.group(1))

    return sorted(tasklets, key=get_number)


def detect_fixed_increment(expr_strings):
    """
    Detect whether a list of expressions forms a fixed-increment progression.

    :param expr_strings: Expression strings to test.
    :returns: ``(increment, smallest_expr)``, or ``(None, None)`` if no
        fixed-stride pattern matches.
    """
    if len(expr_strings) < 2:
        return None, None
    try:
        exprs = [dace.symbolic.SymExpr(s.strip()) for s in expr_strings]
    except Exception:
        return None, None
    symbols = set().union(*(e.free_symbols for e in exprs))
    if len(symbols) != 1:
        return None, None
    base = symbols.pop()
    coeffs = []
    offsets = []
    for e in exprs:
        e = sp.expand(e)
        a = e.coeff(base)
        b = sp.expand(e - a * base)
        coeffs.append(a)
        offsets.append(b)
    if not all(c == coeffs[0] for c in coeffs):
        return None, None
    deltas = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    if not all(d == deltas[0] for d in deltas):
        return None, None
    min_idx = offsets.index(min(offsets))
    return deltas[0], exprs[min_idx]


def _match_assign_fan(
        state: dace.SDFGState, packed_node: dace.nodes.AccessNode,
        direction: str) -> Optional[Tuple[List[dace.nodes.Tasklet], List[Tuple[str, dace.subsets.Range]], int]]:
    """
    Match a fan of ``assign_<i>`` tasklets around a ``_packed`` access node.

    :param state: The state containing ``packed_node``.
    :param packed_node: The ``_packed`` access node to inspect.
    :param direction: ``"gather"``/``"load"`` for the in-side fan,
        ``"scatter"``/``"store"`` for the out-side fan.
    :returns: ``(sorted_tasklets, far_data_and_subset, vector_length)``,
        or ``None`` if no match.
    """
    is_pack_in_side = direction in ("gather", "load")

    neighbour_edges = state.in_edges(packed_node) if is_pack_in_side else state.out_edges(packed_node)
    neighbours = {e.src for e in neighbour_edges} if is_pack_in_side else {e.dst for e in neighbour_edges}
    if not all(isinstance(n, nodes.Tasklet) for n in neighbours):
        return None
    tasklets = {n for n in neighbours if isinstance(n, nodes.Tasklet)}

    numbers = []
    for t in tasklets:
        m = _ASSIGN_LABEL_RE.match(t.label)
        if m is None:
            return None
        numbers.append(int(m.group(1)))
    if set(numbers) != set(range(len(numbers))):
        return None
    vector_length = len(numbers)

    sorted_tasklets = sort_tasklets_by_number(tasklets)
    idx_data_and_subset: List[Tuple[str, dace.subsets.Range]] = []
    idx_datanames: Set[str] = set()
    for t in sorted_tasklets:
        far_edges = state.in_edges(t) if is_pack_in_side else state.out_edges(t)
        if len(far_edges) != 1:
            continue
        far = far_edges[0]
        idx_data_and_subset.append((far.data.data, far.data.subset))
        idx_datanames.add(far.data.data)
    if len(idx_datanames) != 1:
        return None

    return sorted_tasklets, idx_data_and_subset, vector_length


def _single_indirect_neighbour(state: dace.SDFGState, sorted_tasklets, is_pack_in_side: bool):
    """
    Return the single non-packed node every fan tasklet's far edge connects to.

    :param state: The state containing the tasklets.
    :param sorted_tasklets: The fan tasklets, in order.
    :param is_pack_in_side: ``True`` if the packed node is on the in-side.
    :returns: The shared neighbour node.
    """
    indirect_neighbours = set()
    for t in sorted_tasklets:
        far_edges = state.in_edges(t) if is_pack_in_side else state.out_edges(t)
        indirect_neighbours |= {e.src for e in far_edges} if is_pack_in_side else {e.dst for e in far_edges}
    assert len(indirect_neighbours) == 1
    return indirect_neighbours.pop()


def _linearise_subset_through_strides(subset: dace.subsets.Range, strides) -> Optional[dace.symbolic.SymExpr]:
    """
    Linearise a multi-dim point subset through the array strides into a scalar offset.

    :param subset: A point subset (each dim must have ``begin == end``).
    :param strides: The array's per-dimension strides.
    :returns: ``sum_d (b_d * strides[d])``, or ``None`` if any dim is not
        a point access.
    """
    if len(subset) != len(strides):
        return None
    total = sp.Integer(0)
    for d, (b, e, s) in enumerate(subset):
        if b != e or s != 1:
            return None
        total = total + dace.symbolic.SymExpr(str(b)) * dace.symbolic.SymExpr(str(strides[d]))
    return sp.expand(total)


def _bounding_box_per_dim(subsets):
    """
    Compute the per-dimension bounding box across a list of point subsets.

    :param subsets: List of point subsets of equal dimensionality.
    :returns: List of ``(lo, hi, 1)`` tuples suitable for a ``dace.subsets.Range``.
    """
    num_dims = len(subsets[0])
    box = []
    for d in range(num_dims):
        begins = [sp.expand(dace.symbolic.SymExpr(str(s[d][0]))) for s in subsets]
        lo = sp.Min(*begins) if len(begins) > 1 else begins[0]
        hi = sp.Max(*begins) if len(begins) > 1 else begins[0]
        box.append((dace.symbolic.SymExpr(str(lo)), dace.symbolic.SymExpr(str(hi)), 1))
    return box


def detect_multi_dim_strided_apply(sdfg: SDFG,
                                   *,
                                   direction: str,
                                   intrinsic_template: str,
                                   intrinsic_tasklet_name: str,
                                   intrinsic_template_masked: Optional[str] = None) -> int:
    """
    Collapse a multi-dim linear-combination fan around a ``_packed`` node into one intrinsic.

    Recognises patterns such as ``A[i,i]``, ``A[2*i,i]``, ``A[i,2*i]``.
    Per-lane subsets are linearised through the array strides; if the
    offsets form a fixed-increment sequence, the assign tasklets are
    replaced by one CPP strided-load/store intrinsic tasklet whose memlet
    is the bounding box of the touched elements. Only fully matched fans
    are collapsed.

    :param sdfg: The SDFG to transform (recursively, including nested SDFGs).
    :param direction: ``"load"`` or ``"store"``.
    :param intrinsic_template: CPP body template for the collapsed tasklet.
    :param intrinsic_tasklet_name: Label for the collapsed tasklet.
    :param intrinsic_template_masked: Optional masked-variant template,
        used when an iteration mask is in scope.
    :returns: The number of fans collapsed.
    """
    assert direction in ("load", "store"), direction
    is_pack_in_side = direction == "load"

    found = 0
    for state in sdfg.all_states():
        for node in state.nodes():
            if not (isinstance(node, nodes.AccessNode) and PackedNameScheme.is_packed(node.data)):
                continue

            match = _match_assign_fan(state, node, direction)
            if match is None:
                continue
            sorted_tasklets, idx_data_and_subset, vector_length = match

            # Multi-dim only — 1D cases stay on the existing strided detector.
            if not all(len(s) > 1 for _, s in idx_data_and_subset):
                continue

            indirect_dataname = idx_data_and_subset[0][0]
            arr_strides = state.sdfg.arrays[indirect_dataname].strides

            linear_offsets = []
            for _, sub in idx_data_and_subset:
                off = _linearise_subset_through_strides(sub, arr_strides)
                if off is None:
                    break
                linear_offsets.append(off)
            if len(linear_offsets) != vector_length:
                continue

            # Linearised offsets may contain symbols that are constant per
            # lane (e.g. ``N`` in ``N*tile_i + l*(N+1)``); ``detect_fixed_increment``
            # demands a single free symbol so we directly check that consecutive
            # differences are equal here.
            deltas = [sp.expand(linear_offsets[k + 1] - linear_offsets[k]) for k in range(vector_length - 1)]
            if not all(d == deltas[0] for d in deltas[1:]):
                continue
            fixed_increment = deltas[0]

            dtype_cpp = state.sdfg.arrays[node.data].dtype.ctype

            # The fan can be rooted at the surrounding MapEntry (when the
            # AccessNode lives outside the map scope, as for multi-dim arrays)
            # or at an AccessNode (1D arrays moved inside the map). Either is
            # acceptable — we re-plumb the replacement tasklet to the same root.
            indirect = _single_indirect_neighbour(state, sorted_tasklets, is_pack_in_side)
            if not isinstance(indirect, (dace.nodes.AccessNode, dace.nodes.MapEntry, dace.nodes.MapExit)):
                continue

            far_edges: List = []
            for t in sorted_tasklets:
                far_edges.extend(state.in_edges(t) if is_pack_in_side else state.out_edges(t))
            keeper_edge = far_edges[0] if far_edges else None

            subsets = [sub for _, sub in idx_data_and_subset]
            bbox = _bounding_box_per_dim(subsets)

            for t in sorted_tasklets:
                state.remove_node(t)

            iter_mask_name = _find_iter_mask(state.sdfg) if intrinsic_template_masked is not None else None
            template = intrinsic_template_masked if iter_mask_name is not None else intrinsic_template
            intrinsic_code = template.format(vector_length=vector_length, stride=fixed_increment, dtype=dtype_cpp)
            tasklet_inputs = {"_in", "_mask"} if iter_mask_name is not None else {"_in"}
            t1 = state.add_tasklet(intrinsic_tasklet_name, tasklet_inputs, {"_out"}, intrinsic_code,
                                   dace.dtypes.Language.CPP)
            packed_memlet = dace.memlet.Memlet.from_array(node.data, state.sdfg.arrays[node.data])
            indirect_memlet = dace.memlet.Memlet(data=indirect_dataname, subset=dace.subsets.Range(bbox))

            if is_pack_in_side:
                keeper_conn = keeper_edge.src_conn if keeper_edge is not None else None
                state.add_edge(indirect, keeper_conn, t1, "_in", indirect_memlet)
                for e in far_edges:
                    if e in state.edges():
                        state.remove_edge(e)
                    if e.src_conn is not None:
                        e.src.remove_out_connector(e.src_conn)
                if keeper_conn is not None:
                    indirect.add_out_connector(keeper_conn)
                state.add_edge(t1, "_out", node, None, packed_memlet)
            else:
                keeper_conn = keeper_edge.dst_conn if keeper_edge is not None else None
                state.add_edge(t1, "_out", indirect, keeper_conn, indirect_memlet)
                for e in far_edges:
                    if e in state.edges():
                        state.remove_edge(e)
                    if e.dst_conn is not None:
                        e.dst.remove_in_connector(e.dst_conn)
                if keeper_conn is not None:
                    indirect.add_in_connector(keeper_conn)
                state.add_edge(node, None, t1, "_in", packed_memlet)

            if iter_mask_name is not None:
                mask_an = state.add_access(iter_mask_name)
                state.add_edge(mask_an, None, t1, "_mask", dace.memlet.Memlet(f"{iter_mask_name}[0:{vector_length}]"))

            found += 1

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                found += detect_multi_dim_strided_apply(node.sdfg,
                                                        direction=direction,
                                                        intrinsic_template=intrinsic_template,
                                                        intrinsic_tasklet_name=intrinsic_tasklet_name,
                                                        intrinsic_template_masked=intrinsic_template_masked)
    return found


def detect_lane_fanout_apply(sdfg: SDFG,
                             *,
                             direction: str,
                             pattern: str,
                             intrinsic_template: str,
                             intrinsic_tasklet_name: str,
                             intrinsic_template_masked: Optional[str] = None) -> int:
    """
    Collapse a 1-D ``assign_<i>`` fan around a ``_packed`` node into one intrinsic call.

    :param sdfg: The SDFG to transform (recursively, including nested SDFGs).
    :param direction: ``"gather"``/``"load"`` (fan on the in-side) or
        ``"scatter"``/``"store"`` (fan on the out-side).
    :param pattern: ``"contiguous"`` (arbitrary per-lane indices; the
        intrinsic takes an index array) or ``"strided"`` (fixed-increment
        indices; the intrinsic takes a stride).
    :param intrinsic_template: CPP body template. ``contiguous`` uses
        ``{initializer_values}``/``{vector_length}``; ``strided`` uses
        ``{vector_length}``/``{stride}``.
    :param intrinsic_tasklet_name: Label for the collapsed tasklet.
    :param intrinsic_template_masked: Optional masked-variant template,
        used when an iteration mask is in scope.
    :returns: The number of fans collapsed.
    """
    assert direction in ("gather", "scatter", "load", "store"), direction
    assert pattern in ("contiguous", "strided"), pattern
    is_pack_in_side = direction in ("gather", "load")

    found = 0
    for state in sdfg.all_states():
        for node in state.nodes():
            if not (isinstance(node, nodes.AccessNode) and PackedNameScheme.is_packed(node.data)):
                continue

            match = _match_assign_fan(state, node, direction)
            if match is None:
                continue
            sorted_tasklets, idx_data_and_subset, vector_length = match

            # Skip multi-dim subsets — those are handled by
            # ``detect_multi_dim_strided_apply`` (1D vs multi-dim are separate
            # collapse paths so the memlet shape and intrinsic differ).
            if not all(len(s) == 1 for _, s in idx_data_and_subset):
                continue

            dtype_cpp = state.sdfg.arrays[node.data].dtype.ctype
            iter_mask_name = _find_iter_mask(state.sdfg) if intrinsic_template_masked is not None else None
            template = intrinsic_template_masked if iter_mask_name is not None else intrinsic_template

            if pattern == "contiguous":
                initializer_values = ", ".join(str(s) for _, s in idx_data_and_subset)
                intrinsic_code = template.format(initializer_values=initializer_values,
                                                 vector_length=vector_length,
                                                 dtype=dtype_cpp)
            else:  # strided
                initializers = [str(s) for _, s in idx_data_and_subset]
                fixed_increment, base_expr = detect_fixed_increment(initializers)
                if fixed_increment is None:
                    continue
                intrinsic_code = template.format(vector_length=vector_length, stride=fixed_increment, dtype=dtype_cpp)

            indirect = _single_indirect_neighbour(state, sorted_tasklets, is_pack_in_side)
            if not isinstance(indirect, dace.nodes.AccessNode):
                continue

            # Collect the far edges and the connector each used on the
            # indirect-side node; the strided path replumbs connectors.
            far_edges: List = []
            for t in sorted_tasklets:
                if is_pack_in_side:
                    far_edges.extend(state.in_edges(t))
                else:
                    far_edges.extend(state.out_edges(t))
            keeper_edge = far_edges[0] if far_edges else None

            # Remove the assign fan tasklets.
            for t in sorted_tasklets:
                state.remove_node(t)

            tasklet_inputs = {"_in", "_mask"} if iter_mask_name is not None else {"_in"}
            t1 = state.add_tasklet(intrinsic_tasklet_name, tasklet_inputs, {"_out"}, intrinsic_code,
                                   dace.dtypes.Language.CPP)
            packed_memlet = dace.memlet.Memlet.from_array(node.data, state.sdfg.arrays[node.data])

            if pattern == "contiguous":
                indirect_memlet = dace.memlet.Memlet.from_array(indirect.data, state.sdfg.arrays[indirect.data])
                if is_pack_in_side:
                    state.add_edge(indirect, None, t1, "_in", indirect_memlet)
                    state.add_edge(t1, "_out", node, None, packed_memlet)
                else:
                    state.add_edge(node, None, t1, "_in", packed_memlet)
                    state.add_edge(t1, "_out", indirect, None, indirect_memlet)
            else:  # strided
                end = base_expr + vector_length * fixed_increment
                stride_memlet = dace.memlet.Memlet(data=keeper_edge.data.data,
                                                   subset=dace.subsets.Range([(base_expr, end - 1, 1)]))
                if is_pack_in_side:
                    # Reuse the original src-connector on ``indirect`` for the new
                    # tasklet; drop the per-lane connectors the assign fan used.
                    keeper_conn = keeper_edge.src_conn
                    state.add_edge(indirect, keeper_conn, t1, "_in", stride_memlet)
                    for e in far_edges:
                        if e in state.edges():
                            state.remove_edge(e)
                        e.src.remove_out_connector(e.src_conn)
                    indirect.add_out_connector(keeper_conn)
                    state.add_edge(t1, "_out", node, None, packed_memlet)
                else:
                    keeper_conn = keeper_edge.dst_conn
                    state.add_edge(t1, "_out", indirect, keeper_conn, stride_memlet)
                    for e in far_edges:
                        if e in state.edges():
                            state.remove_edge(e)
                        e.dst.remove_in_connector(e.dst_conn)
                    indirect.add_in_connector(keeper_conn)
                    state.add_edge(node, None, t1, "_in", packed_memlet)

            if iter_mask_name is not None:
                mask_an = state.add_access(iter_mask_name)
                state.add_edge(mask_an, None, t1, "_mask", dace.memlet.Memlet(f"{iter_mask_name}[0:{vector_length}]"))

            found += 1

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                found += detect_lane_fanout_apply(node.sdfg,
                                                  direction=direction,
                                                  pattern=pattern,
                                                  intrinsic_template=intrinsic_template,
                                                  intrinsic_tasklet_name=intrinsic_tasklet_name,
                                                  intrinsic_template_masked=intrinsic_template_masked)
    return found
