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
from typing import Dict, List, Optional, Set, Tuple

import sympy as sp

import dace
from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion

from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme, PackedNameScheme

_ASSIGN_LABEL_RE = re.compile(r"^(?:assign|a)_(\d+)$")

_ITER_MASK_PREFIX = "_iter_mask"

# An interstate-edge laneid assignment RHS that reads a single element of
# an index array, e.g. ``edge_idx[3]`` or ``idx[tile_i + 2]``. Group 1 is
# the array name, group 2 the (possibly symbolic) index expression.
_INDEX_ARRAY_READ_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\[(.+)\]\s*$")


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


def _collect_interstate_assignments(sdfg: SDFG) -> Dict[str, Set[str]]:
    """
    Collect every interstate-edge assignment ``lhs -> {rhs, ...}`` in ``sdfg``.

    Only the SDFG's own control-flow-region edges are walked (not nested
    SDFGs): the per-lane laneid symbols are bound on interstate edges of
    the same SDFG that holds the fan.

    :param sdfg: The SDFG whose interstate edges to scan.
    :returns: Map from assigned symbol to the set of distinct RHS strings.
    """
    asg: Dict[str, Set[str]] = {}
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for e in cfg.edges():
            for lhs, rhs in (e.data.assignments or {}).items():
                asg.setdefault(lhs, set()).add(str(rhs))
    return asg


def outside_index_param_coeff(inner_sdfg: SDFG, idxarr: str, vector_length: int) -> Optional[int]:
    """
    Recover the per-lane stride into the index array from the NSDFG boundary.

    The inside contiguity check (``_laneid_<k> = idxarr[begin + k]``)
    proves the fan reads the *view* ``idxarr`` contiguously, but the view
    is the NSDFG-input window into the original index array and may be
    strided: ``idx[c*i]`` feeds a step-1 box
    ``idx[c*tile : c*tile + c*(W-1)]`` (every element touched), so lane
    ``k``'s real index lives at view offset ``c*k``, not ``k``. The true
    per-lane stride ``c`` is the coefficient of the enclosing vectorized
    map param in the feeding memlet's param-bearing dim.

    The collapse can still emit a *strided* gather/scatter — the view
    already contains all touched elements; the intrinsic just indexes it
    by ``c`` (``__vec_lane_idx[l] = _idx[l*c]``). So this returns ``c``
    rather than refusing.

    Conservative: returns ``None`` (caller keeps the per-lane laneid fan)
    when the boundary cannot be proven — ambiguous feeding edge, non-unit
    step, non-constant / non-positive-integer coefficient, or a window
    too short to hold the ``c*(W-1)+1`` touched span. A static window
    with no param dependence is contiguous, returns ``1``.

    :param inner_sdfg: The nested SDFG holding the fan (``state.sdfg``).
    :param idxarr: The recognised index-array / connector name.
    :param vector_length: The fan width ``W``.
    :returns: The integer per-lane stride ``c >= 1``, or ``None``.
    """
    nsdfg_node = inner_sdfg.parent_nsdfg_node
    parent_state = inner_sdfg.parent
    if nsdfg_node is None or parent_state is None:
        # Top-level array: inside contiguity is authoritative (no hidden
        # flattened window across a boundary) ⇒ unit stride.
        return 1
    feeds = list(parent_state.in_edges_by_connector(nsdfg_node, idxarr))
    if len(feeds) != 1:
        return None
    memlet = feeds[0].data
    if memlet.subset is None:
        return None
    map_entry = parent_state.entry_node(nsdfg_node)
    if not isinstance(map_entry, nodes.MapEntry):
        # No enclosing map ⇒ static window; contiguous iff unit-step.
        return 1 if all(s == 1 for (_, _, s) in memlet.subset) else None
    param_sym = dace.symbolic.symbol(map_entry.map.params[-1])
    coeff: Optional[int] = 1
    for (b, e, s) in memlet.subset:
        b_sym = dace.symbolic.pystr_to_symbolic(str(b))
        e_sym = dace.symbolic.pystr_to_symbolic(str(e))
        if param_sym not in (b_sym.free_symbols | e_sym.free_symbols):
            continue
        if dace.symbolic.simplify(s - 1) != 0:
            return None
        c_expr = b_sym.coeff(param_sym)
        if not getattr(c_expr, "is_Integer", False) or int(c_expr) < 1:
            return None
        c = int(c_expr)
        # The window must hold every touched element ``begin + c*k``,
        # k=0..W-1 (over-coverage is safe; the strided read only hits
        # positions 0, c, ..., c*(W-1)).
        span = dace.symbolic.simplify(e_sym - b_sym + 1 - (c * (vector_length - 1) + 1))
        if not (getattr(span, "is_Integer", False) and int(span) >= 0):
            return None
        coeff = c
    return coeff


def _recognize_laneid_index_slice(
        state: dace.SDFGState, idx_data_and_subset: List[Tuple[str, dace.subsets.Range]],
        vector_length: int) -> Optional[Tuple[str, dace.symbolic.SymbolicType, int, List[str]]]:
    """
    Recognise a per-lane laneid fan and the per-lane stride into the index array.

    Each fan tasklet's far edge reads ``<data>[<base>_laneid_<k>]`` for
    ``k = 0 .. W-1``. The W laneid symbols must be defined by interstate-
    edge assignments ``<base>_laneid_<k> = <idxarr>[<begin> + s*<k>]`` for
    the *same* index array ``<idxarr>``, the same (possibly symbolic)
    ``<begin>``, and a constant positive integer per-lane stride ``s``.
    ``expand_interstate_assignments_to_lanes`` already injects the true
    boundary coefficient into the fan (``view(c*i)``), so the inside
    exprs carry the correct ``s`` directly — it is read here, not
    re-derived from the boundary (re-applying ``c`` would double it).
    ``s == 1`` is a genuine contiguous gather; ``s > 1`` is a strided
    index access (``idx[s*i]``) the collapse emits as a strided gather
    (the view holds every touched element; the intrinsic indexes it by
    ``s``).

    :param state: The state holding the fan (its ``sdfg`` owns the
        interstate edges that bind the laneid symbols).
    :param idx_data_and_subset: ``(data, subset)`` per fan tasklet, in
        lane order, as returned by :func:`_match_assign_fan`.
    :param vector_length: The fan width ``W``.
    :returns: ``(idxarr, begin_expr, stride, [laneid_sym_0 .. _{W-1}])``
        or ``None`` if the fan / boundary cannot be proven.
    """
    base_name: Optional[str] = None
    for lane, (_, subset) in enumerate(idx_data_and_subset):
        if len(subset) != 1:
            return None
        begin, end, step = subset[0]
        if begin != end or step != 1:
            return None
        parsed = LaneIdScheme.parse(str(begin))
        if parsed is None:
            return None
        base, parsed_lane = parsed
        if parsed_lane != lane:
            return None
        if base_name is None:
            base_name = base
        elif base != base_name:
            return None
    if base_name is None:
        return None

    asg = _collect_interstate_assignments(state.sdfg)
    idxarr: Optional[str] = None
    begin_expr: Optional[dace.symbolic.SymbolicType] = None
    idx_exprs: List[dace.symbolic.SymbolicType] = []
    laneid_syms: List[str] = []
    for k in range(vector_length):
        sym = LaneIdScheme.make(base_name, k)
        rhs_set = asg.get(sym)
        # The laneid symbol must be bound by exactly one RHS reading
        # ``<idxarr>[<expr_k>]`` with the same array across all lanes.
        if rhs_set is None or len(rhs_set) != 1:
            return None
        m = _INDEX_ARRAY_READ_RE.match(next(iter(rhs_set)))
        if m is None:
            return None
        arr = m.group(1)
        try:
            idx_expr = dace.symbolic.pystr_to_symbolic(m.group(2))
        except Exception:
            return None
        if idxarr is None:
            idxarr = arr
            begin_expr = idx_expr
        elif arr != idxarr:
            return None
        idx_exprs.append(idx_expr)
        laneid_syms.append(sym)
    if idxarr is None or idxarr not in state.sdfg.arrays:
        return None

    # Derive the per-lane stride from the laneid exprs themselves:
    # ``expr_k - begin == s*k`` for a constant positive integer ``s``.
    # ``expand_interstate_assignments_to_lanes`` already injects the true
    # ``view(c*i)`` (the boundary coefficient) into the fan, so the
    # inside exprs carry the correct stride directly — no separate
    # outside-boundary trace is needed (and applying ``c`` again here
    # would double it). ``s == 1`` is the contiguous gather.
    if vector_length == 1:
        stride = 1
    else:
        delta = dace.symbolic.simplify(idx_exprs[1] - begin_expr)
        if not (getattr(delta, "is_Integer", False) and int(delta) >= 1):
            return None
        stride = int(delta)
        for k in range(vector_length):
            if dace.symbolic.simplify(idx_exprs[k] - begin_expr - stride * k) != 0:
                return None
    return idxarr, begin_expr, stride, laneid_syms


def _symbol_referenced_outside_defining_assignment(sdfg: SDFG, sym: str) -> bool:
    """
    Whether ``sym`` is read anywhere other than as the LHS of its own
    interstate-edge assignment.

    Scans interstate assignment RHSs / conditions, ``ConditionalBlock``
    branch conditions, ``LoopRegion`` init/cond/update, tasklet code,
    memlet subsets, and the parent NSDFG ``symbol_mapping``. A laneid
    symbol that survives this scan is kept (loud-safe): dropping a still-
    referenced symbol would silently corrupt the SDFG.

    :param sdfg: The SDFG that binds ``sym``.
    :param sym: The symbol to test.
    :returns: ``True`` if a remaining reader exists.
    """
    only = {sym}
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for e in cfg.edges():
            for lhs, rhs in (e.data.assignments or {}).items():
                if lhs == sym:
                    continue
                if dace.symbolic.symbols_in_code(str(rhs), potential_symbols=only):
                    return True
            if e.data.condition is not None:
                if dace.symbolic.symbols_in_code(e.data.condition.as_string, potential_symbols=only):
                    return True
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, ConditionalBlock):
            for c, _ in block.branches:
                if c is None:
                    continue
                if dace.symbolic.symbols_in_code(c.as_string, potential_symbols=only):
                    return True
    for region in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(region, LoopRegion):
            for attr in ("loop_condition", "update_statement", "init_statement"):
                code = getattr(region, attr, None)
                if code is None:
                    continue
                if dace.symbolic.symbols_in_code(code.as_string, potential_symbols=only):
                    return True
    for s in sdfg.all_states():
        for n in s.nodes():
            if isinstance(n, nodes.Tasklet):
                if dace.symbolic.symbols_in_code(n.code.as_string, potential_symbols=only):
                    return True
        for e in s.edges():
            if e.data.data is None:
                continue
            if dace.symbolic.symbols_in_code(str(e.data.subset), potential_symbols=only):
                return True
            if e.data.other_subset is not None and dace.symbolic.symbols_in_code(str(e.data.other_subset),
                                                                                 potential_symbols=only):
                return True
    if sdfg.parent_nsdfg_node is not None:
        for k, v in sdfg.parent_nsdfg_node.symbol_mapping.items():
            if k == sym:
                continue
            if dace.symbolic.symbols_in_code(str(v), potential_symbols=only):
                return True
    return False


def _drop_collapsed_laneid_syms(sdfg: SDFG, syms: List[str]) -> None:
    """
    Delete the now-dead laneid symbols and their interstate assignments.

    A symbol is dropped only if
    :func:`_symbol_referenced_outside_defining_assignment` proves it has
    no remaining reader (the fan tasklets that read it were already
    removed by the caller). Symbols with a surviving reference are left
    in place — a conservative, correctness-preserving no-op.

    :param sdfg: The SDFG binding the symbols.
    :param syms: Laneid symbols slated for removal.
    """
    for sym in syms:
        if _symbol_referenced_outside_defining_assignment(sdfg, sym):
            continue
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            for e in cfg.edges():
                if e.data.assignments and sym in e.data.assignments:
                    del e.data.assignments[sym]
        if sym in sdfg.symbols:
            sdfg.remove_symbol(sym)


def detect_lane_fanout_apply(sdfg: SDFG,
                             *,
                             direction: str,
                             pattern: str,
                             intrinsic_template: str,
                             intrinsic_tasklet_name: str,
                             intrinsic_template_masked: Optional[str] = None,
                             skip_unmasked: bool = False,
                             collapse_laneid_index_loads: bool = False,
                             intrinsic_template_idxarr: Optional[str] = None,
                             intrinsic_template_idxarr_masked: Optional[str] = None,
                             intrinsic_template_idxarr_conv: Optional[str] = None,
                             intrinsic_template_idxarr_conv_masked: Optional[str] = None) -> int:
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
    :param skip_unmasked: When ``True``, only collapse fans in masked
        states (an ``_iter_mask`` in scope, i.e. the vector remainder);
        leave the per-lane scalar fan in unmasked states (the main loop)
        untouched. The masked remainder must always collapse - per-lane
        scalar fan faults on inactive lanes - so this only opts the main
        loop out to scalar gather/scatter.
    :param collapse_laneid_index_loads: When ``True`` (and the
        ``intrinsic_template_idxarr*`` templates are supplied), a
        ``contiguous`` fan whose per-lane indices are W laneid symbols
        bound to a contiguous index-array slice
        ``<idxarr>[<begin> : <begin> + W]`` is collapsed so the intrinsic
        reads the index array directly via an ``_idx`` connector; the
        now-dead laneid symbols and their interstate-edge assignments are
        removed (only when provably unreferenced elsewhere). A
        non-contiguous *access* is carried by the index values, not by
        striding the index table.
    :param intrinsic_template_idxarr: CPP body template for the
        ``_idx``-direct fast path (uses ``{vector_length}``/``{dtype}``;
        passes the ``_idx`` slice pointer straight into the intrinsic).
        Used when the index array is already ``int64``.
    :param intrinsic_template_idxarr_masked: Masked counterpart, used
        when an iteration mask is in scope.
    :param intrinsic_template_idxarr_conv: Fallback template used when
        the index array is *not* ``int64``: it materialises a local
        ``int64`` buffer from ``_idx`` (an element-width conversion the
        runtime signature requires — not a regeneration of the indices).
    :param intrinsic_template_idxarr_conv_masked: Masked counterpart of
        the conversion fallback.
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
            # ``skip_unmasked``: keep the main loop's per-lane scalar fan.
            # The masked remainder (iter_mask in scope) must still collapse.
            if skip_unmasked and iter_mask_name is None:
                continue
            template = intrinsic_template_masked if iter_mask_name is not None else intrinsic_template

            # Optional laneid index-fan collapse: recognise the W laneid
            # symbols as a fixed-stride ``<idxarr>[begin + stride*k]``
            # slice and emit the variant that reads the index array
            # through an ``_idx`` connector instead of W interstate-edge
            # symbols.
            idxarr_match: Optional[Tuple[str, dace.symbolic.SymbolicType, int, List[str]]] = None
            if (pattern == "contiguous" and collapse_laneid_index_loads and intrinsic_template_idxarr is not None):
                idxarr_match = _recognize_laneid_index_slice(state, idx_data_and_subset, vector_length)

            if pattern == "contiguous" and idxarr_match is not None:
                idx_stride = idxarr_match[2]
                # Direct pointer pass only when the index array is int64
                # (the runtime intrinsic's index type) AND the access is
                # genuinely contiguous (stride 1). Otherwise the
                # conversion variant materialises a local int64 buffer
                # ``__vec_lane_idx[l] = _idx[l*stride]`` — covering a
                # narrower index dtype and/or a strided index access
                # (``idx[c*i]``), reading the right element per lane out
                # of the contiguous boundary window.
                idx_is_int64 = state.sdfg.arrays[idxarr_match[0]].dtype == dace.int64
                if idx_is_int64 and idx_stride == 1:
                    idxarr_template = (intrinsic_template_idxarr_masked
                                       if iter_mask_name is not None else intrinsic_template_idxarr)
                    intrinsic_code = idxarr_template.format(vector_length=vector_length, dtype=dtype_cpp)
                else:
                    idxarr_template = (intrinsic_template_idxarr_conv_masked
                                       if iter_mask_name is not None else intrinsic_template_idxarr_conv)
                    intrinsic_code = idxarr_template.format(vector_length=vector_length,
                                                            dtype=dtype_cpp,
                                                            stride=idx_stride)
            elif pattern == "contiguous":
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

            tasklet_inputs = {"_in"}
            if iter_mask_name is not None:
                tasklet_inputs.add("_mask")
            if idxarr_match is not None:
                tasklet_inputs.add("_idx")
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
                if idxarr_match is not None:
                    idxarr, begin_expr, idx_stride, laneid_syms = idxarr_match
                    idx_an = state.add_access(idxarr)
                    # ``_idx`` maps the full contiguous span the strided
                    # read touches: positions ``begin .. begin +
                    # stride*(W-1)`` (step 1 — the buffer is contiguous;
                    # the intrinsic strides into it via ``_idx[l*stride]``).
                    # ``stride == 1`` reduces to the W-wide window.
                    idx_end = begin_expr + idx_stride * (vector_length - 1)
                    idx_memlet = dace.memlet.Memlet(data=idxarr, subset=dace.subsets.Range([(begin_expr, idx_end, 1)]))
                    state.add_edge(idx_an, None, t1, "_idx", idx_memlet)
                    # The fan tasklets that read the laneid symbols were
                    # just removed; drop those now-dead symbols and their
                    # interstate assignments (only if provably unused).
                    _drop_collapsed_laneid_syms(state.sdfg, laneid_syms)
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
                found += detect_lane_fanout_apply(
                    node.sdfg,
                    direction=direction,
                    pattern=pattern,
                    intrinsic_template=intrinsic_template,
                    intrinsic_tasklet_name=intrinsic_tasklet_name,
                    intrinsic_template_masked=intrinsic_template_masked,
                    skip_unmasked=skip_unmasked,
                    collapse_laneid_index_loads=collapse_laneid_index_loads,
                    intrinsic_template_idxarr=intrinsic_template_idxarr,
                    intrinsic_template_idxarr_masked=intrinsic_template_idxarr_masked,
                    intrinsic_template_idxarr_conv=intrinsic_template_idxarr_conv,
                    intrinsic_template_idxarr_conv_masked=intrinsic_template_idxarr_conv_masked)
    return found
