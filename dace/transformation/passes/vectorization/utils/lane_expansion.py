# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Symbol lane fan-out helpers for the vectorization pipeline.

When a scalar symbol is computed by an interstate edge in an inner SDFG
below a vectorized map, every lane needs its own copy of the symbol.
These helpers build the per-lane variants and stitch them back into the
SDFG.
"""
from typing import Sequence, Set

import sympy

import dace
from dace.transformation.helpers import get_parent_map_and_loop_scopes
import dace.sdfg.utils as sdutil
from dace.sdfg.state import LoopRegion
from dace.symbolic import DaceSympyPrinter

from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme
from dace.transformation.passes.vectorization.utils.lane_fanout import outside_index_param_coeff
from dace.transformation.passes.vectorization.utils.symbolic_polymorphism import (atoms_of, free_symbol_names,
                                                                                  free_symbols, is_integer)


def assert_symbols_in_parent_map_symbols(missing_symbols: Set[str], state: dace.SDFGState,
                                         nsdfg: dace.nodes.NestedSDFG):
    """
    Validate that symbols correspond to loop variables in parent map scopes of a NestedSDFG.

    :param missing_symbols: Symbols to validate (e.g. ``{"i_laneid_0", "j_laneid_1"}``).
    :param state: The SDFG state.
    :param nsdfg: The NestedSDFG node.
    :returns: Set of loop variable names found in the parent scopes.
    :raises AssertionError: If a symbol is not found in the loop scopes.
    """

    # Peel every lane chunk via the canonical helper so legacy
    # ``<base>_laneid_<n>`` and Option B ``<base>_lane<d>id_<n>`` names
    # both reduce to the bare base.
    loop_vars = {LaneIdScheme.base_of(s) for s in missing_symbols}

    sdict = state.scope_dict()
    first_parent_map = sdict[nsdfg]
    parent_maps_and_loops = get_parent_map_and_loop_scopes(state.sdfg, first_parent_map, state)

    loop_symbols = set()
    for p in first_parent_map.map.params:
        loop_symbols.add(p)

    for map_or_loop in parent_maps_and_loops:
        if isinstance(map_or_loop, dace.nodes.MapEntry):
            for p in map_or_loop.map.params:
                loop_symbols.add(p)
        elif isinstance(map_or_loop, LoopRegion):
            loop_symbols.add(map_or_loop.loop_variable)

    for loop_var in loop_vars:
        assert loop_var in loop_symbols or loop_var in nsdfg.symbol_mapping, (
            f"{loop_var} not in parent-scope loop_symbols={loop_symbols} and not in "
            f"nsdfg.symbol_mapping={set(nsdfg.symbol_mapping.keys())}")

    return loop_vars


def find_symbol_assignment(sdfg: dace.SDFG, sym_name: str) -> str:
    """
    Find a symbol's assignment expression by traversing the SDFG backwards.

    :param sdfg: The SDFG to search.
    :param sym_name: Symbol to find.
    :returns: Assignment expression as a string, or ``None`` if not found.
    """

    # Pre-condition for vectorization
    assert all({isinstance(s, dace.SDFGState) for s in sdfg.nodes()})
    sink_state = {s for s in sdfg.nodes() if sdfg.out_degree(s) == 0}.pop()
    edges_to_check = sink_state.parent_graph.in_edges(sink_state)
    while edges_to_check:
        edge = edges_to_check.pop()

        for k, v in edge.data.assignments.items():
            if k == sym_name:
                return v

        edges_to_check += sink_state.parent_graph.in_edges(edge.src)

    return None


def _all_atoms(expr, ignored=()):
    """
    Return all atomic elements (Symbols, Indexed, Function calls) of a SymPy expression.

    :param expr: The SymPy expression to inspect.
    :param ignored: Function classes to skip (e.g. ``(sympy.Number,)``).
    :returns: Set of atoms found in the expression.
    """
    # Use expr.atoms to get all different types of atoms
    atoms = set()

    # Get all symbols
    atoms.update(expr.atoms(sympy.Symbol))

    # Get all Indexed (arrays)
    atoms.update(expr.atoms(sympy.Indexed))

    # Get all function symbols (but not the class, only instances)
    funcs = expr.atoms(sympy.Function)
    for f in funcs:
        if f.func not in ignored:
            atoms.add(f)
            # Also include arguments of the function
            atoms.update(f.args)

    return atoms


def expand_interstate_assignments_to_lanes(inner_sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG,
                                           state: dace.SDFGState, vector_width: int, invariant_data: Set[str],
                                           vector_map_param: str):
    """
    Fan out every interstate-edge assignment into one variant per lane.

    ``sym = expr`` becomes a family ``sym_laneid_<i> = expr`` with the
    vector iterator shifted by the lane index. Fully invariant
    expressions keep only the original symbol; already-lane-encoded LHS
    keys are carried through unchanged (idempotency).

    :param inner_sdfg: The nested SDFG whose interstate edges to expand.
    :param nsdfg_node: The NestedSDFG node containing ``inner_sdfg``.
    :param state: The state holding ``nsdfg_node``.
    :param vector_width: Number of lanes (variants emitted per assignment).
    :param invariant_data: Names whose values are constant across lanes.
    :param vector_map_param: The vectorized map parameter name.
    """
    # `sym = 0`
    # Would become
    # `sym_laneid_0 = 0, sym=sym_laneid_0, sym_laneid_1 = 0, sym_laneid_2 = 0, ....`
    # Assume:
    # `sym = A[_for_it] + 1`
    # Would become:
    # `sym_laneid_0 = A[_for_it + 0] + 1`, `sym = sym_laneid_0`, `sym_laneid_1 = A[_for_it + 1] + 1`, ...

    # Invariant data means that the data is constant across iterators
    # If all free symbols are from invariant data then duplication is not necessar

    # Pre-condition last dimension is the dimension we vectorize
    parent_map_entry = state.scope_dict()[nsdfg_node]
    assert parent_map_entry is not None and isinstance(parent_map_entry, dace.nodes.MapEntry)
    vectorized_param = vector_map_param

    for edge in inner_sdfg.all_interstate_edges():
        new_assignments = dict()
        assignments = edge.data.assignments

        # Idempotency: any LHS that already encodes a lane in its name is taken as
        # fixed (its lane is fully determined by the suffix). Re-expanding it would
        # produce <base>_laneid_<i>_laneid_<j> double-suffixed garbage. Carry the
        # already-expanded assignments through unchanged and drive the per-lane loop
        # only over the plain (un-encoded) keys.
        plain_assignments = {}
        for k, v in assignments.items():
            if LaneIdScheme.is_laneid(k):
                new_assignments[k] = v
            else:
                plain_assignments[k] = v

        for k, v in plain_assignments.items():
            original_v_expr = dace.symbolic.SymExpr(v)
            for i in range(vector_width):
                new_k = LaneIdScheme.make_dim(k, 0, i)
                v_expr = dace.symbolic.SymExpr(v)

                # Lane-variance is carried by the free symbols of the assignment
                # (an array head is no longer a free symbol of a ``Subscript``,
                # and the legacy ``array_accesses`` set was always empty here
                # because ``str(Function)`` never matched an array name, so this
                # restores that effective behaviour).
                variant_array_accesses = {str(s) for s in v_expr.free_symbols} - invariant_data

                if len(variant_array_accesses) == 0:
                    # Whole expression is invariant — keep the original (un-expanded) symbol only.
                    new_assignments[k] = v
                    continue

                if new_k not in inner_sdfg.symbols:
                    inner_sdfg.add_symbol(new_k, inner_sdfg.symbols.get(k, dace.float64))

                # Replace the vector iterator with iter+lane
                v_expr = v_expr.subs(vectorized_param, f"({vectorized_param} + {i})")

                # Other free symbols are duplicated per-lane; symbols that already encode
                # a lane (parse non-trivially) are skipped so we never produce a doubly
                # lane-suffixed name.
                non_map_free_syms = {str(s)
                                     for s in original_v_expr.free_symbols} - ({vectorized_param}.union(
                                         inner_sdfg.free_symbols))
                assert vectorized_param not in non_map_free_syms

                for free_sym in non_map_free_syms:
                    free_sym_str = str(free_sym)
                    assert free_sym_str in inner_sdfg.arrays or free_sym_str in inner_sdfg.symbols

                    if LaneIdScheme.is_laneid(free_sym_str):
                        # Already lane-bound; its lane is fixed by the name. Don't re-encode.
                        continue

                    if free_sym_str in inner_sdfg.symbols:
                        if free_sym_str == vector_map_param:
                            raise AssertionError(
                                f"vector_map_param {vector_map_param!r} appeared in non_map_free_syms; "
                                f"upstream filtering is broken")
                        lane_sym = LaneIdScheme.make_dim(free_sym_str, 0, i)
                        v_expr = v_expr.subs(free_sym, lane_sym)
                        if lane_sym not in inner_sdfg.symbols:
                            inner_sdfg.add_symbol(lane_sym, inner_sdfg.symbols.get(free_sym_str, dace.float64))
                    else:
                        if isinstance(inner_sdfg.arrays[free_sym_str], dace.data.Scalar):
                            v_expr = v_expr.subs(free_sym, f"{free_sym}")
                        else:
                            assert inner_sdfg.arrays[free_sym_str].shape != (1, )
                            # The connector ``free_sym`` is the NSDFG-input
                            # window of an array accessed ``arr[c*i]`` in
                            # the parent. The window is the step-1 bbox of
                            # the W touched elements, so lane ``i`` lives at
                            # offset ``c*i`` — *not* ``i``. Recover ``c``
                            # from the boundary memlet (same helper the
                            # gather/scatter collapse uses, so the per-lane
                            # element identity agrees on both sides).
                            # ``c == 1`` or an unprovable boundary ⇒ the
                            # plain unit-lane offset ``({i})``.
                            c = outside_index_param_coeff(inner_sdfg, free_sym_str, vector_width)
                            if c is not None and c > 1:
                                # Inside bound check (the outside window
                                # is already verified ≥ c*(W-1)+1 by the
                                # helper): the inner connector must hold
                                # the same span, else ``view(c*i)`` reads
                                # out of bounds. The reshape contract says
                                # inner == outside-bbox; if a provably-
                                # constant inner shape contradicts that,
                                # fail loudly rather than emit an OOB
                                # access or silently revert to the wrong
                                # ``view(i)``.
                                inner_desc = inner_sdfg.arrays[free_sym_str]
                                need = c * (vector_width - 1) + 1
                                total = inner_desc.total_size
                                total_simpl = dace.symbolic.simplify(total - need)
                                if is_integer(total_simpl) and int(total_simpl) < 0:
                                    raise RuntimeError(f"Lane fan-out for '{free_sym_str}': inner connector size "
                                                       f"{total} < required strided span {need} (stride c={c}, "
                                                       f"W={vector_width}); the NSDFG-input window and the inner "
                                                       f"connector disagree (reshape inconsistency).")
                                v_expr = v_expr.subs(free_sym, f"{free_sym}({c} * {i})")
                            else:
                                v_expr = v_expr.subs(free_sym, f"{free_sym}({i})")

                # ``DaceSympyPrinter`` prints array reads as ``arr[idx]``
                # (subscript form for names in the ``arrays`` set) and emits
                # ``(a and b)`` / ``(a or b)`` / ``(not a)`` directly for
                # ``sympy.Or``/``And``/``Not``, so the previous two-step
                # ``sympy.pycode`` + ``rewrite_boolean_functions_to_boolean_ops``
                # + ``convert_nonstandard_calls`` chain collapses to one print.
                printer = DaceSympyPrinter(set(inner_sdfg.arrays.keys()))
                new_v = printer.doprint(v_expr)
                new_assignments[new_k] = new_v

                if i == 0:
                    # Keep the original un-suffixed symbol bound to the lane-0 expansion so
                    # downstream consumers that haven't been retargeted yet still see it.
                    new_assignments[k] = new_v

        edge.data.assignments = new_assignments


def _widen_index_connector_to_tile(inner_sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG,
                                   parent_state: dace.SDFGState, idx_conn: str, vector_width: int,
                                   tile_iter_var: str) -> None:
    """Reshape a length-1 index connector to a ``(W,)`` tile and widen its outer edge.

    The frontend loads the per-lane index into a ``(1,)`` connector (``idx[i]``).
    To gather the ``W`` per-lane indices, the inner connector array is reshaped
    to ``(W,)`` and its outer edge subset is grown from ``idx[c*i]`` to the tile
    region ``idx[c*i : c*i + c*(W-1) : c]`` (the dim whose ``begin`` references
    the tile var). The window is ``c``-strided, where ``c`` is the tile var's
    coefficient in ``begin``: lane ``l`` reads ``begin(i + l) = c*i + c*l``, so
    ``b[idx[2*i]]`` gathers ``idx[2*i], idx[2*i+2], ...`` not ``idx[2*i+1]``.
    A unit coefficient recovers the contiguous ``idx[i:i+W]`` window.
    Idempotent: a connector already of shape ``(W,)`` is left untouched.

    :param inner_sdfg: The tile-body SDFG.
    :param nsdfg_node: The body NestedSDFG node.
    :param parent_state: State holding ``nsdfg_node``.
    :param idx_conn: The index connector name.
    :param vector_width: Tile width ``W``.
    :param tile_iter_var: The tile iter-var name.
    """
    arr = inner_sdfg.arrays[idx_conn]
    if tuple(arr.shape) == (vector_width, ):
        return
    dtype = arr.dtype
    sym = dace.symbolic.symbol(tile_iter_var)
    # ``begin = c*i + d`` (affine in the tile var): lane ``l`` reads
    # ``begin + c*l``, so the per-lane index window is ``c``-strided. The outer
    # edge is grown to the CONTIGUOUS bounding window ``idx[c*i : c*i+c*(W-1)]``
    # (``c*(W-1)+1`` elements) and the connector keeps that contiguous shape:
    # the ``c``-strided per-lane pick is realised at the gather, which reads
    # ``_idx[c*l]`` (TileLoad (gather_dims) ``index_strides``). A strided memlet / strided
    # connector would NOT work because the NSDFG argument is a base pointer
    # (``&idx[c*i]``) that drops the source stride. A unit ``c`` recovers the
    # plain ``idx[i:i+W]`` window.
    lane_stride = 1
    for oe in parent_state.in_edges(nsdfg_node):
        if oe.dst_conn != idx_conn or oe.data is None or oe.data.subset is None:
            continue
        new_ranges = []
        for (b, e, s) in oe.data.subset:
            b_syms = free_symbols(b)
            if sym in b_syms:
                coeff = b.coeff(sym)
                affine = sym not in (b - coeff * sym).free_symbols
                lane_stride = coeff if (affine and coeff != 0) else 1
                new_ranges.append((b, b + lane_stride * (vector_width - 1), 1))
            else:
                new_ranges.append((b, e, s))
        oe.data.subset = dace.subsets.Range(new_ranges)
    inner_sdfg.remove_data(idx_conn, validate=False)
    inner_sdfg.add_array(idx_conn, [lane_stride * (vector_width - 1) + 1],
                         dtype,
                         storage=dace.dtypes.StorageType.Register,
                         transient=False)


def fan_out_tile_gather_index_symbols(inner_sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG,
                                      parent_state: dace.SDFGState, vector_width: int, tile_iter_var: str) -> Set[str]:
    """Tile analog of :func:`expand_interstate_assignments_to_lanes` for gather indices.

    A frontend-lowered gather ``src[idx[i]]`` leaves the per-lane index in a
    length-1 connector ``C_idx`` (``= idx[i]``) consumed by an interstate-edge
    assignment ``__sym = C_idx`` (the point gather ``src[__sym]`` reads
    ``__sym``). To gather the ``W`` per-lane indices into an array — the same
    fan the 1D path builds as ``sym_laneid_<i>`` — this widens ``C_idx`` to a
    ``(W,)`` tile (``idx[i] -> idx[i:i+W]``) and fans ``__sym`` out into
    ``__sym_laneid_<l> = C_idx[l]`` per lane. A later collapse pass promotes
    the per-lane fan into a ``TileLoad`` (with ``gather_dims``) and simplifies the symbols away
    (the tile mirror of expand-then-``DetectGather``).

    Unlike the 1D helper, a length-1 index connector is NOT treated as a
    lane-invariant scalar — the tile body has already loaded ``idx`` into it,
    so it is widened and indexed per lane.

    :param inner_sdfg: The tile-body SDFG whose interstate edges to fan out.
    :param nsdfg_node: The body NestedSDFG node (for connector names).
    :param parent_state: State holding ``nsdfg_node`` (for outer edges).
    :param vector_width: Number of lanes ``W`` (K=1 tile).
    :param tile_iter_var: The tile iter-var name (the strided map param).
    :returns: The set of widened index-connector names.
    """
    widened: Set[str] = set()
    for edge in inner_sdfg.all_interstate_edges():
        assignments = edge.data.assignments
        new_assignments = {}
        plain = {}
        for k, v in assignments.items():
            if LaneIdScheme.is_laneid(k):
                new_assignments[k] = v
            else:
                plain[k] = v
        # A connector is a length-1 per-lane index source when the OUTER
        # memlet exposes exactly 1 element per outer iteration AND the outer
        # subset is 1-dim (so the connector can be widened to a 1D ``(W,)``
        # tile by :func:`_widen_index_connector_to_tile`). We can't key off
        # the inner array shape because RefineNestedAccess sometimes leaves
        # the inner array at its full source shape and constrains the access
        # via the boundary memlet (vag-style ``ip[i]`` boundary with inner
        # shape ``(LEN_1D,)``). The 1-dim guard prevents over-triggering on
        # multi-dim boundary connectors (icon-style ``edge_blk[i, jk, m]``
        # where the outer is 3-dim, num_elements 1; widening to ``(W,)``
        # would invalidate downstream multi-dim gather emission).
        outer_extent_one = {}
        outer_has_tile_var = {}
        iter_var_sym = dace.symbolic.pystr_to_symbolic(tile_iter_var)
        for oe in parent_state.in_edges(nsdfg_node):
            if oe.dst_conn is None or oe.data is None or oe.data.subset is None:
                continue
            try:
                ne = oe.data.subset.num_elements()
                one_elem = (dace.symbolic.simplify(ne) == 1)
                one_dim = (len(oe.data.subset) == 1)
                outer_extent_one[oe.dst_conn] = one_elem and one_dim
            except (TypeError, AttributeError):
                outer_extent_one[oe.dst_conn] = False
            # Tile-var-dep guard (mirror of :func:`fan_out_tile_gather_index_symbols_kd`):
            # a length-1 outer subset whose begin has NO tile-var dependency is
            # a loop-invariant scalar (cloudsc's ``zqx[z1, j+1, i+1]`` z1 — a
            # scalar parameter passed via a single-element boundary). Widening
            # it to a ``(W,)`` tile would create per-lane reads ``z1[0..W-1]``
            # past the 1-element source → OOB / segfault. Skip widening so the
            # downstream gather/scatter classifier treats it as a Scalar
            # broadcast.
            any_tv = False
            for (b, _e, _s) in oe.data.subset:
                if iter_var_sym in free_symbols(b):
                    any_tv = True
                    break
            outer_has_tile_var[oe.dst_conn] = any_tv
        for k, v in plain.items():
            v_expr = dace.symbolic.SymExpr(v)
            # The assignment reaches the connector either as a bare symbol
            # (``ip_index = ip`` when the connector is rank-0) or as a
            # ``Subscript`` base (``ip_index = ip[0]`` when the connector is a
            # length-1 array view). ``free_symbols`` only catches the bare
            # form because ``ip[0]`` parses as ``Subscript(ip, 0)`` whose
            # base ``ip`` is not a free symbol. ``dace.symbolic.arrays``
            # collects the subscript bases so we catch both forms.
            referenced = {str(s) for s in v_expr.free_symbols} | dace.symbolic.arrays(v_expr)
            idx_conns = sorted({
                s
                for s in referenced if s in nsdfg_node.in_connectors and not inner_sdfg.arrays[s].transient
                and outer_extent_one.get(s, False) and outer_has_tile_var.get(s, False)
            })
            if not idx_conns:
                new_assignments[k] = v
                continue
            for c in idx_conns:
                _widen_index_connector_to_tile(inner_sdfg, nsdfg_node, parent_state, c, vector_width, tile_iter_var)
                widened.add(c)
            # The assignment value uses ``c`` either as a bare ``Symbol`` (rank-0
            # connector form) OR as the base of a ``Subscript`` ``c[0]`` (rank-1
            # length-1 view form). For the bare case ``sympy.subs`` replaces the
            # symbol with ``c(lane)`` (rendered as ``c[lane]`` by the dace
            # printer). For the subscript case the inner array was just widened
            # to ``(W,)`` by :func:`_widen_index_connector_to_tile` and the
            # subscript index ``0`` -> ``lane`` substitution is what produces
            # the per-lane read ``c[lane]``. Detect each case per connector.
            from dace.symbolic import Subscript
            subscript_bases = {str(node.args[0]) for node in atoms_of(v_expr, Subscript)}
            for lane in range(vector_width):
                lane_expr = v_expr
                for c in idx_conns:
                    if c in subscript_bases:
                        # ``c[0]`` form: rewrite the subscript index 0 -> lane.
                        lane_expr = lane_expr.replace(
                            lambda node: isinstance(node, Subscript) and str(node.args[0]) == c,
                            lambda node: Subscript(node.args[0], sympy.Integer(lane)))
                    else:
                        # Bare symbol form: substitute the symbol with ``c(lane)``.
                        lane_expr = lane_expr.subs(sympy.Symbol(c), dace.symbolic.SymExpr(f"{c}({lane})"))
                lane_v = DaceSympyPrinter(set(inner_sdfg.arrays.keys())).doprint(lane_expr)
                new_assignments[LaneIdScheme.make_dim(k, 0, lane)] = lane_v
                if lane == 0:
                    new_assignments[k] = lane_v
        edge.data.assignments = new_assignments
    return widened


def _widen_index_connector_to_tile_kd(inner_sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG,
                                      parent_state: dace.SDFGState, idx_conn: str, widths: Sequence[int],
                                      tile_iter_vars: Sequence[str]) -> None:
    """K-dim analog of :func:`_widen_index_connector_to_tile`.

    Reshape a length-1 boundary connector that feeds an interstate-edge
    gather index from ``(1,)`` to a ``(W_0, ..., W_{K-1})`` register-tile,
    sourced from a K-dim window of the outer boundary array. Each outer-
    subset dim whose ``begin`` references a tile iter-var is widened by
    that iter-var's W; non-tile dims stay degenerate. The K-tile fans the
    per-lane gather index across all K lanes — the gather collapse then
    reads ``c[__l0, ..., __l_{K-1}]`` per lane.

    Idempotent: a connector already at the K-tile shape is left untouched.

    :param inner_sdfg: The tile-body SDFG.
    :param nsdfg_node: The body NestedSDFG node.
    :param parent_state: State holding ``nsdfg_node``.
    :param idx_conn: The index connector name.
    :param widths: Per-tile-var width ``W_p``, K entries innermost-last.
    :param tile_iter_vars: Tile iter-var names, K entries innermost-last.
    """
    arr = inner_sdfg.arrays[idx_conn]
    W_tuple = tuple(int(w) for w in widths)
    if tuple(arr.shape) == W_tuple:
        return
    dtype = arr.dtype
    iter_syms = [dace.symbolic.symbol(v) for v in tile_iter_vars]
    # Walk every outer in-edge into this connector and grow each tile-var-
    # carrying subset dim by its W. A dim whose begin doesn't reference any
    # tile var stays degenerate (extent kept). Track the inner shape AND
    # the source-array per-dim strides (the connector is a strided view of
    # the source's tile block; the inner descriptor strides must match the
    # source so flat-offset reads ``_src[l0*stride_0 + l1*stride_1]`` hit
    # the right per-lane element).
    inner_shape: list = []
    inner_strides: list = []
    for oe in parent_state.in_edges(nsdfg_node):
        if oe.dst_conn != idx_conn or oe.data is None or oe.data.subset is None:
            continue
        src_arr = parent_state.sdfg.arrays[oe.data.data]
        new_ranges = []
        inner_shape = []
        inner_strides = []
        for d, (b, e, s) in enumerate(oe.data.subset):
            b_syms = free_symbols(b)
            iv_match_idx = None
            for p, isym in enumerate(iter_syms):
                if isym in b_syms:
                    iv_match_idx = p
                    break
            if iv_match_idx is not None:
                w = W_tuple[iv_match_idx]
                coeff = b.coeff(iter_syms[iv_match_idx])
                affine = iter_syms[iv_match_idx] not in (b - coeff * iter_syms[iv_match_idx]).free_symbols
                lane_stride = coeff if (affine and coeff != 0) else 1
                new_ranges.append((b, b + lane_stride * (w - 1), 1))
                inner_shape.append(lane_stride * (w - 1) + 1)
                inner_strides.append(src_arr.strides[d])
            else:
                # Degenerate / non-tile dim: keep the original extent.
                new_ranges.append((b, e, s))
                try:
                    extent = int(dace.symbolic.simplify(e - b + 1))
                except (TypeError, ValueError):
                    extent = 1
                inner_shape.append(max(1, extent))
                inner_strides.append(src_arr.strides[d])
        oe.data.subset = dace.subsets.Range(new_ranges)
    if not inner_shape:
        return
    inner_sdfg.remove_data(idx_conn, validate=False)
    inner_sdfg.add_array(idx_conn,
                         inner_shape,
                         dtype,
                         strides=inner_strides,
                         storage=dace.dtypes.StorageType.Register,
                         transient=False)
    # Source-stride symbols (``B`` row stride ``X``) must be in the NSDFG's
    # symbol mapping + inner symbols.
    parent_sdfg = parent_state.sdfg
    for stride in inner_strides:
        try:
            stride_syms = dace.symbolic.pystr_to_symbolic(str(stride)).free_symbols
        except Exception:
            continue
        for fs in stride_syms:
            sname = str(fs)
            if sname not in nsdfg_node.symbol_mapping:
                nsdfg_node.symbol_mapping[sname] = dace.symbolic.pystr_to_symbolic(sname)
            if sname not in inner_sdfg.symbols:
                inner_sdfg.add_symbol(sname, parent_sdfg.symbols.get(sname, dace.int64))


def fan_out_tile_gather_index_symbols_kd(inner_sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG,
                                         parent_state: dace.SDFGState, widths: Sequence[int],
                                         tile_iter_vars: Sequence[str]) -> Set[str]:
    """K-dim analog of :func:`fan_out_tile_gather_index_symbols`.

    For each interstate-edge assignment ``__sym = c`` or ``__sym = c[0]``
    where ``c`` is a length-1 boundary connector that the outer memlet
    binds to at least one tile iter-var, widen ``c`` to a K-shape
    ``(W_0, ..., W_{K-1})`` register tile via
    :func:`_widen_index_connector_to_tile_kd` and rewrite the assignment
    to read ``c[0, 0, ..., 0]`` (lane 0 value). The K-shape index tile is
    the downstream gather collapse's per-lane index source — the
    multidim-gather emission walks the Subscript indices and substitutes
    each iter-var-bound dim with its ``__l<p>`` so lane ``(l0, ..., l_{K-1})``
    reads ``c[l0, ..., l_{K-1}]``.

    Unlike the K=1 fan, this does NOT mint per-lane laneid assignments —
    K-dim laneid naming explodes (``__sym_laneid_l0_l1_..._l(K-1)``) and
    the gather collapse doesn't need it: it walks the BASE assignment's
    Subscript and substitutes ``__l<p>`` directly per dim.

    :param inner_sdfg: The tile-body SDFG.
    :param nsdfg_node: The body NestedSDFG node.
    :param parent_state: State holding ``nsdfg_node``.
    :param widths: Per-tile-var width ``W_p``, K entries innermost-last.
    :param tile_iter_vars: Tile iter-var names, K entries innermost-last.
    :returns: The set of widened index-connector names.
    """
    widened: Set[str] = set()
    iter_var_set = set(tile_iter_vars)
    # Cache outer-extent and tile-var-presence flags per connector.
    outer_has_tile_var = {}
    outer_extent_one = {}
    for oe in parent_state.in_edges(nsdfg_node):
        if oe.dst_conn is None or oe.data is None or oe.data.subset is None:
            continue
        any_tv = False
        for (b, _e, _s) in oe.data.subset:
            fs = free_symbol_names(b)
            if fs & iter_var_set:
                any_tv = True
                break
        outer_has_tile_var[oe.dst_conn] = any_tv
        try:
            ne = oe.data.subset.num_elements()
            outer_extent_one[oe.dst_conn] = bool(dace.symbolic.simplify(ne) == 1)
        except (TypeError, AttributeError):
            outer_extent_one[oe.dst_conn] = False
    for edge in inner_sdfg.all_interstate_edges():
        assignments = dict(edge.data.assignments)
        new_assignments = {}
        for k, v in assignments.items():
            if LaneIdScheme.is_laneid(k):
                new_assignments[k] = v
                continue
            v_expr = dace.symbolic.SymExpr(v)
            # Catch both bare ``__sym = c`` (rank-0 connector reference) and
            # subscript ``__sym = c[0]`` forms — :func:`dace.symbolic.arrays`
            # collects subscript bases, ``free_symbols`` catches bare names.
            referenced = {str(s) for s in v_expr.free_symbols} | dace.symbolic.arrays(v_expr)
            idx_conns = sorted({
                s
                for s in referenced if s in nsdfg_node.in_connectors and not inner_sdfg.arrays[s].transient
                and outer_extent_one.get(s, False) and outer_has_tile_var.get(s, False)
            })
            if not idx_conns:
                new_assignments[k] = v
                continue
            for c in idx_conns:
                _widen_index_connector_to_tile_kd(inner_sdfg, nsdfg_node, parent_state, c, widths, tile_iter_vars)
                widened.add(c)
            # Rewrite the base assignment to read lane (0, ..., 0) of each
            # widened K-tile connector — symbol-form ``__sym = c`` becomes
            # ``__sym = c[0, 0, ..., 0]``; subscript-form ``__sym = c[0]``
            # gets its single index extended to K zeros.
            from dace.symbolic import Subscript
            subscript_bases = {str(node.args[0]) for node in atoms_of(v_expr, Subscript)}
            base_expr = v_expr
            for c in idx_conns:
                arr_shape = inner_sdfg.arrays[c].shape
                K = len(arr_shape)
                zeros = ", ".join("0" for _ in range(K))
                if c in subscript_bases:
                    # Rewrite each Subscript with base ``c`` to ``c[0, ..., 0]``.
                    base_expr = base_expr.replace(lambda node: isinstance(node, Subscript) and str(node.args[0]) == c,
                                                  lambda node: Subscript(*([node.args[0]] + [sympy.Integer(0)] * K)))
                else:
                    # Bare-symbol form: substitute ``c`` with ``c(0, ..., 0)``.
                    base_expr = base_expr.subs(sympy.Symbol(c), dace.symbolic.SymExpr(f"{c}({zeros})"))
            base_v = DaceSympyPrinter(set(inner_sdfg.arrays.keys())).doprint(base_expr)
            new_assignments[k] = base_v
        edge.data.assignments = new_assignments
    return widened


def _index_symbols(inner_sdfg: dace.SDFG) -> Set[str]:
    """Collect symbols used for indirect accessing (inside a memlet subset).

    A symbol that appears in any memlet subset is a gather/scatter index
    (``src[sym]``). Such symbols must stay symbols so the gather fan-out can
    widen them into an index tile; every other interstate symbol may be
    demoted to a scalar.

    :param inner_sdfg: The nested SDFG whose memlets to scan.
    :returns: Symbol names referenced by any memlet subset.
    """
    index_syms: Set[str] = set()
    for state in inner_sdfg.all_states():
        for edge in state.edges():
            if edge.data.subset is None:
                continue
            available = state.symbols_defined_at(edge.dst)
            index_syms |= {
                str(s)
                for s in edge.data.free_symbols if str(s) in inner_sdfg.symbols or str(s) in available
            }
    return index_syms


def demote_non_index_symbols(inner_sdfg: dace.SDFG) -> Set[str]:
    """Demote every interstate-edge symbol to a Register scalar unless it is
    used for indirect accessing.

    ``ScalarToSymbolPromotion`` demotes integer-valued tile computations to
    interstate-edge symbol assignments (``sym = A_slice[0] + B_slice[0]``)
    consumed by a trivial store tasklet. On the tile path such symbols are
    loop-dependent (they cannot be broadcast), and fanning them per lane
    explodes combinatorially in K dimensions. Reversing the promotion —
    turning ``sym = expr`` back into a scalar dataflow tasklet via
    :func:`dace.sdfg.utils.demote_symbol_to_scalar` — lets the standard
    TileLoad / TileBinop / TileStore promotion lower the computation
    tile-natively, with no ``_laneid_`` symbols minted.

    Symbols used as an array index (``src[sym]``) are NOT demoted: they stay
    symbols for the gather fan-out (:func:`fan_out_tile_gather_index_symbols`)
    to widen into an index tile. Running this before the fan-out also shrinks
    how much the fan-out has to expand.

    :param inner_sdfg: The tile-body SDFG whose interstate symbols to demote.
    :returns: The set of demoted symbol names.
    """
    index_syms = _index_symbols(inner_sdfg)
    assigned: Set[str] = set()
    for edge in inner_sdfg.all_interstate_edges():
        assigned |= set(edge.data.assignments.keys())
    demotable = sorted(assigned - index_syms)
    for sym in demotable:
        stype = inner_sdfg.symbols.get(sym, dace.int64)
        sdutil.demote_symbol_to_scalar(inner_sdfg, sym, stype)
    return set(demotable)


def try_demoting_vectorizable_symbols(inner_sdfg: dace.SDFG) -> Set[str]:
    """
    Demote interstate-edge symbols that do not depend on array data into scalar data nodes.

    Symbols used on memlet subsets are never demoted. Demoting frees a
    symbol from the lane-fan-out path.

    :param inner_sdfg: The nested SDFG whose symbols to consider.
    :returns: Set of symbol names that were demoted.
    """
    assigned_symbols = dict()
    for edge in inner_sdfg.all_interstate_edges():
        for k, v in edge.data.assignments.items():
            if k not in assigned_symbols:
                assigned_symbols[k] = set()
            assigned_symbols[k].add(v)

    demotable_symbols = set()
    for sym, sym_assignments in assigned_symbols.items():
        # Check that the access is to arrays and map param is involved
        all_function_args = set()
        for sym_assignment in sym_assignments:
            sym_assign_expr = dace.symbolic.SymExpr(sym_assignment)
            # Collect the index symbols of every array access. Array accesses are
            # ``Subscript`` nodes: ``args[0]`` is the array head, ``args[1:]`` the
            # indices, so gather the free symbols of the indices.
            for sub in sym_assign_expr.atoms(dace.symbolic.Subscript):
                if str(sub.args[0]) in inner_sdfg.arrays:
                    for arg in sub.args[1:]:
                        all_function_args = all_function_args.union({str(s) for s in arg.free_symbols})

        # If all function args are s
        # if the depend set has no arrays or scalars we can do it
        data_in_dependence_set = {d for d in all_function_args if d in inner_sdfg.arrays}
        if len(data_in_dependence_set) == 0:
            demotable_symbols.add(sym)

    # Symbols used on memlets can't be demoted
    access_syms = set()
    for state in inner_sdfg.all_states():
        for edge in state.edges():
            if edge.data.subset is not None:
                dst = edge.dst
                available_syms = state.symbols_defined_at(dst)
                syms_used = {
                    str(s)
                    for s in edge.data.free_symbols if str(s) in inner_sdfg.symbols or str(s) in available_syms
                }
                access_syms = access_syms.union(syms_used)

    demotable_symbols = demotable_symbols - access_syms

    for demotable_symbol in demotable_symbols:
        stype = inner_sdfg.symbols[demotable_symbol]
        sdutil.demote_symbol_to_scalar(inner_sdfg, demotable_symbol, stype)

    return demotable_symbols


def resolve_missing_laneid_symbols(inner_sdfg, nsdfg, state, vector_map_param):
    """
    Reconstruct missing ``loop_var_laneid_<id>`` symbols in a vectorized-map nested SDFG.

    Lane-suffixed symbol variants may appear without being present in
    ``nsdfg.symbol_mapping``. Parent-map symbols are left untouched;
    others are aliased (or offset, for the vector parameter) via an
    assignment state inserted before the start block.

    :param inner_sdfg: The inner SDFG in which missing free symbols appear.
    :param nsdfg: The NestedSDFG node holding the symbol mapping.
    :param state: The state containing ``nsdfg``; used to look up parent map symbols.
    :param vector_map_param: The vector-lane map iterator name; symbols
        derived from it are rewritten as ``vector_map_param + laneid``.
    :raises AssertionError: If unexpected missing symbols remain after processing.
    :raises NotImplementedError: If a missing symbol lacks a ``_laneid_<i>`` suffix.
    """
    # Find missing symbols
    missing_symbols = set(inner_sdfg.free_symbols - set(nsdfg.symbol_mapping.keys()))

    # Determine which of the missing symbols correspond to parent map symbols
    map_symbols = assert_symbols_in_parent_map_symbols(missing_symbols, state, nsdfg)

    # Any symbol not in map_symbols must be auto-constructed
    unresolved = missing_symbols - map_symbols
    if len(unresolved) != 0:
        assignments = {}

        for missing_sym in unresolved:
            parsed = LaneIdScheme.parse(missing_sym)
            if parsed is None:
                raise NotImplementedError(f"Unexpected free symbol {missing_sym!r} without `_laneid_<i>` suffix; "
                                          f"cannot auto-construct")
            base, laneid = parsed

            if base == vector_map_param:
                # vector iterator -> add lane offset
                assignments[missing_sym] = f"{base} + {laneid}"
            else:
                # other iterators -> simply alias
                assignments[missing_sym] = base

        # Insert assignment state before the start block
        inner_sdfg.add_state_before(
            inner_sdfg.start_block,
            "pre_missing_assignment",
            is_start_state=True,
            assignments=assignments,
        )

    # Ensure no missing symbols remain
    remaining = set(inner_sdfg.free_symbols - set(nsdfg.symbol_mapping.keys()))
    assert len(remaining) == 0, \
        f"Remaining missing symbols after fix: {remaining}"
