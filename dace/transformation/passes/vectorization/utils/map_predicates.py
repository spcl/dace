# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Boolean predicates and defensive assertions on maps / SDFGs.

Policy (locked): ``assert_X`` siblings kept alongside their ``X`` counterparts; every
loud-failure helper stays available. Removing them shifts silent corruption into the pipeline.
"""
import sympy

import dace
from dace import SDFGState
from dace.sdfg.state import BreakBlock, ConditionalBlock
from dace.transformation.passes.vectorization.utils.symbolic_polymorphism import free_symbols


def has_maps(sdfg: dace.SDFG) -> bool:
    """True if the SDFG or any nested SDFG contains a MapEntry.

    :param sdfg: The SDFG to inspect.
    :returns: ``True`` if any MapEntry exists in the hierarchy.
    """
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            return True
    return False


def is_innermost_map(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if a map is innermost -- no nested maps, including inside nested SDFGs.

    :param state: The state containing the map entry.
    :param map_entry: The map entry node to test.
    :returns: ``True`` if the map has no inner maps.
    """
    nodes_between = state.all_nodes_between(map_entry, state.exit_node(map_entry))
    if any(isinstance(node, dace.nodes.MapEntry) for node in nodes_between):
        return False
    return not any(isinstance(node, dace.nodes.NestedSDFG) and has_maps(node.sdfg) for node in nodes_between)


def _sdfg_has_self_recurrent_assign(root_sdfg: dace.SDFG) -> bool:
    """True if some interstate assignment in ``root_sdfg`` (recursively) is self-referential
    -- ``k = f(k)`` (e.g. ``k = k + inc``, ``k = (k + i) + 1``).

    Self-recurrent symbol = loop-carried recurrence depending on iteration history → enclosing
    map cannot be tiled/vectorized. Refused REGARDLESS of use (index/arithmetic/...): never
    vectorize over an un-eliminated recurrence. Linearizable recurrences (plain affine IV) are
    rewritten away upstream (``InductionVariableSubstitution`` / ``NormalizeStridedMaps``), so
    whatever survives here is genuine.

    Reductions NOT caught: an accumulator is a WCR dataflow edge (``x ─[wcr:+]→ acc``), never an
    interstate symbol assignment.
    """
    for sd in root_sdfg.all_sdfgs_recursive():
        for e in sd.all_interstate_edges():
            for lhs, rhs in (e.data.assignments or {}).items():
                try:
                    rhs_syms = {str(s) for s in dace.symbolic.pystr_to_symbolic(rhs).free_symbols}
                except Exception:  # noqa: BLE001 -- unparseable rhs is not a recurrence we model
                    rhs_syms = set()
                if lhs in rhs_syms:
                    return True
    return False


def _loop_bound_uses_symbols(region, param_syms: set) -> bool:
    """True if ``region``'s init / condition / update references any name in ``param_syms``."""
    for code in (region.loop_condition, region.init_statement, region.update_statement):
        if code is None:
            continue
        try:
            used = {str(s) for s in dace.symbolic.symbols_in_code(code.as_string)}
        except Exception:  # noqa: BLE001 -- unparseable loop head is not a dependence we model
            continue
        if used & param_syms:
            return True
    return False


def _sdfg_loops_depend_on_symbols(sdfg: dace.SDFG, param_syms: set) -> bool:
    """True if some ``LoopRegion`` in ``sdfg`` (descending nested SDFGs, remapping ``param_syms``
    through each ``symbol_mapping``) has a bound referencing a symbol in ``param_syms``."""
    from dace.sdfg.state import LoopRegion
    for region in sdfg.all_control_flow_regions(recursive=False):
        if isinstance(region, LoopRegion) and _loop_bound_uses_symbols(region, param_syms):
            return True
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, dace.nodes.NestedSDFG):
                continue
            inner_syms = {
                inner_key
                for inner_key, outer_val in node.symbol_mapping.items()
                if param_syms & {str(s)
                                 for s in dace.symbolic.pystr_to_symbolic(str(outer_val)).free_symbols}
            }
            if inner_syms and _sdfg_loops_depend_on_symbols(node.sdfg, inner_syms):
                return True
    return False


def map_body_has_param_dependent_loop(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if the map body has a nested loop whose bound depends on a tiled map param.

    A triangular ``for i in range(1, j+1)`` inside a ``j``-map (TSVC s232) has a per-lane trip
    count once ``j`` is tiled -- the tiled body runs ONE tile-start bound for all W lanes and
    applies the inner (recurrence) body uniformly, under-computing the higher rows. No single
    strided loop honours W divergent bounds, so the map is refused (kept scalar, bit-exact) rather
    than tiled incorrectly.
    """
    params = {str(p) for p in map_entry.map.params}
    for node in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if not isinstance(node, dace.nodes.NestedSDFG):
            continue
        inner_syms = {
            inner_key
            for inner_key, outer_val in node.symbol_mapping.items()
            if params & {str(s)
                         for s in dace.symbolic.pystr_to_symbolic(str(outer_val)).free_symbols}
        }
        if inner_syms and _sdfg_loops_depend_on_symbols(node.sdfg, inner_syms):
            return True
    return False


def is_tile_eligible(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if an (assumed innermost) ``map_entry`` can be safely tiled/vectorized.

    Refuses a body with a self-referential loop-carried recurrence (``k = f(k)``, e.g. TSVC
    s141's ``k = (k + i) + 1`` feeding ``flat_2d_array[k]``): per-iteration recurrence → map
    stays scalar, refused REGARDLESS of use. Also refuses a body whose nested loop bound depends
    on a tiled map param (triangular ``for i in range(1, j+1)``, TSVC s232): W lanes need divergent
    trip counts a single tiled loop cannot honour. Graceful-refuse gate: ineligible map left as
    correct scalar rather than tiled incorrectly.
    """
    for node in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if isinstance(node, dace.nodes.NestedSDFG) and _sdfg_has_self_recurrent_assign(node.sdfg):
            return False
    if map_body_has_param_dependent_loop(state, map_entry):
        return False
    return True


def map_body_has_library_node(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if the map's body contains an OPAQUE library node (recursively, incl. nested SDFGs).

    Opaque = an external un-expanded primitive the tile emitters cannot see through (a sort, a
    ``ScatterConflictCheck``, a ``Reduce``, a BLAS call): a map wrapping one is left scalar
    until the node is expanded (after vectorization). This keeps the scatter guard's
    ``ScatterConflictCheck`` opaque to tiling.

    The vectorizer's OWN tile-op library nodes (``TileLoad`` / ``TileStore`` / ``TileBinop`` /
    ...) are EXCLUDED: they are inserted into the body DURING tiling, so treating them as
    opaque would make a half-tiled map refuse its own remaining tile passes.
    """
    from dace.libraries.tileops.nodes import (TileBinop, TileITE, TileLoad, TileMaskGen, TileReduce, TileStore,
                                              TileUnop)
    tile_ops = (TileBinop, TileITE, TileLoad, TileMaskGen, TileReduce, TileStore, TileUnop)

    def _opaque(n) -> bool:
        return isinstance(n, dace.nodes.LibraryNode) and not isinstance(n, tile_ops)

    for node in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if _opaque(node):
            return True
        if isinstance(node, dace.nodes.NestedSDFG) and any(_opaque(n) for n, _ in node.sdfg.all_nodes_recursive()):
            return True
    return False


def is_vectorizable_map(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """Innermost AND tile-eligible AND no library node inside: the shared tile-candidate gate.

    All tile passes select through this predicate so an un-vectorizable map (non-innermost,
    recurrence-indexed, or wrapping an opaque library node) is refused CONSISTENTLY -- never
    tiled by one pass while another skips it.
    """
    if map_body_has_library_node(state, map_entry):
        return False
    return is_innermost_map(state, map_entry) and is_tile_eligible(state, map_entry)


def is_gpu_resident_map(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if a map executes on the GPU device.

    GPU-resident = itself ``GPU_Device``-scheduled (a kernel), OR a non-kernel (e.g.
    ``Sequential``) map nested inside a ``GPU_Device`` map. Nesting followed through enclosing
    map scopes in the state AND across NestedSDFG boundaries (``sdfg.parent_nsdfg_node``), so a
    sequential body map inside a nested SDFG still finds its parent GPU kernel.

    GPU tile vectorizer's applicability rule: half2 ``__device__`` intrinsics only compile
    inside a GPU kernel, so only GPU-resident innermost maps may be tiled; host-side maps skipped.

    :param state: The state containing the map entry.
    :param map_entry: The map entry node to test.
    :returns: ``True`` if the map runs inside a GPU kernel.
    """
    gpu_device = dace.dtypes.ScheduleType.GPU_Device
    if map_entry.map.schedule == gpu_device:
        return True
    # Walk scope tree up, crossing NestedSDFG boundaries; accept if any ancestor map is GPU_Device.
    node = map_entry
    cur_state = state
    while cur_state is not None:
        scope = cur_state.scope_dict()
        parent = scope.get(node)
        while parent is not None:
            if isinstance(parent, dace.nodes.MapEntry) and parent.map.schedule == gpu_device:
                return True
            parent = scope.get(parent)
        # Top scope of this state → ascend into enclosing NestedSDFG node (parent state) and
        # keep walking. ``parent_nsdfg_node`` is None at the top-level SDFG → terminates.
        owning_sdfg = cur_state.sdfg
        nsdfg_node = owning_sdfg.parent_nsdfg_node
        if nsdfg_node is None:
            return False
        node = nsdfg_node
        cur_state = owning_sdfg.parent
    return False


def map_consists_of_single_nsdfg_or_no_nsdfg(graph: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if a map contains a single NestedSDFG or none at all.

    :param graph: The state containing the map.
    :param map_entry: The map entry to check.
    :returns: ``True`` if the map contains a single NestedSDFG or no NestedSDFG.
    """
    all_nodes = {
        k
        for k in graph.all_nodes_between(map_entry, graph.exit_node(map_entry))
        if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
    }
    return (len(all_nodes) == 1 and isinstance(next(
        iter(all_nodes)), dace.nodes.NestedSDFG)) or not any(isinstance(_n, dace.nodes.NestedSDFG) for _n in all_nodes)


def get_single_nsdfg_inside_map(graph: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> dace.nodes.NestedSDFG:
    """Return the sole NestedSDFG inside a map, or ``None`` if not exactly one.

    :param graph: The state containing the map.
    :param map_entry: The map entry to inspect.
    :returns: The single NestedSDFG node, or ``None``.
    """
    all_nodes = {
        k
        for k in graph.all_nodes_between(map_entry, graph.exit_node(map_entry))
        if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
    }
    if (len(all_nodes) == 1 and isinstance(next(iter(all_nodes)), dace.nodes.NestedSDFG)):
        return next(iter(all_nodes))
    return None


def has_only_states(sdfg: dace.SDFG) -> bool:
    """True if every top-level node of an SDFG is a plain SDFGState (no control-flow regions).

    :param sdfg: The SDFG to inspect.
    :returns: ``True`` if no control-flow regions are present.
    """
    return all({isinstance(n, dace.SDFGState) for n in sdfg.nodes()})


def has_only_states_or_single_block_with_break_only(sdfg: dace.SDFG) -> bool:
    """True if an SDFG has only states, or only conditional blocks whose sole branch is a break.

    :param sdfg: The SDFG to inspect.
    :returns: ``True`` if the SDFG matches either shape.
    """
    ifs = {n for n in sdfg.nodes() if isinstance(n, ConditionalBlock)}
    all_ifs_are_only_break = all({
        len(ifb.branches) == 1 and len(ifb.branches[0][1].nodes()) == 1
        and isinstance(ifb.branches[0][1].nodes()[0], BreakBlock)
        for ifb in ifs
    })
    non_ifs_non_states = {
        n
        for n in sdfg.nodes() if not isinstance(n, ConditionalBlock) and not isinstance(n, SDFGState)
    }
    return (all({isinstance(n, dace.SDFGState)
                 for n in sdfg.nodes()}) or (all_ifs_are_only_break and len(non_ifs_non_states) == 0))


def _no_edge_attr_state(state, attr: str, recursive: bool) -> bool:
    """True iff no edge in ``state`` has the attribute set. ``recursive=True`` descends into NSDFGs."""
    for edge in state.edges():
        value = edge.data.wcr if attr == "wcr" else edge.data.other_subset
        if value is not None:
            return False
    if recursive:
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                if not _no_edge_attr_sdfg(node.sdfg, attr, True):
                    return False
    return True


def _no_edge_attr_sdfg(sdfg: dace.SDFG, attr: str, recursive: bool) -> bool:
    """True iff no edge in any state of ``sdfg`` has the attribute set."""
    for state in sdfg.all_states():
        if not _no_edge_attr_state(state, attr, recursive):
            return False
    return True


def no_other_subset(state, recursive: bool = True) -> bool:
    """True iff no edge in ``state`` has ``other_subset`` set; recurses into NSDFGs by default."""
    return _no_edge_attr_state(state, "other_subset", recursive)


def no_wcr(state, recursive: bool = True) -> bool:
    """True iff no edge in ``state`` has WCR set; recurses into NSDFGs by default."""
    return _no_edge_attr_state(state, "wcr", recursive)


def last_dim_of_map_is_contiguous_accesses(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if the last dimension of a map performs contiguous accesses.

    :param state: The state containing the map.
    :param map_entry: The map entry to check.
    :returns: ``True`` if every memlet's unit-stride dim involves the last map parameter.
    """
    nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
    edges = state.all_edges(*nodes)
    # TODO: requires the map param to appear literally in the memlet; misses indirected
    # forms (``_s2 = map_param + 1; A[_s2]``). Needs richer analysis.
    for edge in edges:
        memlet: dace.memlet.Memlet = edge.data
        if memlet.subset is None:
            continue
        stride_one_idx = [i for i, s in enumerate(state.sdfg.arrays[edge.data.data].strides) if s == 1][0]
        b, e, s = memlet.subset[stride_one_idx]
        b_free_syms = free_symbols(b)
        e_free_syms = free_symbols(e)
        all_syms = {str(s) for s in b_free_syms.union(e_free_syms)}
        last_param = str(list(map_entry.map.params)[-1])
        if last_param not in all_syms and all_syms != set():
            return False
    return True


def count_param_in_expr(expr, param_str: str):
    """Count occurrences of a parameter in a SymPy expression, including function-call args.

    Matches by symbol name (not SymPy ``==``): DaCe symbols with the same name but different
    metadata can compare unequal.

    :param expr: The SymPy expression to scan.
    :param param_str: The parameter name to count.
    :returns: Number of occurrences.
    """
    if not isinstance(expr, sympy.Basic):
        return 0

    count = 0
    # standalone symbol occurrences (match by name)
    for atom in expr.atoms(sympy.Symbol):
        if str(atom) == param_str:
            count += 1

    # nested function-call argument occurrences
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.FunctionClass):
            continue  # function name, not an arg
        if isinstance(node, sympy.Function):
            for arg in node.args:
                count += count_param_in_expr(arg, param_str)

    return count


def map_param_appears_in_multiple_dimensions(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if the last map parameter appears across multiple subset dimensions.

    :param state: The containing state.
    :param map_entry: The map entry node.
    :returns: ``True`` if the last parameter appears in more than one dimension.
    """

    last_param = str(map_entry.map.params[-1])

    nodes_between = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
    edges = state.all_edges(*nodes_between)

    for edge in edges:
        memlet: dace.memlet.Memlet = edge.data

        # flag if last param appears >1 across this memlet's subset dims
        if memlet.subset is not None:
            subset_appearances = 0
            for (b, e, s) in memlet.subset:
                if free_symbols(b):
                    subset_appearances += count_param_in_expr(b, last_param)

            if subset_appearances >= 2:
                return True

    return False


def is_linear_in_param(expr, param_str: str) -> bool:
    """True if ``expr`` is linear in ``param_str`` (form ``c*p + d``, ``c``/``d`` constant in ``p``).

    A bare int/float literal counts as linear (coefficient 0).

    :param expr: The expression to classify.
    :param param_str: The parameter symbol name.
    :returns: ``True`` if ``expr`` is linear in the parameter.
    """
    if not isinstance(expr, sympy.Basic):
        return True  # plain int/float literal
    # Use the parameter symbol AS IT APPEARS in ``expr`` (carrying its real assumptions), not a
    # freshly fabricated bare ``sympy.Symbol`` -- a same-name symbol with mismatched assumptions is
    # a DISTINCT sympy object, so ``in expr.free_symbols`` / ``Poly`` would miss it and the
    # expression would look spuriously constant in the parameter (the ``i - i`` canonicalization
    # class: mismatched-assumption same-name symbols never cancel).
    param_sym = next((s for s in expr.free_symbols if s.name == param_str), None)
    if param_sym is None:
        return True  # expr is constant in the parameter -> linear
    try:
        poly = sympy.Poly(expr, param_sym)
    except (sympy.PolynomialError, sympy.GeneratorsNeeded):
        return False
    if poly.degree() > 1:
        return False
    # coefficients must not themselves contain ``param_sym``
    for c in poly.all_coeffs():
        if param_sym in free_symbols(c):
            return False
    return True


def map_param_dim_usage_is_linear_combo(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if multi-dimension uses of the last map parameter are all linear in it (strided-lowerable).

    For every memlet where the last param appears in >1 dim, each such dim must be a point access
    whose begin expr is linear in the param. Memlets where the param is absent or used in one dim
    do not block the classification.

    :param state: The containing state.
    :param map_entry: The map entry to inspect.
    :returns: ``True`` if all multi-dim uses are linear (strided-lowerable).
    """
    last_param = str(map_entry.map.params[-1])
    nodes_between = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
    edges = state.all_edges(*nodes_between)
    for edge in edges:
        memlet: dace.memlet.Memlet = edge.data
        if memlet.subset is None:
            continue
        dims_with_param = []
        for d, (b, e, _) in enumerate(memlet.subset):
            if free_symbols(b) and count_param_in_expr(b, last_param) > 0:
                dims_with_param.append((d, b, e))
        if len(dims_with_param) < 2:
            continue
        for _, b, e in dims_with_param:
            if b != e:
                return False
            if not is_linear_in_param(b, last_param):
                return False
    return True


def map_has_branching_memlets(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    """True if any map-entry out-connector feeds more than one edge.

    :param state: The state containing the map.
    :param map_entry: The map entry to inspect.
    :returns: ``True`` if a single out-connector branches to multiple edges.
    """
    for out_conn in map_entry.out_connectors:
        out_egdges_of_out_conn = set(state.out_edges_by_connector(map_entry, out_conn))
        if len(out_egdges_of_out_conn) > 1:
            return True
    return False


def sdfg_has_nested_sdfgs(sdfg: dace.SDFG):
    """True if an SDFG contains any NestedSDFG node.

    :param sdfg: The SDFG to inspect.
    :returns: ``True`` if a NestedSDFG node is present.
    """
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                return True
    return False


def has_nsdfg_depth_more_than_one(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    """True if a map body contains a NestedSDFG that itself contains a NestedSDFG.

    :param state: The state containing the map.
    :param map_entry: The map entry to inspect.
    :returns: ``True`` if nested-SDFG depth exceeds one.
    """
    for node in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if isinstance(node, dace.nodes.NestedSDFG):
            if sdfg_has_nested_sdfgs(node.sdfg):
                return True
    return False
