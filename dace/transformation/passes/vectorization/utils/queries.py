# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Read-only query helpers used by the vectorization pipeline.

These helpers do not mutate the SDFG; they extract access subsets,
map-parameter relationships, or scalar conversions used by the
emission and prep passes.
"""
import typing
from typing import Dict, Optional, Set, Tuple

import sympy

import dace
from dace.transformation.passes.vectorization.utils.lane_access import LaneAccessKind, classify_lane_access


def to_ints(sym_epxr: dace.symbolic.SymExpr) -> typing.Union[int, None]:
    """Try to convert a symbolic expression to an integer.

    :param sym_epxr: The symbolic expression to convert.
    :returns: The integer value if conversion succeeds, otherwise ``None``.
    """
    try:
        return int(sym_epxr)
    except Exception:
        return None


def collect_non_unit_stride_accesses_in_map(sdfg: dace.SDFG, state: dace.SDFGState,
                                            map_entry: dace.nodes.MapEntry) -> Set[str]:
    """Find non-unit-stride array accesses inside a map.

    An access is non-unit-stride when the map parameter appears multiplied
    in the stride-1-dim begin expression (e.g. ``A[i*2]``), or when the
    parameter spans more than one dimension (diagonal / linear-combo).

    :param sdfg: The SDFG to analyze.
    :param state: State containing the map.
    :param map_entry: The map whose body accesses are inspected.
    :returns: Dictionary mapping array names to a boolean indicating
        vectorizability.
    :raises NotImplementedError: if a memlet carries an ``other_subset``.
    """
    # Pre condition first parent maps is over the contiguous dimension and right most param if multi-dimensional
    parent_map = map_entry
    assert isinstance(parent_map, dace.nodes.MapEntry)
    map_param = parent_map.map.params[-1]

    # Collect all subsets
    all_nodes = state.all_nodes_between(parent_map, state.exit_node(parent_map))
    all_accesses_to_arrays = {e.data.data: list() for e in state.all_edges(*all_nodes) if e.data.data is not None}
    for e in state.all_edges(*all_nodes):
        if e.data.data is not None:
            all_accesses_to_arrays[e.data.data].append(e.data.subset)

    for edge in state.all_edges(*all_nodes):
        if edge.data.other_subset is not None:
            raise NotImplementedError("other subset support not implemented")

    array_is_vectorizable = {k: True for k in all_accesses_to_arrays}

    # Since no nestedSDFG, no indirectness may occur just check stridedness

    for arr_name, accesses in all_accesses_to_arrays.items():
        for access_subset in accesses:
            # ``classify_lane_access`` is the single source of truth for
            # how the access moves per lane. Strided (``A[2*i]``),
            # transposed (lane var in a non-stride-1 dim, e.g. a branch
            # condition ``zqx[z1, i, j]``) and diagonal (``A[i, i]`` /
            # ``A[2*i, i]``) all fan out per lane and cannot be a
            # contiguous ``vector_copy`` — route them through the
            # packed / strided gather path. Contiguous and lane-constant
            # accesses stay vectorizable.
            if classify_lane_access(access_subset, sdfg.arrays[arr_name].strides, map_param).fans_out_per_lane:
                array_is_vectorizable[arr_name] = False

    return array_is_vectorizable


def collect_accesses_to_array_name(sdfg: dace.SDFG) -> Dict[Tuple[str, dace.subsets.Range], str]:
    """Collect all access subsets for each array in the SDFG.

    :param sdfg: The SDFG to analyze.
    :returns: Dictionary mapping array names to a set of accessed subsets.
    :raises NotImplementedError: if a memlet carries an ``other_subset``.
    """
    d = dict()
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.other_subset is not None:
                raise NotImplementedError("other subset support not implemented")
            if edge.data.data is not None:
                if edge.data.data not in d:
                    d[edge.data.data] = set()
                d[edge.data.data].add(edge.data.subset)
    return d


def collect_all_memlets_to_dataname(sdfg: dace.SDFG) -> Dict[str, Set[dace.subsets.Range]]:
    """Collect all unique memlet subsets for each data array in the SDFG.

    Groups memlet subsets by the data array they access. Does not check
    interstate edges or conditionals.

    :param sdfg: The SDFG to analyze.
    :returns: Dictionary mapping data array names to sets of accessed subsets.
    """
    dataname_to_memlets = dict()
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.data is not None:
                if edge.data.data not in dataname_to_memlets:
                    dataname_to_memlets[edge.data.data] = set()
                dataname_to_memlets[edge.data.data].add(edge.data.subset)

    return dataname_to_memlets


def parse_int_or_default(value, default=8):
    """Parse ``value`` as an int, returning ``default`` on failure.

    :param value: Value to convert.
    :param default: Fallback when conversion fails.
    :returns: The parsed int, or ``default``.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def collect_vectorizable_arrays(sdfg: dace.SDFG, parent_nsdfg_node: dace.nodes.NestedSDFG, parent_state: dace.SDFGState,
                                invariant_scalars: Set[str], vector_width: int) -> Dict[str, bool]:
    """Determine which arrays can be vectorized through a NestedSDFG.

    An array is not vectorizable if its access is strided in the map
    parameter (``A[i*2]``) or if a non-map free symbol is defined by an
    indirect (array-indexed) interstate assignment.

    :param sdfg: The SDFG to analyze.
    :param parent_nsdfg_node: NestedSDFG node.
    :param parent_state: State containing the NestedSDFG.
    :param invariant_scalars: Scalar names invariant across lanes (these do
        not prevent vectorization).
    :param vector_width: SIMD lane count; a 1-D array already sized exactly
        to this width is an already-packed buffer and is skipped (a
        concrete-size real array, e.g. ``A[64]``, must NOT be skipped or
        an indirect gather ``A[B[i]]`` is misclassified as contiguous).
    :returns: Dictionary mapping array names to a boolean indicating
        vectorizability.
    :raises NotImplementedError: if a memlet carries an ``other_subset``.
    :raises Exception: if a strided multiplicative access is encountered
        (case not yet analyzed).
    """
    # Lazy import to avoid an obvious cycle: ``utils.lane_expansion``
    # itself imports from ``utils.name_schemes`` but not from this
    # module — keep the import inside the function so callers don't
    # have to reason about load order between ``queries`` and
    # ``lane_expansion``.
    from dace.transformation.passes.vectorization.utils.lane_expansion import (
        _all_atoms,
        find_symbol_assignment,
    )

    # Pre condition first parent maps is over the contiguous dimension and right most param if multi-dimensional
    parent_map = parent_state.scope_dict()[parent_nsdfg_node]
    assert isinstance(parent_map, dace.nodes.MapEntry)
    map_param = parent_map.map.params[-1]
    parent_syms_defined = parent_state.symbols_defined_at(parent_nsdfg_node)

    all_accesses_to_arrays = collect_accesses_to_array_name(sdfg)

    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.other_subset is not None:
                raise NotImplementedError("other subset support not implemented")

    # Drop ``_iter_mask`` arrays entirely from the classification: they are
    # transient W-wide bool buffers filled per lane by GenerateIterationMask
    # (P3) and are NOT user data the vectorizer should touch — neither
    # vectorize (they are already W-wide) nor pack (they are not strided).
    # Their fill memlet is ``[0:W]`` which would fail the point-access
    # assert below; their packing path crashes on tasklet edges.
    # Same treatment for ``__strided_buf_*`` arrays + the bbox-shaped
    # connector arrays they alias inside the NSDFG body: both are emitted
    # by ``_setup_strided_inside_nsdfg`` and carry full-array memlets in
    # the prep / finish state (subset ``[0:W]`` / ``[0:bbox-1]``) that
    # similarly fail the point-access assert.  The body itself only ever
    # accesses these arrays point-wise via ``__strided_buf_*`` already.
    # Identify the connector arrays the strided handlers aliased.  Two
    # naming conventions:
    # - 1D / multi-dim:    ``__strided_buf_<conn>``  → base is ``<conn>``
    # - Multi-elem (TSVC s127): ``__strided_buf_<conn>_p<N>`` → base
    #   is ``<conn>`` (strip the ``_p<N>`` suffix too)
    import re as _re
    _phase_suffix = _re.compile(r"_p\d+$")
    strided_buf_aliases: set = set()
    for arr_name in list(all_accesses_to_arrays.keys()):
        if not arr_name.startswith("__strided_buf_"):
            continue
        rest = arr_name[len("__strided_buf_"):]
        # Add the literal rest (covers 1D / multi-dim case).
        strided_buf_aliases.add(rest)
        # Also add the rest with any trailing ``_p<N>`` phase suffix
        # stripped (covers multi-elem-per-iter case).
        stripped = _phase_suffix.sub("", rest)
        if stripped != rest:
            strided_buf_aliases.add(stripped)
    for arr_name in list(all_accesses_to_arrays.keys()):
        if arr_name == "_iter_mask" or arr_name.startswith("_iter_mask_"):
            del all_accesses_to_arrays[arr_name]
            continue
        if arr_name.startswith("__strided_buf_") or arr_name in strided_buf_aliases:
            del all_accesses_to_arrays[arr_name]
            continue
        # ``_cond_*`` bool transients are emitted by BranchNormalization /
        # SameWriteSetIfElseToMergeCFG to carry per-lane condition results.
        # After ``replace_arrays_with_new_shape`` reshapes them to ``(W,)``
        # and their memlets are rewritten to ``[0:W-1]``, the point-access
        # ``b == e`` assert below would trip — exclude them, same as the
        # ``_iter_mask`` and ``__strided_buf_`` cases above.
        if arr_name.startswith("_cond_"):
            del all_accesses_to_arrays[arr_name]
            continue
        # Any already-W-wide 1D array (transient OR non-transient
        # connector — non-transient covers e.g. NSDFG-boundary
        # accumulator descriptors like ``__tmp_70_18_r`` from
        # the SpMV reduction pattern) is past the "should this be
        # vectorized?" decision.  Its memlets carry the full W-wide
        # range subset after ``replace_arrays_with_new_shape``, which
        # would trip the point-access ``b == e`` assert below.  Skip
        # everything whose array descriptor is already ``(vector_width,)``.
        try:
            arr_obj = sdfg.arrays.get(arr_name)
        except Exception:
            arr_obj = None
        if arr_obj is not None:
            shape = tuple(arr_obj.shape)
            if len(shape) == 1:
                try:
                    # Only skip buffers already sized *exactly* to the
                    # vector width (the W-wide packed transients/connectors
                    # this guard targets). The old ``> 1`` test also
                    # deleted every concrete-size real array (e.g.
                    # ``A[64]``) before the indirection check below ran, so
                    # an indirect gather ``A[B[i]]`` was misclassified as a
                    # contiguous load. Symbolic ``N`` raised here (caught)
                    # and was correctly classified — concrete sizes were
                    # not; this restores parity.
                    if int(shape[0]) == vector_width:
                        del all_accesses_to_arrays[arr_name]
                        continue
                except (TypeError, ValueError):
                    pass

    array_is_vectorizable = {k: True for k in all_accesses_to_arrays}

    for arr_name, accesses in all_accesses_to_arrays.items():
        for access_subset in accesses:
            # Transposed (lane var in a non-stride-1 dim, e.g. a branch
            # condition ``zqx[z1, i, j]``) and diagonal (lane var in
            # >1 dims) accesses fan out per lane: the W lanes are
            # ``arr.strides[d]`` apart in memory — a strided gather, not
            # a contiguous load. ``classify_lane_access`` is the single
            # source of truth. Mark the array non-vectorizable so it
            # routes through the packed / strided gather path; a
            # contiguous ``vector_copy`` would read W adjacent elements
            # along the wrong (constant) dim. STRIDED (``A[2*i]``) is
            # left to the explicit strided-Mul check below (it carries
            # its own unanalysed-case raise); CONTIGUOUS / CONSTANT fall
            # through to the indirect-symbol provenance walk.
            if classify_lane_access(access_subset, sdfg.arrays[arr_name].strides,
                                    map_param).kind in (LaneAccessKind.TRANSPOSED, LaneAccessKind.DIAGONAL):
                array_is_vectorizable[arr_name] = False
                continue

            # Get the stride 1 dimension
            stride_one_dim = {i for i, stride in enumerate(sdfg.arrays[arr_name].strides) if stride == 1}.pop()
            b, e, s = access_subset[stride_one_dim]
            assert b == e
            assert s == 1

            # Evaluate the expression (b == e)
            access_expr = b  # use b since b==e
            if isinstance(access_expr, (dace.symbolic.SymExpr, sympy.Expr)):
                # Strided iff the map_param is a *direct* factor of some Mul
                # term — e.g. ``2*i``.  Expressions like ``i + LEN_1D // 2``
                # contain a Mul atom ``LEN_1D / 2`` whose free symbols don't
                # include the map_param; subset-propagation can also leave
                # ``i - Min(i, i + LEN_1D // 2)``, whose Mul atom
                # ``-Min(i, i + LEN_1D // 2)`` carries ``i`` transitively via
                # ``Min`` but doesn't make the access strided.  Inspect the
                # Mul's direct args so neither case is misclassified.
                def _is_strided_mul(term: sympy.Mul) -> bool:
                    for f in term.args:
                        if isinstance(f, sympy.Symbol) and str(f) == map_param:
                            return True
                    return False

                if any(_is_strided_mul(term) for term in access_expr.atoms(sympy.Mul)):
                    # A stride-1-dim begin with the map param as a direct
                    # Mul factor (e.g. ``A[2*i]`` through the NSDFG
                    # boundary) is a constant-stride gather. Marking the
                    # array non-vectorizable routes it to the packed /
                    # strided handler (``_setup_strided_nsdfg_edges_inline``
                    # / the ``(W-1)*S+K`` strided-load lowering), which is
                    # the correct support for this shape and is verified
                    # numerically (``test_strided_through_nsdfg``). A
                    # genuinely-unsupported strided sub-shape still fails
                    # loudly — but at that handler, with concrete context,
                    # rather than via a blanket early abort here.
                    array_is_vectorizable[arr_name] = False

            if isinstance(b, (dace.symbolic.SymExpr, dace.symbolic.symbol, sympy.Expr)):
                if isinstance(b, (dace.symbolic.SymExpr, sympy.Expr)):
                    free_syms = {str(s) for s in b.free_symbols}
                else:
                    free_syms = {b}
                for free_sym in free_syms:
                    # Accessing map param is ok
                    if str(free_sym) == map_param:
                        continue
                    else:
                        # Other free symbols should not have indirect accesses
                        # Analysis tries find the first assignment in the CFG
                        assignment = find_symbol_assignment(sdfg, str(free_sym))
                        assert not (
                            assignment is None and str(free_sym) not in parent_syms_defined
                        ), f"Could not find an iedge assignment for {free_sym}, assignment {assignment}, parent symbols defined {parent_syms_defined}. {sdfg.label}, {sdfg.parent_nsdfg_node}: map param {map_param}"
                        # Loop invariant symbol passed from outside
                        if assignment is None:
                            continue

                        assignment_expr = dace.symbolic.SymExpr(assignment)
                        # The array names this assignment reads. ``arrays`` reports the
                        # ``Subscript`` heads, matching what the old ``atoms(Function)``
                        # ``.name`` form reported before #2378 made array accesses
                        # ``Subscript`` nodes (arithmetic functions like ``int_floor``
                        # carry no subscript and are handled by ``unhandled_replication``).
                        funcs = dace.symbolic.arrays(assignment_expr)
                        # Any array on the right-hand-side -> big problem
                        # Check for scalar / array accesses like this too
                        scalars = {str(s)
                                   for s in assignment_expr.free_symbols if str(s) in sdfg.arrays} - invariant_scalars
                        # If scalar is invariant it should be ok?
                        #      {s
                        #       for s in assignment_expr.free_symbols if str(s) in sdfg.arrays} - invariant_scalars)
                        # A map-param-dependent ``int_floor`` / ``int_ceil`` in a symbol's
                        # interstate assignment is a lane-replication index (``a[i // D]``)
                        # that the multiplex path (``detect_halve_index``) did NOT rewrite --
                        # e.g. a data-dependent divisor buried in a nested SDFG. Reaching here
                        # means the normal contiguous widening would mis-lower it, so refuse
                        # (the loop-invariant ``int_floor(const, D)`` case does not reference
                        # the map param, so it is not flagged).
                        unhandled_replication = (map_param in {str(s) for s in assignment_expr.free_symbols} and any(
                            type(f).__name__ in ("int_floor", "int_ceil")
                            for f in assignment_expr.atoms(sympy.Function)))
                        if len(funcs) != 0 or len(scalars) != 0 or unhandled_replication:
                            array_is_vectorizable[arr_name] = False

            # Go through non unit stride dimensions in case it those dimensions have unstructuredness
            for i, (b, e, s) in enumerate(access_subset):
                if i == stride_one_dim:
                    continue
                free_syms = set()
                if hasattr(b, "free_syms"):
                    free_syms = {str(s) for s in b.free_syms}
                if hasattr(b, "free_symbols"):
                    free_syms = {str(s) for s in b.free_symbols}

                if free_syms != set():
                    for free_sym in free_syms:
                        # Accessing map param is ok
                        if str(free_sym) == map_param:
                            continue
                        else:
                            # Other free symbols should not have indirect accesses
                            # Analysis tries find the first assignment in the CFG
                            assignment = find_symbol_assignment(sdfg, str(free_sym))

                            # If assignment is None, it is probably coming from parent map
                            parent_syms_defined = parent_state.symbols_defined_at(parent_nsdfg_node)
                            if assignment is None:
                                assert str(
                                    free_sym
                                ) in parent_syms_defined, f"Could not find an iedge assignment for {free_sym} it is also not defined in symbols defined in nsdfg entry {parent_syms_defined}"
                                continue

                            assignment_expr = dace.symbolic.SymExpr(assignment)
                            # Define functions to ignore (common arithmetic + piecewise + rounding)
                            ignored = {
                                sympy.sin, sympy.cos, sympy.tan, sympy.exp, sympy.log, sympy.sqrt, sympy.Abs,
                                sympy.floor, sympy.ceiling, sympy.Min, sympy.Max, sympy.asin, sympy.acos, sympy.atan,
                                sympy.sinh, sympy.cosh, sympy.tanh, sympy.asinh, sympy.acosh, sympy.atanh
                            }
                            all_atoms = _all_atoms(assignment_expr, ignored)
                            all_atoms_str = {str(s) for s in all_atoms}

                            # Map parameter appears in inddirect access, array is not vectorizable
                            if map_param in all_atoms_str:
                                array_is_vectorizable[arr_name] = False

    return array_is_vectorizable


def collect_element_write_subsets(state: dace.SDFGState) -> Optional[Dict[str, dace.subsets.Range]]:
    """Return ``{arr_name: subset}`` for every element-wise write in ``state``.

    A write is element-wise iff its memlet subset has
    ``num_elements_exact() == 1``. Multiple writes to the same array keep
    only the last subset seen.

    :param state: State to inspect.
    :returns: Mapping of array name to its element-wise write subset, or
        ``None`` if any in-edge to an AccessNode is not element-wise.
    """
    out: Dict[str, dace.subsets.Range] = {}
    for n in state.nodes():
        if not isinstance(n, dace.nodes.AccessNode):
            continue
        for e in state.in_edges(n):
            if e.data.data is None:
                continue
            try:
                if e.data.subset.num_elements_exact() != 1:
                    return None
            except Exception:
                return None
            out[n.data] = e.data.subset
    return out
