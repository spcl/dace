# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end pass: detect scatter loops, guard their index arrays, parallelize.

A *scatter loop* is a ``LoopRegion`` whose body writes to a non-transient array at
an index of the form ``arr[idx[f(i)]]`` -- the write slot is data-dependent
through an index array ``idx`` read at a (possibly strided) function of the loop
variable. ``LoopToMap`` refuses such loops by default because two iterations may
write the same slot; the user's contract is that ``idx`` is a permutation (no
duplicates → no write-write race), and ``LoopToMap``'s ``permissive`` mode lifts
the loop under that assumption.

This pass operationalises that contract end-to-end:

1. **Detect** every scatter loop in the SDFG (see
   :func:`_scatter_idx_arrays_for_loop`), recognising the three lowered forms a
   ``out[idx[f(i)]] = ...`` scatter takes: an interstate-bound index
   (``sym := idx[f(i)]`` used in a write subset), an inline-subscript index
   (``idx[f(i)]`` written literally inside a write-memlet subset), and a
   nested-SDFG data-dependent write whose index traces back to an integer array
   read at ``[f(i)]``. ``f(i)`` is any expression referencing the loop variable,
   so both unit-stride (``idx[i]``) and symbolic-stride (``idx[SSYM*i]``)
   scatters are covered. The union of source-array names is the set of ``idx``
   arrays.
2. **Guard** each detected ``idx`` array via
   :func:`~dace.transformation.passes.scatter_conflict_guard.insert_scatter_guard`,
   which inserts an ``IntegerSort`` + adjacent-equal-pair check + ``__builtin_trap()``
   at the earliest legal CFG state.
3. **Parallelize** by applying ``LoopToMap`` in ``permissive`` mode, which lifts
   the scatter loops (and any other previously refused permissive cases) into
   parallel Maps.

The ordering is intentional: guards are emitted *before* permissive lifts, so on
collision the abort fires before any consumer reads the corrupted output.
"""
import ast
from typing import Optional, Set

from dace import SDFG, data, properties
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.scatter_conflict_guard import insert_scatter_guard


@properties.make_properties
@xf.explicit_cf_compatible
class ScatterToGuardedMaps(ppl.Pass):
    """Detect scatter loops, insert per-array runtime guards, then permissively lift to maps.

    Two collision policies are supported via :attr:`emit_unparallelized_else_branch`:

    - ``False`` (default): the guard's check tasklet calls ``__builtin_trap()``
      whenever a duplicate is detected; the parallelised Map runs unconditionally
      afterwards. The contract is "permutation or abort" -- callers committed to
      that contract get the simpler CFG.
    - ``True``: the guard emits only the sort + duplicate-count steps (no trap)
      and the scatter loop is wrapped in a ``ConditionalBlock`` keyed on the
      duplicate-count symbol. The ``True`` branch keeps a deep copy of the
      original sequential ``LoopRegion``; the ``False`` branch holds the
      ``LoopToMap``-lifted parallel Map. The check tasklet routes at runtime,
      so collisions degrade to sequential execution rather than aborting.

    Idempotence: the underlying guard utility refuses to emit a second guard for the
    same ``idx`` array (the ``_scatter_guard_sorted_<name>`` transient acts as the
    presence marker). Re-running this pass on an SDFG it has already guarded is a
    no-op for the guard step; the ``LoopToMap`` step still re-applies and is itself
    idempotent on already-lifted Maps.
    """

    CATEGORY: str = 'Optimization Preparation'

    emit_unparallelized_else_branch = properties.Property(
        dtype=bool,
        default=False,
        desc="When True, emit a ``ConditionalBlock`` dispatching at runtime on "
        "the duplicate-count symbol: the True branch runs a sequential clone "
        "of the original scatter loop; the False branch runs the parallel "
        "Map lift. The duplicate-trap is suppressed; collisions degrade to "
        "sequential execution instead of ``__builtin_trap()``.",
    )

    assume_no_conflicts = properties.Property(
        dtype=bool,
        default=False,
        desc="When True, ASSUME every scatter ``idx`` array is a permutation "
        "(no duplicate targets): skip the sort + duplicate-count guard entirely "
        "and lift the scatter loop to an unconditional parallel Map. Unsound if "
        "the assumption is violated at runtime (write races); the caller owns "
        "that contract. Takes precedence over ``emit_unparallelized_else_branch``.",
    )

    def __init__(self, emit_unparallelized_else_branch: bool = False, assume_no_conflicts: bool = False):
        super().__init__()
        self.emit_unparallelized_else_branch = emit_unparallelized_else_branch
        self.assume_no_conflicts = assume_no_conflicts

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        """Run the full pipeline. Returns the number of distinct ``idx`` arrays guarded,
        or ``None`` if no scatter loop was found.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap

        scatter_loops, idx_arrays = detect_scatter_loops_and_idx_arrays(sdfg)

        if self.assume_no_conflicts:
            # Caller asserts every idx array is a permutation: skip the sort +
            # duplicate-count guard entirely and lift each scatter loop to an
            # unconditional parallel Map (no ConditionalBlock, no trap).
            for loop in scatter_loops:
                parent = loop.parent_graph
                if parent is None or loop not in parent.nodes():
                    continue
                instance = LoopToMap()
                instance.loop = loop
                try:
                    instance.apply(parent, _owning_sdfg(sdfg, loop))
                except Exception:
                    pass
            return len(idx_arrays) or None

        # Track each idx_array's duplicate-count symbol so the else-branch
        # dispatcher knows which symbol to gate on per scatter loop. None when
        # the trap mode is on.
        dup_count_syms: dict = {}
        for idx_name in sorted(idx_arrays):
            try:
                trap_sym = insert_scatter_guard(sdfg, idx_name, emit_trap=not self.emit_unparallelized_else_branch)
                if trap_sym is not None:
                    dup_count_syms[idx_name] = trap_sym
            except ValueError as exc:
                if 'already exists' not in str(exc):
                    raise

        for loop in scatter_loops:
            parent = loop.parent_graph
            if parent is None or loop not in parent.nodes():
                continue  # already removed by a sibling lift
            owner_sdfg = _owning_sdfg(sdfg, loop)

            if self.emit_unparallelized_else_branch and dup_count_syms:
                # Find the dup-count symbol for any idx array this loop
                # scatters into. Loops with multiple idx arrays would need ALL
                # of them to be conflict-free for the parallel branch to be
                # safe; we OR the counts together so any positive count routes
                # to the sequential branch.
                loop_idx = _scatter_idx_arrays_for_loop(loop, owner_sdfg)
                loop_idx_syms = [dup_count_syms[i] for i in loop_idx if i in dup_count_syms]
                if loop_idx_syms:
                    cond = ' + '.join(loop_idx_syms) + ' > 0'
                    _wrap_loop_in_dispatcher(parent, loop, cond, LoopToMap)
                    continue

            instance = LoopToMap()
            instance.loop = loop
            try:
                instance.apply(parent, owner_sdfg)
            except Exception:
                pass
        return len(idx_arrays) or None


def detect_scatter_idx_arrays(sdfg: SDFG) -> Set[str]:
    """Find every ``idx`` array name used as an indirect-write index in any LoopRegion.

    See :func:`detect_scatter_loops_and_idx_arrays` for the underlying scan; this
    helper drops the loops set and returns only the idx-array names.
    """
    _, idx_arrays = detect_scatter_loops_and_idx_arrays(sdfg)
    return idx_arrays


def detect_scatter_loops_and_idx_arrays(sdfg: SDFG):
    """Scan ``sdfg`` (and nested SDFGs) for scatter loops; return
    ``(scatter_loops, idx_arrays)``.

    A ``LoopRegion`` qualifies as a scatter loop iff any interstate edge in the
    region binds a symbol via ``sym := arr[loop_var]`` AND a write-memlet to a
    non-transient array inside the region's body references that symbol.

    :param sdfg: The SDFG to scan; nested SDFGs are walked too.
    :returns: ``(list[LoopRegion], set[str])`` -- deterministic-order list of
              the scatter ``LoopRegion`` instances + set of ``idx`` array names
              resolved against the owning SDFG's ``arrays`` table.
    """
    scatter_loops: list = []
    idx_arrays: Set[str] = set()
    for sd in sdfg.all_sdfgs_recursive():
        for region in sd.all_control_flow_regions():
            if not (isinstance(region, LoopRegion) and region.loop_variable):
                continue
            loop_arrays = _scatter_idx_arrays_for_loop(region, sd)
            if loop_arrays:
                scatter_loops.append(region)
                idx_arrays |= loop_arrays
    return scatter_loops, idx_arrays


def _scatter_idx_arrays_for_loop(region: LoopRegion, sdfg: SDFG) -> Set[str]:
    """Return the scatter index-array names driving an indirect WRITE in ``region``.

    Recognises the three lowered forms an ``out[idx[f(i)]] = ...`` scatter takes,
    where ``f(i)`` is any expression referencing the loop variable (a bare
    ``idx[i]`` or a strided ``idx[c*i + d]``):

    1. **Interstate-bound index** -- ``sym := idx[f(i)]`` on a loop interstate
       edge, with ``sym`` referenced in a write-memlet subset to a non-transient
       array (TSVC ``s4113``/``s491``/``vas`` and the symbolic-stride
       ``s4113_ssym``).
    2. **Inline-subscript index** -- the index array appears literally inside a
       write-memlet subset, ``out[idx[f(i)]]`` (the symbolic-stride ``vas_ssym``).
    3. **Nested-SDFG data-dependent write** -- a ``NestedSDFG`` writes a
       non-transient array with a data-dependent (fewer accesses than the subset
       spans) memlet whose write index traces back to an integer array read at
       ``[f(i)]`` (``ext_scatter_store``, lowered from a ``dace.map`` scatter).

    :param region: The candidate loop region.
    :param sdfg: The SDFG owning ``region``'s arrays.
    :returns: The set of ``idx`` array names (empty if ``region`` is not a scatter).
    """
    loop_var = region.loop_variable
    bindings = _collect_indirect_bindings(region, sdfg)
    loop_arrays: Set[str] = set()
    for state in region.all_states():
        for node in state.data_nodes():
            if state.in_degree(node) == 0:
                continue
            desc = sdfg.arrays.get(node.data)
            if desc is None or desc.transient:
                continue
            for e in state.in_edges(node):
                if e.data is None or e.data.subset is None:
                    continue
                # Form 1: interstate-bound index symbol referenced in the write subset.
                for sym in e.data.subset.free_symbols:
                    arr = bindings.get(str(sym))
                    if arr is not None:
                        loop_arrays.add(arr)
                # Form 2: index array inline-subscripted inside the write subset.
                loop_arrays |= _inline_indirect_idx_arrays(e.data.subset, loop_var, sdfg)
        # Form 3: nested-SDFG data-dependent write.
        loop_arrays |= _nested_dynamic_scatter_idx_arrays(state, sdfg, loop_var)
    return loop_arrays


def _collect_indirect_bindings(region: LoopRegion, sdfg: SDFG) -> dict[str, str]:
    """Map each symbol bound by ``region``'s interstate edges to its source data
    array, when the binding is of the form ``sym := arr[f(loop_var)]``.

    Conservative: only a bare index-array subscript ``arr[<expr>]`` whose index
    ``<expr>`` references the loop variable is recognised (see
    :func:`_resolve_indirect_source`) -- the bare ``arr[loop_var]`` and the
    strided ``arr[c*loop_var + d]``. Non-subscript compound expressions like
    ``arr[loop_var] + 1`` are skipped; they do not arise from the DaCe Python
    frontend's scatter lowering, and extending the recognition surface risks
    misclassifying non-scatter interstate computations.
    """
    bindings: dict[str, str] = {}
    loop_var = region.loop_variable
    for e in region.edges():
        for lhs, rhs in (e.data.assignments or {}).items():
            arr = _resolve_indirect_source(rhs, loop_var, sdfg)
            if arr is not None:
                bindings[lhs] = arr
    return bindings


def _owning_sdfg(root: SDFG, loop: LoopRegion) -> SDFG:
    """Walk the SDFG tree to find the SDFG that owns ``loop``. Used so
    ``LoopToMap.apply`` reads / writes the correct arrays table on nested
    SDFGs.
    """
    for sd in root.all_sdfgs_recursive():
        if loop in list(sd.all_control_flow_regions()):
            return sd
    return root  # defensive fallback


def _resolve_indirect_source(rhs_str: str, loop_var: str, sdfg: SDFG) -> Optional[str]:
    """Return ``arr`` if ``rhs_str`` is ``arr[f(loop_var)]`` (``arr`` a data
    descriptor in ``sdfg`` and the index a function of ``loop_var``); ``None``
    otherwise.

    The index ``f(loop_var)`` may be the bare loop variable (``arr[loop_var]``,
    unit-stride scatters) or any expression referencing it (``arr[c*loop_var +
    d]``, symbolic-stride scatters such as ``ip[SSYM*i]``). Requiring the loop
    variable to appear keeps loop-invariant indices (not per-iteration scatters)
    out.
    """
    try:
        tree = ast.parse(str(rhs_str), mode='eval').body
    except (SyntaxError, ValueError, TypeError):
        return None
    if not isinstance(tree, ast.Subscript):
        return None
    if not isinstance(tree.value, ast.Name):
        return None
    arr = tree.value.id
    if arr not in sdfg.arrays:
        return None
    idx = tree.slice
    # Python <3.9 wraps the slice in ast.Index; unwrap.
    if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
        idx = idx.value
    if loop_var not in {n.id for n in ast.walk(idx) if isinstance(n, ast.Name)}:
        return None
    return arr


def _inline_indirect_idx_arrays(subset, loop_var: str, sdfg: SDFG) -> Set[str]:
    """Data-array names inline-subscripted inside a memlet ``subset`` with an
    index referencing ``loop_var`` -- the ``out[idx[f(i)]]`` form where the index
    array ``idx`` is embedded directly in the write subset rather than bound on an
    interstate edge.

    :param subset: A memlet subset (its ``str`` is parsed for ``idx[...]`` nodes).
    :param loop_var: The loop variable that a genuine scatter index must reference.
    :param sdfg: The SDFG whose ``arrays`` table qualifies the subscript bases.
    :returns: The set of index-array names found (empty if none).
    """
    arrays: Set[str] = set()
    try:
        tree = ast.parse(str(subset), mode='eval').body
    except (SyntaxError, ValueError, TypeError):
        return arrays
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
            continue
        arr = node.value.id
        if arr not in sdfg.arrays:
            continue
        idx = node.slice
        if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
            idx = idx.value
        if loop_var in {n.id for n in ast.walk(idx) if isinstance(n, ast.Name)}:
            arrays.add(arr)
    return arrays


def _nested_dynamic_scatter_idx_arrays(state, sdfg: SDFG, loop_var: str) -> Set[str]:
    """Integer index-array names driving a nested-SDFG data-dependent write in ``state``.

    Matches the shape a ``dace.map`` scatter (``dst[idx[i]] = ...``) lowers to: a
    ``NestedSDFG`` writes a non-transient array with a memlet whose accessed
    volume is smaller than the subset it spans (a single element scattered into a
    whole-array range). The write index lives inside the nested SDFG; this traces
    it back through the nested SDFG's input connectors to the outer integer array
    read at ``[f(loop_var)]`` and returns that array (the one whose distinctness
    the guard must check).

    :param state: The loop-body state to scan.
    :param sdfg: The SDFG owning ``state``'s arrays.
    :param loop_var: The loop variable the index read must reference.
    :returns: The set of integer index-array names found (empty if none).
    """
    from dace.libraries.sort.nodes._helpers import is_integer_dtype

    arrays: Set[str] = set()
    for node in state.data_nodes():
        desc = sdfg.arrays.get(node.data)
        if desc is None or desc.transient:
            continue
        for e in state.in_edges(node):
            m = e.data
            if m is None or m.subset is None or not isinstance(e.src, nodes.NestedSDFG):
                continue
            # Data-dependent write: fewer accesses (volume) than the subset spans.
            if m.volume == m.subset.num_elements():
                continue
            idx_conns = _write_index_input_connectors(e.src, e.src_conn)
            for ie in state.in_edges(e.src):
                if ie.dst_conn not in idx_conns or not isinstance(ie.src, nodes.AccessNode):
                    continue
                src_desc = sdfg.arrays.get(ie.src.data)
                if (src_desc is None or src_desc.transient or not isinstance(src_desc, data.Array)
                        or not is_integer_dtype(src_desc.dtype)):
                    continue
                if ie.data is None or ie.data.subset is None:
                    continue
                if loop_var in {str(sym) for sym in ie.data.subset.free_symbols}:
                    arrays.add(ie.src.data)
    return arrays


def _write_index_input_connectors(nsdfg_node: nodes.NestedSDFG, out_conn: str) -> Set[str]:
    """Input-connector names of ``nsdfg_node`` that appear in the subset writing
    the output array ``out_conn`` inside the nested SDFG -- i.e. the connectors
    that carry the data-dependent write index.
    """
    in_conns = set(nsdfg_node.in_connectors.keys())
    idx_conns: Set[str] = set()
    for st in nsdfg_node.sdfg.all_states():
        for dn in st.data_nodes():
            if dn.data != out_conn:
                continue
            for e in st.in_edges(dn):
                if e.data is None or e.data.subset is None:
                    continue
                idx_conns |= {str(sym) for sym in e.data.subset.free_symbols if str(sym) in in_conns}
    return idx_conns


def _wrap_loop_in_dispatcher(parent, loop: LoopRegion, condition_expr: str, loop_to_map_cls) -> None:
    """Replace ``loop`` in ``parent`` with a ``ConditionalBlock`` that picks
    between a sequential clone (taken when ``condition_expr`` is true -- the
    "collision detected, fall back" branch) and a parallelised lift of the
    original loop (taken otherwise).

    The clone is a deep copy of the LoopRegion so the sequential branch keeps
    the original semantics regardless of what ``LoopToMap`` does on the other
    branch. The ConditionalBlock is spliced in at ``loop``'s former position;
    the parent's edges to/from ``loop`` are redirected to the new block.

    :param parent: The ``ControlFlowRegion`` that holds ``loop`` as one of its
        nodes. Must support ``add_node``/``remove_node``/``add_edge``.
    :param loop: The scatter loop to wrap. Must be in ``parent.nodes()``.
    :param condition_expr: The guard expression (e.g. ``"__dup_count > 0"``)
        for the ``True`` branch (sequential clone). The ``False`` branch is
        unguarded.
    :param loop_to_map_cls: ``LoopToMap`` class (injected to avoid a top-level
        import cycle through ``dace.transformation.interstate``).
    """
    import copy as _copy
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion

    if loop not in parent.nodes():
        return

    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))
    was_start = getattr(parent, 'start_block', None) is loop

    sequential_clone = _copy.deepcopy(loop)
    sequential_clone.label = loop.label + '_seq_fallback'
    # Pin the fallback so no later parallelizer re-lifts it, and so a parallelism
    # counter can treat this guarded region as fully parallel (the pinned clone is
    # the collision fallback, not a genuinely-sequential loop) -- same marker the
    # specialize-family fallbacks carry (loop_specialization.py).
    sequential_clone.pinned_sequential = True

    cb = ConditionalBlock(loop.label + '_dispatch')
    parent.add_node(cb, is_start_block=was_start, ensure_unique_name=True)  # derived label; wired by object ref

    seq_branch = ControlFlowRegion(loop.label + '_seq_branch', sdfg=parent.sdfg)
    seq_branch.add_node(sequential_clone, is_start_block=True)

    par_branch = ControlFlowRegion(loop.label + '_par_branch', sdfg=parent.sdfg)
    par_branch.add_node(loop, is_start_block=True)
    parent.remove_node(loop)

    cb.add_branch(condition_expr, seq_branch)
    cb.add_branch(None, par_branch)

    for e in in_edges:
        parent.add_edge(e.src, cb, e.data)
    for e in out_edges:
        parent.add_edge(cb, e.dst, e.data)

    # Lift the loop inside the False branch to a Map.
    owner_sdfg = parent.sdfg
    while owner_sdfg.parent_sdfg is not None:
        owner_sdfg = owner_sdfg.parent_sdfg
    instance = loop_to_map_cls()
    instance.loop = loop
    try:
        instance.apply(par_branch, owner_sdfg)
    except Exception:
        # If the lift fails on the parallel branch the sequential clone in the
        # other branch still produces the right result; codegen will compile
        # both arms unchanged.
        pass


__all__ = ['ScatterToGuardedMaps', 'detect_scatter_idx_arrays', 'detect_scatter_loops_and_idx_arrays']
