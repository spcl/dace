# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runtime guard: abort if a scatter index array has duplicates.

Permissive ``LoopToMap`` lifts an indirect write ``out[idx[i]] = ...`` into a parallel
scatter Map *assuming* ``idx`` is a permutation (duplicates → write-write races). This
pass checks that contract at runtime with an opaque ``ScatterConflictCheck`` libnode
(sort a copy + count adjacent-equal → host ``int64`` count); an interstate edge binds the
count to a symbol, and a trap tasklet ``__builtin_trap()``s if it is positive.

- Opaque libnode → the tile vectorizer never looks inside it, so the guard leaves the
  scatter Map it guards fully vectorizable. Only Maps are tiled; the guard is a libnode +
  a top-level trap tasklet, neither a Map.
- Host scalar out on every backend (CPU + CUDA) → the symbol bind + trap run on the host
  wherever ``idx`` lives.
- Emitted at the earliest CFG point where ``idx`` is fully defined.
- Abort-only; the sequential-scatter fallback is the caller's, outside the SDFG.
"""
from typing import Iterable, Optional, Set

import numpy as np

import dace
from dace import SDFG, SDFGState, data, dtypes, memlet as mm, properties, subsets, symbolic
from dace.frontend.python import astutils
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

#: Prefix for the collision-count scalar the guard allocates (one per guarded idx).
_COUNT_PREFIX = '_scatter_guard_count_'


@properties.make_properties
@xf.explicit_cf_compatible
class GuardScatterConflicts(ppl.Pass):
    """Insert a sort-based runtime guard for each named scatter index array.

    :param idx_names: Names of integer arrays whose values must be a permutation
        (no duplicates) at runtime. The pass emits, for each name, a sort of the
        array plus an adjacent-equal-pair scan that calls ``__builtin_trap()`` on
        violation. Emission point is the earliest CFG state where the array is
        fully defined: the SDFG's entry if the array has no internal writes; the
        topologically-latest state with an in-degree-positive AccessNode of the
        array otherwise.
    """

    CATEGORY: str = 'Optimization Preparation'

    def __init__(self, idx_names: Iterable[str]):
        super().__init__()
        self._idx_names = list(idx_names)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        # Single-shot: re-running re-emits guards (idempotent only on a fresh-name basis).
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        emitted = 0
        for idx_name in self._idx_names:
            if idx_name not in sdfg.arrays:
                continue
            insert_scatter_guard(sdfg, idx_name)
            emitted += 1
        return emitted or None


def scatter_index_is_provably_injective(sdfg: SDFG, idx_name: str) -> bool:
    """True iff every runtime value of ``idx_name`` is provably distinct at compile time.

    The runtime ``ScatterConflictCheck`` guard exists to prove, at run time, that the whole
    index array holds no duplicate values (a duplicate = two scatter iterations writing the
    same output slot = a race). When that fact is already decidable statically, the guard is
    pure overhead and can be dropped (*Lever 1*). This is the static side of the same
    predicate the guard checks dynamically, so eliding on a ``True`` verdict is behaviourally
    identical to running the guard and having it pass.

    Sound and deliberately conservative -- returns ``True`` ONLY for the forms it can prove,
    and ``False`` (keep the guard) on any uncertainty. Two forms are recognised:

    1. **Compile-time constant** -- ``idx_name`` is an SDFG constant whose integer values are
       all distinct (a compile-time permutation / injective sequence).
    2. **Affine full-domain producer** -- ``idx_name`` is a transient written by exactly one
       ``LoopRegion`` ``for j in range(0, M): idx[j] = a*j + b`` where ``M`` is the array's
       full extent, the write lands at the point ``[j]`` (so iterating ``j`` over ``[0, M)``
       covers every element), and the stored value is an affine function of ``j`` with a
       non-zero integer leading coefficient. An affine map with ``a != 0`` over a contiguous
       integer domain is injective, so the values are pairwise distinct.

    Anything outside these forms -- a parameter array (unknown contents), multiple or partial
    writers, a non-affine or data-dependent producer, a strided/offset write position, or a
    non-``LoopRegion`` (e.g. already-lifted Map) producer -- returns ``False``.

    :param sdfg: The SDFG owning ``idx_name``.
    :param idx_name: The candidate scatter index array.
    :returns: ``True`` iff the values are provably all-distinct; ``False`` otherwise.
    """
    # Inline import: the ``sort`` library eagerly pulls in its environments; keep it off the
    # module-load path (mirrors ``insert_scatter_guard`` below).
    from dace.libraries.sort.nodes._helpers import is_integer_dtype

    desc = sdfg.arrays.get(idx_name)
    if not isinstance(desc, data.Array) or len(desc.shape) != 1 or not is_integer_dtype(desc.dtype):
        return False

    # Form 1: compile-time constant with distinct integer values.
    const_val = sdfg.constants.get(idx_name)
    if isinstance(const_val, np.ndarray) and const_val.dtype.kind in ('i', 'u'):
        flat = const_val.reshape(-1)
        return int(np.unique(flat).size) == int(flat.size)

    # Form 2: transient fully written by a single injective affine producer loop.
    if not desc.transient:
        return False

    writers = [(st, n) for st in sdfg.all_states() for n in st.data_nodes()
               if n.data == idx_name and st.in_degree(n) > 0]
    if len(writers) != 1:  # multiple / partial / no writers -> cannot prove full-domain coverage
        return False
    state, node = writers[0]
    in_edges = list(state.in_edges(node))
    if len(in_edges) != 1:
        return False
    write = in_edges[0]
    if not isinstance(write.src, nodes.Tasklet):  # a copy from another array -> unknown values
        return False

    region = state.parent_graph
    if not isinstance(region, LoopRegion) or not region.loop_variable:
        return False
    loop_var = symbolic.pystr_to_symbolic(str(region.loop_variable))
    init = loop_analysis.get_init_assignment(region)
    end = loop_analysis.get_loop_end(region)
    stride = loop_analysis.get_loop_stride(region)
    if init is None or end is None or stride is None:
        return False
    # Loop must sweep the array's full extent [0, M) with unit stride, else some element is
    # left uninitialised (garbage the guard would still sort) or the coverage is non-contiguous.
    extent = desc.shape[0]
    if symbolic.simplify(init) != 0 or symbolic.simplify(stride - 1) != 0 or symbolic.simplify(end - (extent - 1)) != 0:
        return False

    # Write position must be the bare point ``[loop_var]`` so sweeping j covers exactly [0, M).
    ndrange = list(write.data.subset.ndrange())
    if len(ndrange) != 1:
        return False
    begin, stop, _ = ndrange[0]
    if (symbolic.simplify(symbolic.pystr_to_symbolic(str(begin)) - loop_var) != 0
            or symbolic.simplify(symbolic.pystr_to_symbolic(str(stop)) - loop_var) != 0):
        return False

    # Stored value must be an affine function of the loop variable with a non-zero integer
    # leading coefficient (a*j + b with a != 0 -> distinct values over the contiguous domain).
    code = write.src.code.code
    statements = code if isinstance(code, list) else [code]
    finder = astutils.FindAssignment()
    for stmt in statements:
        finder.visit(stmt)
    if finder.multiple:
        return False
    value_expr = finder.assignments.get(write.src_conn)
    if value_expr is None:
        return False
    return value_is_injective_affine(value_expr, region.loop_variable)


def value_is_injective_affine(value_expr: str, loop_var: str) -> bool:
    """True iff ``value_expr`` is an affine ``a*loop_var + b`` with a non-zero integer ``a``.

    Such a map is injective over any contiguous integer domain, so the values it produces as
    ``loop_var`` sweeps a range are pairwise distinct. Any input-dependent or non-affine
    expression (a runtime read, ``loop_var % k``, ``loop_var * loop_var``) yields ``a == 0`` or
    fails the affinity check and returns ``False``. Parsed through the symbol registry via
    :func:`dace.symbolic.pystr_to_symbolic` so loop-variable assumptions never break the match.

    :param value_expr: The stored value as a string expression (a producer tasklet's RHS).
    :param loop_var: The producing loop's iteration variable.
    :returns: ``True`` iff ``value_expr`` is ``a*loop_var + b`` with a non-zero integer ``a``.
    """
    j = symbolic.pystr_to_symbolic(str(loop_var))
    expr = symbolic.pystr_to_symbolic(str(value_expr))
    lead = expr.coeff(j, 1)
    const = expr.coeff(j, 0)
    if symbolic.simplify(expr - (lead * j + const)) != 0:  # non-affine in loop_var
        return False
    return bool(lead.is_Integer) and lead != 0


def insert_scatter_guard(sdfg: SDFG,
                         idx_name: str,
                         emit_trap: bool = True,
                         elide_if_injective: bool = True) -> Optional[str]:
    """Emit a sort+compare+abort guard for ``idx_name`` at the earliest legal CFG point.

    :param sdfg: The SDFG to mutate in place.
    :param idx_name: The integer array whose runtime values must be all distinct.
    :param emit_trap: When ``True`` (default), the chain ends in a state whose
        tasklet calls ``__builtin_trap()`` if the duplicate count is positive.
        When ``False``, the trap state is omitted and the duplicate-count
        symbol is returned for the caller to thread into its own runtime
        check (e.g. a ``ConditionalBlock`` selecting between a parallel and
        a fallback sequential branch).
    :param elide_if_injective: When ``True`` (default -- *Lever 1*), first run the
        compile-time injectivity analysis (:func:`scatter_index_is_provably_injective`).
        If it proves ``idx_name``'s runtime values are all distinct, no guard is emitted
        at all (a plain scatter with no runtime check): the runtime ``ScatterConflictCheck``
        would inevitably find a zero duplicate-count, so the three O(n) passes it runs are
        pure overhead. Pass ``False`` to force a guard regardless (e.g. structural tests).
    :returns: ``None`` when ``emit_trap=True`` OR when the guard was elided as provably
              conflict-free; the duplicate-count symbol name
              (``__scatter_guard_check_<count>``) when ``emit_trap=False`` and a guard was
              emitted. Callers in the latter mode dispatch on ``sym > 0``; a ``None`` return
              there means "no dispatch needed -- proven safe, lift unconditionally".
    :raises ValueError: If ``idx_name`` is not a 1-D integer Array in ``sdfg.arrays``,
        or if a guard already exists for it (its ``_scatter_guard_count_`` scalar is present).
    """
    from dace.libraries.sort.nodes._helpers import is_integer_dtype

    desc = sdfg.arrays.get(idx_name)
    if desc is None:
        raise ValueError(f"insert_scatter_guard: '{idx_name}' is not a data descriptor in this SDFG.")
    if not isinstance(desc, data.Array):
        raise ValueError(f"insert_scatter_guard: '{idx_name}' must be an Array; got {type(desc).__name__}.")
    if len(desc.shape) != 1:
        raise ValueError(f"insert_scatter_guard: '{idx_name}' must be 1-D; got shape {tuple(desc.shape)}.")
    if not is_integer_dtype(desc.dtype):
        raise ValueError(f"insert_scatter_guard: '{idx_name}' must have an integer dtype; got {desc.dtype}.")

    # Lever 1: static-injective elision. When the index array is provably a set of distinct
    # values at compile time, the runtime conflict check is dead weight -- drop it entirely.
    if elide_if_injective and scatter_index_is_provably_injective(sdfg, idx_name):
        return None

    count_name = f"{_COUNT_PREFIX}{idx_name}"
    if count_name in sdfg.arrays:
        raise ValueError(f"insert_scatter_guard: a guard for '{idx_name}' already exists "
                         f"(scalar '{count_name}' is present). Refusing to emit duplicate guards.")

    # Capture the original CFG entry + definer states BEFORE adding the guard states, whose
    # new states would otherwise pollute the source-node set both queries depend on.
    def_states = _find_definition_states(sdfg, idx_name) & set(sdfg.states())
    original_start = sdfg.start_block

    check_state, trap_state, count_name, trap_sym = _build_guard_states(sdfg, idx_name, emit_trap=emit_trap)
    _splice_guard_into_cfg(sdfg, idx_name, check_state, trap_state, count_name, trap_sym, def_states, original_start)
    sdfg.reset_cfg_list()
    return None if emit_trap else trap_sym


def _find_definition_states(sdfg: SDFG, idx_name: str) -> Set[SDFGState]:
    """Return every state that has an in-degree-positive AccessNode for ``idx_name``."""
    return {
        st
        for sd in sdfg.all_sdfgs_recursive()
        for st in sd.all_states() if any(n.data == idx_name and st.in_degree(n) > 0 for n in st.data_nodes())
    }


def _build_guard_states(sdfg: SDFG,
                        idx_name: str,
                        emit_trap: bool = True) -> tuple[SDFGState, Optional[SDFGState], str, str]:
    """Build (but do not splice) the guard states: check, [trap].

    ``check`` runs the opaque ``ScatterConflictCheck`` libnode over ``idx_name`` into the
    ``int64`` host collision count. ``trap`` (emitted iff ``emit_trap``) reads the count via
    the ``trap_sym`` interstate binding and ``__builtin_trap()``s if positive; a count (not a
    boolean) gives a free duplicate-pair diagnostic on abort.

    :returns: ``(check_state, trap_state, count_name, trap_sym)``.
    """
    from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck

    n_expr = sdfg.arrays[idx_name].shape[0]
    count_name, _ = sdfg.add_scalar(f"{_COUNT_PREFIX}{idx_name}", dtypes.int64, transient=True)
    trap_sym = f"__scatter_guard_check_{count_name}"
    sdfg.add_symbol(trap_sym, dtypes.int64)

    # check: opaque ScatterConflictCheck libnode (sort a copy + count adjacent-equal) -> host
    # count. A library node, so the tile vectorizer never looks through it -> the guard leaves
    # the scatter Map fully vectorizable.
    check_state = sdfg.add_state(f"_scatter_guard_check_{idx_name}")
    idx_read = check_state.add_read(idx_name)
    count_write = check_state.add_write(count_name)
    check_node = ScatterConflictCheck(f"conflict_check_{idx_name}")
    check_state.add_node(check_node)
    check_state.add_edge(idx_read, None, check_node, ScatterConflictCheck.INPUT_CONNECTOR_NAME,
                         mm.Memlet(data=idx_name, subset=subsets.Range([(0, n_expr - 1, 1)])))
    check_state.add_edge(check_node, ScatterConflictCheck.OUTPUT_CONNECTOR_NAME, count_write, None,
                         mm.Memlet(data=count_name, subset='0'))

    # trap: top-level tasklet reading only ``trap_sym`` (bound to the count on the incoming
    # edge), so no connectors. ``side_effects`` keeps DeadDataflowElimination from pruning it --
    # else the whole guard chain feeding ``trap_sym`` looks dead and the scatter goes unguarded.
    trap_state: Optional[SDFGState] = None
    if emit_trap:
        trap_state = sdfg.add_state(f"_scatter_guard_trap_{idx_name}")
        trap_tasklet = trap_state.add_tasklet(f"check_assumption_{idx_name}", {}, {},
                                              f"if ({trap_sym} > 0) {{ __builtin_trap(); }}",
                                              language=dtypes.Language.CPP)
        trap_tasklet.side_effects = True

    return check_state, trap_state, count_name, trap_sym


def _splice_guard_into_cfg(sdfg: SDFG, idx_name: str, check_state: SDFGState, trap_state: Optional[SDFGState],
                           count_name: str, trap_sym: str, def_states: Set[SDFGState], original_start) -> None:
    """Splice ``check -> [trap] -> downstream`` in at the earliest legal CFG point.

    The edge leaving the last guard state binds the count to ``trap_sym``: onto ``check ->
    trap`` when a trap is emitted, else onto the edge out of ``check`` (the caller then
    dispatches on the symbol, e.g. a parallel-vs-sequential ``ConditionalBlock``).

    - ``def_states`` empty (``idx_name`` has no internal writes) → the chain becomes the start.
    - Otherwise → inserted right after the topologically-latest top-level definer state.
    """
    if trap_state is not None:
        sdfg.add_edge(check_state, trap_state, dace.InterstateEdge(assignments={trap_sym: count_name}))
        chain_tail = trap_state
        downstream_iedge_assignments = None
    else:
        chain_tail = check_state
        downstream_iedge_assignments = {trap_sym: count_name}

    if def_states:
        topo_order = list(sdfg.bfs_nodes(original_start))
        last_def = max(def_states, key=topo_order.index)
        for e in list(sdfg.out_edges(last_def)):
            sdfg.remove_edge(e)
            if downstream_iedge_assignments is not None:
                merged = dict(e.data.assignments or {})
                merged.update(downstream_iedge_assignments)
                sdfg.add_edge(chain_tail, e.dst, dace.InterstateEdge(condition=e.data.condition, assignments=merged))
            else:
                sdfg.add_edge(chain_tail, e.dst, e.data)
        sdfg.add_edge(last_def, check_state, dace.InterstateEdge())
    else:
        sdfg.add_edge(chain_tail, original_start, dace.InterstateEdge(assignments=downstream_iedge_assignments))
        sdfg.start_block = sdfg.node_id(check_state)


# Public re-exports for the explicit-API style ("call this function") on top of
# the Pass class. Callers that already know which idx arrays need guarding
# typically use ``insert_scatter_guard`` directly; callers driving a batch via
# the Pass pipeline use ``GuardScatterConflicts``.
__all__ = ['GuardScatterConflicts', 'insert_scatter_guard', 'scatter_index_is_provably_injective']
