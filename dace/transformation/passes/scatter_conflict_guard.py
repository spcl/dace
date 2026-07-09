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

import dace
from dace import SDFG, SDFGState, data, dtypes, memlet as mm, properties, subsets
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf

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


def insert_scatter_guard(sdfg: SDFG, idx_name: str, emit_trap: bool = True) -> Optional[str]:
    """Emit a sort+compare+abort guard for ``idx_name`` at the earliest legal CFG point.

    :param sdfg: The SDFG to mutate in place.
    :param idx_name: The integer array whose runtime values must be all distinct.
    :param emit_trap: When ``True`` (default), the chain ends in a state whose
        tasklet calls ``__builtin_trap()`` if the duplicate count is positive.
        When ``False``, the trap state is omitted and the duplicate-count
        symbol is returned for the caller to thread into its own runtime
        check (e.g. a ``ConditionalBlock`` selecting between a parallel and
        a fallback sequential branch).
    :returns: ``None`` when ``emit_trap=True``; the duplicate-count symbol
              name (``__scatter_guard_check_<count>``) when ``emit_trap=False``.
              Callers in the latter mode dispatch on ``sym > 0``.
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
__all__ = ['GuardScatterConflicts', 'insert_scatter_guard']
