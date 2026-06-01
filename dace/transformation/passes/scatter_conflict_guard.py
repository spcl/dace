# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Insert a runtime guard that aborts the program if a scatter's index array has duplicates.

Under permissive ``LoopToMap`` we lift a loop with an indirect write
``out[idx[i]] = ...`` into a parallel scatter Map, *assuming* the user's
contract that ``idx`` is a permutation (no duplicates → no write-write races).
This pass adds a runtime check of that contract: it sorts a copy of ``idx`` and
verifies adjacent elements differ; if any pair is equal the program aborts.

Design choices (decided in conversation -- see [[project_scatter_conflict_guard]]):

- **Sort + adjacent-equal scan**, not race-write-mark + reduce. Sort is integer-
  specialised (ska_sort / cub), independent of ``out``'s size, detects *before*
  the scatter runs, and uses zero non-value-equivalent races.
- **Emit as early as possible**, at the earliest CFG state where ``idx`` is fully
  defined (kernel entry for input arrays; right after the last write state for
  computed arrays). Earlier emission means earlier abort and fewer wasted cycles.
- **Abort only.** No collision flag, no log, no recovery path. The comparison
  Map's body calls ``__builtin_trap()`` directly when an adjacent pair is equal;
  the host process / GPU kernel terminates immediately. No flag indirection means
  no race-writes anywhere in the guard.
- **Tests use permutation ``idx``**, so the guard never fires in CI. The fallback
  (a sequential scatter) lives outside the SDFG entirely -- it's the caller's
  recovery path, not part of the generated kernel.
"""
from typing import Iterable, Optional, Set

import dace
from dace import SDFG, SDFGState, data, dtypes, memlet as mm, properties, subsets
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf

#: Prefix for the sorted-index transient the guard allocates.
_SORTED_PREFIX = '_scatter_guard_sorted_'
_COUNT_PREFIX = '_scatter_guard_count_'

#: Body of the per-element comparison tasklet. Each iteration contributes
#: ``1`` on collision (adjacent equal) and ``0`` otherwise; the WCR ``+`` on
#: the output memlet aggregates these into a per-thread counter that the
#: backend reduces sequentially at end-of-region:
#:
#: - On **CPU**, DaCe lowers WCR ``+`` to ``#pragma omp parallel for
#:   reduction(+:_scatter_guard_count_<idx>)``. The shared counter is touched
#:   exactly once per thread (final reduction), so there is no false sharing
#:   on the per-iteration write path -- writes go to per-thread private copies.
#: - On **GPU**, the WCR lowers to either a block-reduction or per-thread
#:   atomic-add. Either way the post-map sequential trap state checks
#:   ``count > 0`` and traps if so. The trap is OUTSIDE the parallel region
#:   in both backends, well-defined under both OpenMP and CUDA.
#:
#: We use ``+`` (counts collisions) rather than ``|`` (boolean OR) because the
#: count carries more diagnostic information at no additional cost, and the
#: per-iteration ALU work (``__cnt = (cur == nxt)``) is the same.
#:
#: Python language so type inference picks up the literal ``int`` output --
#: a CPP body would leave the output connector untyped and trip
#: ``dead_dataflow_elimination`` mid-pipeline.
_COMPARE_TASKLET_CODE = "__cnt = 1 if __cur == __nxt else 0"


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
        or if a guard already exists for it (presence of a ``_scatter_guard_sorted_``
        transient with the same suffix).
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

    sorted_name = f"{_SORTED_PREFIX}{idx_name}"
    if sorted_name in sdfg.arrays:
        raise ValueError(f"insert_scatter_guard: a guard for '{idx_name}' already exists (transient "
                         f"'{sorted_name}' is present). Refusing to emit duplicate guards.")

    # Capture the original CFG entry block + definer states BEFORE allocating the
    # transient or adding the new guard states. Both queries depend on the SDFG's
    # source-node set, which the new states would otherwise pollute.
    def_states = _find_definition_states(sdfg, idx_name) & set(sdfg.states())
    original_start = sdfg.start_block if def_states else sdfg.start_block

    sdfg.add_transient(sorted_name, desc.shape, desc.dtype)
    init_state, sort_state, compare_state, trap_state, count_name, trap_sym = _build_guard_states(sdfg,
                                                                                                  idx_name,
                                                                                                  sorted_name,
                                                                                                  emit_trap=emit_trap)
    _splice_guard_into_cfg(sdfg, idx_name, init_state, sort_state, compare_state, trap_state, count_name, trap_sym,
                           def_states, original_start)
    sdfg.reset_cfg_list()
    return None if emit_trap else trap_sym


def _find_definition_states(sdfg: SDFG, idx_name: str) -> Set[SDFGState]:
    """Return every state that has an in-degree-positive AccessNode for ``idx_name``."""
    return {
        st
        for sd in sdfg.all_sdfgs_recursive()
        for st in sd.all_states() if any(n.data == idx_name and st.in_degree(n) > 0 for n in st.data_nodes())
    }


def _build_guard_states(
        sdfg: SDFG,
        idx_name: str,
        sorted_name: str,
        emit_trap: bool = True) -> tuple[SDFGState, SDFGState, SDFGState, Optional[SDFGState], str, str]:
    """Build (but do not splice) the four guard states: init, sort, compare, trap.

    The compare map writes per-iteration ``0`` or ``1`` (collision indicator) under a
    WCR ``+`` reduction into a single ``int64`` counter -- the **collision count**.
    DaCe lowers WCR ``+`` to ``#pragma omp parallel for reduction(+:count)`` on CPU
    (per-thread private accumulator, single tree-merge at end of region) and to a
    block-reduce / cub::DeviceReduce on GPU (one global atomic per launch). Neither
    backend writes to the shared counter from within the loop body, so there is no
    false sharing on the hot path. The post-map sequential trap state then reads
    the count via an interstate-edge symbol binding and traps if it is positive --
    the trap is OUTSIDE the parallel region, well-defined under both OpenMP and CUDA.

    Using a count (rather than a boolean flag) gives a free diagnostic: when the trap
    fires in production, the post-mortem can report the number of duplicate pairs.

    :returns: ``(init_state, sort_state, compare_state, trap_state, count_name, trap_sym)``.
    """
    from dace.libraries.sort.nodes.integer_sort import (IntegerSort, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME)

    desc = sdfg.arrays[idx_name]
    n_expr = desc.shape[0]

    # Length-1 collision counter; WCR-summed by the compare map.
    count_name, _ = sdfg.add_scalar(f"{_COUNT_PREFIX}{idx_name}", dtypes.int64, transient=True, find_new_name=True)
    trap_sym = f"__scatter_guard_check_{count_name}"
    sdfg.add_symbol(trap_sym, dtypes.int64)

    # State 0: zero the counter (DaCe transients are uninitialised; WCR + assumes 0 identity).
    # Python language so type inference reads the literal ``0`` (the CPP form
    # ``__cnt = 0;`` has no declared type and forces ``dead_dataflow_elimination`` to
    # bail out when it inspects the connector).
    init_state = sdfg.add_state(f"_scatter_guard_init_{idx_name}")
    count_init_write = init_state.add_write(count_name)
    init_tasklet = init_state.add_tasklet(f"init_{count_name}", {}, {'__cnt'},
                                          "__cnt = 0",
                                          language=dtypes.Language.Python)
    init_state.add_edge(init_tasklet, '__cnt', count_init_write, None, mm.Memlet(data=count_name, subset='0'))

    # State 1: sort.
    sort_state = sdfg.add_state(f"_scatter_guard_sort_{idx_name}")
    idx_read = sort_state.add_read(idx_name)
    sorted_write = sort_state.add_write(sorted_name)
    sort_node = IntegerSort(f"sort_{idx_name}")
    sort_state.add_node(sort_node)
    sort_state.add_edge(idx_read, None, sort_node, INPUT_CONNECTOR_NAME,
                        mm.Memlet(data=idx_name, subset=subsets.Range([(0, n_expr - 1, 1)])))
    sort_state.add_edge(sort_node, OUTPUT_CONNECTOR_NAME, sorted_write, None,
                        mm.Memlet(data=sorted_name, subset=subsets.Range([(0, n_expr - 1, 1)])))

    # State 2: adjacent-equal scan. Map over i in [0, N-2]; tasklet reads sorted[i]
    # and sorted[i+1] and emits ``__cnt = (cur == nxt) ? 1 : 0``. WCR ``+``
    # aggregates per-thread (reduction(+:count) on CPU / cub::DeviceReduce on GPU).
    compare_state = sdfg.add_state(f"_scatter_guard_compare_{idx_name}")
    sorted_in = compare_state.add_read(sorted_name)
    count_compare_write = compare_state.add_write(count_name)
    compare_state.add_mapped_tasklet(
        f"compare_{idx_name}",
        {'__guard_i': f"0:({n_expr}) - 1"},
        {
            '__cur': mm.Memlet(data=sorted_name, subset=subsets.Range([('__guard_i', '__guard_i', 1)])),
            '__nxt': mm.Memlet(data=sorted_name, subset=subsets.Range([('__guard_i + 1', '__guard_i + 1', 1)])),
        },
        _COMPARE_TASKLET_CODE,
        {'__cnt': mm.Memlet(data=count_name, subset='0', wcr='lambda a, b: a + b')},
        external_edges=True,
        input_nodes={sorted_name: sorted_in},
        output_nodes={count_name: count_compare_write},
        language=dtypes.Language.Python,
    )

    # State 3: trap if the count is positive. The trap is OUTSIDE the parallel
    # region, which is the whole point of the reduction pattern. The count value is
    # bound to the symbol ``trap_sym`` on the incoming interstate edge, so the
    # tasklet has no connectors (DaCe convention: tasklets that read only symbols
    # may have no in-connectors).
    trap_state: Optional[SDFGState] = None
    if emit_trap:
        trap_state = sdfg.add_state(f"_scatter_guard_trap_{idx_name}")
        trap_state.add_tasklet(f"trap_{idx_name}", {}, {},
                               f"if ({trap_sym} > 0) {{ __builtin_trap(); }}",
                               language=dtypes.Language.CPP)

    return init_state, sort_state, compare_state, trap_state, count_name, trap_sym


def _splice_guard_into_cfg(sdfg: SDFG, idx_name: str, init_state: SDFGState, sort_state: SDFGState,
                           compare_state: SDFGState, trap_state: Optional[SDFGState], count_name: str, trap_sym: str,
                           def_states: Set[SDFGState], original_start) -> None:
    """Splice the guard states in at the earliest legal CFG point.

    Internal chain: ``init -> sort -> compare -> [trap] -> downstream`` (plain
    interstate edges, except the edge OUT of ``compare`` which binds the count
    value to ``trap_sym``). When ``trap_state`` is ``None``, the chain ends at
    ``compare`` and the count symbol propagates directly to downstream code
    (the caller is responsible for routing on it, e.g. a ``ConditionalBlock``).

    - If ``def_states`` is empty (``idx_name`` has no internal writes), the chain
      becomes the new start.
    - Otherwise, the chain is inserted right after the topologically-latest
      top-level definer state.
    """
    sdfg.add_edge(init_state, sort_state, dace.InterstateEdge())
    sdfg.add_edge(sort_state, compare_state, dace.InterstateEdge())
    if trap_state is not None:
        sdfg.add_edge(compare_state, trap_state, dace.InterstateEdge(assignments={trap_sym: count_name}))
        chain_tail = trap_state
        downstream_iedge_assignments = None
    else:
        chain_tail = compare_state
        # When the trap is absent, the count value must be bound on the edge
        # leaving the chain (the caller will dispatch on the symbol downstream).
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
        sdfg.add_edge(last_def, init_state, dace.InterstateEdge())
    else:
        sdfg.add_edge(chain_tail, original_start, dace.InterstateEdge(assignments=downstream_iedge_assignments))
        sdfg.start_block = sdfg.node_id(init_state)


# Public re-exports for the explicit-API style ("call this function") on top of
# the Pass class. Callers that already know which idx arrays need guarding
# typically use ``insert_scatter_guard`` directly; callers driving a batch via
# the Pass pipeline use ``GuardScatterConflicts``.
__all__ = ['GuardScatterConflicts', 'insert_scatter_guard']
