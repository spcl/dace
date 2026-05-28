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
import copy
from typing import Iterable, Optional, Set

import dace
from dace import SDFG, SDFGState, data, dtypes, memlet as mm, properties, subsets
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


#: Prefix for the sorted-index transient the guard allocates.
_SORTED_PREFIX = '_scatter_guard_sorted_'

#: Body of the per-element comparison tasklet. Race-write semantics are not needed
#: because ``__builtin_trap()`` terminates the program before any other thread
#: observes a partial state. Portable across GCC/Clang (CPU) and NVCC (CUDA).
_COMPARE_TASKLET_CODE = "if (__cur == __nxt) { __builtin_trap(); }"


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


def insert_scatter_guard(sdfg: SDFG, idx_name: str) -> None:
    """Emit a sort+compare+abort guard for ``idx_name`` at the earliest legal CFG point.

    :param sdfg: The SDFG to mutate in place.
    :param idx_name: The integer array whose runtime values must be all distinct.
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
    sort_state, compare_state = _build_guard_states(sdfg, idx_name, sorted_name)
    _splice_guard_into_cfg(sdfg, idx_name, sort_state, compare_state, def_states, original_start)
    sdfg.reset_cfg_list()


def _find_definition_states(sdfg: SDFG, idx_name: str) -> Set[SDFGState]:
    """Return every state that has an in-degree-positive AccessNode for ``idx_name``."""
    return {
        st
        for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states()
        if any(n.data == idx_name and st.in_degree(n) > 0 for n in st.data_nodes())
    }


def _build_guard_states(sdfg: SDFG, idx_name: str, sorted_name: str) -> tuple[SDFGState, SDFGState]:
    """Build (but do not splice) the two guard states: the sort and the adjacent-compare.

    :returns: ``(sort_state, compare_state)`` -- two fresh detached states added to ``sdfg``.
    """
    from dace.libraries.sort.nodes.integer_sort import (IntegerSort, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME)

    desc = sdfg.arrays[idx_name]
    n_expr = desc.shape[0]

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
    # and sorted[i+1]; calls ``__builtin_trap()`` on equality.
    compare_state = sdfg.add_state(f"_scatter_guard_compare_{idx_name}")
    sorted_in = compare_state.add_read(sorted_name)
    compare_state.add_mapped_tasklet(
        f"compare_{idx_name}",
        {'__guard_i': f"0:({n_expr}) - 1"},
        {
            '__cur': mm.Memlet(data=sorted_name, subset=subsets.Range([('__guard_i', '__guard_i', 1)])),
            '__nxt': mm.Memlet(data=sorted_name, subset=subsets.Range([('__guard_i + 1', '__guard_i + 1', 1)])),
        },
        _COMPARE_TASKLET_CODE,
        # No data outputs: the tasklet's only effect is a trap on collision.
        {},
        external_edges=True,
        input_nodes={sorted_name: sorted_in},
        language=dtypes.Language.CPP,
    )

    return sort_state, compare_state


def _splice_guard_into_cfg(sdfg: SDFG, idx_name: str, sort_state: SDFGState, compare_state: SDFGState,
                           def_states: Set[SDFGState], original_start) -> None:
    """Splice the two guard states in at the earliest legal CFG point.

    - If ``def_states`` is empty (``idx_name`` has no internal writes), the
      guard becomes the new start chain: ``sort_state -> compare_state -> original_start``.
    - Otherwise, the guard is inserted right after the topologically-latest
      top-level definer state: every out-edge of that state is redirected to
      ``compare_state``, and a fresh edge from the definer to ``sort_state`` is added.

    ``sort_state -> compare_state`` is always connected by a plain interstate edge.
    """
    sdfg.add_edge(sort_state, compare_state, dace.InterstateEdge())

    if def_states:
        # Topological order of the ORIGINAL CFG (the guard states aren't reachable yet).
        topo_order = list(sdfg.bfs_nodes(original_start))
        last_def = max(def_states, key=topo_order.index)
        for e in list(sdfg.out_edges(last_def)):
            sdfg.remove_edge(e)
            sdfg.add_edge(compare_state, e.dst, e.data)
        sdfg.add_edge(last_def, sort_state, dace.InterstateEdge())
    else:
        # No internal definers: the guard runs at SDFG entry.
        sdfg.add_edge(compare_state, original_start, dace.InterstateEdge())
        sdfg.start_block = sdfg.node_id(sort_state)


# Public re-exports for the explicit-API style ("call this function") on top of
# the Pass class. Callers that already know which idx arrays need guarding
# typically use ``insert_scatter_guard`` directly; callers driving a batch via
# the Pass pipeline use ``GuardScatterConflicts``.
__all__ = ['GuardScatterConflicts', 'insert_scatter_guard']
