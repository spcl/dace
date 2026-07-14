# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU stream wiring pass: follow-up to a :class:`GPUStreamSchedulingStrategy` that reads the
``Node.gpu_stream_id`` set by the strategy, allocates the root-scope ``gpu_streams`` transient,
wires each consumer's stream connector to ``gpu_streams[<i>]``, and delegates sync-tasklet insertion.

Gated by :func:`is_stream_wiring_applied` for idempotency across pipeline re-application; the
scheduling pass stays idempotent via the per-node Property.
"""
from typing import Any, Dict, Optional, Set, Type, Union

from dace import SDFG
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import GPUStreamSchedulingStrategy
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_stream_wiring_applied
from dace.transformation.passes.gpu_specialization.stream_lowering_helpers import (allocate_stream_array,
                                                                                   wire_stream_connectors)


@transformation.explicit_cf_compatible
class GPUStreamWiring(ppl.Pass):
    """Allocate ``gpu_streams`` + wire connectors + insert sync tasklets.

    Holds a strategy reference only for the strategy-specific :meth:`insert_sync_tasklets`;
    allocation and connector wiring are strategy-agnostic.
    """

    def __init__(self, strategy: GPUStreamSchedulingStrategy):
        if not isinstance(strategy, GPUStreamSchedulingStrategy):
            raise TypeError(f"strategy must be a GPUStreamSchedulingStrategy, got {type(strategy).__name__}.")
        self._strategy = strategy

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {type(self._strategy)}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        if sdfg.parent_sdfg is not None:
            raise ValueError(f"GPUStreamWiring: must run on the root SDFG. Got nested SDFG "
                             f"'{sdfg.name}' (parent '{sdfg.parent_sdfg.name}').")
        if is_stream_wiring_applied(sdfg):
            return None
        assignments = _collect_assignments(sdfg)
        num_streams = max(assignments.values(), default=-1) + 1

        allocate_stream_array(sdfg, num_streams)
        wire_stream_connectors(sdfg, assignments)
        self._strategy.insert_sync_tasklets(sdfg, assignments)
        return num_streams


def _collect_assignments(sdfg: SDFG) -> Dict[nodes.Node, int]:
    """Transient dict view of every persisted ``Node.gpu_stream_id`` across the SDFG hierarchy.

    The per-node property is the durable source of truth; this view is rebuilt on demand.
    """
    out: Dict[nodes.Node, int] = {}
    for sub_sdfg in sdfg.all_sdfgs_recursive():
        for state in sub_sdfg.states():
            for node in state.nodes():
                if node.gpu_stream_id is not None:
                    out[node] = node.gpu_stream_id
    return out
