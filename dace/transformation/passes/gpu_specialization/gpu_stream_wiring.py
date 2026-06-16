# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU stream wiring pass.

Single-shot follow-up to a :class:`GPUStreamSchedulingStrategy`. Reads
``Node.gpu_stream_id`` set by the strategy, allocates the ``gpu_streams``
transient at the SDFG-root scope, wires each consumer's stream connector
to a ``gpu_streams[<i>]`` source, and delegates strategy-specific sync
tasklet insertion. Gated by :func:`is_stream_wiring_applied` so it is
idempotent across pipeline re-application -- the *scheduling* pass stays
idempotent via the per-node Property.
"""
import warnings
from typing import Any, Dict, Optional, Set, Type, Union

from dace import SDFG
from dace.config import Config
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import GPUStreamSchedulingStrategy
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_stream_wiring_applied
from dace.transformation.passes.gpu_specialization.stream_lowering_helpers import (allocate_stream_array,
                                                                                   wire_stream_connectors)


@transformation.explicit_cf_compatible
class GPUStreamWiring(ppl.Pass):
    """Allocate ``gpu_streams`` + wire connectors + insert sync tasklets.

    The strategy reference is needed for :meth:`insert_sync_tasklets`
    (strategy-specific). Allocation and connector wiring are strategy-agnostic.
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

        max_concurrent = int(Config.get('compiler', 'cuda', 'max_concurrent_streams'))
        warnings.warn(
            f"GPUStreamWiring: allocating {num_streams} stream(s) "
            f"(max_concurrent_streams={max_concurrent}).",
            UserWarning,
            stacklevel=2)

        allocate_stream_array(sdfg, num_streams)
        wire_stream_connectors(sdfg, assignments)
        self._strategy.insert_sync_tasklets(sdfg, assignments)
        return num_streams


def _collect_assignments(sdfg: SDFG) -> Dict[nodes.Node, int]:
    """Dict view of every persisted ``Node.gpu_stream_id`` across the SDFG hierarchy.

    Used by the wiring helpers + the strategy's sync-tasklet inserter. The
    per-node property is the durable source; this is just a transient view.
    """
    out: Dict[nodes.Node, int] = {}
    for sub_sdfg in sdfg.all_sdfgs_recursive():
        for state in sub_sdfg.states():
            for node in state.nodes():
                if node.gpu_stream_id is not None:
                    out[node] = node.gpu_stream_id
    return out
