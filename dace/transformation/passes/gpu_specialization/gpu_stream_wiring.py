# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wires the streams a :class:`GPUStreamSchedulingStrategy` assigned: allocates ``gpu_streams``, connects each
consumer to ``gpu_streams[i]`` and has the strategy insert its syncs. Applied once: see :func:`is_stream_wiring_applied`.
"""

from typing import Any

from dace import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (
    GPUStreamSchedulingStrategy,
    allocate_stream_array,
    wire_stream_connectors,
)
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (
    is_stream_wiring_applied,
    persisted_stream_assignments,
)


@transformation.explicit_cf_compatible
class GPUStreamWiring(ppl.Pass):
    """Allocate ``gpu_streams``, wire the connectors and insert the strategy's sync tasklets."""

    def __init__(self, strategy: GPUStreamSchedulingStrategy):
        if not isinstance(strategy, GPUStreamSchedulingStrategy):
            raise TypeError(f"strategy must be a GPUStreamSchedulingStrategy, got {type(strategy).__name__}.")
        self._strategy = strategy

    def depends_on(self) -> set[type[ppl.Pass] | ppl.Pass]:
        return {type(self._strategy)}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        if sdfg.parent_sdfg is not None:
            raise ValueError(
                f"GPUStreamWiring: must run on the root SDFG. Got nested SDFG "
                f"'{sdfg.name}' (parent '{sdfg.parent_sdfg.name}')."
            )
        if is_stream_wiring_applied(sdfg):
            return None
        assignments = persisted_stream_assignments(sdfg)
        num_streams = max(assignments.values(), default=-1) + 1

        allocate_stream_array(sdfg, num_streams)
        wire_stream_connectors(sdfg, assignments)
        self._strategy.insert_sync_tasklets(sdfg, assignments)
        return num_streams
