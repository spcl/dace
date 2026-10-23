# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Memory engine (TMA) bulk copy between global and shared memory.
"""
from typing import List, Optional, Set, Tuple

import dace
from dace import data, library, nodes, dtypes, symbolic
from dace.libraries.standard.environments.memory_engine import MemoryEngine
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard.nodes.copy.common import (
    _build_memory_engine_barrier_init_code, _build_memory_engine_bulk_copy_code, _build_memory_engine_desc_code,
    mark_memory_engine_shared_static, memory_engine_bulk_reason, memory_engine_geometry, MemoryEngineGeometry,
    CUtensorMap, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, MEMORY_ENGINE_BARRIER_INPUT_NAME,
    MEMORY_ENGINE_BARRIER_OUTPUT_NAME, MEMORY_ENGINE_DESC_INPUT_NAME, MEMORY_ENGINE_DESC_OUTPUT_NAME,
    MEMORY_ENGINE_GLOBAL_INPUT_NAME)

_KERNEL_SCHEDULES = (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_Persistent)
"""Schedules that start a GPU kernel."""


@library.register_expansion(CopyLibraryNode, "MemoryEngineBulk")
class ExpandMemoryEngineBulk(ExpandTransformation):
    """Memory engine based bulk copy between ``GPU_Global`` and ``GPU_Shared``.

    Currently only the TMA engine on NVIDIA Hopper+ GPUs is supported. The copy becomes a
    block-collective tasklet issuing ``cp.async.bulk.tensor`` from a leader thread. Postprocessing
    adds the support graph:
    - a host tasklet in the closest CPU scope that encodes the ``CUtensorMap``, which reaches the kernel
      as a ``const __grid_constant__`` parameter.
    - for loads, a ``GPU_Shared`` mbarrier and a tasklet initializing it once at the top of the kernel.

    The shared side is pinned to static shared memory (``GPU_Shared(dynamic=False)``), because currently
    only static shared memory is aligned to 128B as the memory engine requires.

    Caller must place this expansion outside any enclosing ``GPU_ThreadBlock`` map. This expansion *is*
    the thread-block-level operation.

    TODO: Add support for dynamic shared memory.
    """
    environments = [MemoryEngine]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        node.validate(parent_sdfg, parent_state, allow_cross_storage=True)
        reason = memory_engine_bulk_reason(node, parent_state)
        if reason is not None:
            raise ValueError(f"Cannot expand {node} with MemoryEngineBulk: {reason}")
        geometry, _ = memory_engine_geometry(node, parent_state)

        inputs = {
            INPUT_CONNECTOR_NAME: dtypes.pointer(geometry.global_desc.dtype),
            # By value through a dynamic memlet: a const reference to the kernel parameter, never a copy
            MEMORY_ENGINE_DESC_INPUT_NAME: CUtensorMap,
        }
        if geometry.is_load:
            inputs[MEMORY_ENGINE_BARRIER_INPUT_NAME] = dtypes.pointer(dtypes.uint64)
        return nodes.Tasklet(node.name,
                             inputs=inputs,
                             outputs={OUTPUT_CONNECTOR_NAME: dtypes.pointer(geometry.global_desc.dtype)},
                             code=_build_memory_engine_bulk_copy_code(node, geometry),
                             language=dace.Language.CPP)

    @staticmethod
    def postprocessing(sdfg: SDFG, state: SDFGState, expansion: nodes.Tasklet):
        raise NotImplementedError()
