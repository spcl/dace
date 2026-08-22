# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Block-collective copy into shared memory.
"""
from typing import TYPE_CHECKING

import dace
from dace import library, nodes, dtypes
from dace.sdfg.scope import is_in_scope
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard.nodes.copy.common import (_build_shmem_collective_copy_code, INPUT_CONNECTOR_NAME,
                                                       OUTPUT_CONNECTOR_NAME)

if TYPE_CHECKING:
    pass


@library.expansion
class ExpandSharedMemoryCollective(ExpandTransformation):
    """Block-collective Shared <-> Shared/Global copy: a single Tasklet emitting
    ``dace::CopyND<...>::Copy + __syncthreads()``, with ``_in``/``_out`` connectors matching the
    libnode's directly (no NSDFG wrapper -- the parent kernel's ``__shared__`` array binds
    straight in, no scope-id name mangling).

    Caller must place this outside any enclosing ``GPU_ThreadBlock`` map -- this expansion *is*
    the thread-block-level operation. Shared <-> Register goes through ``MappedTasklet`` instead
    (auto selector routes it there)."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_sdfg,
                                                                            parent_state,
                                                                            allow_cross_storage=True)

        valid_storages = {dtypes.StorageType.GPU_Shared, dtypes.StorageType.GPU_Global}
        if inp.storage not in valid_storages or out.storage not in valid_storages:
            raise ValueError(f"SharedMemoryCollective requires GPU_Shared / GPU_Global storages "
                             f"(got {inp.storage} -> {out.storage}). Use MappedTasklet for "
                             "Shared <-> Register thread-level copies.")
        if inp.storage != dtypes.StorageType.GPU_Shared and out.storage != dtypes.StorageType.GPU_Shared:
            raise ValueError("SharedMemoryCollective requires at least one side to be GPU_Shared.")

        if is_in_scope(parent_sdfg, parent_state, node, [dtypes.ScheduleType.GPU_ThreadBlock]):
            raise ValueError("SharedMemoryCollective IS the thread-block-level operation "
                             "and must not be nested inside a GPU_ThreadBlock map.")

        return nodes.Tasklet(node.name,
                             inputs={INPUT_CONNECTOR_NAME: dace.dtypes.pointer(inp.dtype)},
                             outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                             code=_build_shmem_collective_copy_code(node, parent_state, inp, in_subset, out,
                                                                    out_subset),
                             language=dace.Language.CPP)
