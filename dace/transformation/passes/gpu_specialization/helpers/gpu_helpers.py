# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared naming / code-snippet helpers for the GPU-specialization passes.

Provides the canonical names used when threading per-kernel GPU stream
handles through an SDFG -- the stream array, the per-node connector
prefix, the legacy runtime placeholder symbol (``__dace_current_stream``)
recognized by the DaCe codegen, and the connector name added to
``CopyLibraryNode`` / ``MemsetLibraryNode`` by the stream-connection
pass.
"""
from dace import Config
from dace.codegen import common

# Name of the `stream` in-connector on CopyLibraryNode / MemsetLibraryNode.
# Kept in sync with the ``_STREAM_CONN`` constant in the library-node
# modules so the stream passes can add the connector without importing
# the private constant.
COPY_MEMSET_STREAM_CONNECTOR = "stream"


def get_gpu_stream_array_name() -> str:
    return "gpu_streams"


def get_gpu_stream_connector_name() -> str:
    return "__stream_"


def get_dace_runtime_gpu_stream_name() -> str:
    return "__dace_current_stream"


def get_default_gpu_stream_name() -> str:
    return "__default_stream"


def generate_sync_debug_call() -> str:
    """Generate backend sync + error-check calls when ``syncdebug`` is enabled."""
    backend: str = common.get_gpu_backend()
    sync_call: str = ""
    if Config.get_bool('compiler', 'cuda', 'syncdebug'):
        sync_call = (f"DACE_GPU_CHECK({backend}GetLastError());\n"
                     f"DACE_GPU_CHECK({backend}DeviceSynchronize());\n")

    return sync_call
