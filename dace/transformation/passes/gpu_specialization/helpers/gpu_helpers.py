# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import Config
from dace.codegen import common


def get_gpu_stream_array_name() -> str:
    return "gpu_streams"


def get_gpu_stream_connector_name() -> str:
    return "_stream_"


def generate_sync_debug_call() -> str:
    """
    Generate backend sync and error-check calls as a string if
    synchronous debugging is enabled.

    Parameters
    ----------
    backend : str
        Backend API prefix (e.g., 'cuda').

    Returns
    -------
    str
        The generated debug call code, or an empty string if debugging is disabled.
    """
    backend: str = common.get_gpu_backend()
    sync_call: str = ""
    if Config.get_bool('compiler', 'cuda', 'syncdebug'):
        sync_call = (f"DACE_GPU_CHECK({backend}GetLastError());\n"
                     f"DACE_GPU_CHECK({backend}DeviceSynchronize());\n")

    return sync_call
