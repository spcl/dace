import ctypes

import pytest
import numpy as np

from dace.libraries.ort_api import ORTCAPIInterface, OrtCUDAProviderOptions


@pytest.mark.ort
def test_basic():
    with ORTCAPIInterface() as api:
        env = ctypes.c_void_p()
        api.CreateEnv("ORT_LOGGING_LEVEL_WARNING", "ort_api",
                      ctypes.byref(env))

        session_opts = ctypes.c_void_p()
        api.CreateSessionOptions(ctypes.byref(session_opts))

        session = ctypes.c_void_p()
        api.CreateKernelSession(session_opts, ctypes.byref(session), 12)
        api.ReleaseKernelSession(session)
        api.ReleaseEnv(env)


@pytest.mark.ort
@pytest.mark.gpu
def test_basic_gpu():
    with ORTCAPIInterface() as api:
        env = ctypes.c_void_p()
        api.CreateEnv("ORT_LOGGING_LEVEL_WARNING", "ort_api",
                      ctypes.byref(env))

        session_opts = ctypes.c_void_p()
        api.CreateSessionOptions(ctypes.byref(session_opts))

        cuda_opts = OrtCUDAProviderOptions(
            device_id=0,
            cudnn_conv_algo_search=api.get_enum_value("DEFAULT"),
            cuda_mem_limit=np.iinfo(ctypes.c_size_t).max,
            do_copy_in_default_stream=1,
            has_user_compute_stream=0,
            user_compute_stream=0)

        api.SessionOptionsAppendExecutionProvider_CUDA(session_opts,
                                                       ctypes.byref(cuda_opts))
        session = ctypes.c_void_p()
        api.CreateKernelSession(session_opts, ctypes.byref(session), 12)
        api.ReleaseKernelSession(session)

        api.ReleaseSessionOptions(session_opts)
        api.ReleaseEnv(env)
