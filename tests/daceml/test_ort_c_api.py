import ctypes
import tempfile

import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto

from dace.libraries.ort_api import ORTCAPIInterface, OrtCUDAProviderOptions


def _minimal_onnx_model_bytes(opset: int = 12) -> bytes:
    # Graph: Y = Identity(X), X/Y are [1] float
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "min_graph", [X], [Y])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", opset)],
        ir_version=8,  # broadly supported by ORT 1.2x
        producer_name="test"
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _create_session(api: ORTCAPIInterface, env, session_opts, model_bytes: bytes):
    """
    Try CreateSessionFromArray first; fall back to CreateSession via a temp file
    if the wrapper doesn't expose CreateSessionFromArray.
    """
    session = ctypes.c_void_p()

    if hasattr(api, "CreateSessionFromArray"):
        # Build a stable ctypes buffer
        buf = (ctypes.c_char * len(model_bytes)).from_buffer_copy(model_bytes)
        api.CreateSessionFromArray(env, buf, ctypes.c_size_t(len(model_bytes)),
                                   session_opts, ctypes.byref(session))
    elif hasattr(api, "CreateSession"):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model_bytes)
            path = f.name.encode("utf-8")
        api.CreateSession(env, path, session_opts, ctypes.byref(session))
    else:
        raise RuntimeError("ORTCAPIInterface missing CreateSession* methods")

    return session


def test_basic():
    model_bytes = _minimal_onnx_model_bytes(opset=12)

    with ORTCAPIInterface() as api:
        env = ctypes.c_void_p()
        api.CreateEnv("ORT_LOGGING_LEVEL_WARNING", "ort_api", ctypes.byref(env))

        session_opts = ctypes.c_void_p()
        api.CreateSessionOptions(ctypes.byref(session_opts))

        session = _create_session(api, env, session_opts, model_bytes)

        # Clean up
        api.ReleaseSession(session)
        api.ReleaseSessionOptions(session_opts)
        api.ReleaseEnv(env)

@pytest.mark.gpu
def test_basic_gpu():
    model_bytes = _minimal_onnx_model_bytes(opset=12)

    with ORTCAPIInterface() as api:
        env = ctypes.c_void_p()
        api.CreateEnv("ORT_LOGGING_LEVEL_WARNING", "ort_api", ctypes.byref(env))

        session_opts = ctypes.c_void_p()
        api.CreateSessionOptions(ctypes.byref(session_opts))

        if hasattr(api, "CreateCUDAProviderOptions") and hasattr(api, "SessionOptionsAppendExecutionProvider_CUDA_V2"):
            # ---- V2 path (recommended) ----
            opts = ctypes.c_void_p()
            api.CreateCUDAProviderOptions(ctypes.byref(opts))

            # Build keys/values arrays (bytes)
            keys = (ctypes.c_char_p * 4)(
                b"device_id",
                b"do_copy_in_default_stream",
                b"cudnn_conv_algo_search",
                b"gpu_mem_limit",            # v2 key name
            )
            vals = (ctypes.c_char_p * 4)(
                b"0",
                b"1",
                b"DEFAULT",                  # EXHAUSTIVE|HEURISTIC|DEFAULT
                str(np.iinfo(np.uintp).max).encode("ascii"),
            )

            api.UpdateCUDAProviderOptions(opts, keys, vals, 4)
            api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_opts, opts)
            api.ReleaseCUDAProviderOptions(opts)
        else:
            # ---- v1 fallback (make sure your ctypes.Structure matches your ORT!) ----
            from dace.libraries.ort_api import OrtCUDAProviderOptions  # your struct

            cuda_opts = OrtCUDAProviderOptions()
            # Zero the whole struct to avoid junk defaults
            ctypes.memset(ctypes.byref(cuda_opts), 0, ctypes.sizeof(cuda_opts))

            cuda_opts.device_id = 0
            # If you keep using v1, use your enum getter as before:
            cuda_opts.cudnn_conv_algo_search = api.get_enum_value("OrtCudnnConvAlgoSearchDefault")
            # IMPORTANT: v1 field is often named cuda_mem_limit; 0 means 'default/unlimited' in practice.
            # If you set it, keep it small/realistic or 0 to avoid overflow into other fields.
            cuda_opts.cuda_mem_limit = 0
            # Ensure a valid arena strategy (0 or 1) if your struct has the field
            if hasattr(cuda_opts, "arena_extend_strategy"):
                cuda_opts.arena_extend_strategy = 0
            cuda_opts.do_copy_in_default_stream = 1
            cuda_opts.has_user_compute_stream = 0
            cuda_opts.user_compute_stream = 0

            api.SessionOptionsAppendExecutionProvider_CUDA(session_opts, ctypes.byref(cuda_opts))

        session = _create_session(api, env, session_opts, model_bytes)

        # Clean up
        api.ReleaseSession(session)
        api.ReleaseSessionOptions(session_opts)
        api.ReleaseEnv(env)


if __name__ == "__main__":
    # test_basic()
    test_basic_gpu()
