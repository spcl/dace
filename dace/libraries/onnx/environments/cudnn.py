import os
import logging

import dace
import dace.library
from dace.libraries.standard import CUDA

log = logging.getLogger(__name__)


@dace.library.environment
class cuDNN:
    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []
    state_fields = ["dace::cudnn::CudnnHandle *cudnn_handle;"]
    dependencies = [CUDA]

    headers = {
        'cuda': ["../include/dace_cudnn.h"],
        'frame': ["../include/dace_cudnn.h"]
    }
    init_code = """
        __state->cudnn_handle = new dace::cudnn::CudnnHandle;
    """
    finalize_code = """
        delete __state->cudnn_handle;
    """

    @staticmethod
    def handle_setup_code(node, init_stream=True):
        location = node.location
        if not location or "gpu" not in node.location:
            location = 0
        else:
            try:
                location = int(location["gpu"])
            except ValueError:
                raise ValueError("Invalid GPU identifier: {}".format(location))

        code = f"""\
const int __dace_cuda_device = {location};
cudnnHandle_t &__dace_cudnn_handle = __state->cudnn_handle->Get(__dace_cuda_device);
{"cudnnSetStream(__dace_cudnn_handle, __dace_current_stream);" if init_stream else ''}\n"""

        return code.format(location=location)

    @staticmethod
    def cmake_includes():
        if 'CUDNN_HOME' in os.environ:
            return [os.path.join(os.environ['CUDNN_HOME'], 'include')]
        else:
            log.warning("CUDNN_HOME was not set, compilation may fail")
            return []

    @staticmethod
    def cmake_libraries():
        if 'CUDNN_HOME' in os.environ:
            prefix = dace.Config.get('compiler', 'library_prefix')
            suffix = dace.Config.get('compiler', 'library_extension')
            libfile = os.path.join(os.environ['CUDNN_HOME'], 'lib64',
                                   prefix + 'cudnn.' + suffix)
            if os.path.isfile(libfile):
                return [libfile]

        log.warning("CUDNN_HOME was not set, compilation may fail")
        return ['cudnn']
