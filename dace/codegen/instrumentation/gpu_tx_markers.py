# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import os

from dace import dtypes, registry
from dace.codegen import common
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.sdfg.sdfg import SDFG


@registry.autoregister_params(type=dtypes.InstrumentationType.GPU_TX_MARKERS)
class GPUTXMarkersProvider(InstrumentationProvider):
    """ Timing instrumentation that adds NVTX range to the top level SDFG. """

    def __init__(self):
        self.backend = common.get_gpu_backend()
        # Check if ROCm TX libraries and headers are available
        rocm_path = os.getenv('ROCM_PATH', '/opt/rocm')
        roctx_header_paths = [
            os.path.join(rocm_path, 'roctracer/include/roctx.h'),
            os.path.join(rocm_path, 'include/roctracer/roctx.h')
        ]
        roctx_library_path = os.path.join(rocm_path, 'lib', 'libroctx64.so')
        self.enable_rocTX = any(os.path.isfile(path) for path in roctx_header_paths) and os.path.isfile(roctx_library_path)
        super().__init__()

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream, codegen) -> None:
        top_level_sdfg = sdfg.parent is None
        if top_level_sdfg:
            if self.backend == 'cuda':
                sdfg.append_global_code('#include <nvtx3/nvToolsExt.h>', 'frame')
                local_stream.write(f'nvtxRangePush("{sdfg.name}");')
            elif self.backend == 'hip':
                if self.enable_rocTX:
                    sdfg.append_global_code('#include <roctx.h>', 'frame')
                    local_stream.write(f'roctxRangePush("{sdfg.name}");')
            else:
                raise NameError('GPU backend "%s" not recognized' % self.backend)

    def on_sdfg_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        top_level_sdfg = sdfg.parent is None
        if top_level_sdfg:
            if self.backend == 'cuda':
                local_stream.write('nvtxRangePop();')
            elif self.backend == 'hip':
                if self.enable_rocTX:
                    local_stream.write('roctxRangePop();')
            else:
                raise NameError('GPU backend "%s" not recognized' % self.backend)
