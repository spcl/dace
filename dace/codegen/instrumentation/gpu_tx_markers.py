# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes.util

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
        self.enable_rocTX = ctypes.util.find_library('roctx64')
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
