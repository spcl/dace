# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import os

from dace import dtypes, registry
from dace.codegen import common
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ControlFlowRegion, SDFGState


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
        self.enable_rocTX = any(os.path.isfile(path)
                                for path in roctx_header_paths) and os.path.isfile(roctx_library_path)
        super().__init__()

    def print_range_push(self, name: str, stream: CodeIOStream) -> None:
        if self.backend == 'cuda':
            stream.write('#ifndef __CUDA_ARCH__')
            stream.write(f'nvtxRangePush("{name}");')
            stream.write('#endif')
        elif self.backend == 'hip':
            if self.enable_rocTX:
                stream.write('#ifndef __HIP_DEVICE_COMPILE__')
                stream.write(f'roctxRangePush("{name}");')
                stream.write('#endif')
        else:
            raise NameError(f'GPU backend "{self.backend}" not recognized')

    def print_range_pop(self, stream: CodeIOStream) -> None:
        if self.backend == 'cuda':
            stream.write('#ifndef __CUDA_ARCH__')
            stream.write('nvtxRangePop();')
            stream.write('#endif')
        elif self.backend == 'hip':
            if self.enable_rocTX:
                stream.write('#ifndef __HIP_DEVICE_COMPILE__')
                stream.write('roctxRangePop();')
                stream.write('#endif')
        else:
            raise NameError(f'GPU backend "{self.backend}" not recognized')

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream, codegen) -> None:
        top_level = sdfg.parent is None
        if top_level:
            if self.backend == 'cuda':
                sdfg.append_global_code('#include <nvtx3/nvToolsExt.h>', 'frame')
            elif self.backend == 'hip':
                if self.enable_rocTX:
                    sdfg.append_global_code('#include <roctx.h>', 'frame')
            else:
                raise NameError('GPU backend "%s" not recognized' % self.backend)

        self.print_range_push(f'sdfg_{sdfg.name}', local_stream)

    def on_sdfg_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        self.print_range_pop(local_stream)

    def on_state_begin(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, local_stream: CodeIOStream,
                       global_stream: CodeIOStream) -> None:
        self.print_range_push(f'state_{state.label}', local_stream)

    def on_state_end(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, local_stream: CodeIOStream,
                     global_stream: CodeIOStream) -> None:
        self.print_range_pop(local_stream)
