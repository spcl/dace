# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Union
from dace import config, dtypes, registry
from dace.codegen.prettycode import CodeIOStream
from dace.sdfg import nodes, is_devicelevel_gpu
from dace.codegen import common
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ControlFlowRegion, SDFGState


@registry.autoregister_params(type=dtypes.InstrumentationType.NVTX)
class NVTXProvider(InstrumentationProvider):
    """ Timing instrumentation that adds NVTX range to the top level SDFG. """
    def __init__(self):
        self.backend = common.get_gpu_backend()
        super().__init__()

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream, codegen) -> None:
        top_level_sdfg = sdfg.parent is None
        if top_level_sdfg:
            sdfg.append_global_code('#include <nvtx3/nvToolsExt.h>', 'frame')
            local_stream.write(f'nvtxRangePushA("{sdfg.name}");')

    def on_sdfg_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        top_level_sdfg = sdfg.parent is None
        if top_level_sdfg:
            local_stream.write('nvtxRangePop();')
