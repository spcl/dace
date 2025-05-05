# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, registry
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.sdfg.sdfg import SDFG


@registry.autoregister_params(type=dtypes.InstrumentationType.NVTX)
class NVTXProvider(InstrumentationProvider):
    """ Timing instrumentation that adds NVTX range to the top level SDFG. """

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream, codegen) -> None:
        top_level_sdfg = sdfg.parent is None
        if top_level_sdfg:
            sdfg.append_global_code('#include <nvtx3/nvToolsExt.h>', 'frame')
            local_stream.write(f'nvtxRangePush("{sdfg.name}");')

    def on_sdfg_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        top_level_sdfg = sdfg.parent is None
        if top_level_sdfg:
            local_stream.write('nvtxRangePop();')
