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
    """ Timing instrumentation that adds NVTX ranges. """
    def __init__(self):
        self.backend = common.get_gpu_backend()
        super().__init__()

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream, codegen) -> None:
        # global_stream.write('#include <nvtx3/nvToolsExt.h> // [NVTXProvider][on_sdfg_begin] global_stream')

        # local_stream.write('// [NVTXProvider][on_sdfg_begin] local_stream')
        # sdfg.append_global_code('// [NVTXProvider][on_sdfg_begin] append_global_code cuda', 'cuda')
        sdfg.append_global_code('#include <nvtx3/nvToolsExt.h> // [NVTXProvider][on_sdfg_begin] append_global_code frame', 'frame')
        # sdfg.append_init_code('// [NVTXProvider][on_sdfg_begin] append_init_code cuda', 'cuda')
        # sdfg.append_init_code('// [NVTXProvider][on_sdfg_begin] append_init_code frame', 'frame')
        # sdfg.append_exit_code('// [NVTXProvider][on_sdfg_begin] append_exit_code cuda', 'cuda')
        # sdfg.append_exit_code('// [NVTXProvider][on_sdfg_begin] append_exit_code frame', 'frame')

    # def on_sdfg_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
    #     global_stream.write('// [NVTXProvider][on_sdfg_end] global_stream')
    #     local_stream.write('// [NVTXProvider][on_sdfg_end] local_stream')
    #     sdfg.append_global_code('// [NVTXProvider][on_sdfg_end] append_global_code cuda', 'cuda')
    #     sdfg.append_global_code('// [NVTXProvider][on_sdfg_end] append_global_code frame', 'frame')
    #     sdfg.append_init_code('// [NVTXProvider][on_sdfg_end] append_init_code', 'cuda')
    #     sdfg.append_init_code('// [NVTXProvider][on_sdfg_end] append_init_code frame', 'frame')
    #     sdfg.append_exit_code('// [NVTXProvider][on_sdfg_end] append_exit_code', 'cuda')
    #     sdfg.append_exit_code('// [NVTXProvider][on_sdfg_end] append_exit_code frame', 'frame')
