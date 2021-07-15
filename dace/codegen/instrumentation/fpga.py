# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, registry
from dace.codegen.instrumentation.provider import InstrumentationProvider


@registry.autoregister_params(type=dtypes.InstrumentationType.FPGA)
class FPGAInstrumentationProvider(InstrumentationProvider):
    """Dummy provider to register the instrumentation type."""
    def __init__(self):
        super().__init__()
