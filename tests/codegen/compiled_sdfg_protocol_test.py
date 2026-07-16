# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Both compiled-SDFG backends satisfy the shared ``CompiledSDFGProtocol``."""
import dace
from dace.codegen.compiled_sdfg import CompiledSDFGProtocol
from dace.config import set_temporary


def test_compiled_sdfg_protocol_ctypes():
    N = dace.symbol('N')

    @dace.program
    def protocol_probe_ctypes(A: dace.float64[N]):
        A[:] = A + 1.0

    csdfg = protocol_probe_ctypes.to_sdfg().compile()
    assert isinstance(csdfg, CompiledSDFGProtocol)


def test_compiled_sdfg_protocol_nanobind():
    N = dace.symbol('N')

    @dace.program
    def protocol_probe_nanobind(A: dace.float64[N]):
        A[:] = A + 1.0

    with set_temporary('compiler', 'interface', value='nanobind'):
        csdfg = protocol_probe_nanobind.to_sdfg().compile()
    assert isinstance(csdfg, CompiledSDFGProtocol)


if __name__ == '__main__':
    test_compiled_sdfg_protocol_ctypes()
    test_compiled_sdfg_protocol_nanobind()
