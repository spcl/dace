# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Both compiled-SDFG backends satisfy the shared ``CompiledSDFGProtocol``."""
import warnings

import numpy as np
import pytest

import dace
from dace.codegen.compiled_sdfg import CompiledSDFG, CompiledSDFGProtocol, CtypesCompiledSDFG
from dace.config import set_temporary


def test_compiled_sdfg_protocol_ctypes():
    N = dace.symbol('N')

    @dace.program
    def protocol_probe_ctypes(A: dace.float64[N]):
        A[:] = A + 1.0

    # Pin the ctypes interface so this exercises the ctypes backend even when the
    # CI leg exports DACE_compiler_interface=nanobind (the env var wins over
    # set_temporary in Config.get, so drop it first).
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv('DACE_compiler_interface', raising=False)
        with set_temporary('compiler', 'interface', value='ctypes'):
            csdfg = protocol_probe_ctypes.to_sdfg().compile()
    assert isinstance(csdfg, CompiledSDFGProtocol)
    assert type(csdfg) is CtypesCompiledSDFG


def test_ctypes_compiled_sdfg_rename_and_deprecation():
    """The ctypes backend is ``CtypesCompiledSDFG``; ``CompiledSDFG`` is a
    deprecated subclass kept for compatibility (warns only on construction)."""
    # The deprecated name is a subclass of the renamed class.
    assert issubclass(CompiledSDFG, CtypesCompiledSDFG)

    N = dace.symbol('N')

    @dace.program
    def rename_probe(A: dace.float64[N]):
        A[:] = A + 1.0

    # Pin the ctypes interface so this exercises the ctypes rename even when the
    # CI leg exports DACE_compiler_interface=nanobind (the env var wins over
    # set_temporary in Config.get, so drop it first).
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv('DACE_compiler_interface', raising=False)
        with set_temporary('compiler', 'interface', value='ctypes'):
            # A normal ctypes compile returns the renamed class and does NOT warn.
            with warnings.catch_warnings():
                warnings.simplefilter('error', DeprecationWarning)
                csdfg = rename_probe.to_sdfg().compile()
            assert type(csdfg) is CtypesCompiledSDFG

            # Constructing the deprecated ``CompiledSDFG`` directly warns.
            csdfg(A=np.zeros(8), N=np.int32(8))  # ensure the library is loaded
            with pytest.warns(DeprecationWarning, match='CompiledSDFG is deprecated'):
                deprecated = CompiledSDFG(csdfg.sdfg, csdfg._lib, csdfg.sdfg.arg_names)
            assert isinstance(deprecated, CtypesCompiledSDFG)


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
    test_ctypes_compiled_sdfg_rename_and_deprecation()
    test_compiled_sdfg_protocol_nanobind()
