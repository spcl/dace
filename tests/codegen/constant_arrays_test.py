# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.codegen import codegen


def test_nsdfg_input():
    """ Tests constexpr array passed as input argument to a NestedSDFG. """

    @dace.program
    def constexpr_nsdfg():
        a = np.array([1., 2., 3.])
        b = np.max(a)

    with dace.config.set_temporary('compiler', 'inline_sdfgs', value=False):
        constexpr_nsdfg()


def test_tasklet_input_cpu():
    """ Tests constexpr array passed as input argument to a Tasklet (CPU)."""

    @dace.program
    def constexpr_tasklet_cpu():
        a = np.array([1., 2., 3.])
        b = np.max(a)

    with dace.config.set_temporary('optimizer', 'autooptimize', value=True):
        constexpr_tasklet_cpu()


@pytest.mark.parametrize('value,expected', [
    (np.array([True, False, True]), ['true', 'false']),
    (np.array([1 + 2j, 3 - 4j], dtype=np.complex64), ['dace::complex64(1.0, 2.0)', 'dace::complex64(3.0, -4.0)']),
    (np.array([1 + 2j, 3 - 4j]), ['dace::complex128(1.0, 2.0)', 'dace::complex128(3.0, -4.0)']),
])
def test_constant_array_elements_are_cpp_literals(value, expected):
    """A constant array is emitted through sym2cpp, so no element carries Python syntax."""
    sdfg = dace.SDFG(f'const_literals_{value.dtype}')
    sdfg.add_constant('cst', value)
    sdfg.add_array('out', [1], dace.int32)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 0')
    state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))

    lines = [line for c in codegen.generate_code(sdfg) for line in c.clean_code.splitlines()]
    decl = next(line for line in lines if 'cst[' in line)
    assert 'True' not in decl and 'False' not in decl and 'j)' not in decl, decl
    for literal in expected:
        assert literal in decl, (literal, decl)


if __name__ == "__main__":
    test_nsdfg_input()
    test_tasklet_input_cpu()
