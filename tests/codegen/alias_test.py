# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests aliasing analysis. """
import pytest
import dace
from dace import config

AliasedArray = dace.data.Array(dace.float64, (20, ), may_alias=True)

# These tests count literal "__restrict__" occurrences, one per nested-SDFG call boundary kept
# in the generated code. Only the legacy generator keeps each dace.program as its own call with
# its own pointer parameter; experimental_readable inlines connectors and indexes the outer array
# directly (see compiler.cpu.implementation docs), so the count is generator-specific. Same pin
# used by cpp_test.py's test_ndcopy_to_strided_copy_declines_broadcast_source for the same reason.


@pytest.mark.parametrize('may_alias', (False, True))
def test_simple_program(may_alias):
    desc = AliasedArray if may_alias else dace.float64[20]

    @dace.program
    def tester(a: desc, b: desc, c: desc):
        c[:] = a + b

    with config.set_temporary('compiler', 'cpu', 'implementation', value='legacy'):
        code = tester.to_sdfg().generate_code()[0]

    if may_alias:
        assert code.clean_code.count('__restrict__') == 0
    else:
        assert code.clean_code.count('__restrict__') >= 3


def test_multi_nested():

    @dace.program
    def nested(a: dace.float64[20], b: dace.float64[20]):
        b[:] = a + 1

    @dace.program
    def interim(a: dace.float64[20], b: dace.float64[20]):
        nested(a, b)

    @dace.program
    def tester(a: AliasedArray, b: dace.float64[20]):
        interim(a, b)

    with config.set_temporary('compiler', 'cpu', 'implementation', value='legacy'):
        code = tester.to_sdfg(simplify=False).generate_code()[0]

    # Restrict keyword should show up once per aliased array, even if nested programs say otherwise
    assert code.clean_code.count('__restrict__') == 4  # = [__program, tester, interim, nested]


def test_inference():

    @dace.program
    def nested(a: dace.float64[2, 20], b: dace.float64[2, 20]):
        b[:] = a + 1

    @dace.program
    def interim(a: dace.float64[3, 20]):
        nested(a[:2], a[1:])

    @dace.program
    def tester(a: dace.float64[20]):
        interim(a)

    with config.set_temporary('compiler', 'cpu', 'implementation', value='legacy'):
        code = tester.to_sdfg(simplify=False).generate_code()[0]

    # Restrict keyword should never show up in "nested", since arrays are aliased,
    # but should show up in [__program, tester, interim]
    assert code.clean_code.count('__restrict__') == 3


@pytest.mark.parametrize('may_alias', (False, True))
def test_out_connector_pointer_alias(may_alias):
    """A pointer out-connector aliases the written array, so it must honor ``may_alias`` too."""
    sdfg = dace.SDFG('out_conn_alias_%s' % may_alias)
    sdfg.add_array('A', [20], dace.float64, may_alias=may_alias)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('w', {}, ['out'], 'out[0] = 1.0;', language=dace.Language.CPP)
    tasklet.out_connectors['out'] = dace.pointer(dace.float64)
    state.add_edge(tasklet, 'out', state.add_write('A'), None, dace.Memlet('A[0:20]'))

    code = sdfg.generate_code()[0].clean_code

    if may_alias:
        assert code.count('__restrict__') == 0
    else:
        assert code.count('__restrict__') >= 1


if __name__ == '__main__':
    test_simple_program(False)
    test_simple_program(True)
    test_multi_nested()
    test_inference()
    test_out_connector_pointer_alias(False)
    test_out_connector_pointer_alias(True)
