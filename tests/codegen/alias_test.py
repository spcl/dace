# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests aliasing analysis. """
import re

import pytest
import dace

#: A generated function definition, which is where aliasing analysis decides the qualifier.
SIGNATURE = re.compile(r'^\s*(?:DACE_EXPORTED\s+)?(?:inline\s+)?void\s+\w+\(')


def restrict_on_parameters(code: str) -> int:
    """``__restrict__`` occurrences on function parameters.

    Counting every occurrence in the file also counts the qualifier on local pointers the code
    generator introduces inside a body (a copy's endpoints, say). Those are a lowering detail and
    say nothing about whether two of the program's arrays may alias, which is what these assert.
    """
    return sum(line.count('__restrict__') for line in code.splitlines() if SIGNATURE.match(line))


AliasedArray = dace.data.Array(dace.float64, (20, ), may_alias=True)


@pytest.mark.parametrize('may_alias', (False, True))
def test_simple_program(may_alias):
    desc = AliasedArray if may_alias else dace.float64[20]

    @dace.program
    def tester(a: desc, b: desc, c: desc):
        c[:] = a + b

    code = tester.to_sdfg().generate_code()[0]

    if may_alias:
        assert restrict_on_parameters(code.clean_code) == 0
    else:
        assert restrict_on_parameters(code.clean_code) >= 3


@pytest.mark.parametrize('may_alias', (False, True))
def test_local_pointer_into_a_tasklet(may_alias):
    """The local pointer a tasklet reads an array through is qualified only when the descriptor
    allows it. ``may_alias`` is a promise the SDFG makes -- the persistent BFS frontiers are two
    names for memory that gets swapped between iterations -- so qualifying that local hands the
    compiler a licence the program denies, and the reordering it then does is a miscompile.
    """
    sdfg = dace.SDFG(f'alias_local_pointer_{may_alias}')
    sdfg.add_array('A', [20], dace.float64, may_alias=may_alias)
    sdfg.add_array('B', [20], dace.float64)
    state = sdfg.add_state('s0')
    tasklet = state.add_tasklet('copy20', {'_a'}, {'_b'},
                                'for (int i = 0; i < 20; ++i) _b[i] = _a[i];',
                                language=dace.Language.CPP)
    tasklet.in_connectors['_a'] = dace.dtypes.pointer(dace.float64)
    tasklet.out_connectors['_b'] = dace.dtypes.pointer(dace.float64)
    state.add_edge(state.add_read('A'), None, tasklet, '_a', dace.Memlet('A[0:20]'))
    state.add_edge(tasklet, '_b', state.add_write('B'), None, dace.Memlet('B[0:20]'))

    code = sdfg.generate_code()[0].clean_code
    declaration = next(line for line in code.splitlines() if '_a = ' in line)
    assert ('__restrict__' in declaration) is (not may_alias), declaration


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

    code = tester.to_sdfg(simplify=False).generate_code()[0]

    # Restrict keyword should show up once per aliased array, even if nested programs say otherwise
    assert restrict_on_parameters(code.clean_code) == 4  # = [__program, tester, interim, nested]


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

    code = tester.to_sdfg(simplify=False).generate_code()[0]

    # Restrict keyword should never show up in "nested", since arrays are aliased,
    # but should show up in [__program, tester, interim]
    assert restrict_on_parameters(code.clean_code) == 3


if __name__ == '__main__':
    test_simple_program(False)
    test_simple_program(True)
    test_local_pointer_into_a_tasklet(False)
    test_local_pointer_into_a_tasklet(True)
    test_multi_nested()
    test_inference()
