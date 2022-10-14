# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests aliasing analysis. """
import pytest
import dace

AliasedArray = dace.data.Array(dace.float64, (20, ), may_alias=True)


@pytest.mark.parametrize('may_alias', (False, True))
def test_simple_program(may_alias):
    desc = AliasedArray if may_alias else dace.float64[20]

    @dace.program
    def tester(a: desc, b: desc, c: desc):
        c[:] = a + b

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

    code = tester.to_sdfg(simplify=False).generate_code()[0]

    # Restrict keyword should never show up in "nested", since arrays are aliased,
    # but should show up in [__program, tester, interim]
    assert code.clean_code.count('__restrict__') == 3


if __name__ == '__main__':
    test_simple_program(False)
    test_simple_program(True)
    test_multi_nested()
    test_inference()
