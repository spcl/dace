# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

B, C, E, F = (dace.symbol(s) for s in 'BCEF')


@dace.program
def nested(A: dace.float32[B, C]):
    return A * 2


@dace.program
def inline_symbol(D: dace.float32[E, F]):
    return nested(D) * 5


def test_inline_symbol():
    sdfg = inline_symbol.to_sdfg(simplify=False)

    # NOTE: The original test only checked if the SDFG was valid when `validate_undefs` was set
    #   to `True`. However, since the SDFG is also valid when `validate_undefs` is set to `False`,
    #   as it can be seen below, this test actually does not serves any meaning. It would be more
    #   meaningful if one of the cases, probably the `validate_undefs=True` case, would fail.
    with dace.config.set_temporary('experimental', 'validate_undefs', value=False):
        sdfg.validate()
    with dace.config.set_temporary('experimental', 'validate_undefs', value=True):
        sdfg.validate()


if __name__ == '__main__':
    test_inline_symbol()
