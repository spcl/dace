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

    # NOTE: Instead of checking if the SDFG is valid one should first, with `validate_undefs`
    #   set to `False` if validation passes. Then enable it and ensure that the validation
    #   fails, because as it stands now even when the feature is disabled validation passes.
    with dace.config.set_temporary('experimental', 'validate_undefs', value=False):
        assert sdfg.is_valid()
    with dace.config.set_temporary('experimental', 'validate_undefs', value=True):
        assert sdfg.is_valid()


if __name__ == '__main__':
    test_inline_symbol()
