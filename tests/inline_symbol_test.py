# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace

B, C, E, F = (dace.symbol(s) for s in 'BCEF')


@dace.program
def nested(A: dace.float32[B, C]):
    return A * 2


@dace.program
def inline_symbol(D: dace.float32[E, F]):
    return nested(D) * 5


def test_inline_symbol():
    sdfg = inline_symbol.to_sdfg(strict=False)
    dace.Config.set('experimental', 'validate_undefs', value=True)
    sdfg.validate()


if __name__ == '__main__':
    test_inline_symbol()
