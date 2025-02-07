"""
Regression tests to check that init only contains scalars in the signature
"""

import dace
from dace import dtypes


def test_init_contains_only_symbols_cpu():
    sdfg = dace.SDFG("test_init_contains_only_symbols_cpu")
    sdfg.add_scalar("A_useless_scalar", dace.float32)
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("M", dace.int64)
    state = sdfg.add_state()
    state.add_tasklet(
        "tasklet", {}, {},
        "// Hello this is my tasklet",
        dtypes.Language.CPP,
        code_init=
        'if (N != 123 || M != 456) { printf("N: %ld, M: %ld\\n", N, M); exit(1);}'
    )
    sdfg(N=123, A_useless_scalar=1.0, M=456)
