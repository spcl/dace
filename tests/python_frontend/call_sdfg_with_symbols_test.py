# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

N = dace.symbol("N")


@dace.program
def add_one(A: dace.int64[N, N], result: dace.int64[N, N]):
    result[:] = A + 1


def call_test():
    @dace.program
    def add_one_more(A: dace.int64[N, N]):
        result = dace.define_local([N, N], dace.int64)
        add_one(A, result)
        return result + 1

    A = np.random.randint(0, 10, size=(11, 11), dtype=np.int64)
    result = add_one_more(A=A.copy())
    assert np.allclose(result, A + 2)


def call_sdfg_test():
    add_one_sdfg = add_one.to_sdfg()

    @dace.program
    def add_one_more(A: dace.int64[N, N]):
        result = dace.define_local([N, N], dace.int64)
        add_one_sdfg(A=A, result=result)
        return result + 1

    A = np.random.randint(0, 10, size=(11, 11), dtype=np.int64)
    result = add_one_more(A=A.copy())
    assert np.allclose(result, A + 2)


other_N = dace.symbol("N")


@dace.program
def add_one_other_n(A: dace.int64[other_N - 1, other_N - 1],
                    result: dace.int64[other_N - 1, other_N - 1]):
    result[:] = A + 1


def call_sdfg_same_symbol_name_test():
    add_one_sdfg = add_one_other_n.to_sdfg()

    @dace.program
    def add_one_more(A: dace.int64[N, N]):
        result = dace.define_local([N, N], dace.int64)
        add_one_sdfg(A=A, result=result)
        return result + 1

    A = np.random.randint(0, 10, size=(11, 11), dtype=np.int64)
    result = add_one_more(A=A.copy())
    assert np.allclose(result, A + 2)


if __name__ == "__main__":
    call_test()
    call_sdfg_test()
    call_sdfg_same_symbol_name_test()
