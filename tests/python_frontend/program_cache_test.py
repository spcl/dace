# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol("N", dtype=dace.int64)


def test_cache_same_args():
    """
    Tests that two subsequent calls to a program does not trigger
    recompilation.
    """

    @dace.program
    def test(x):
        return x * x

    test(5)
    assert len(test._cache.cache) == 1
    test(5)
    assert len(test._cache.cache) == 1


def test_cache_different_args():
    """
    Tests that two subsequent calls to a program with different shapes does
    trigger recompilation.
    """

    @dace.program
    def test(x):
        return x * x

    a = np.random.rand(2)
    b = np.random.rand(3)
    ra = test(a)
    assert len(test._cache.cache) == 1
    rb = test(b)
    assert len(test._cache.cache) == 2

    assert np.allclose(a * a, ra)
    assert np.allclose(b * b, rb)


def test_cache_return_values():

    @dace.program
    def test(x):
        return x * x

    a = test(5)
    b = test(6)

    assert a == 25 and b == 36


def test_cache_argument_names():

    @dace.program
    def test(C: dace.float32[20], A: dace.float64[30]):
        A *= 5
        C *= 2

    sdfg = test.to_sdfg()
    a = np.random.rand(20).astype(np.float32)
    c = np.random.rand(30)
    rega = a * 2
    regc = c * 5
    sdfg(a, c)

    assert np.allclose(a, rega) and np.allclose(c, regc)


def test_cache_skips_autoopt_specialized_symbols():
    """
    Tests that a call whose symbol values were baked in by auto-optimization is never served to a
    call with another value of the symbol.
    """

    @dace.program(auto_optimize=True)
    def test(x: dace.float64[10]):
        if N == 0:
            x[:] = 1.0
        else:
            x[:] = 2.0

    for n, expected in ((0, 1.0), (5, 2.0)):
        x = np.zeros(10, dtype=np.float64)
        test(x, N=n)
        assert np.allclose(x, expected), f"N={n} ran an SDFG specialized for another value: {x[0]}"


@dace.program
def doubled(a: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        out[i] = a[i] * 2.0


@dace.program
def calls_doubled(a: dace.float64[N], out: dace.float64[N]):
    doubled(a, out)


def _compiled_entries(program) -> int:
    return sum(entry.compiled_sdfg is not None for entry in program._cache.cache.values())


def test_autooptimized_program_is_not_reused_for_other_symbol_values():
    """Auto-optimization specializes the SDFG for the call's symbol values, so a call with other values recompiles."""
    with dace.config.set_temporary("optimizer", "autooptimize", value=True):
        doubled._cache.clear()
        for n in (5, 6, 5):
            out = np.zeros(n)
            doubled(np.arange(n, dtype=np.float64), out)
            assert np.allclose(out, 2.0 * np.arange(n)), f"wrong result for N={n}"
        assert _compiled_entries(doubled) == 2, "a repeated call with the same values must hit the cache"


def test_nesting_an_autooptimized_program_does_not_inherit_its_symbol_values():
    """A program nesting one that was auto-optimized for N=5 must still see N as a symbol."""
    with dace.config.set_temporary("optimizer", "autooptimize", value=True):
        doubled._cache.clear()
        calls_doubled._cache.clear()
        doubled(np.arange(5, dtype=np.float64), np.zeros(5))
        out = np.zeros(6)
        calls_doubled(np.arange(6, dtype=np.float64), out)
        assert np.allclose(out, 2.0 * np.arange(6))


if __name__ == "__main__":
    test_cache_same_args()
    test_cache_different_args()
    test_cache_return_values()
    test_cache_argument_names()
    test_cache_skips_autoopt_specialized_symbols()
    test_autooptimized_program_is_not_reused_for_other_symbol_values()
    test_nesting_an_autooptimized_program_does_not_inherit_its_symbol_values()
