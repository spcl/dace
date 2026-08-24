# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock

LEN_1D = dace.symbol('LEN_1D')


@dace.program
def s3113_max_abs(a: dace.float64[LEN_1D], b: dace.float64[2]):
    maxv = dace.float64(0)
    maxv = abs(a[0])
    for i in range(LEN_1D):
        av = abs(a[i])
        if av > maxv:
            maxv = av
    b[0] = maxv


@dace.program
def simple_condition(i: dace.int32):
    return i % 2 == 0


@dace.program
def simple_condition2(fib: dace.int32, F: dace.int32, i: dace.int32, N: dace.int32):
    return fib < F and i < N


@dace.program
def simple_if(A: dace.int32[10]):
    for i in range(10):
        if i % 2 == 0:
            A[i] += 2 * i
        else:
            A[i] += 3 * i


@dace.program
def chained_bounds(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        if 0.0 <= a[i] <= 1.0:
            b[i] = a[i]
        else:
            b[i] = -1.0


@dace.program
def chained_symbolic_bounds(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        if 0 <= i < LEN_1D - 1:
            b[i] = a[i + 1]
        else:
            b[i] = 0.0


@dace.program
def chained_computed_middle(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        if 0.0 <= a[i] * 2.0 < 1.0:
            b[i] = 1.0
        else:
            b[i] = 0.0


def test_simple_if():
    A = np.random.randint(1, 10, size=(10, ), dtype=np.int32)
    ref = np.copy(A)
    for i in range(10):
        if i % 2 == 0:
            ref[i] += 2 * i
        else:
            ref[i] += 3 * i
    simple_if(A)
    assert (np.array_equal(A, ref))


@dace.program
def call_if(A: dace.int32[10]):
    for i in range(10):
        if simple_condition(i):
            A[i] += 2 * i
        else:
            A[i] += 3 * i


def test_call_if():
    A = np.random.randint(1, 10, size=(10, ), dtype=np.int32)
    ref = np.copy(A)
    for i in range(10):
        if i % 2 == 0:
            ref[i] += 2 * i
        else:
            ref[i] += 3 * i
    sdfg = call_if.to_sdfg()
    call_if(A)
    assert (np.array_equal(A, ref))


@dace.program
def call_if2(A: dace.int32[10]):
    A[0] = 0
    i = np.int32(1)
    fib = np.int32(1)
    while True:
        if simple_condition2(fib, 50, i, 10):
            A[i] = fib
            fib += A[i]
            i += 1
        else:
            break


def test_call_if2():
    A = np.random.randint(1, 10, size=(10, ), dtype=np.int32)
    ref = np.copy(A)
    ref[0] = 0
    i = 1
    fib = 1
    while fib < 50 and i < 10:
        ref[i] = fib
        fib += ref[i]
        i += 1
    call_if2(A)
    assert (np.array_equal(A, ref))


@dace.program
def simple_while(A: dace.int32[10]):
    i = 0
    while i < 10:
        A[i] += 2 * i
        i += 1


def test_simple_while():
    A = np.random.randint(1, 10, size=(10, ), dtype=np.int32)
    ref = np.copy(A)
    for i in range(10):
        ref[i] += 2 * i
    simple_while(A)
    assert (np.array_equal(A, ref))


@dace.program
def call_while(A: dace.int32[10]):
    A[0] = 0
    i = np.int32(1)
    fib = np.int32(1)
    while simple_condition2(fib, 50, i, 10):
        A[i] = fib
        fib += A[i]
        i += 1


def test_call_while():
    A = np.random.randint(1, 10, size=(10, ), dtype=np.int32)
    ref = np.copy(A)
    ref[0] = 0
    i = 1
    fib = 1
    while fib < 50 and i < 10:
        ref[i] = fib
        fib += ref[i]
        i += 1
    call_while(A)
    assert (np.array_equal(A, ref))


def test_if_return_both():

    @dace.program
    def if_return_both(i: dace.int64):
        if i < 5:
            return 0
        else:
            return 1
        return 2

    assert if_return_both(4)[0] == 0
    assert if_return_both(7)[0] == 1


def test_if_return_chain():

    @dace.program
    def if_return_chain(i: dace.int64):
        if i < 2:
            return 0
        if i < 4:
            return 1
        if i < 6:
            return 2
        if i < 8:
            return 3
        return 4

    assert if_return_chain(0)[0] == 0
    assert if_return_chain(2)[0] == 1
    assert if_return_chain(4)[0] == 2
    assert if_return_chain(7)[0] == 3
    assert if_return_chain(15)[0] == 4


def test_if_test_call():

    @dace.program
    def if_test_call(a, b):
        if bool(a):
            return a
        else:
            return b

    assert if_test_call(0, 2)[0] == if_test_call.f(0, 2)
    assert if_test_call(1, 2)[0] == if_test_call.f(1, 2)


_K = dace.symbol("guard_flag")


@dace.program
def guard_only_symbol(a: dace.float64[10]):
    if _K > 0:
        for i in range(10):
            a[i] = a[i] + 1.0


def test_guard_only_symbol_is_registered():
    """A symbol used ONLY in an if-guard (``if K > 0``) -- never in an array shape or a
    loop condition -- must still be registered as an SDFG symbol so ``arglist`` / codegen
    do not raise ``KeyError`` on it (TSVC ``fuse_move_ifs`` / ``config_select_branch``)."""
    sdfg = guard_only_symbol.to_sdfg(simplify=True)
    assert "guard_flag" in sdfg.symbols
    sdfg.arglist()  # must not raise KeyError('guard_flag')

    a = np.ones(10, np.float64)
    sdfg(a=a, guard_flag=1)
    assert np.allclose(a, 2.0), "guard true -> body runs"
    b = np.ones(10, np.float64)
    sdfg(a=b, guard_flag=0)
    assert np.allclose(b, 1.0), "guard false -> body skipped"


def test_simple_guard_reads_the_current_version():
    """A ``Name``-only guard is unparsed verbatim, so it must resolve through the parser's
    variable map: reassigning versions a scalar (``maxv`` -> ``maxv_0``) and the raw spelling
    reads the stale pre-assignment data. TSVC s3113 is a max-reduction, so a guard stuck on the
    dead ``maxv`` is always true and ``b[0]`` silently becomes ``abs(a[-1])``, not the max."""
    sdfg = s3113_max_abs.to_sdfg(simplify=False)
    cond, branch = next((c.as_string, br) for blk, _ in sdfg.all_nodes_recursive() if isinstance(blk, ConditionalBlock)
                        for c, br in blk.branches if c is not None)
    updated = {e.dst.data for st in branch.all_states() for e in st.edges() if isinstance(e.dst, nodes.AccessNode)}
    assert updated <= dace.symbolic.free_symbols_and_functions(cond), \
        f"guard {cond} does not name the accumulator {sorted(updated)} its body updates"

    a = np.random.default_rng(0).random(64)
    b = np.zeros(2)
    sdfg(a=a, b=b, LEN_1D=64)
    assert np.isclose(b[0], np.abs(a).max()), "guarded max reduction must yield max(|a|)"


def test_a_chained_compare_is_the_conjunction_of_its_links():
    """``0.0 <= a[i] <= 1.0`` is the commonest bounds test in ported code and used to raise a
    BODYLESS NotImplementedError. Python defines it as ``(0.0 <= a[i]) and (a[i] <= 1.0)``."""
    a = np.linspace(-1.0, 2.0, 16)
    b = np.zeros(16)
    chained_bounds(a=a, b=b, LEN_1D=16)
    assert np.allclose(b, np.where((a >= 0.0) & (a <= 1.0), a, -1.0))


def test_a_chained_compare_over_symbols_guards_the_branch():
    """The links may be symbolic relations rather than data: ``0 <= i < LEN_1D - 1`` must fold into
    ONE guard that is the conjunction of the two links -- the form sympy manipulates and codegen
    lowers to ``&&``, rather than a relation the conjunction cannot take."""
    sdfg = chained_symbolic_bounds.to_sdfg(simplify=False)
    guards = [
        cond.as_string for blk, _ in sdfg.all_nodes_recursive() if isinstance(blk, ConditionalBlock)
        for cond, _branch in blk.branches if cond is not None
    ]
    assert guards == ['((0 <= i) and (i < (LEN_1D - 1)))'], guards
    assert 'if (((0 <= i) && (i < (LEN_1D - 1))))' in sdfg.generate_code()[0].clean_code

    a = np.arange(16, dtype=np.float64)
    b = np.zeros(16)
    chained_symbolic_bounds(a=a, b=b, LEN_1D=16)
    expected = np.concatenate([a[1:], [0.0]])
    assert np.allclose(b, expected)


def test_a_chained_compare_reads_each_operand_once():
    """Python evaluates the middle operand of ``x < y < z`` ONCE. Rebuilding the chain from AST
    nodes would visit it per link, so ``a[i] * 2.0`` would be computed -- and ``a[i]`` read --
    twice. One multiplying tasklet is the whole point of evaluating the operands up front."""
    sdfg = chained_computed_middle.to_sdfg(simplify=False)
    products = [
        n.label for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet) and '*' in n.code.as_string
    ]
    assert len(products) == 1, f"the middle operand is computed {len(products)} times: {products}"

    a = np.linspace(-1.0, 1.0, 16)
    b = np.zeros(16)
    chained_computed_middle(a=a, b=b, LEN_1D=16)
    assert np.allclose(b, np.where((a * 2.0 >= 0.0) & (a * 2.0 < 1.0), 1.0, 0.0))


if __name__ == "__main__":
    test_simple_if()
    test_call_if()
    test_call_if2()
    test_simple_while()
    test_call_while()
    test_if_return_both()
    test_if_return_chain()
    test_if_test_call()
    test_guard_only_symbol_is_registered()
    test_simple_guard_reads_the_current_version()
    test_a_chained_compare_is_the_conjunction_of_its_links()
    test_a_chained_compare_over_symbols_guards_the_branch()
    test_a_chained_compare_reads_each_operand_once()
