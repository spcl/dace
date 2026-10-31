# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from tests.codegen.sve.common import get_code

N = dace.symbol("N")
M = dace.symbol("M")


def test_contiguous_map():

    @dace.program
    def program(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    code = get_code(program)

    assert "svld1(" in code
    assert "svst1(" in code


def test_stride_map():

    @dace.program
    def program(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N:2]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    code = get_code(program)

    assert "svld1_gather" in code
    assert "svst1_scatter" in code
    assert "(0, 2)" in code


def test_fake_stride():

    @dace.program
    def program(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N:2]:
            with dace.tasklet:
                a << A[i / 2]
                b >> B[i]
                b = a

    code = get_code(program)

    # Load is contiguous even though it doesn't look like it
    assert "svld1(" in code

    # Store is stride
    assert "svst1_scatter" in code


def test_matrix_stride():

    @dace.program
    def program(A: dace.float64[N, M], B: dace.float64[M, N]):
        for i, j in dace.map[0:N, 0:M]:
            with dace.tasklet:
                a << A[i, j]
                b >> B[j, i]
                b = a

    code = get_code(program)

    # Contiguous load of entries
    assert "svld1" in code
    # Stride N store
    assert "svst1_scatter" in code
    assert "(0, N)" in code


def test_indirect_load_explicit():

    @dace.program
    def program(A: dace.int64[N], B: dace.int64[N], C: dace.int64[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[:]
                b << B[i]
                c >> C[i]
                c = a[b]

    code = get_code(program)

    assert "svld1_gather_index" in code


def test_indirect_load_implicit():

    @dace.program
    def program(A: dace.int64[N], B: dace.int64[N], C: dace.int64[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[B[i]]
                c >> C[i]
                c = a

    code = get_code(program)

    # This is still an indirect load (uses Indirection tasklet)
    assert "svld1_gather_index" in code


def test_an_indirect_read_lives_in_an_sve_register_declared_inside_the_loop():

    @dace.program
    def program(idx: dace.uint32[N], x: dace.float32[M], B: dace.float32[N]):
        for j in dace.map[0:N]:
            with dace.tasklet:
                a << x[idx[j]]
                b >> B[j]
                b = a

    code = get_code(program)

    assert "dace::vec" not in code and "__dace_sve_len" not in code
    declared = [line.strip() for line in code.splitlines() if line.strip().startswith("svfloat32_t __")]
    loop = code.index("while(svptest_any")
    assert declared and all(code.index(line) > loop for line in declared)
