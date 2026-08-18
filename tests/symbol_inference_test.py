# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def symbol_inference(A: dace.float64[N, N], B: dace.float64[M + 1, M * 2]):
    for i, j in dace.map[0:N, 0:N]:
        with dace.tasklet:
            a >> A[i, j]
            a = N

    for i, j in dace.map[0:M + 1, 0:M * 2]:
        with dace.tasklet:
            b >> B[i, j]
            b = M


@dace.program
def symbol_inference_joint(A: dace.float64[N + M], B: dace.float64[N + 2 * M]):
    for i in dace.map[0:N + M]:
        with dace.tasklet:
            a >> A[i]
            a = N

    for i in dace.map[0:N + 2 * M]:
        with dace.tasklet:
            b >> B[i]
            b = M


def test_symbol_inference():
    real_N = 5
    real_M = 7
    A = np.random.rand(real_N, real_N)
    B = np.random.rand(real_M + 1, real_M * 2)
    symbol_inference(A, B)
    assert np.allclose(A, np.full_like(A, real_N))
    assert np.allclose(B, np.full_like(B, real_M))


def test_symbol_inference_joint():
    real_N = 3
    real_M = 2
    A = np.random.rand(real_N + real_M)
    B = np.random.rand(real_N + real_M * 2)
    symbol_inference_joint(A, B)
    assert np.allclose(A, np.full_like(A, real_N))
    assert np.allclose(B, np.full_like(B, real_M))


def test_stride_symbols_align_by_section_on_rank_mismatch():
    """A rank-4 view with a length-1 axis passed to a rank-3 symbolic descriptor must solve the
    stride symbols from the STRIDES of the squeezed argument -- not from a concatenated
    shape+strides+offset zip whose sections shift by the rank difference. The old zip solved
    ``I_stride`` against the trailing shape entry ``1`` (pyFV3 Remapping: strides ``(1, 19, 1)``
    instead of ``(19, 1, 361)``, uninitialized reads at runtime)."""
    from dace.frontend.python.parser import infer_symbols_from_datadescriptor

    inner = dace.SDFG('inner_stencil_sig')
    i_size, j_size, k_size = (dace.symbol(s) for s in ('__q_I_size', '__q_J_size', '__q_K_size'))
    i_str, j_str, k_str = (dace.symbol(s) for s in ('__q_I_stride', '__q_J_stride', '__q_K_stride'))
    inner.add_array('q', [i_size, j_size, k_size], dace.float64, strides=[i_str, j_str, k_str])

    outer_view = dace.data.Array(dace.float64, (19, 19, 80, 1), strides=(19, 1, 361, 28880))
    inferred = {str(k): v for k, v in infer_symbols_from_datadescriptor(inner, {'q': outer_view}).items()}

    assert inferred['__q_I_size'] == 19 and inferred['__q_J_size'] == 19 and inferred['__q_K_size'] == 80
    assert inferred['__q_I_stride'] == 19, f"I stride solved against a shape entry: {inferred}"
    assert inferred['__q_J_stride'] == 1
    assert inferred['__q_K_stride'] == 361


def test_equal_rank_inference_unchanged():
    """Sanity: the section-wise alignment must not change the ordinary equal-rank case."""
    from dace.frontend.python.parser import infer_symbols_from_datadescriptor

    inner = dace.SDFG('inner_equal_rank')
    rows, cols = dace.symbol('__rows'), dace.symbol('__cols')
    rstr = dace.symbol('__rstr')
    inner.add_array('q', [rows, cols], dace.float64, strides=[rstr, 1])

    arg = np.zeros((5, 7))
    inferred = {str(k): v for k, v in infer_symbols_from_datadescriptor(inner, {'q': arg}).items()}
    assert inferred['__rows'] == 5 and inferred['__cols'] == 7 and inferred['__rstr'] == 7


if __name__ == '__main__':
    test_symbol_inference()
    test_symbol_inference_joint()
    test_stride_symbols_align_by_section_on_rank_mismatch()
    test_equal_rank_inference_unchanged()
