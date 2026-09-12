# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``prepare_for_layout``'s packed-strides precondition: the whole layout algebra assumes a packed
C/Fortran representation, so ``normalize_to_packed_c`` is the gate that must be both LOUD (a genuinely
padded array is refused, not warned about) and ACCURATE (a canonicalized array is not a false alarm).

The accuracy half is not cosmetic. ``RelaxIntegerPowers`` -- which ``canonicalize`` runs, so every
program through ``prepare_for_layout`` sees it -- respells a packed rank-3 stride ``N**2`` as
``ipow(N, 2)``. ``ipow`` is an opaque ``Function`` to SymPy and never simplifies against an equal
``Pow``, so a raw structural stride comparison calls every symbolic rank>=3 array padded.
"""
import warnings

import pytest

import dace
from dace.transformation.layout.prepare import normalize_to_packed_c, prepare_for_layout

N = dace.symbol("N")


@dace.program
def rank3_scale(A: dace.float64[N, N, N], B: dace.float64[N, N, N]):
    for i, j, k in dace.map[0:N, 0:N, 0:N]:
        B[i, j, k] = A[i, j, k] * 2.0


def test_relax_ipow_undoes_the_canonicalization_respelling():
    """The shared normalizer: ``ipow(b, e)`` and ``b ** e`` are the same value, and only after this
    rewrite do they compare equal."""
    ipow_squared = dace.symbolic.ipow(N, 2)
    assert dace.symbolic.simplify(ipow_squared - N**2) != 0  # opaque Function: no reduction on its own
    assert dace.symbolic.simplify(dace.symbolic.relax_ipow(ipow_squared) - N**2) == 0
    assert dace.symbolic.relax_ipow(4) == 4  # non-symbolic values pass through untouched


def test_canonicalized_rank3_array_reads_as_packed():
    """A plainly packed rank-3 array must not be called padded just because canonicalization respelled
    its stride. This is the exact shape ``prepare_for_layout`` produces, so a false alarm here fires on
    every symbolic rank>=3 program the stack handles."""
    packed = dace.data.Array(dace.float64, [N, N, N], strides=[dace.symbolic.ipow(N, 2), N, 1])
    assert packed.is_packed_c_strides()
    padded = dace.data.Array(dace.float64, [N, N, N], strides=[2 * N * N, N, 1])
    assert not padded.is_packed_c_strides()  # a real pad is still detected


def test_prepare_leaves_no_packed_stride_complaint():
    """End to end: the program that produces ``ipow`` strides passes the gate silently."""
    sdfg = rank3_scale.to_sdfg(simplify=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepare_for_layout(sdfg)
    assert not [w for w in caught if "non-packed strides" in str(w.message)]
    strides = sdfg.arrays["A"].strides
    assert any(isinstance(s, dace.symbolic.ipow) for s in strides), strides  # the respelling did happen
    assert sdfg.arrays["A"].is_packed_c_strides()


def test_packed_stride_cache_follows_a_direct_shape_assignment():
    """The packed-stride cache used to refresh only via ``set_shape``, but passes assign ``.shape``
    directly, and a symbol replacement rebuilds the shape entries as NEW sympy objects (canonicalize
    re-registers symbols with nonnegative assumptions). Two same-named symbols with different assumptions
    are DIFFERENT to sympy, so the stale cache made ``J*K`` compare unequal to ``J*K`` and a packed array
    read as padded -- and ``copy_node``/``subsets`` pick a layout path off that answer."""
    i, j, k = (dace.symbol(s) for s in "IJK")
    desc = dace.data.Array(dace.float64, [i, j, k], strides=[j * k, k, 1])
    assert desc.is_packed_c_strides()
    plain = [dace.symbolic.symbol(s.name, s.dtype) for s in (i, j, k)]  # same names, default assumptions
    desc.shape = tuple(plain)
    desc.strides = (plain[1] * plain[2], plain[2], 1)
    assert desc.is_packed_c_strides()  # stale cache would say False here
    desc.shape = (i, j, k)
    desc.strides = (2 * j * k, k, 1)
    assert not desc.is_packed_c_strides()  # a real pad is still caught after a refresh


def test_non_packed_array_is_refused_not_warned():
    """Downstream (``composed_permutation``, the clone ``add_array``) ASSUMES packed strides, so warning
    here only defers the failure to a place where it reads as a miscompile."""
    sdfg = dace.SDFG("padded")
    sdfg.add_array("A", [N, N], dace.float64, strides=[2 * N, 1])
    sdfg.add_state("s", is_start_block=True)
    with pytest.raises(NotImplementedError, match="non-packed strides"):
        normalize_to_packed_c(sdfg)


def test_fortran_strides_are_accepted():
    """The gate accepts either packed order -- a column-major input is a legitimate starting layout."""
    sdfg = dace.SDFG("fortran")
    sdfg.add_array("A", [N, N], dace.float64, strides=[1, N])
    sdfg.add_state("s", is_start_block=True)
    normalize_to_packed_c(sdfg)  # must not raise


if __name__ == "__main__":
    test_relax_ipow_undoes_the_canonicalization_respelling()
    test_canonicalized_rank3_array_reads_as_packed()
    test_prepare_leaves_no_packed_stride_complaint()
    test_non_packed_array_is_refused_not_warned()
    test_fortran_strides_are_accepted()
    print("prepare tests PASS")
