# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the Shuffle layout primitive.

The shuffle registry (``register_shuffle``) mints sympy-function + C-lowering artifacts for a
user-supplied bijection, and :class:`ShuffleElements` physically renumbers a dimension by it. A
shuffle is TRANSPARENT: it changes only the physical layout, so a shuffled kernel's output must
be bit-identical to the unshuffled one -- that equality is the oracle. A deliberately WRONG
inverse must BREAK that equality (the negative test proves the rewrite genuinely uses sigma^-1).
"""
import copy
import numpy
import dace

from dace.libraries.layout.shuffle import register_shuffle, emit_shuffle_globals, _symbol_params
from dace.libraries.layout.algebra import Shuffle
from dace.libraries.layout.lowering import relayout_map
from dace.transformation.layout.shuffle_elements import ShuffleElements

N = dace.symbol("N")


def test_layoutchange_rejects_net_shuffle():
    """A net Shuffle in a relayout op-sequence is refused (it is the ShuffleElements pass's job),
    but a cancelling Shuffle chain still lowers (simplify_ops removes it)."""
    import pytest
    with pytest.raises(NotImplementedError):
        relayout_map([16], [Shuffle(0, "sig")])
    # Shuffle o Shuffle^-1 cancels -> no net shuffle -> lowers fine (identity copy).
    _, out_map, out_shape = relayout_map([16], [Shuffle(0, "sig"), Shuffle(0, "sig", inverted=True)])
    assert not out_map.shuffles and list(out_shape) == [16]


# --------------------------------------------------------------------------- #
#  Registry unit tests
# --------------------------------------------------------------------------- #
def test_registry_params_and_folding():
    xor = register_shuffle("t_xor3", "i ^ 3", "i ^ 3")
    assert xor.params == ()  # no SDFG symbols
    assert xor.apply_inverse(7) == 4  # 7 ^ 3 folds to 4
    assert str(xor.apply_inverse("k")) == "shuffle_inv_t_xor3(k)"  # symbolic -> a call
    assert "return (i ^ 3);" in xor.c_definitions()


def test_registry_symbol_param_detected():
    aff = register_shuffle("t_aff", "(3*i + 1) % M", "((i - 1) * 7) % M")
    assert aff.params == ("M", )  # M auto-detected as a symbol parameter
    assert str(aff.apply_forward("e")) == "shuffle_t_aff(e, M)"
    assert "long long M)" in aff.c_definitions()


def test_registry_inverse_roundtrips_numerically():
    cyc = register_shuffle("t_cyc", "(i + 1) % M", "(i + M - 1) % M")
    fwd, inv = cyc.numeric_forward(), cyc.numeric_inverse()
    for i in range(10):
        assert inv(fwd(i, M=10), M=10) == i  # sigma^-1 . sigma == id over the domain


def test_registry_excludes_call_callees():
    """A function callee (pow/isqrt/...) is NOT mistaken for an SDFG symbol parameter."""
    assert _symbol_params("pow(i, 2)", "isqrt(i)") == ()
    assert _symbol_params("(3*i) % M", "(i + M) % M") == ("M", )


def test_registry_emits_floored_mod():
    """A modulo lowers through the floored-mod helper, matching Python's % (not C's truncation)."""
    sh = register_shuffle("t_shift", "(i + 3) % 8", "(i - 3) % 8")
    assert "shuffle_pymod(" in sh.c_definitions()


def test_emit_shuffle_globals_name_collision_raises():
    """Two shuffles whose emitted C names collide with different bodies fail loudly."""
    import pytest
    register_shuffle("collide", "i ^ 1", "i ^ 1")  # emits shuffle_inv_collide (body i^1)
    register_shuffle("inv_collide", "i ^ 2", "i ^ 3")  # forward name == shuffle_inv_collide (body i^2)
    sdfg = dace.SDFG("collide_sdfg")
    with pytest.raises(ValueError):
        emit_shuffle_globals(sdfg, ["collide", "inv_collide"])


# --------------------------------------------------------------------------- #
#  ShuffleElements: transparency oracle
# --------------------------------------------------------------------------- #
@dace.program
def scale(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] * 3.0


@dace.program
def incr(A: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        A[i] = A[i] + 1.0


@dace.program
def rowscale(A: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] * 2.0


def _apply_shuffle(program, name, shuffle_map, simplify=True):
    sdfg = copy.deepcopy(program.to_sdfg(simplify=simplify))
    sdfg.name = name
    ShuffleElements(shuffle_map=shuffle_map).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def test_shuffle_xor_readonly_transparent():
    """XOR swizzle on a read-only input: C = A*3 is unchanged (only A's layout moves)."""
    register_shuffle("xor3", "i ^ 3", "i ^ 3")
    sh = _apply_shuffle(scale, "scale_xor", {"A": ("xor3", 0)})
    assert "shuffled_A" in sh.arrays

    _N = 8
    A = numpy.random.rand(_N)
    C0 = numpy.zeros(_N)
    C1 = numpy.zeros(_N)
    scale.to_sdfg()(A=A.copy(), C=C0, N=_N)
    sh(A=A.copy(), C=C1, N=_N)
    assert numpy.allclose(C1, C0)


def test_shuffle_affine_inplace_transparent():
    """A non-self-inverse cyclic shift (symbol param) on an in-place update: gather + scatter."""
    register_shuffle("cyc", "(i + 1) % N", "(i + N - 1) % N")
    sh = _apply_shuffle(incr, "incr_cyc", {"A": ("cyc", 0)})

    _N = 8
    A0 = numpy.random.rand(_N)
    A1 = A0.copy()
    incr.to_sdfg()(A=A0, N=_N)
    sh(A=A1, N=_N)
    assert numpy.allclose(A1, A0)


def test_shuffle_wrong_inverse_breaks_transparency():
    """Teeth: a wrong sigma^-1 (identity) makes the shuffled output DIFFER from the reference."""
    register_shuffle("cyc_bad", "(i + 1) % N", "i")
    sh = _apply_shuffle(scale, "scale_cycbad", {"A": ("cyc_bad", 0)})

    _N = 8
    A = numpy.random.rand(_N)
    C0 = numpy.zeros(_N)
    C1 = numpy.zeros(_N)
    scale.to_sdfg()(A=A.copy(), C=C0, N=_N)
    sh(A=A.copy(), C=C1, N=_N)
    assert not numpy.allclose(C1, C0)


def test_shuffle_2d_single_dim_transparent():
    """Shuffle only dim 1 of a 2D array; the elementwise result is unchanged."""
    register_shuffle("xor3", "i ^ 3", "i ^ 3")
    sh = _apply_shuffle(rowscale, "rowscale_xor", {"A": ("xor3", 1)})

    _N = 8
    A = numpy.random.rand(_N, _N)
    C0 = numpy.zeros((_N, _N))
    C1 = numpy.zeros((_N, _N))
    rowscale.to_sdfg()(A=A.copy(), C=C0, N=_N)
    sh(A=A.copy(), C=C1, N=_N)
    assert numpy.allclose(C1, C0)


def test_shuffle_2d_multi_dim_transparent():
    """Shuffle BOTH dims of a 2D array with distinct sigma each; the elementwise result is unchanged."""
    register_shuffle("xor3", "i ^ 3", "i ^ 3")
    register_shuffle("cyc", "(i + 1) % N", "(i + N - 1) % N")
    sh = _apply_shuffle(rowscale, "rowscale_multi", {"A": [("xor3", 0), ("cyc", 1)]})
    assert "shuffled_A" in sh.arrays

    _N = 8
    A = numpy.random.rand(_N, _N)
    C0 = numpy.zeros((_N, _N))
    C1 = numpy.zeros((_N, _N))
    rowscale.to_sdfg()(A=A.copy(), C=C0, N=_N)
    sh(A=A.copy(), C=C1, N=_N)
    assert numpy.allclose(C1, C0)


def test_shuffle_list_form_single_dim_matches_tuple():
    """The list form [(name, dim)] is accepted and equivalent to the bare (name, dim) tuple."""
    register_shuffle("xor3", "i ^ 3", "i ^ 3")
    sh = _apply_shuffle(scale, "scale_listform", {"A": [("xor3", 0)]})
    assert "shuffled_A" in sh.arrays

    _N = 8
    A = numpy.random.rand(_N)
    C0 = numpy.zeros(_N)
    C1 = numpy.zeros(_N)
    scale.to_sdfg()(A=A.copy(), C=C0, N=_N)
    sh(A=A.copy(), C=C1, N=_N)
    assert numpy.allclose(C1, C0)


def test_shuffle_duplicate_dim_raises():
    """Shuffling the same dimension twice is refused."""
    import pytest
    register_shuffle("xor3", "i ^ 3", "i ^ 3")
    register_shuffle("cyc", "(i + 1) % N", "(i + N - 1) % N")
    sdfg = copy.deepcopy(rowscale.to_sdfg(simplify=True))
    with pytest.raises(ValueError):
        ShuffleElements(shuffle_map={"A": [("xor3", 0), ("cyc", 0)]}).apply_pass(sdfg, {})


def test_shuffle_negative_mod_transparent():
    """A cyclic shift whose inverse (i - 3) % 8 produces a negative dividend stays transparent:
    the emitted C floors like Python (would give an OOB negative index under C's truncating %)."""
    register_shuffle("shift3", "(i + 3) % 8", "(i - 3) % 8")
    sh = _apply_shuffle(scale, "scale_shift3", {"A": ("shift3", 0)})

    _N = 8
    A = numpy.random.rand(_N)
    C0 = numpy.zeros(_N)
    C1 = numpy.zeros(_N)
    scale.to_sdfg()(A=A.copy(), C=C0, N=_N)
    sh(A=A.copy(), C=C1, N=_N)
    assert numpy.allclose(C1, C0)


@dace.program
def rows_partial(A: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N - 1, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] + 1.0


def test_shuffle_partial_write_nonshuffled_dim_preserved():
    """Shuffle dim 1 of C while the kernel writes only rows 0:N-1: the gather-on-write initializes
    the transient so the untouched last row round-trips (would be clobbered without it)."""
    register_shuffle("xor3", "i ^ 3", "i ^ 3")
    sh = _apply_shuffle(rows_partial, "rows_partial_xor", {"C": ("xor3", 1)})

    _N = 8
    A = numpy.random.rand(_N, _N)
    Cin = numpy.random.rand(_N, _N)  # caller-provided; last row must survive
    C0 = Cin.copy()
    C1 = Cin.copy()
    rows_partial.to_sdfg()(A=A.copy(), C=C0, N=_N)
    sh(A=A.copy(), C=C1, N=_N)
    assert numpy.allclose(C1, C0) and numpy.allclose(C1[-1], Cin[-1])


@dace.program
def partial_shuffled(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N - 1] @ dace.ScheduleType.Sequential:
        C[i] = A[i] + 1.0


def test_shuffle_partial_on_shuffled_dim_raises():
    """A partial (non-point, non-full) range on the shuffled dimension cannot be sigma^-1-composed
    and is refused rather than silently miscompiled."""
    import pytest
    register_shuffle("xor3", "i ^ 3", "i ^ 3")
    sdfg = copy.deepcopy(partial_shuffled.to_sdfg(simplify=True))
    with pytest.raises(NotImplementedError):
        ShuffleElements(shuffle_map={"A": ("xor3", 0)}).apply_pass(sdfg, {})


def test_shuffle_emits_c_functions():
    """The gather calls sigma, consumers call sigma^-1, and both C defs are emitted."""
    register_shuffle("cyc", "(i + 1) % N", "(i + N - 1) % N")
    sh = _apply_shuffle(scale, "scale_emit", {"A": ("cyc", 0)})
    code = sh.generate_code()[0].clean_code
    assert "shuffle_cyc(long long" in code and "shuffle_inv_cyc(long long" in code
    assert "shuffle_cyc(" in code and "shuffle_inv_cyc(" in code


# --------------------------------------------------------------------------- #
#  Nested SDFG: the shuffled array flows whole into a nested SDFG
# --------------------------------------------------------------------------- #
def _build_nested_scale_sdfg(name):
    """Outer SDFG: A[N] flows WHOLE into a nested SDFG that computes C[i] = A[i]*4 -> C[N]."""
    inner = dace.SDFG("inner_scale")
    inner.add_array("Ac", [N], dace.float64)
    inner.add_array("Cc", [N], dace.float64)
    istate = inner.add_state("is", is_start_block=True)
    istate.add_mapped_tasklet(
        name="iscale",
        map_ranges={"i": "0:N"},
        inputs={"a": dace.Memlet.simple("Ac", "i")},
        code="c = a * 4.0",
        outputs={"c": dace.Memlet.simple("Cc", "i")},
        external_edges=True,
    )

    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    nsdfg = state.add_nested_sdfg(inner, {"Ac"}, {"Cc"}, symbol_mapping={"N": N})
    rA = state.add_read("A")
    wC = state.add_write("C")
    state.add_edge(rA, None, nsdfg, "Ac", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(nsdfg, "Cc", wC, None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg


def test_shuffle_nested_sdfg_transparent():
    """Shuffling A (which flows whole into a nested SDFG) rewrites the inner accesses; the
    result is unchanged."""
    register_shuffle("xor3", "i ^ 3", "i ^ 3")
    ref = _build_nested_scale_sdfg("nested_ref")

    sh = _build_nested_scale_sdfg("nested_xor")
    ShuffleElements(shuffle_map={"A": ("xor3", 0)}).apply_pass(sh, {})
    sh.validate()

    _N = 8
    A = numpy.random.rand(_N)
    C0 = numpy.zeros(_N)
    C1 = numpy.zeros(_N)
    ref(A=A.copy(), C=C0, N=_N)
    sh(A=A.copy(), C=C1, N=_N)
    assert numpy.allclose(C1, C0)


if __name__ == "__main__":
    test_registry_params_and_folding()
    test_registry_symbol_param_detected()
    test_registry_inverse_roundtrips_numerically()
    test_shuffle_xor_readonly_transparent()
    test_shuffle_affine_inplace_transparent()
    test_shuffle_wrong_inverse_breaks_transparency()
    test_shuffle_2d_single_dim_transparent()
    test_shuffle_2d_multi_dim_transparent()
    test_shuffle_list_form_single_dim_matches_tuple()
    test_shuffle_duplicate_dim_raises()
    test_shuffle_emits_c_functions()
    test_shuffle_nested_sdfg_transparent()
    print("shuffle tests PASS")
