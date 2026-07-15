# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Additional unit tests for the layout algebra (dace.libraries.layout.algebra).

These cover branches and edge cases not exercised by ``algebra_test.py``: op ``inverse``
methods, finest/coarsest digit selection, error paths, ``dim_sizes`` bookkeeping, shuffle
chaining, JSON serialization round-trips, and the digit index-expression lowering checked
bit-exact against a numpy oracle. Everything here is pure-Python algebra (no SDFG compilation).
"""
import json

import numpy as np
import pytest
import sympy

from dace.libraries.layout.algebra import (Digit, LayoutMap, Permute, Block, Unblock, Pad, Shuffle, Zip, Unzip,
                                           identity_map, compose_ops, simplify_ops, is_identity, physical_index_exprs,
                                           op_to_dict, op_from_dict, ops_to_list, ops_from_list)

N = sympy.Symbol('N', nonnegative=True, integer=True)
M = sympy.Symbol('M', nonnegative=True, integer=True)


def packed_offset(m: LayoutMap, index_by_dim: dict) -> int:
    """Row-major (packed-C) linear offset built from the per-digit index expressions.

    Substitutes concrete logical indices into ``physical_index_exprs`` and folds the resulting
    digit coordinates with the digit extents as radices, C order (the storage the layout defines).
    """
    exprs = physical_index_exprs(m)
    subs = {sympy.Symbol(f"__i{d}", nonnegative=True, integer=True): v for d, v in index_by_dim.items()}
    off = 0
    for expr, dg in zip(exprs, m.digits):
        coord = int(expr.subs(subs))
        off = off * int(dg.extent) + coord
    return off


# --------------------------- Digit / LayoutMap --------------------------- #
def test_digit_coerces_fields_to_sympy():
    d = Digit(0, 16, 8)
    assert isinstance(d.stride, sympy.Basic) and isinstance(d.extent, sympy.Basic)
    assert d.stride == sympy.Integer(16) and d.extent == sympy.Integer(8)
    # A string radix is sympified too, and compares equal to the integer form.
    assert Digit(1, '4', '32') == Digit(1, 4, 32)


def test_identity_map_explicit_dims():
    m = identity_map([N, M], dims=[2, 5])
    assert m.digits == (Digit(2, 1, N), Digit(5, 1, M))
    assert m.dim_sizes == {2: N, 5: M}
    assert m.shape() == (N, M)
    assert m.element is None and m.shuffles == ()


def test_block_integer_extent_is_exact():
    m = compose_ops([Block(0, 16)], shape=[128])
    outer, inner = m.digits
    assert outer == Digit(0, 16, 8) and inner == Digit(0, 1, 16)
    assert isinstance(outer.extent, sympy.Integer)


# --------------------------- inverse() methods --------------------------- #
def test_inverse_methods_return_expected_ops():
    assert Permute((2, 0, 1)).inverse() == Permute((1, 2, 0))
    assert Block(3, 8).inverse() == Unblock(3, 8)
    assert Unblock(3, 8).inverse() == Block(3, 8)
    assert Pad(0, 8).inverse() == Pad(0, -8)
    assert Shuffle(0, 'rcm').inverse() == Shuffle(0, 'rcm', inverted=True)
    assert Zip(('re', 'im')).inverse() == Unzip(('re', 'im'))
    assert Unzip(('re', 'im')).inverse() == Zip(('re', 'im'))


def test_permute_inverse_roundtrips_layout():
    p = Permute((2, 0, 1))
    base = identity_map([N, M, N])
    assert compose_ops([p, p.inverse()], base=base) == base


# --------------------------- finest / coarsest selection --------------------------- #
def test_block_splits_finest_digit_of_dim():
    # After Block(0,16) the finest digit is (0,1,16); Block(0,4) must split THAT one.
    m = compose_ops([Block(0, 16), Block(0, 4)], shape=[N])
    assert m.digits == (Digit(0, 16, sympy.ceiling(N / 16)), Digit(0, 4, 4), Digit(0, 1, 4))


def test_unblock_merges_finest_inner_first():
    # Block twice then Unblock(0,4): merges the (0,1,4) inner with its (0,4,4) partner.
    m = compose_ops([Block(0, 16), Block(0, 4), Unblock(0, 4)], shape=[N])
    assert m.digits == (Digit(0, 16, sympy.ceiling(N / 16)), Digit(0, 1, 16))


def test_pad_grows_coarsest_digit_and_dim_size():
    # Coarsest digit of dim 0 after blocking is the stride-16 outer; Pad must grow that one.
    m = compose_ops([Block(0, 16), Pad(0, 3)], shape=[64])
    coarsest = max(m.digits, key=lambda dg: dg.stride)
    assert coarsest == Digit(0, 16, 4 + 3)
    assert m.dim_sizes[0] == sympy.Integer(64 + 3)
    # The finest (inner) digit is untouched by the pad.
    assert Digit(0, 1, 16) in m.digits


# --------------------------- error paths --------------------------- #
def test_compose_ops_without_base_or_shape_raises():
    with pytest.raises(ValueError):
        compose_ops([Pad(0, 1)])


def test_permute_wrong_length_raises():
    with pytest.raises(AssertionError):
        compose_ops([Permute((0, 1, 2))], shape=[N, M])


def test_block_and_pad_missing_dim_raise():
    with pytest.raises(ValueError):
        compose_ops([Block(5, 2)], shape=[N])
    with pytest.raises(ValueError):
        compose_ops([Pad(5, 2)], shape=[N])


def test_unblock_without_inner_digit_raises():
    with pytest.raises(ValueError):
        compose_ops([Unblock(0, 16)], shape=[N])


def test_unblock_without_outer_partner_raises():
    # An inner-shaped digit with no coarser partner at stride*factor cannot be merged.
    m = LayoutMap(dim_sizes={0: 4}, digits=(Digit(0, 1, 4), ))
    with pytest.raises(ValueError):
        Unblock(0, 4).apply(m)


def test_zip_on_existing_struct_raises():
    with pytest.raises(ValueError):
        compose_ops([Zip(('re', 'im')), Zip(('a', 'b'))], shape=[N])


def test_unzip_field_mismatch_raises():
    with pytest.raises(ValueError):
        compose_ops([Zip(('re', 'im')), Unzip(('a', 'b'))], shape=[N])


# --------------------------- element / shuffle semantics --------------------------- #
def test_zip_then_unzip_element_roundtrip():
    zipped = compose_ops([Zip(('re', 'im'))], shape=[N])
    assert zipped.element == ('re', 'im')
    back = compose_ops([Unzip(('re', 'im'))], base=zipped)
    assert back.element is None


def test_shuffle_chain_accumulates_and_sorts_by_dim():
    m = compose_ops([Shuffle(2, 'a'), Shuffle(0, 'b'), Shuffle(0, 'c')], shape=[N, M, N])
    # Sorted by dim; same-dim tokens accumulate in application order.
    assert m.shuffles == ((0, (('b', False), ('c', False))), (2, (('a', False), )))


# --------------------------- simplify_ops peephole identities --------------------------- #
def test_block_then_unblock_is_id_but_pads_when_indivisible():
    # simplify cancels the pair structurally...
    assert is_identity([Block(0, 16), Unblock(0, 16)])
    # ...but the *materialized* layout is padded up (ceil(100/16)*16 == 112 != 100).
    m = compose_ops([Block(0, 16), Unblock(0, 16)], shape=[100])
    assert m.digits == (Digit(0, 1, 112), )


def test_permute_then_inverse_is_id():
    p = Permute((3, 0, 1, 2))
    assert is_identity([p, p.inverse()])
    assert simplify_ops([p, p.inverse()]) == []


def test_three_cycle_permute_reduces_to_id_via_fixpoint():
    # (1,2,0) is a 3-cycle: applied three times it is the identity.
    assert is_identity([Permute((1, 2, 0)), Permute((1, 2, 0)), Permute((1, 2, 0))])


def test_pad_pad_fuses_and_zero_total_cancels():
    assert simplify_ops([Pad(0, 4), Pad(0, 8)]) == [Pad(0, 12)]
    # Fusion producing a zero total collapses to nothing (exercises the total==0 branch).
    assert is_identity([Pad(0, 4), Pad(0, 4), Pad(0, -8)])


def test_pad_different_dims_do_not_fuse():
    out = simplify_ops([Pad(0, 4), Pad(1, 4)])
    assert out == [Pad(0, 4), Pad(1, 4)]


def test_shuffle_inverse_is_id_but_variants_kept():
    assert is_identity([Shuffle(0, 'rcm'), Shuffle(0, 'rcm', inverted=True)])
    # Different dim or different name must not cancel.
    assert simplify_ops([Shuffle(0, 'rcm'), Shuffle(1, 'rcm', inverted=True)]) == \
        [Shuffle(0, 'rcm'), Shuffle(1, 'rcm', inverted=True)]
    assert not is_identity([Shuffle(0, 'a'), Shuffle(0, 'b', inverted=True)])


def test_unzip_then_zip_is_id():
    assert is_identity([Unzip(('re', 'im')), Zip(('re', 'im'))])
    assert not is_identity([Unzip(('re', 'im')), Zip(('a', 'b'))])


def test_nested_cancellation_reaches_outer_pair():
    # Inner Block/Unblock cancel, exposing the outer Pad pair which then also cancels.
    assert is_identity([Pad(0, 4), Block(0, 8), Unblock(0, 8), Pad(0, -4)])


# --------------------------- is_identity exactness --------------------------- #
def test_is_identity_exactness():
    assert is_identity([])
    assert simplify_ops([]) == []
    # Each single non-trivial op is NOT an identity.
    for op in [Block(0, 8), Unblock(0, 8), Pad(0, 4), Permute((1, 0)), Shuffle(0, 'p'), Zip(('re', 'im'))]:
        assert not is_identity([op])
    # A zero pad is a semantic no-op but does not structurally cancel on its own.
    assert not is_identity([Pad(0, 0)])
    assert simplify_ops([Pad(0, 0)]) == [Pad(0, 0)]
    # Two distinct blockings of one dim survive (no false cancellation).
    assert not is_identity([Block(0, 16), Block(0, 4)])


# --------------------------- JSON serialization round-trips --------------------------- #
def test_ops_json_roundtrip_all_op_kinds():
    ops = [
        Permute((2, 0, 1)),
        Block(1, 8),
        Unblock(1, 8),
        Pad(0, 12),
        Shuffle(3, 'rcm', inverted=True),
        Zip(('re', 'im')),
        Unzip(('re', 'im')),
    ]
    encoded = ops_to_list(ops)
    # The encoding must survive a real JSON serialize/deserialize (proves it is JSON-safe).
    text = json.dumps(encoded)
    decoded = ops_from_list(json.loads(text))
    assert decoded == ops


def test_op_to_dict_pad_amount_is_string_and_roundtrips():
    d = op_to_dict(Pad(0, 12))
    assert d == {'op': 'Pad', 'dim': 0, 'amount': '12'}
    assert op_from_dict(d) == Pad(0, 12)


def test_op_to_dict_unknown_op_raises():
    with pytest.raises(TypeError):
        op_to_dict(object())


def test_op_from_dict_unknown_kind_raises():
    with pytest.raises(ValueError):
        op_from_dict({'op': 'Nope'})


# --------------------------- digit index expression lowering (numpy oracle) --------------------------- #
def test_physical_index_exprs_identity_matches_ravel():
    shape = [4, 5, 3]
    m = identity_map(shape)
    # Packed-C offset from the digit exprs must equal numpy's row-major ravel for every index.
    got = np.empty(shape, dtype=np.int64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                got[i, j, k] = packed_offset(m, {0: i, 1: j, 2: k})
    oracle = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    assert np.array_equal(got, oracle)
    assert np.allclose(got, oracle)


def test_physical_index_exprs_block_reconstructs_linear_index():
    n = 64
    m = compose_ops([Block(0, 16)], shape=[n])
    # Blocking a single dim by a divisor is a lossless reshape: packed offset == original index.
    got = np.array([packed_offset(m, {0: i}) for i in range(n)], dtype=np.int64)
    oracle = np.arange(n, dtype=np.int64)
    assert np.array_equal(got, oracle)


def test_physical_index_exprs_shapes_and_symbols():
    m = compose_ops([Block(0, 16)], shape=[N])
    exprs = physical_index_exprs(m)
    assert len(exprs) == len(m.digits) == 2
    # The finest digit's expression carries a modulus by its (constant) extent.
    idx = sympy.Symbol('__i0', nonnegative=True, integer=True)
    assert exprs[1] == idx % 16


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"ok  {name}")
    print("algebra extra tests PASS")