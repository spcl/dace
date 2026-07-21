# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the layout algebra (dace.libraries.layout.algebra)."""
import sympy

from dace.symbolic import int_ceil
from dace.libraries.layout.algebra import (Digit, Permute, Block, Unblock, Pad, Shuffle, Zip, Unzip, identity_map,
                                           compose_ops, simplify_ops, is_identity)

N = sympy.Symbol('N', nonnegative=True, integer=True)
M = sympy.Symbol('M', nonnegative=True, integer=True)


# --------------------------- compose (semantics) --------------------------- #
def test_identity_map():
    m = identity_map([N, M])
    assert m.digits == (Digit(0, 1, N), Digit(1, 1, M))
    assert m.shape() == (N, M)


def test_permute_apply():
    m = compose_ops([Permute((1, 0))], shape=[N, M])
    assert m.shape() == (M, N)
    assert m.digits == (Digit(1, 1, M), Digit(0, 1, N))


def test_block_apply():
    # [N] blocked by 16 -> outer (0,16,ceil(N/16)) in place, inner (0,1,16) appended.
    m = compose_ops([Block(0, 16)], shape=[N])
    assert m.digits == (Digit(0, 16, int_ceil(N, 16)), Digit(0, 1, 16))


def test_block_then_unblock_map_roundtrip_divisible():
    n = 128
    m = compose_ops([Block(0, 16), Unblock(0, 16)], shape=[n])
    # ceil(128/16)*16 == 128 exactly.
    assert m.digits == (Digit(0, 1, 128), )


# --------------------------- optimizer (rewrite rules) --------------------------- #
def test_block_unblock_cancels():
    assert is_identity([Block(0, 16), Unblock(0, 16)])
    assert is_identity([Unblock(0, 16), Block(0, 16)])
    assert simplify_ops([Block(0, 16), Unblock(0, 16)]) == []


def test_permute_inverse_cancels():
    assert is_identity([Permute((1, 0)), Permute((1, 0))])  # self-inverse
    assert is_identity([Permute((1, 2, 0)), Permute((2, 0, 1))])  # tau then tau^-1
    assert not is_identity([Permute((1, 2, 0))])


def test_two_permutes_fuse_to_one():
    out = simplify_ops([Permute((1, 0, 2)), Permute((0, 2, 1))])
    assert len(out) == 1 and isinstance(out[0], Permute)
    # Composed perm equals applying (1,0,2) then (0,2,1): r[i] = p[q[i]].
    p, q = (1, 0, 2), (0, 2, 1)
    assert out[0].perm == tuple(p[q[i]] for i in range(3))


def test_pad_fuses_and_cancels():
    assert simplify_ops([Pad(0, 4), Pad(0, 8)]) == [Pad(0, 12)]
    assert is_identity([Pad(0, 8), Pad(0, -8)])
    assert simplify_ops([Pad(0, 0)]) == [Pad(0, 0)]  # single zero-pad left as-is (no pair)


def test_shuffle_inverse_cancels():
    assert is_identity([Shuffle(0, 'rcm'), Shuffle(0, 'rcm', inverted=True)])
    assert not is_identity([Shuffle(0, 'rcm'), Shuffle(0, 'other', inverted=True)])


def test_zip_unzip_cancels():
    assert is_identity([Zip(('re', 'im')), Unzip(('re', 'im'))])
    assert is_identity([Unzip(('re', 'im')), Zip(('re', 'im'))])
    assert not is_identity([Zip(('re', 'im'))])


def test_mixed_sequence_reduces():
    # permute, block, unblock, permute-back -> nothing.
    ops = [Permute((1, 0)), Block(0, 8), Unblock(0, 8), Permute((1, 0))]
    assert is_identity(ops)


def test_block_block_kept():
    # Two distinct blockings of the same dim do not cancel; both survive.
    out = simplify_ops([Block(0, 16), Block(0, 4)])
    assert out == [Block(0, 16), Block(0, 4)]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"ok  {name}")
    print("algebra tests PASS")
