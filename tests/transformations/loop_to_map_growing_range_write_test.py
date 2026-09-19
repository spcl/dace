# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A write whose extent GROWS with the iteration is not uniquely indexed by it.

``LoopToMap`` decides that question on the LOWER bound of each dimension: iteration ``i`` starts
at ``a*i+b``, and ``|a| >= 1`` means consecutive iterations start at different elements. A
dimension whose lower bound folded to the literal ``0`` used to be matched on its UPPER bound
instead, which accepts ``a[0:i+1]`` -- an extent that grows -- as injective, when in truth every
iteration rewrites element 0.

Nothing reached the branch while scope summaries were whole-array approximations. Rebuilding them
from the body is what produces the growing form: tsvc_2_5 ``wf_diff_skew`` is a wavefront whose
body writes ``a[i, j]``, and propagating that out of the ``i`` loop gives ``a[0:i+1, 0:LEN-1]``.
The loop carries ``(1, 0)`` and ``(1, -1)`` dependences and was lifted and miscompiled. That
end-to-end shape lives in ``tests/canonicalize``, next to the propagation that produces it; this
file pins the rule itself.
"""
import sympy as sp

from dace import subsets, symbolic
from dace.transformation.interstate.loop_to_map import _check_range


def check(subset_str: str) -> bool:
    itersym = symbolic.pystr_to_symbolic('i')
    a = sp.Wild('a', exclude=[itersym])
    b = sp.Wild('b', exclude=[itersym])
    return _check_range(subsets.Range.from_string(subset_str), a, itersym, b, 1)


def test_a_growing_range_is_not_uniquely_indexed():
    """The bug at unit scale, with the two neighbours that must keep answering the other way."""
    assert not check('0:i+1'), 'every iteration rewrites element 0'
    assert not check('0:2*i+1'), 'a stride does not help when the range still starts at 0'
    assert check('i'), 'a point write moves with the iteration'
    assert check('i:i+4'), 'so does a fixed-width band'
    assert check('2*i:2*i+2'), 'and a strided one'


def test_a_zero_based_point_write_is_still_refused():
    """``[0]`` and ``[0:N]`` do not mention the iteration variable at all, so neither is indexed
    by it -- the verdict has to stay False without the branch that used to inspect them."""
    assert not check('0')
    assert not check('0:N')


if __name__ == '__main__':
    test_a_growing_range_is_not_uniquely_indexed()
    test_a_zero_based_point_write_is_still_refused()
