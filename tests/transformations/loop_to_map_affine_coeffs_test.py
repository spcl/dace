# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``_affine_coeffs`` must derive the coefficients, not search for them.

``LoopToMap.can_be_applied`` asks whether two accesses can alias, which needs ``a`` and ``b`` from
``a*i + b`` for each index. Spelling that as ``expand`` + ``coeff`` + ``simplify`` is superlinear
in the size of the index expression: on a two-level tiled stencil, whose innermost index carries
every enclosing tile origin, the expansion blows up and the ``coeff`` walk over the resulting sum
does not come back -- the pipeline hung outright.

The deep case below is the regression: it is instant when the coefficients are derived (derivative
and evaluation at zero, both structural) and does not finish when they are searched for.
"""
import pytest

from dace import symbolic
from dace.transformation.interstate.loop_to_map import _affine_coeffs


def _sym(src):
    return symbolic.pystr_to_symbolic(src)


@pytest.mark.parametrize('expr,coeff,offset', [
    ('2*i + 3', 2, 3),
    ('i', 1, 0),
    ('7', 0, 7),
    ('N - i', -1, 'N'),
    ('j*i + 1', 'j', 1),
])
def test_affine_expressions_give_their_coefficients(expr, coeff, offset):
    a, b = _affine_coeffs(_sym(expr), _sym('i'))
    assert a == _sym(str(coeff))
    assert b == _sym(str(offset))


@pytest.mark.parametrize('expr', ['i**2', '(i + 1)*(i + 2)', 'int_floor(i, 8)'])
def test_non_affine_expressions_are_refused(expr):
    assert _affine_coeffs(_sym(expr), _sym('i')) is None


def test_a_deeply_nested_tile_index_does_not_blow_up():
    """The shape that hung: an index summing many tile origins, each itself a sum.

    Built as a product of sums so that EXPANDING it is exponential, while the coefficient of ``i``
    is trivially derivable. Correctness is the assertion; termination is the point.
    """
    factors = ' * '.join(f'(t{d} + o{d} + 1)' for d in range(12))
    a, b = _affine_coeffs(_sym(f'i * {factors} + {factors}'), _sym('i'))
    assert a == _sym(factors), 'the coefficient of i is the product, unexpanded'
    assert b == _sym(factors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
