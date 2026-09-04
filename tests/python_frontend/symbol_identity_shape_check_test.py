# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""A symbol NAME denotes one value, but sympy folds dtype/assumptions into symbol identity. A
shape built from ``dace.symbol('N', dace.int32)`` and one built from ``dace.symbol('N',
dace.int64)`` are the same shape by name but different sympy objects, so a raw ``!=``/``==``
shape check answers "different" and rejects arrays whose shapes actually agree -- with a
self-refuting message naming the same symbol on both sides. These replacement functions are
called directly (not through a full program parse) because parsing both bounds from one string
mints a single symbol instance and the mismatch cannot arise; ``Memlet.from_array`` and a
descriptor built straight from the mismatched symbol reproduce the two-instance situation that
a rebuilt descriptor vs. a reparsed bound produces in practice.
"""
import ast
import warnings

import numpy as np

import dace
from dace import symbolic
from dace.frontend.python.memlet_parser import _fill_missing_slices
from dace.frontend.python.replacements.array_manipulation import _concat, _stack
from dace.frontend.python.replacements.linalg import _matmult, _tensordot, dot


class _FakeVisitor:
    """Minimal ProgramVisitor stand-in: only get_target_name/add_temp_transient/views are used
    by the replacement functions below before any AST-only state is touched."""

    def __init__(self, sdfg):
        self.sdfg = sdfg
        self.views = {}
        self._counter = 0

    def get_target_name(self, output_index=None, default=None):
        self._counter += 1
        return f'_t{self._counter}'

    def add_temp_transient(self, *args, **kwargs):
        kwargs['find_new_name'] = True
        return self.sdfg.add_transient(self.get_target_name(), *args, **kwargs)


def test_dot_shape_check_equalizes_symbols():
    wide = dace.symbol('N', dace.int32)
    narrow = dace.symbol('N', dace.int64)
    assert wide is not narrow and wide != narrow  # the premise: identity says these differ

    sdfg = dace.SDFG('dot_symbol_instances')
    sdfg.add_array('a', [wide], dace.float64)
    sdfg.add_array('b', [narrow], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    state = sdfg.add_state()

    dot(None, sdfg, state, 'a', 'b', op_out='out')  # raised SyntaxError() before the fix
    sdfg.validate()
    sdfg.expand_library_nodes()

    rng = np.random.default_rng(0)
    a = rng.random(6)
    b = rng.random(6)
    out = np.zeros((1, ), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        sdfg.compile()(a=a, b=b, out=out, N=6)
    assert np.allclose(out, np.dot(a, b))


def test_stack_shape_check_equalizes_symbols():
    wide = dace.symbol('N', dace.int32)
    narrow = dace.symbol('N', dace.int64)

    sdfg = dace.SDFG('stack_symbol_instances')
    sdfg.add_array('a', [wide], dace.float64)
    sdfg.add_array('b', [narrow], dace.float64)
    state = sdfg.add_state()

    # raised "Array shapes are not equal ((N,) != (N,) at index 0)" before the fix
    name = _stack(_FakeVisitor(sdfg), sdfg, state, ('a', 'b'))
    assert symbolic.shapes_equal(sdfg.arrays[name].shape, (2, wide))


def test_concatenate_shape_check_equalizes_symbols():
    wide = dace.symbol('N', dace.int32)
    narrow = dace.symbol('N', dace.int64)

    sdfg = dace.SDFG('concat_symbol_instances')
    sdfg.add_array('a', [3, wide], dace.float64)
    sdfg.add_array('b', [3, narrow], dace.float64)
    state = sdfg.add_state()

    # raised "Array shapes do not match at index 0" before the fix
    name = _concat(_FakeVisitor(sdfg), sdfg, state, ('a', 'b'), axis=0)
    sdfg.arrays[name].transient = False
    sdfg.validate()

    rng = np.random.default_rng(1)
    a = rng.random((3, 4))
    b = rng.random((3, 4))
    out = np.zeros((6, 4))
    sdfg.compile()(a=a, b=b, N=4, **{name: out})
    assert np.allclose(out, np.concatenate([a, b], axis=0))


def test_tensordot_contracting_modes_equalize_symbols():
    wide = dace.symbol('N', dace.int32)
    narrow = dace.symbol('N', dace.int64)

    sdfg = dace.SDFG('tensordot_symbol_instances')
    sdfg.add_array('a', [wide, 4], dace.float64)
    sdfg.add_array('b', [narrow, 5], dace.float64)
    state = sdfg.add_state()

    # raised "The input tensors' contracting modes must have the same length." before the fix
    name = _tensordot(_FakeVisitor(sdfg), sdfg, state, 'a', 'b', axes=([0], [0]))
    assert symbolic.shapes_equal(sdfg.arrays[name].shape, (4, 5))


def test_matmul_vector_forms_equalize_symbols():
    wide = dace.symbol('N', dace.int32)
    narrow = dace.symbol('N', dace.int64)

    shapes = {
        'vecvec': ([wide], [narrow]),
        'matvec': ([3, wide], [narrow]),
        'vecmat': ([wide], [narrow, 3]),
        'matmat': ([3, wide], [narrow, 3]),
    }
    for label, (shape_a, shape_b) in shapes.items():
        sdfg = dace.SDFG(f'matmul_symbol_instances_{label}')
        sdfg.add_array('a', shape_a, dace.float64)
        sdfg.add_array('b', shape_b, dace.float64)
        state = sdfg.add_state()

        # each shape combination emitted a spurious "may not match" UserWarning before the fix
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _matmult(_FakeVisitor(sdfg), sdfg, state, 'a', 'b')
        assert not any('may not match' in str(w.message) for w in caught), (label, caught)


def test_boolean_array_index_shape_equalizes_symbols():
    wide = dace.symbol('N', dace.int32)
    narrow = dace.symbol('N', dace.int64)

    arr = dace.data.Array(dace.float64, [wide])
    mask = dace.data.Array(dace.bool, [narrow])
    das = {'mask': mask}
    ast_ndslice = [ast.Name(id='mask')]

    # raised "Invalid indexing into array "mask". Shape of boolean index must match ..." before the fix
    ndslice, offsets, new_axes, arrdims = _fill_missing_slices(das, ast_ndslice, arr, [0])
    assert arrdims == {0: 'mask'}


if __name__ == '__main__':
    test_dot_shape_check_equalizes_symbols()
    test_stack_shape_check_equalizes_symbols()
    test_concatenate_shape_check_equalizes_symbols()
    test_tensordot_contracting_modes_equalize_symbols()
    test_matmul_vector_forms_equalize_symbols()
    test_boolean_array_index_shape_equalizes_symbols()
