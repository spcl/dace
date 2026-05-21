# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for symbolic serialization, including the symbol-dtype consistency problem.

The problem: a map or loop iterator is not registered in ``sdfg.symbols``, and its
dtype is inferred from its range (int64 for an integer range). A free ``symbol()``,
however, defaults to ``DEFAULT_SYMBOL_TYPE`` (int32) -- so a memlet index parsed
from a string (``A[i]``) yields an int32 ``i`` while the same iterator is int64 in
its scope. Equality is name-based, so symbolic math stays correct (``i - i == 0``),
but SymPy caches ``Add``/``Min``/``Max`` by a name-based key and conflates the two
instances. Serialization therefore flips non-deterministically: the in-scope ``i``
may be saved as bare ``$i`` (int32) or ``symbol($i, dtype=dace.int64)`` depending on
which instance the cache happened to hold.

The fix does not change the default dtype or symbol equality. Instead, serialization
resolves each symbol's dtype from the enclosing SDFG scope (``symbols_defined_at``)
and emits that, so the saved form is a deterministic function of the scope rather
than of the SymPy cache. These tests pin that down at the SDFG level, where the
scope authority exists; a bare symbolic expression has no scope and so cannot be
canonicalized in isolation.
"""

import filecmp
import json
import tempfile

import numpy as np
import pytest

import dace
from dace import subsets, symbolic
from dace.codegen.common import sym2cpp
from dace.properties import DictProperty, ListProperty
from dace.sdfg.infer_types import infer_connector_types
import sympy


def test_symbolic_serialization_roundtrip_preserves_metadata():
    typed_sym = symbolic.symbol('test_sym', dtype=dace.uint64, nonnegative=True)
    expr = symbolic.SymExpr(typed_sym + symbolic.TypedConstant(np.int16(2)),
                            typed_sym + symbolic.TypedConstant(np.int16(4)))

    serialized = symbolic.serialize_symbolic(expr)
    restored = symbolic.deserialize_symbolic(serialized)

    assert serialized.startswith('SymExpr(')
    assert '$test_sym' in serialized
    assert isinstance(restored, symbolic.SymExpr)

    restored_sym = next(iter(restored.expr.free_symbols))
    assert isinstance(restored_sym, symbolic.symbol)
    assert restored_sym.name == 'test_sym'
    assert restored_sym.dtype == dace.uint64
    assert restored_sym.is_nonnegative

    constants = [atom for atom in restored.expr.atoms() if isinstance(atom, symbolic.TypedConstant)]
    assert len(constants) == 1
    assert constants[0].dtype == dace.int16
    assert int(constants[0]) == 2


def test_range_json_roundtrip_uses_symbolic_deserializer():
    sym = symbolic.symbol('N', dtype=dace.uint64, nonnegative=True)
    rng = subsets.Range([(symbolic.TypedConstant(np.int16(2)), sym + symbolic.TypedConstant(np.int16(6)),
                          symbolic.TypedConstant(np.int16(2)), symbolic.TypedConstant(np.uint8(4)))])

    restored = subsets.Range.from_json(rng.to_json())
    start, end, step = restored.ranges[0]
    tile = restored.tile_sizes[0]

    assert isinstance(start, symbolic.TypedConstant)
    assert start.dtype == dace.int16
    assert int(start) == 2

    restored_sym = next(iter(end.free_symbols))
    assert isinstance(restored_sym, symbolic.symbol)
    assert restored_sym.dtype == dace.uint64
    assert restored_sym.is_nonnegative

    assert isinstance(step, symbolic.TypedConstant)
    assert step.dtype == dace.int16
    assert isinstance(tile, symbolic.TypedConstant)
    assert tile.dtype == dace.uint8


def test_symstr_codegen_for_typed_constants():
    expr = symbolic.deserialize_symbolic('2i16 + $N')

    python_expr = symbolic.symstr(expr)
    cpp_expr = symbolic.symstr(expr, cpp_mode=True)

    assert 'dace.int16(2)' in python_expr
    assert 'int16_t(2)' in cpp_expr


def test_symbol_name_clash_roundtrip():
    expr = symbolic.symbol('sin', dtype=dace.uint64, nonnegative=True) + symbolic.TypedConstant(np.uint64(1))

    serialized = symbolic.serialize_symbolic(expr)
    restored = symbolic.deserialize_symbolic(serialized)

    restored_sym = next(iter(restored.free_symbols))
    assert isinstance(restored_sym, symbolic.symbol)
    assert restored_sym.name == 'sin'
    assert restored_sym.dtype == dace.uint64


def test_complex_symbol_roundtrip_preserves_dtype():
    expr = symbolic.symbol('c', dtype=dace.complex128)

    restored = symbolic.deserialize_symbolic(symbolic.serialize_symbolic(expr))

    restored_sym = next(iter(restored.free_symbols))
    assert isinstance(restored_sym, symbolic.symbol)
    assert restored_sym.name == 'c'
    assert restored_sym.dtype == dace.complex128


@pytest.mark.parametrize('dtype', [dace.complex64, dace.complex128])
def test_suffixless_dtype_constant_roundtrips_via_cast_form(dtype):
    """A constant whose dtype has no literal suffix (complex) serializes
    via the parseable ``dace.<dtype>(value)`` form and round-trips."""
    tc = symbolic.deserialize_symbolic(f'dace.{dtype.to_string()}(5)')
    assert isinstance(tc, symbolic.TypedConstant) and tc.dtype == dtype

    serialized = symbolic.serialize_symbolic(tc)
    restored = symbolic.deserialize_symbolic(serialized)
    assert isinstance(restored, symbolic.TypedConstant)
    assert restored.dtype == dtype and int(restored.value) == 5
    assert symbolic.serialize_symbolic(restored) == serialized


def test_complex_literal_parses_as_complex128():
    tc = symbolic.deserialize_symbolic('4j')
    assert isinstance(tc, symbolic.TypedConstant) and tc.dtype == dace.complex128
    assert complex(sympy.re(tc.value), sympy.im(tc.value)) == complex(0, 4)


def test_complex_constant_parse_save_roundtrip():
    tc = symbolic.deserialize_symbolic('complex(3.0, 4.2)')
    assert isinstance(tc, symbolic.TypedConstant) and tc.dtype == dace.complex128
    assert complex(sympy.re(tc.value), sympy.im(tc.value)) == complex(3.0, 4.2)

    serialized = symbolic.serialize_symbolic(tc)
    assert serialized == '(3.0 + 4.2j)c128'
    restored = symbolic.deserialize_symbolic(serialized)
    assert restored.dtype == dace.complex128
    assert symbolic.serialize_symbolic(restored) == serialized

    c64 = symbolic.TypedConstant(complex(1, 2), dace.complex64)
    s64 = symbolic.serialize_symbolic(c64)
    assert s64 == '(1.0 + 2.0j)c64'
    r64 = symbolic.deserialize_symbolic(s64)
    assert r64.dtype == dace.complex64 and symbolic.serialize_symbolic(r64) == s64


def test_sym2cpp_emits_uint64_literals():
    expr = symbolic.TypedConstant(np.uint64(1)) + symbolic.symbol('N', dtype=dace.uint64)

    cpp_expr = sym2cpp(expr)

    assert '1ULL' in cpp_expr


def test_sym2cpp_emits_float32_literals():
    expr = symbolic.TypedConstant(np.float32(1.5)) + symbolic.symbol('N', dtype=dace.float32)

    cpp_expr = sym2cpp(expr)

    assert '1.5f' in cpp_expr


def test_undefined_symbol_serialization_uses_dollar_prefix():
    undefined = symbolic.UndefinedSymbol()

    serialized = symbolic.serialize_symbolic(undefined)
    restored = symbolic.deserialize_symbolic(serialized)

    assert serialized == '$?'
    assert isinstance(restored, symbolic.UndefinedSymbol)


def test_symbol_assumption_roundtrip_preserves_bool_metadata():
    serialized = 'symbol($i, nonnegative=True)'

    restored = symbolic.deserialize_symbolic(serialized)

    assert symbolic.serialize_symbolic(restored) == serialized


def test_same_name_symbols_with_different_dtypes_serialize_independently():
    typed = symbolic.symbol('i', dtype=dace.int16)
    default = symbolic.symbol('i')

    assert symbolic.serialize_symbolic(typed) == 'symbol($i, dtype=dace.int16)'
    assert symbolic.serialize_symbolic(default) == '$i'


def test_typed_symbol_deserialization_does_not_strip_dtype():
    serialized = '2*(1 + symbol($M, dtype=dace.int16))*symbol($M, dtype=dace.int16)'

    restored = symbolic.deserialize_symbolic(serialized)

    assert symbolic.serialize_symbolic(restored) == serialized


def test_rational_addition_roundtrip_preserves_serialization():
    serialized = '-10/3 + 1/3*$i'

    restored = symbolic.deserialize_symbolic(serialized)

    assert symbolic.serialize_symbolic(restored) == serialized


def test_pystr_to_symbolic_preserves_typed_symbols():
    # Prime the parser cache with an equal untyped SymPy expression.
    cache_seed_expr = symbolic.deserialize_symbolic('-1 + $N')
    cache_seed_restored = symbolic.pystr_to_symbolic(cache_seed_expr)
    typed_expr = symbolic.deserialize_symbolic('-1 + symbol($N, dtype=dace.int16)')

    typed_restored = symbolic.pystr_to_symbolic(typed_expr)

    assert cache_seed_restored is cache_seed_expr
    assert symbolic.serialize_symbolic(cache_seed_restored) == '-1 + $N'
    assert typed_restored is typed_expr
    assert symbolic.serialize_symbolic(typed_restored) == '-1 + symbol($N, dtype=dace.int16)'


def test_power_deserialization_preserves_typed_symbols_after_plain_power():
    plain_power = symbolic.deserialize_symbolic('$N**2')

    typed_power = symbolic.deserialize_symbolic('symbol($N, dtype=dace.int16)**2')

    assert symbolic.serialize_symbolic(plain_power) == '$N**2'
    assert symbolic.serialize_symbolic(typed_power) == 'symbol($N, dtype=dace.int16)**2'


@pytest.mark.parametrize('simplify', [None, False])
def test_pystr_to_symbolic_keeps_basic_unsimplified_by_default(simplify):
    expr = sympy.Add(symbolic.symbol('N'), 1, 1, evaluate=False)

    restored = symbolic.pystr_to_symbolic(expr, simplify=simplify)

    assert restored is expr
    assert len(restored.args) == 3


def test_pystr_to_symbolic_simplifies_basic_when_requested():
    expr = sympy.Add(symbolic.symbol('N'), 1, 1, evaluate=False)

    restored = symbolic.pystr_to_symbolic(expr, simplify=True)

    assert restored == symbolic.symbol('N') + 2
    assert len(restored.args) == 2


def test_range_json_roundtrip_preserves_typed_symbol_minus_one():
    json_range = {
        'type': 'Range',
        'ranges': [{
            'start': '0',
            'end': '-1 + symbol($N, dtype=dace.int16)',
            'step': '1',
            'tile': '1',
        }],
    }

    rng = subsets.Range.from_json(json_range)

    assert symbolic.serialize_symbolic(rng.ranges[0][0]) == '0'
    assert symbolic.serialize_symbolic(rng.ranges[0][1]) == '-1 + symbol($N, dtype=dace.int16)'
    assert symbolic.serialize_symbolic(rng.ranges[0][2]) == '1'
    assert symbolic.serialize_symbolic(rng.tile_sizes[0]) == '1'


def test_scalar_memlet_connector_type_after_symbolic_range_roundtrip():
    i = symbolic.symbol('i', dtype=dace.int64)
    stencil_i = symbolic.symbol('stencil_i', dtype=dace.int64)
    start = 2 * i - 2 * stencil_i + 1
    rng = subsets.Range([(start, start, 1)])
    restored = subsets.Range.from_json(rng.to_json())
    restored.replace({i: stencil_i})

    assert 'i' not in restored.free_symbols
    assert restored.ranges[0][0] == 1
    assert restored.ranges[0][1] == 1
    assert restored.num_elements() == 1

    sdfg = dace.SDFG('scalar_memlet_connector_after_symbolic_roundtrip')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_scalar('B', dace.float64)
    state = sdfg.add_state()
    read = state.add_read('A')
    tasklet = state.add_tasklet('use_scalar', {'inp'}, {'out'}, 'out = inp')
    write = state.add_write('B')
    state.add_edge(read, None, tasklet, 'inp', dace.Memlet(data='A', subset=restored))
    state.add_edge(tasklet, 'out', write, None, dace.Memlet('B[0]'))

    infer_connector_types(sdfg)

    assert tasklet.in_connectors['inp'] == dace.float64


def test_list_property_symbolic_type_json_roundtrip_supports_plain_names():
    prop = ListProperty(element_type=sympy.Basic)

    assert prop.to_json(['START']) == ['$START']

    restored = prop.from_json(prop.to_json(['START']))

    assert len(restored) == 1
    assert restored[0] == symbolic.symbol('START')


def test_dict_property_symbolic_type_json_roundtrip_supports_plain_names():
    prop = DictProperty(key_type=str, value_type=sympy.Basic)

    assert prop.to_json({'N': 'N'}) == {'N': '$N'}

    restored = prop.from_json(prop.to_json({'N': 'N'}))

    assert restored == {'N': symbolic.symbol('N')}


@pytest.mark.parametrize('expr', [
    sympy.Add(symbolic.TypedConstant(np.int16(10)),
              sympy.Mul(sympy.S.NegativeOne, symbolic.symbol('i'), evaluate=False),
              evaluate=False),
    sympy.Mul(symbolic.TypedConstant(np.int16(10)), symbolic.symbol('i'), evaluate=False),
    sympy.Mul(symbolic.TypedConstant(np.int16(10)), sympy.Pow(symbolic.symbol('i'), -1, evaluate=False),
              evaluate=False),
    sympy.Pow(symbolic.TypedConstant(np.int16(10)), symbolic.symbol('i'), evaluate=False),
    sympy.Mod(symbolic.symbol('i'), symbolic.TypedConstant(np.int16(3)), evaluate=False),
])
def test_typed_binary_operator_roundtrip_preserves_serialization(expr):
    serialized = symbolic.serialize_symbolic(expr)
    restored = symbolic.deserialize_symbolic(serialized)

    assert symbolic.serialize_symbolic(restored) == serialized


def test_plain_integer_roundtrip_converts_to_sympy_integer():
    restored = symbolic.deserialize_symbolic(symbolic.serialize_symbolic(sympy.Integer(10)))

    assert restored == 10
    assert isinstance(restored, sympy.Integer)


def test_plain_sympy_integer_serializes_without_typed_suffix():
    assert symbolic.serialize_symbolic(sympy.Integer(42)) == '42'


@pytest.mark.parametrize('value', [-7, 2**80])
def test_plain_python_integer_deserialization_uses_sympy_integer(value):
    restored = symbolic.deserialize_symbolic(value)

    assert restored == value
    assert isinstance(restored, sympy.Integer)


def test_untyped_literals_keep_plain_sympy_form():
    """Untyped literals deserialize to plain SymPy numbers (no implicit
    DaCe-typing)."""
    assert symbolic.deserialize_symbolic('5') == sympy.Integer(5)
    assert isinstance(symbolic.deserialize_symbolic('5'), sympy.Integer)
    assert isinstance(symbolic.deserialize_symbolic('5.0'), sympy.Float)


@pytest.mark.parametrize(
    'text,ctype',
    [
        ('double(5)', 'double'),  # float64 cast-wrapper form
        ('int(5)', 'int'),  # int32 cast-wrapper form
        ('float(5)', 'float'),  # float32
        ('int64_t(5)', 'int64_t'),  # explicit-suffix dtype as a ctype cast
    ])
def test_cpp_ctype_cast_parses_to_typed_constant(text, ctype):
    """``double(5)``/``int(5)`` (the C++ printer's cast fallback) must round-trip
    into a TypedConstant."""
    restored = symbolic.deserialize_symbolic(text)
    assert isinstance(restored, symbolic.TypedConstant), text
    assert restored.dtype.ctype == ctype


@pytest.mark.parametrize('op', ['Min', 'Max'])
@pytest.mark.parametrize('literal', [
    '5.0',
    '5.0f64',
    '5f32',
    '7f64',
    '(3.0 + 4.0j)c128',
    '(3.0 - 4.0j)c128',
    '(3.0 + 4.0j)c64',
])
def test_minmax_with_float_literal_roundtrip(op, literal):
    serialized = f'{op}({literal}, $N)'
    restored = symbolic.deserialize_symbolic(serialized)
    assert symbolic.serialize_symbolic(restored) == serialized


def test_minmax_with_ctype_cast_int_literal_normalizes_then_stable():
    first = symbolic.serialize_symbolic(symbolic.deserialize_symbolic('Max(double(7), 5.0)'))
    assert first == 'Max(7f64, 5.0)'
    second = symbolic.serialize_symbolic(symbolic.deserialize_symbolic(first))
    assert first == second


@pytest.mark.parametrize('serialized', [
    '5i16',
    '7i64',
    '5f32',
    '5.0f64',
    '(3.0 + 4.0j)c128',
    '(3.0 - 4.0j)c128',
    '(-3.0 + 4.0j)c128',
    '(3.0 + 4.0j)c64',
    '(4.0j)c128',
    '(-4.0j)c128',
    '(4.0j)c64',
    '(-4.0j)c64',
])
def test_typed_constant_canonical_form_roundtrips(serialized):
    restored = symbolic.deserialize_symbolic(serialized)
    assert isinstance(restored, symbolic.TypedConstant)
    assert symbolic.serialize_symbolic(restored) == serialized


@pytest.mark.parametrize('input_form,canonical', [
    ('4j', '(4.0j)c128'),
    ('-4j', '(-4.0j)c128'),
    ('(0.0 + 4.0j)c128', '(4.0j)c128'),
    ('complex(3.0, 4.0)', '(3.0 + 4.0j)c128'),
    ('dace.complex128(complex(3.0, 4.0))', '(3.0 + 4.0j)c128'),
    ('dace.complex64(complex(3.0, 4.0))', '(3.0 + 4.0j)c64'),
    ('dace.complex128(complex(3.0, -4.0))', '(3.0 - 4.0j)c128'),
])
def test_legacy_complex_form_normalizes_to_canonical_suffix(input_form, canonical):
    first = symbolic.serialize_symbolic(symbolic.deserialize_symbolic(input_form))
    assert first == canonical
    second = symbolic.serialize_symbolic(symbolic.deserialize_symbolic(first))
    assert first == second


def _poison_symbol_cache(name, dtype):
    """Seed SymPy's process-global cache with a non-default-typed same-name symbol
    inside compound expressions. SymPy caches ``Add``/``Min``/``Max`` by a
    name-based key, so this makes a later default-typed ``name`` resolve to the
    cached non-default instance -- the race that makes serialization flaky."""
    other = symbolic.symbol('lidx', dtype=dtype)
    return sympy.Min(10, symbolic.symbol(name, dtype=dtype) + other)


def _map_index_sdfg():
    """A map over an integer range whose body memlet indexes by the iterator.
    The iterator is int64 in the map scope, but ``A[i]`` parses ``i`` as the
    int32 default -- the exact same-name/different-dtype situation."""
    sdfg = dace.SDFG('map_index')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('t',
                             map_ranges={'i': '0:20'},
                             inputs={'a': dace.Memlet('A[i]')},
                             outputs={'b': dace.Memlet('B[i]')},
                             code='b = a + 1.0',
                             external_edges=True)
    return sdfg


@pytest.mark.parametrize('cache_dtype', [dace.int32, dace.int64])
def test_map_iterator_serializes_with_scope_dtype(cache_dtype):
    """Regardless of which dtype the SymPy cache holds for ``i``, the iterator is
    serialized with the int64 dtype its map scope declares, not the instance's."""
    _poison_symbol_cache('i', cache_dtype)
    sdfg = _map_index_sdfg()
    js = json.dumps(sdfg.to_json())
    assert 'symbol($i, dtype=dace.int64)' in js
    assert '"$i"' not in js and '$i ' not in js


def test_sdfg_serialization_independent_of_symbol_cache_order():
    """The serialized form of an SDFG is a function of its scope, not of the
    order in which same-name symbols of different dtypes entered the cache."""
    sdfg = _map_index_sdfg()
    _poison_symbol_cache('i', dace.int32)
    first = json.dumps(sdfg.to_json())
    _poison_symbol_cache('i', dace.int64)
    second = json.dumps(sdfg.to_json())
    assert first == second


def test_save_load_save_idempotent_under_cache_poisoning():
    """The codegen save->load->save idempotence check stays byte-stable even when
    the cache is poisoned with a conflicting same-name dtype between the saves."""
    sdfg = _map_index_sdfg()
    with tempfile.TemporaryDirectory() as tmp:
        sdfg.save(f'{tmp}/a.sdfg', hash=False)
        loaded = dace.SDFG.from_file(f'{tmp}/a.sdfg')
        _poison_symbol_cache('i', dace.int64)
        loaded.save(f'{tmp}/b.sdfg', hash=False)
        assert filecmp.cmp(f'{tmp}/a.sdfg', f'{tmp}/b.sdfg', shallow=False)


def test_declared_symbol_dtype_wins_over_array_instance():
    """A symbol declared int64 in ``sdfg.symbols`` but appearing as the int32
    default instance inside an array shape must serialize with its declared
    dtype. ``symbols_defined_at`` lets an array's free-symbol instance dtype
    shadow the declaration, which is cache-unstable, so the declaration wins."""
    N = dace.symbol('N', dace.int64)
    sdfg = dace.SDFG('declared')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('t',
                             map_ranges={'i': '0:N'},
                             inputs={'a': dace.Memlet('A[i]')},
                             outputs={'b': dace.Memlet('B[i]')},
                             code='b = a',
                             external_edges=True)
    _poison_symbol_cache('N', dace.int32)
    js = json.dumps(sdfg.to_json())
    assert 'symbol($N, dtype=dace.int64)' in js
    assert 'symbol($N, dtype=dace.int32)' not in js


def test_map_param_dtype_differs_from_default_symbol():
    """A map parameter's dtype is inferred from its range bounds (int64 for an
    integer range), while a free ``symbol()`` defaults to ``DEFAULT_SYMBOL_TYPE``
    (int32). This is the source of same-name/different-dtype collisions."""
    N = dace.symbol('N')
    sdfg = dace.SDFG('map_param_dtype')
    sdfg.add_array('A', [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', {'i': '0:N'})
    t = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    w = state.add_access('A')
    state.add_edge(me, None, t, None, dace.Memlet())
    state.add_edge(t, 'o', mx, 'IN_A', dace.Memlet('A[i]'))
    state.add_edge(mx, 'OUT_A', w, None, dace.Memlet('A[0:N]'))
    mx.add_in_connector('IN_A')
    mx.add_out_connector('OUT_A')

    assert me.new_symbols(sdfg, state, {})['i'] == dace.int64
    assert symbolic.symbol('i').dtype == symbolic.DEFAULT_SYMBOL_TYPE


def test_loop_variable_registered_as_int64_symbol():
    """The Python frontend registers a ``range`` loop variable in
    ``sdfg.symbols`` with an int64 dtype, even though a free ``symbol()`` of the
    same name defaults to int32."""
    N = dace.symbol('N')

    @dace.program
    def loopprog(A: dace.float64[N]):
        for i in range(N):
            A[i] = 1.0

    sdfg = loopprog.to_sdfg()
    assert sdfg.symbols['i'] == dace.int64
    assert symbolic.symbol('i').dtype == symbolic.DEFAULT_SYMBOL_TYPE


def test_add_symbol_allows_dtype_collision_with_map_param():
    """``add_symbol`` does not see ``i`` as already in use by the int64 map
    parameter (map params are scope-local, not in ``sdfg.symbols``), so declaring
    it int32 leaves ``sdfg.symbols`` disagreeing with ``symbols_defined_at``. This
    is benign for serialization, which resolves each symbol's dtype from its scope
    (``symbols_defined_at``), not from ``sdfg.symbols``."""
    N = dace.symbol('N')
    sdfg = dace.SDFG('collision')
    sdfg.add_array('A', [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', {'i': '0:N'})
    t = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    w = state.add_access('A')
    state.add_edge(me, None, t, None, dace.Memlet())
    state.add_edge(t, 'o', mx, 'IN_A', dace.Memlet('A[i]'))
    state.add_edge(mx, 'OUT_A', w, None, dace.Memlet('A[0:N]'))
    mx.add_in_connector('IN_A')
    mx.add_out_connector('OUT_A')

    sdfg.add_symbol('i', dace.int32)
    assert sdfg.symbols['i'] == dace.int32
    assert state.symbols_defined_at(t)['i'] == dace.int64


def test_dynamic_map_range_connector_serializes_without_crash():
    """Edge case: a map whose range bounds come from *untyped* dynamic connectors
    (``start:stop``). ``MapEntry.new_symbols`` reports ``start``/``stop`` with the
    connector's dtype, which is ``void`` (``type=None``) for an untyped connector.
    The scope authority must not try to render that unprintable type onto the
    symbol -- it falls back to the symbol's own dtype, so ``start``/``stop`` stay
    bare ``$start``/``$stop``."""
    sdfg = dace.SDFG('dyn_range')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('lo', [1], dace.int32)
    sdfg.add_array('hi', [1], dace.int32)
    state = sdfg.add_state()
    lo, hi, w = state.add_access('lo'), state.add_access('hi'), state.add_access('A')
    me, mx = state.add_map('m', {'idx': 'start:stop'})
    me.add_in_connector('start')
    me.add_in_connector('stop')
    t = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    state.add_edge(lo, None, me, 'start', dace.Memlet('lo[0]'))
    state.add_edge(hi, None, me, 'stop', dace.Memlet('hi[0]'))
    state.add_edge(me, None, t, None, dace.Memlet())
    state.add_edge(t, 'o', mx, 'IN_A', dace.Memlet('A[idx]'))
    state.add_edge(mx, 'OUT_A', w, None, dace.Memlet('A[0:20]'))
    mx.add_in_connector('IN_A')
    mx.add_out_connector('OUT_A')

    assert me.new_symbols(sdfg, state, {})['start'].type is None  # the void connector
    js = json.dumps(sdfg.to_json())  # must not raise on the void type
    assert '$start' in js and '$stop' in js


def test_structure_free_symbols_tolerates_serialized_symbolic_member():
    """Edge case: a Structure member that is symbolic (``Tensor.value_count``)
    round-trips back as its serialized string form (``'$nnz'``) rather than a
    symbol. ``free_symbols`` -- which serialization now calls via
    ``symbols_defined_at`` -- must recover its symbols instead of assuming every
    member is a data descriptor."""
    nnz = dace.symbol('nnz')
    csr = dace.data.Tensor(dace.float32, (dace.symbol('M'), dace.symbol('N')), [(dace.data.TensorIndexDense(), 0),
                                                                                (dace.data.TensorIndexCompressed(), 1)],
                           nnz, 'CSR')
    restored = dace.data.Tensor.from_json(csr.to_json())
    assert isinstance(restored.members['value_count'], str)  # the serialized member
    assert nnz in restored.free_symbols


if __name__ == '__main__':
    pytest.main([__file__])
