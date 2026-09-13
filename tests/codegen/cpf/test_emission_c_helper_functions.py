# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every C helper family CPF renders is a function, and the rendered unit computes what numpy does.

One program per family: the min/max and maths functions the printers name by type, the integer
helpers, and the helpers a native body calls with a loop of their own -- the prefix scans, the
find-first, the sort and the duplicate check. Each case pins the helper call it needs, so a program
that stops reaching its family fails instead of passing vacuously.
"""
import math
import re
from typing import Callable, Dict, Tuple

import numpy as np
import pytest

import dace
from dace.codegen.cpf import render as render_sdfg
from dace.libraries.sort.nodes.integer_sort import IntegerSort
from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck
from dace.libraries.standard.nodes import FindFirst
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from tests.codegen.cpf.conftest import assert_standalone, build_standalone, call_standalone

N = dace.symbol('N')

#: What a C unit must not carry: a macro definition, the C23 type query and the pragma operator the
#: statement macros needed, and an ``auto`` declaration.
MACRO_CONSTRUCTS = (re.compile(r'^[ \t]*#[ \t]*define\b', re.M), re.compile(r'\btypeof(?:_unqual)?\b'),
                    re.compile(r'\b_Pragma\b'), re.compile(r'\bauto\b'))

#: ``(sdfg, helper calls the rendering must contain, arguments, expected outputs by name)``.
Case = Tuple[dace.SDFG, Tuple[str, ...], Dict[str, object], Dict[str, np.ndarray]]


@dace.program
def clamp(x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = min(max(x[i], 0.0), 1.0)


@dace.program
def floored(x: dace.int64[N], q: dace.int64[N], r: dace.int64[N]):
    for i in dace.map[0:N]:
        q[i] = x[i] // 3
        r[i] = x[i] % 3


@dace.program
def rounded_root(x: dace.float32[N], y: dace.float32[N]):
    for i in dace.map[0:N]:
        y[i] = math.floor(math.sqrt(abs(x[i])))


@dace.program
def every_third(x: dace.float64[N], y: dace.float64[N]):
    y[::3] = x[::3]


@dace.program
def running_product(x: dace.int64[N], y: dace.int64[N]):
    y[:] = np.cumprod(x)


def scan_sdfg(op: ScanOp, exclusive: bool, identity, source: dace.typeclass, target: dace.typeclass) -> dace.SDFG:
    """``dst`` as the scan of ``src``, through the sequential expansion CPF renders."""
    sdfg = dace.SDFG('scan')
    sdfg.add_array('src', [N], source)
    sdfg.add_array('dst', [N], target)
    state = sdfg.add_state()
    node = Scan('scan', op=op, exclusive=exclusive, identity=identity)
    node.implementation = 'pure'
    state.add_node(node)
    state.add_edge(state.add_read('src'), None, node, Scan.INPUT_CONNECTOR_NAME, dace.Memlet('src[0:N]'))
    state.add_edge(node, Scan.OUTPUT_CONNECTOR_NAME, state.add_write('dst'), None, dace.Memlet('dst[0:N]'))
    return sdfg


def find_first_sdfg() -> dace.SDFG:
    """The first index at which a ``float64`` array and an ``int32`` array both satisfy a test."""
    sdfg = dace.SDFG('find_first')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('k', [N], dace.int32)
    sdfg.add_array('out', [1], dace.int64)
    state = sdfg.add_state()
    node = FindFirst('ff', predicate='_a[__i] > 0.5 && _k[__i] == 3', begin=0, end=N)
    node.add_in_connector('_a', dace.pointer(dace.float64))
    node.add_in_connector('_k', dace.pointer(dace.int32))
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_a', dace.Memlet.from_array('a', sdfg.arrays['a']))
    state.add_edge(state.add_read('k'), None, node, '_k', dace.Memlet.from_array('k', sdfg.arrays['k']))
    state.add_edge(node, '_out_idx', state.add_write('out'), None, dace.Memlet('out[0]'))
    return sdfg


def sort_sdfg() -> dace.SDFG:
    """``ordered`` as the sorted copy of ``keys``."""
    sdfg = dace.SDFG('sort')
    sdfg.add_array('keys', [N], dace.int64)
    sdfg.add_array('ordered', [N], dace.int64)
    state = sdfg.add_state()
    node = IntegerSort('sort')
    node.implementation = 'pure'
    state.add_node(node)
    state.add_edge(state.add_read('keys'), None, node, IntegerSort.INPUT_CONNECTOR_NAME, dace.Memlet('keys[0:N]'))
    state.add_edge(node, IntegerSort.OUTPUT_CONNECTOR_NAME, state.add_write('ordered'), None,
                   dace.Memlet('ordered[0:N]'))
    return sdfg


def duplicate_check_sdfg() -> dace.SDFG:
    """``count[0]`` as whether ``ip`` repeats a value, with no tag array wired."""
    sdfg = dace.SDFG('duplicate_check')
    sdfg.add_array('ip', [N], dace.int32)
    sdfg.add_array('count', [1], dace.int64)
    state = sdfg.add_state()
    node = ScatterConflictCheck('check')
    node.implementation = 'pure'
    state.add_node(node)
    state.add_edge(state.add_read('ip'), None, node, ScatterConflictCheck.INPUT_CONNECTOR_NAME, dace.Memlet('ip[0:N]'))
    state.add_edge(node, ScatterConflictCheck.OUTPUT_CONNECTOR_NAME, state.add_write('count'), None,
                   dace.Memlet('count[0]'))
    return sdfg


def minmax_case() -> Case:
    x = np.linspace(-1.0, 2.0, 61)
    return (clamp.to_sdfg(simplify=True), ('cpf_max_float64(', 'cpf_min_float64('), {
        'x': x,
        'y': np.zeros_like(x),
        'N': x.size
    }, {
        'y': np.minimum(np.maximum(x, 0.0), 1.0)
    })


def floored_case() -> Case:
    x = np.arange(-20, 21, dtype=np.int64)
    return (floored.to_sdfg(simplify=True), ('cpf_py_floor_int64(', 'cpf_py_mod_int64('), {
        'x': x,
        'q': np.zeros_like(x),
        'r': np.zeros_like(x),
        'N': x.size
    }, {
        'q': x // 3,
        'r': x % 3
    })


def maths_case() -> Case:
    x = np.linspace(-50.0, 50.0, 101).astype(np.float32)
    return (rounded_root.to_sdfg(simplify=True), ('fabsf(', 'sqrtf(', 'floorf('), {
        'x': x,
        'y': np.zeros_like(x),
        'N': x.size
    }, {
        'y': np.floor(np.sqrt(np.abs(x)))
    })


def int_ceil_case() -> Case:
    x = np.arange(1.0, 11.0)
    expected = np.zeros_like(x)
    expected[::3] = x[::3]
    return (every_third.to_sdfg(simplify=True), ('cpf_int_ceil_int64(', ), {
        'x': x,
        'y': np.zeros_like(x),
        'N': x.size
    }, {
        'y': expected
    })


def scan_product_case() -> Case:
    x = np.array([1, 2, 1, 2, 2, 1, 3, 1, 2, 1], dtype=np.int64)
    return (running_product.to_sdfg(simplify=True), ('cpf_scan_incl_product_int64_int64_int64(', ), {
        'x': x,
        'y': np.zeros_like(x),
        'N': x.size
    }, {
        'y': np.cumprod(x)
    })


def scan_widening_case() -> Case:
    # 300 ones: folded at the int8 input type the rank would wrap at 128.
    src = np.ones(300, dtype=np.int8)
    return (scan_sdfg(ScanOp.SUM, True, 0, dace.int8, dace.int64), ('cpf_scan_excl_sum_int8_int64_int64(', ), {
        'src': src,
        'dst': np.zeros(300, dtype=np.int64),
        'N': 300
    }, {
        'dst': np.concatenate(([0], np.cumsum(src.astype(np.int64))[:-1]))
    })


def scan_max_case() -> Case:
    src = np.random.default_rng(3).uniform(-1.0, 1.0, 257)
    return (scan_sdfg(ScanOp.MAX, False, None, dace.float64,
                      dace.float64), ('cpf_scan_incl_max_float64_float64_float64(', ), {
                          'src': src,
                          'dst': np.zeros_like(src),
                          'N': src.size
                      }, {
                          'dst': np.maximum.accumulate(src)
                      })


def scan_min_case() -> Case:
    src = np.random.default_rng(4).integers(-1000, 1000, 257, dtype=np.int32)
    return (scan_sdfg(ScanOp.MIN, False, None, dace.int32, dace.int32), ('cpf_scan_incl_min_int32_int32_int32(', ), {
        'src': src,
        'dst': np.zeros_like(src),
        'N': src.size
    }, {
        'dst': np.minimum.accumulate(src)
    })


def find_first_case() -> Case:
    n = 5000
    a = np.zeros(n)
    k = np.zeros(n, dtype=np.int32)
    a[100] = 1.0
    k[300] = 3
    a[517], k[517] = 1.0, 3
    return (find_first_sdfg(), ('cpf_find_first_', ), {
        'a': a,
        'k': k,
        'out': np.zeros(1, dtype=np.int64),
        'N': n
    }, {
        'out': np.flatnonzero((a > 0.5) & (k == 3))[:1]
    })


def sort_case() -> Case:
    keys = np.random.default_rng(5).integers(-10**12, 10**12, 257, dtype=np.int64)
    return (sort_sdfg(), ('cpf_sort_int64(', ), {
        'keys': keys.copy(),
        'ordered': np.zeros_like(keys),
        'N': keys.size
    }, {
        'ordered': np.sort(keys),
        'keys': keys
    })


def duplicate_check_case(repeat: bool) -> Case:
    ip = np.random.default_rng(6).permutation(64).astype(np.int32)
    if repeat:
        ip[40] = ip[7]
    return (duplicate_check_sdfg(), ('cpf_detect_collision_sized_int32(', ), {
        'ip': ip,
        'count': np.zeros(1, dtype=np.int64),
        'N': ip.size
    }, {
        'count': np.array([int(np.unique(ip).size != ip.size)], dtype=np.int64)
    })


CASES: Dict[str, Callable[[], Case]] = {
    'minmax_float64': minmax_case,
    'floored_int64': floored_case,
    'maths_float32': maths_case,
    'int_ceil': int_ceil_case,
    'scan_product_int64': scan_product_case,
    'scan_exclusive_widening': scan_widening_case,
    'scan_max_float64': scan_max_case,
    'scan_min_int32': scan_min_case,
    'find_first_two_arrays': find_first_case,
    'sort_int64': sort_case,
    'duplicate_check_permutation': lambda: duplicate_check_case(False),
    'duplicate_check_repeat': lambda: duplicate_check_case(True),
}


@pytest.mark.parametrize('label', sorted(CASES))
def test_each_c_helper_family_renders_as_functions_and_computes_what_numpy_does(label):
    """The C unit calls the typed function for its family, carries no macro, and agrees with numpy
    element for element."""
    sdfg, helpers, arguments, expected = CASES[label]()
    sdfg.name = 'cpf_c_family_%s' % label
    rendering = render_sdfg(sdfg, language='c')
    code = rendering.code
    for helper in helpers:
        assert helper in code, f'{label}: the rendering no longer calls {helper}, so this case asserts nothing:\n{code}'
    for construct in MACRO_CONSTRUCTS:
        hit = construct.search(code)
        assert hit is None, f'{label}: the C unit carries {hit.group(0)!r}:\n{code}'
    assert_standalone(code, sdfg.name, language='c')
    call_standalone(build_standalone(code, sdfg.name, language='c'), rendering.sdfg, arguments)
    for name, want in expected.items():
        assert np.array_equal(arguments[name], want), f'{label}/{name}: CPF gives {arguments[name]}, numpy {want}'
