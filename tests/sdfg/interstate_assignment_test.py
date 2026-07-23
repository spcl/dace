# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np


def test_key_replacement_same_name():

    sdfg = dace.SDFG('key_replacement_same_name')
    sdfg.add_array('inp', [1], dace.int32)
    sdfg.add_array('out', [1], dace.int32)

    first = sdfg.add_state('first_state')
    second = sdfg.add_state('second_state')
    edge = sdfg.add_edge(first, second, dace.InterstateEdge(assignments={'s': 'inp[0]'}))

    task = second.add_tasklet('t', {}, {'__out'}, '__out = s')
    access = second.add_access('out')
    second.add_edge(task, '__out', access, None, dace.Memlet('out[0]'))

    sdfg.replace('s', 's')
    assert 's' in edge.data.assignments
    sdfg.replace_dict({'s': 's'})
    assert 's' in edge.data.assignments

    rng = np.random.default_rng()
    inp = rng.integers(1, 100, 1)
    inp = np.array(inp, dtype=np.int32)
    out = np.zeros([1], dtype=np.int32)

    sdfg(inp=inp, out=out)
    assert out[0] == inp[0]


def test_boolean_literal_comparison_roundtrip():
    """A guard comparing a data access to a boolean literal (``A[i] != True``) must survive SDFG
    serialization. SymPy folds ``Ne(A[i], True)`` to ``True`` ("only Booleans can equal Booleans"),
    silently dropping the guard on round-trip; the parser rewrites it to boolean logic instead."""
    import sympy
    from dace import symbolic

    # Parser level: comparisons against a boolean literal must not collapse to a constant.
    for expr in ('b[0] != True', 'b[0] == True', 'b[0] != False', 'b[0] == False', '(b[0] != True) and (n > 0)'):
        parsed = symbolic.pystr_to_symbolic(expr)
        assert parsed not in (sympy.true, sympy.false), f'{expr!r} folded to {parsed!r}'
        assert 'b' in str(parsed), f'{expr!r} dropped the operand: {parsed!r}'

    # Ordinary comparisons are untouched.
    assert '<' in str(symbolic.pystr_to_symbolic('zqx[i] < rlmin'))

    # The guard survives an SDFG serialization round-trip (previously it silently became ``True``).
    sdfg = dace.SDFG('bool_guard_roundtrip')
    sdfg.add_array('b', [8], dace.bool)
    sdfg.add_symbol('n', dace.int32)
    sdfg.add_symbol('cond', dace.bool)
    s0 = sdfg.add_state(is_start_block=True)
    s1 = sdfg.add_state()
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'cond': '(b[0] != True) and (n > 0)'}))

    reloaded = dace.SDFG.from_json(sdfg.to_json())
    cond = next(
        str(e.data.assignments['cond']) for e in reloaded.all_interstate_edges() if 'cond' in e.data.assignments)
    assert 'b' in cond, f'boolean guard dropped on serialization round-trip: {cond!r}'


if __name__ == '__main__':
    test_key_replacement_same_name()
    test_boolean_literal_comparison_roundtrip()
