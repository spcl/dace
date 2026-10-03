# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that shadowing a name inside a C++ tasklet declares it with the type it was declared
with, rather than deducing one from whatever expression a pass substituted. """

import numpy as np

import dace


def test_cpp_shadow_declares_the_symbol_type():
    """``auto`` deduces from the replacement expression, so ``i = 2`` narrows an int64 symbol to
    int and arithmetic on it can then overflow where the symbol's own type would not."""
    sdfg = dace.SDFG('cpp_shadow_type')
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_array('out', [1], dace.int64)

    state = sdfg.add_state('s', is_start_block=True)
    tasklet = state.add_tasklet('w', {}, {'o'}, 'o = i + 1;', language=dace.Language.CPP)
    state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))

    state.replace_dict({'i': '2'})

    code = tasklet.code.as_string
    assert 'int64_t i = 2;' in code, code
    assert 'auto i' not in code, code


def test_cpp_shadow_of_a_scalar_declares_the_descriptor_type():
    """A read-only ``Scalar`` is reached the same way; its type comes off the descriptor."""
    sdfg = dace.SDFG('cpp_shadow_scalar')
    sdfg.add_scalar('param', dace.int64)
    sdfg.add_array('out', [1], dace.int64)

    state = sdfg.add_state('s', is_start_block=True)
    tasklet = state.add_tasklet('w', {}, {'o'}, 'o = param + 1;', language=dace.Language.CPP)
    state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))

    state.replace_dict({'param': '2'})

    code = tasklet.code.as_string
    assert 'int64_t param = 2;' in code, code
    assert 'auto param' not in code, code


def test_cpp_shadow_keeps_the_integer_semantics_of_the_symbol():
    """The deduced type changes results, not just spelling: a floating replacement for an integer
    symbol makes ``auto`` a ``double``, and the division below stops being integer division."""
    sdfg = dace.SDFG('cpp_shadow_semantics')
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_array('out', [1], dace.float64)

    state = sdfg.add_state('s', is_start_block=True)
    tasklet = state.add_tasklet('w', {}, {'o'}, 'o = i / 2;', language=dace.Language.CPP)
    state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))

    state.replace_dict({'i': '5.0'})

    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    assert out[0] == 2.0, f'expected integer division on an int64 symbol, got {out[0]}'


def test_cpp_shadow_of_a_map_parameter_deduces():
    """A map parameter is neither a symbol nor a descriptor, so nothing declares its type here.
    ``auto`` is right for it: it copies the type of the index variable it is renamed to."""
    sdfg = dace.SDFG('cpp_shadow_map_param')
    sdfg.add_array('out', [8], dace.int64)

    state = sdfg.add_state('s', is_start_block=True)
    entry, exit_node = state.add_map('m', {'i': '0:8'})
    tasklet = state.add_tasklet('w', {}, {'o'}, 'o = i + 1;', language=dace.Language.CPP)
    state.add_edge(entry, None, tasklet, None, dace.Memlet())
    exit_node.add_in_connector('IN_o')
    exit_node.add_out_connector('OUT_o')
    state.add_edge(tasklet, 'o', exit_node, 'IN_o', dace.Memlet('out[i]'))
    state.add_edge(exit_node, 'OUT_o', state.add_write('out'), None, dace.Memlet('out[0:8]'))
    sdfg.validate()

    assert 'i' not in sdfg.symbols and 'i' not in sdfg.arrays
    state.replace_dict({'i': 'k'})
    assert 'auto i = k;' in tasklet.code.as_string, tasklet.code.as_string


def test_cpp_same_name_rebind_emits_no_shadow():
    """A symbol re-minted with new sympy assumptions (``AssumeSymbolConstraints`` stamps
    ``nonnegative=True``) keeps its NAME. Shadowing it would emit ``int64_t i = i;`` -- a local
    initialized from itself, i.e. an uninitialized read -- and every use in the tasklet would then
    see garbage. Measured: a canonicalization trap guard built this way aborted at runtime on a
    valid input."""
    sdfg = dace.SDFG('cpp_same_name_rebind')
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_array('out', [1], dace.int64)

    state = sdfg.add_state('s', is_start_block=True)
    tasklet = state.add_tasklet('w', {}, {'o'}, 'o = i + 1;', language=dace.Language.CPP)
    state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))

    state.replace_dict({'i': dace.symbolic.symbol('i', dtype=dace.int64, nonnegative=True)})

    code = tasklet.code.as_string
    assert 'i = i;' not in code, code
    assert code == 'o = i + 1;', code

    out = np.zeros(1, dtype=np.int64)
    sdfg(out=out, i=7)
    assert out[0] == 8, f'the tasklet must read the real symbol, got {out[0]}'


if __name__ == '__main__':
    test_cpp_shadow_declares_the_symbol_type()
    test_cpp_shadow_of_a_scalar_declares_the_descriptor_type()
    test_cpp_shadow_keeps_the_integer_semantics_of_the_symbol()
    test_cpp_shadow_of_a_map_parameter_deduces()
    test_cpp_same_name_rebind_emits_no_shadow()
