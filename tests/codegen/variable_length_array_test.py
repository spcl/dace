# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Stack or heap placement of register arrays, chosen by ``Data.stack_vla``. """

import contextlib
import json
import os
import sys
import tempfile

import numpy as np
import pytest

import dace
from dace.dtypes import AllocationLifetime, StackAllocation
from dace.transformation.passes.resolve_stack_allocation import ResolveStackAllocation

N = dace.symbol('N', dtype=dace.int64)


def register_scratch_sdfg(name: str,
                          size,
                          placement: StackAllocation,
                          lifetime: AllocationLifetime = AllocationLifetime.Scope) -> dace.SDFG:
    """ ``b = a + 1`` through a register transient ``tmp`` that two states share. """
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', (size, ), dace.float64)
    sdfg.add_array('b', (size, ), dace.float64)
    _, tmp = sdfg.add_transient('tmp', (size, ), dace.float64, storage=dace.StorageType.Register, lifetime=lifetime)
    tmp.stack_vla = placement
    rng = f'0:{size}'
    first = sdfg.add_state('first', is_start_block=True)
    first.add_mapped_tasklet('fill', {'i': rng}, {'inp': dace.Memlet('a[i]')},
                             'o = inp + 1.0', {'o': dace.Memlet('tmp[i]')},
                             external_edges=True)
    second = sdfg.add_state_after(first, 'second')
    second.add_mapped_tasklet('drain', {'i': rng}, {'inp': dace.Memlet('tmp[i]')},
                              'o = inp', {'o': dace.Memlet('b[i]')},
                              external_edges=True)
    sdfg.validate()
    return sdfg


def run_scratch(sdfg: dace.SDFG, size: int, **symbols):
    a = np.random.rand(size)
    b = np.zeros(size)
    sdfg(a=a, b=b, **symbols)
    np.testing.assert_allclose(b, a + 1.0, rtol=0, atol=0)


PLACEMENT_CASES = [
    (16, StackAllocation.Auto, StackAllocation.Stack),
    (2047, StackAllocation.Auto, StackAllocation.Stack),
    (2048, StackAllocation.Auto, StackAllocation.Heap),
    (N, StackAllocation.Auto, StackAllocation.Heap),
    (16, StackAllocation.Heap, StackAllocation.Heap),
    (4096, StackAllocation.Stack, StackAllocation.Stack),
    (N, StackAllocation.Stack, StackAllocation.Stack),
]


@contextlib.contextmanager
def captured_stderr_fd():
    """ Collects what compiled code writes to file descriptor 2 into the one-item list it yields. """
    captured = ['']
    sys.stderr.flush()
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile(mode='w+') as sink:
        os.dup2(sink.fileno(), 2)
        try:
            yield captured
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            sink.seek(0)
            captured[0] = sink.read()


@pytest.mark.parametrize('size, placement, resolved', PLACEMENT_CASES)
def test_auto_placement_resolves_by_size_and_explicit_placement_is_kept(size, placement, resolved):
    sdfg = register_scratch_sdfg('resolve_placement', size, placement)
    ResolveStackAllocation().apply_pass(sdfg, {})
    assert sdfg.arrays['tmp'].stack_vla is resolved


def test_auto_placement_of_a_large_constant_array_respects_max_stack_array_size():
    """A byte limit lowered below 2048 elements must still move the array to the heap."""
    sdfg = register_scratch_sdfg('resolve_bytes', 1024, StackAllocation.Auto)
    with dace.config.set_temporary('compiler', 'max_stack_array_size', value=1024):
        ResolveStackAllocation().apply_pass(sdfg, {})
    assert sdfg.arrays['tmp'].stack_vla is StackAllocation.Heap


def test_a_symbolic_stack_array_is_a_variable_length_array():
    sdfg = register_scratch_sdfg('vla_stack', N, StackAllocation.Stack)
    code = sdfg.generate_code()[0].clean_code
    assert 'double tmp[Max(1, N)];' in code, code
    assert 'tmp = new' not in code, code
    run_scratch(sdfg, 32, N=32)


def test_a_zero_extent_stack_array_has_a_positive_bound():
    """A zero-length VLA is undefined behaviour, and Fortran automatic arrays are often empty."""
    sdfg = register_scratch_sdfg('vla_zero', N, StackAllocation.Stack)
    args = dace.Config.get('compiler', 'cpu', 'args')
    with dace.config.set_temporary('compiler', 'cpu', 'args', value=f'{args} -fsanitize=vla-bound'), \
            dace.config.set_temporary('compiler', 'cpu', 'libs', value='ubsan'), captured_stderr_fd() as stderr:
        run_scratch(sdfg, 0, N=0)
    assert 'runtime error' not in stderr[0], stderr[0]


def test_a_zeroed_symbolic_stack_array_is_cleared_by_memset():
    """A VLA cannot take a brace initializer."""
    sdfg = register_scratch_sdfg('vla_setzero', N, StackAllocation.Stack)
    for node in sdfg.start_block.data_nodes():
        if node.data == 'tmp':
            node.setzero = True
    code = sdfg.generate_code()[0].clean_code
    assert 'memset(tmp, 0, sizeof(double)*(Max(1, N)));' in code, code


def test_an_auto_symbolic_register_array_stays_on_the_heap():
    sdfg = register_scratch_sdfg('vla_auto', N, StackAllocation.Auto)
    with pytest.warns(UserWarning, match='Variable-length array tmp'):
        code = sdfg.generate_code()[0].clean_code
    assert 'tmp = new' in code, code
    assert 'double tmp[' not in code, code


def test_a_small_constant_register_array_stays_aligned_on_the_stack():
    sdfg = register_scratch_sdfg('stack_constant', 16, StackAllocation.Auto)
    code = sdfg.generate_code()[0].clean_code
    assert 'double tmp[16]  DACE_ALIGN(64);' in code, code
    assert 'tmp = new' not in code, code
    run_scratch(sdfg, 16)


def test_a_large_constant_register_array_moves_to_the_heap():
    sdfg = register_scratch_sdfg('heap_constant', 4096, StackAllocation.Auto)
    with pytest.warns(UserWarning,
                      match='Register array tmp with 4096 elements was allocated on the heap instead of the stack'):
        code = sdfg.generate_code()[0].clean_code
        run_scratch(sdfg, 4096)
    assert 'tmp = new' in code, code


def test_a_global_lifetime_keeps_a_symbolic_stack_array_on_the_heap():
    """A VLA dies with its block, while a Global array is declared outside it. Persistent and External
    register arrays are rejected by validation."""
    sdfg = register_scratch_sdfg('vla_global', N, StackAllocation.Stack, AllocationLifetime.Global)
    with pytest.warns(UserWarning, match='Variable-length array tmp'):
        code = sdfg.generate_code()[0].clean_code
        run_scratch(sdfg, 8, N=8)
    assert 'double tmp[' not in code, code


def test_a_split_declaration_keeps_a_symbolic_stack_array_on_the_heap():
    """A size that only an interstate edge defines is declared in one block and allocated in another,
    and a VLA cannot bridge that split."""
    K = dace.symbol('K', dtype=dace.int64)
    sdfg = dace.SDFG('vla_split_declaration')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_array('a', (N, ), dace.float64)
    sdfg.add_array('b', (N, ), dace.float64)
    _, tmp = sdfg.add_transient('tmp', (K, ), dace.float64, storage=dace.StorageType.Register)
    tmp.stack_vla = StackAllocation.Stack
    init = sdfg.add_state('init', is_start_block=True)
    first = sdfg.add_state('first')
    second = sdfg.add_state('second')
    sdfg.add_edge(init, first, dace.InterstateEdge(assignments={'K': 'N'}))
    sdfg.add_edge(first, second, dace.InterstateEdge())
    first.add_mapped_tasklet('fill', {'i': '0:K'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp + 1.0', {'o': dace.Memlet('tmp[i]')},
                             external_edges=True)
    second.add_mapped_tasklet('drain', {'i': '0:K'}, {'inp': dace.Memlet('tmp[i]')},
                              'o = inp', {'o': dace.Memlet('b[i]')},
                              external_edges=True)
    sdfg.validate()

    with pytest.warns(UserWarning, match='Variable-length array tmp'):
        code = sdfg.generate_code()[0].clean_code
        run_scratch(sdfg, 8, N=8)
    assert 'double tmp[' not in code, code


def test_stack_placement_survives_clone_and_serialization():
    sdfg = register_scratch_sdfg('placement_roundtrip', N, StackAllocation.Stack)
    desc = sdfg.arrays['tmp']
    assert desc.clone().stack_vla is StackAllocation.Stack
    reloaded = dace.SDFG.from_json(json.loads(json.dumps(sdfg.to_json())))
    assert reloaded.arrays['tmp'].stack_vla is StackAllocation.Stack


if __name__ == '__main__':
    for case in PLACEMENT_CASES:
        test_auto_placement_resolves_by_size_and_explicit_placement_is_kept(*case)
    test_auto_placement_of_a_large_constant_array_respects_max_stack_array_size()
    test_a_symbolic_stack_array_is_a_variable_length_array()
    test_a_zero_extent_stack_array_has_a_positive_bound()
    test_a_zeroed_symbolic_stack_array_is_cleared_by_memset()
    test_an_auto_symbolic_register_array_stays_on_the_heap()
    test_a_small_constant_register_array_stays_aligned_on_the_stack()
    test_a_large_constant_register_array_moves_to_the_heap()
    test_a_global_lifetime_keeps_a_symbolic_stack_array_on_the_heap()
    test_a_split_declaration_keeps_a_symbolic_stack_array_on_the_heap()
    test_stack_placement_survives_clone_and_serialization()
