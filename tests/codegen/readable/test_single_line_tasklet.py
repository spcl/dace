# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
A tasklet whose connectors were all inlined is emitted as one brace-free line ending in ``// <label>``. A tasklet that
keeps a local, such as a write-conflict output, keeps its scope.
"""
import numpy as np
import pytest

import dace
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, available_targets,
                                             gpu_available, generated_for, run_variant)

LEGACY_SEPARATOR = '///////////////////'


def add_2d_sdfg(name):
    """``C[i, j] = A[i, j] + B[i, j]`` over a map."""
    sdfg = dace.SDFG(name)
    for arr in ('A', 'B', 'C'):
        sdfg.add_array(arr, [6, 7], dace.float64)
    state = sdfg.add_state('main')
    ra, rb, wc = state.add_read('A'), state.add_read('B'), state.add_write('C')
    entry, exit_node = state.add_map('m', dict(i='0:6', j='0:7'))
    tasklet = state.add_tasklet('add', {'a', 'b'}, {'c'}, 'c = a + b')
    state.add_memlet_path(ra, entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(rb, entry, tasklet, dst_conn='b', memlet=dace.Memlet('B[i, j]'))
    state.add_memlet_path(tasklet, exit_node, wc, src_conn='c', memlet=dace.Memlet('C[i, j]'))
    sdfg.validate()
    return sdfg


def wcr_reduction_sdfg(name):
    """``s += A[i]``, whose conflict-resolving output cannot be inlined."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [16], dace.float64)
    sdfg.add_array('s', [1], dace.float64)
    state = sdfg.add_state('main')
    ra, ws = state.add_read('A'), state.add_write('s')
    entry, exit_node = state.add_map('m', dict(i='0:16'))
    tasklet = state.add_tasklet('acc', {'a'}, {'o'}, 'o = a')
    state.add_memlet_path(ra, entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, exit_node, ws, src_conn='o', memlet=dace.Memlet('s[0]', wcr='lambda x, y: x + y'))
    sdfg.validate()
    return sdfg


def tasklet_body_line(code):
    """The line that stores into ``C`` through the index functions."""
    lines = [ln.strip() for ln in code.splitlines() if 'C_idx(' in ln and 'A_idx(' in ln and 'B_idx(' in ln]
    assert lines, 'no C[C_idx(..)] = A[A_idx(..)] + B[B_idx(..)] line found:\n' + code
    return lines[0]


def test_single_line_no_block():
    """Legacy frames the body with separator lines; the readable form is one statement."""
    experimental = generated_for(add_2d_sdfg, 'sl_exp', EXPERIMENTAL)
    legacy = generated_for(add_2d_sdfg, 'sl_leg', LEGACY)

    body = tasklet_body_line(experimental)
    assert '// add' in body, body
    assert '{' not in body and '}' not in body, body
    assert body.count(';') == 1, body
    assert LEGACY_SEPARATOR not in experimental
    assert 'double a =' not in experimental and 'double b =' not in experimental
    assert 'double c;' not in experimental

    assert LEGACY_SEPARATOR in legacy


def test_wcr_tasklet_keeps_block():
    """A conflict-resolving output keeps the tasklet's scope."""
    experimental = generated_for(wcr_reduction_sdfg, 'wcr_exp', EXPERIMENTAL)
    assert 'wcr' in experimental.lower() or 'reduce' in experimental.lower() or 'atomic' in experimental.lower(), \
        experimental
    assert '{' in experimental and '}' in experimental


def test_single_line_bit_exact(target):
    """The one-line add matches legacy."""
    base = dict(A=np.random.rand(6, 7), B=np.random.rand(6, 7), C=np.zeros((6, 7)))
    legacy = run_variant(add_2d_sdfg, f'sl_run_leg_{target}', LEGACY, base, target)
    experimental = run_variant(add_2d_sdfg, f'sl_run_exp_{target}', EXPERIMENTAL, base, target)
    assert_outputs_equivalent(legacy, experimental, target, label='single_line_add')
    assert np.allclose(experimental['C'], base['A'] + base['B'])


def test_wcr_reduction_bit_exact(target):
    """The reduction with a scoped tasklet matches legacy."""
    base = dict(A=np.random.rand(16), s=np.zeros(1))
    legacy = run_variant(wcr_reduction_sdfg, f'wcr_run_leg_{target}', LEGACY, base, target)
    experimental = run_variant(wcr_reduction_sdfg, f'wcr_run_exp_{target}', EXPERIMENTAL, base, target)
    assert_outputs_equivalent(legacy, experimental, target, label='wcr_reduction')
    assert np.allclose(experimental['s'][0], base['A'].sum())


@pytest.mark.gpu
def test_single_line_inside_kernel(require_gpu):
    """The connector-free single-line tasklet appears inside the ``__global__``
    kernel: the CUDA generator emits device tasklets through the shared CPU
    generator, so the readable form flows into device code too."""
    code = generated_for(add_2d_sdfg, 'sl_gpu_inspect', EXPERIMENTAL, gpu=True)
    assert '__global__' in code, 'no CUDA kernel emitted'
    body = tasklet_body_line(code)
    assert '{' not in body and '}' not in body, body
    assert LEGACY_SEPARATOR not in code


if __name__ == "__main__":
    test_single_line_no_block()
    test_wcr_tasklet_keeps_block()
    for target in available_targets():
        test_single_line_bit_exact(target)
        test_wcr_reduction_bit_exact(target)
    if gpu_available():
        test_single_line_inside_kernel(None)
