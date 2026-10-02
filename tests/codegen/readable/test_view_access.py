# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
A view carries its own strides, so an access through it uses an index function built from the view's strides, not the
source's, and needs no connector temporary.
"""
import numpy as np
import pytest

import dace
from dace import subsets
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, available_targets,
                                             gpu_available, generated_for, run_variant)


def strided_view_copy_sdfg(name):
    """``C[i, j] = V[i, j]`` where ``V`` is the view ``A[:, ::2]`` with strides ``[16, 2]``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [8, 16], dace.float64)
    sdfg.add_array('C', [8, 8], dace.float64)
    sdfg.add_view('V', [8, 8], dace.float64, strides=[16, 2])
    state = sdfg.add_state('main')
    a, v, c = state.add_access('A'), state.add_access('V'), state.add_access('C')
    state.add_edge(a, None, v, 'views', dace.Memlet(data='A', subset=subsets.Range([(0, 7, 1), (0, 15, 2)])))
    entry, exit_node = state.add_map('m', dict(i='0:8', j='0:8'))
    tasklet = state.add_tasklet('cpy', {'inp'}, {'out'}, 'out = inp')
    state.add_memlet_path(v, entry, tasklet, dst_conn='inp', memlet=dace.Memlet(data='V', subset='i, j'))
    state.add_memlet_path(tasklet, exit_node, c, src_conn='out', memlet=dace.Memlet(data='C', subset='i, j'))
    sdfg.validate()
    return sdfg


def reference(A):
    """The expected result."""
    return A[:, ::2].copy()


def view_index_body(code):
    """The ``return`` line of ``V_idx``."""
    lines = [ln.strip() for ln in code.splitlines() if 'V_idx(' in ln and 'return' in ln]
    assert lines, 'experimental codegen emitted no V_idx index function:\n' + code
    return lines[0]


def test_view_idx_uses_view_strides():
    """``V_idx`` linearizes with the view's strides ``[16, 2]``."""
    code = generated_for(strided_view_copy_sdfg, 'view_inspect', EXPERIMENTAL)
    body = view_index_body(code)
    assert '16 * __d0' in body, body
    assert '2 * __d1' in body, body
    assert any('V[V_idx(' in ln for ln in code.splitlines()), 'no V[V_idx(..)] access emitted:\n' + code
    assert 'double inp =' not in code and 'double inp;' not in code


def test_view_no_pure_fallback():
    """A view input does not force the classic connector copy."""
    code = generated_for(strided_view_copy_sdfg, 'view_nofallback', EXPERIMENTAL)
    assert 'V[V_idx(' in code
    assert '///////////////////' not in code


def test_view_access_bit_exact(target):
    """The strided-view copy matches legacy and ``A[:, ::2]``."""
    base = dict(A=np.random.rand(8, 16), C=np.zeros((8, 8)))
    legacy = run_variant(strided_view_copy_sdfg, f'view_run_leg_{target}', LEGACY, base, target)
    experimental = run_variant(strided_view_copy_sdfg, f'view_run_exp_{target}', EXPERIMENTAL, base, target)
    assert_outputs_equivalent(legacy, experimental, target, label='strided_view_copy')
    assert np.allclose(experimental['C'], reference(base['A']))


@pytest.mark.gpu
def test_view_idx_inside_kernel(require_gpu):
    """The view index function appears in the device kernel."""
    code = generated_for(strided_view_copy_sdfg, 'view_gpu_inspect', EXPERIMENTAL, gpu=True)
    assert '__global__' in code, 'no CUDA kernel emitted'
    assert 'V_idx(' in code, 'view index function missing from device code'
    body = view_index_body(code)
    assert '2 * __d1' in body, body


if __name__ == "__main__":
    test_view_idx_uses_view_strides()
    test_view_no_pure_fallback()
    for target in available_targets():
        test_view_access_bit_exact(target)
    if gpu_available():
        test_view_idx_inside_kernel(None)
