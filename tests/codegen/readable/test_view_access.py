# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
View access for the experimental "readable" code generator.

A DaCe View is ``source_ptr + offset`` carrying its OWN strides/step, so
``view[i, j]`` addresses ``i * stride0 + j * stride1`` into the source. The
readable generator treats an ``ArrayView`` as a first-class pointer: an access
routes through a generated ``V_idx(...)`` index function built from the VIEW's
strides (not the source's), exactly like any array -- ``V[V_idx(i, j)]`` -- with
no copy-in / copy-out connector temporary. A View is never a reason to fall back
to the classic connector lowering.

The tests assert the emitted ``V_idx`` uses the view strides (codegen
inspection, no compile) and that the lowered access reproduces the legacy result
(bit-exact on CPU, tight ``allclose`` on GPU), on CPU and inside ``__global__``
kernels.
"""
import copy

import numpy as np
import pytest

import dace
from dace import subsets
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, run_isolated,
                                             use_implementation)


def strided_view_copy_sdfg(name):
    """``C[i, j] = V[i, j]`` where ``V`` is the strided view ``A[:, ::2]``.

    ``V`` has shape ``(8, 8)`` and strides ``[16, 2]`` into ``A`` (shape
    ``(8, 16)``), a genuine non-unit-stride ArrayView node (not a subset folded
    onto ``A``), so the generator must emit ``V_idx`` from the view's strides.
    """
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
    """The value ``strided_view_copy_sdfg`` computes: ``A[:, ::2]``."""
    return A[:, ::2].copy()


def generated_for(build, name, implementation, gpu=False):
    with use_implementation(implementation):
        sdfg = build(name)
        if gpu:
            sdfg.apply_gpu_transformations()
        return '\n'.join((obj.clean_code or obj.code) for obj in sdfg.generate_code())


def view_index_body(code):
    """Body of the emitted ``V_idx`` index function (the ``return ...;`` line)."""
    lines = [ln.strip() for ln in code.splitlines() if 'V_idx(' in ln and 'return' in ln]
    assert lines, 'experimental codegen emitted no V_idx index function:\n' + code
    return lines[0]


# --------------------------------------------------------------------------- #
# Codegen shape (no compile)
# --------------------------------------------------------------------------- #
def test_view_idx_uses_view_strides(require_experimental):
    """``V_idx`` linearizes with the VIEW's strides ``[16, 2]`` -- ``16*d0 + 2*d1``
    -- and the access is connector-free (``V[V_idx(...)]`` in the body)."""
    code = generated_for(strided_view_copy_sdfg, 'view_inspect', EXPERIMENTAL)
    body = view_index_body(code)
    assert '16 * __d0' in body, body
    assert '2 * __d1' in body, body
    # The view is accessed directly through its index function, no copy-in temp.
    assert any('V[V_idx(' in ln for ln in code.splitlines()), 'no V[V_idx(..)] access emitted:\n' + code
    assert 'double inp =' not in code and 'double inp;' not in code


def test_view_no_pure_fallback(require_experimental):
    """A View input never forces the classic connector copy: the readable body
    references the view pointer directly."""
    code = generated_for(strided_view_copy_sdfg, 'view_nofallback', EXPERIMENTAL)
    # Connector name gone from the body; view name present as an indexed pointer.
    assert 'V[V_idx(' in code
    assert '///////////////////' not in code


# --------------------------------------------------------------------------- #
# Numerical equivalence (compile + run): CPU (fork) and GPU (in-process)
# --------------------------------------------------------------------------- #
def run_variant(build, name, implementation, target, base):

    def work():
        sdfg = build(name)
        if target == 'gpu':
            sdfg.apply_gpu_transformations()
        arrays = copy.deepcopy(base)
        sdfg.compile()(**arrays)
        return {k: v for k, v in arrays.items() if isinstance(v, np.ndarray)}

    with use_implementation(implementation):
        return work() if target == 'gpu' else run_isolated(work)


def test_view_access_bit_exact(require_experimental, target):
    """The strided-view copy reproduces the legacy result (bit-exact CPU / tight
    tolerance GPU) and equals the analytical ``A[:, ::2]``."""
    base = dict(A=np.random.rand(8, 16), C=np.zeros((8, 8)))
    legacy = run_variant(strided_view_copy_sdfg, f'view_run_leg_{target}', LEGACY, target, base)
    experimental = run_variant(strided_view_copy_sdfg, f'view_run_exp_{target}', EXPERIMENTAL, target, base)
    assert_outputs_equivalent(legacy, experimental, target, label='strided_view_copy')
    assert np.allclose(experimental['C'], reference(base['A']))


@pytest.mark.gpu
def test_view_idx_inside_kernel(require_experimental, require_gpu):
    """``V[V_idx(...)]`` (view strides) appears inside the ``__global__`` kernel."""
    code = generated_for(strided_view_copy_sdfg, 'view_gpu_inspect', EXPERIMENTAL, gpu=True)
    assert '__global__' in code, 'no CUDA kernel emitted'
    assert 'V_idx(' in code, 'view index function missing from device code'
    body = view_index_body(code)
    assert '2 * __d1' in body, body


if __name__ == '__main__':
    from tests.codegen.readable.conftest import experimental_available, gpu_available
    if not experimental_available():
        print('experimental readable codegen not ready; skipping')
    else:
        test_view_idx_uses_view_strides(None)
        test_view_no_pure_fallback(None)
        test_view_access_bit_exact(None, 'cpu')
        if gpu_available():
            test_view_access_bit_exact(None, 'gpu')
            test_view_idx_inside_kernel(None, None)
        print('ok')
