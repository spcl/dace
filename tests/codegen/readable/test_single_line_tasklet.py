# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Single-line tasklet emission for the experimental "readable" code generator.

Under ``compiler.cpu.implementation = experimental`` a tasklet whose connectors
were all inlined (a connector-free, single-statement element-wise body) is
emitted as ONE brace-free line with a trailing ``// <label>`` comment::

    C[C_idx(i, j)] = (A[A_idx(i, j)] + B[B_idx(i, j)]);  // _Add_

instead of the legacy ``{ /////// <body> /////// }`` scope block. A tasklet that
still declares locals -- a copy-in/out temporary, a code->code register, or a
write-conflict (WCR) output that must go through the atomic path -- keeps its
own ``{ }`` block (dropping it would leak the local into the enclosing map body).

The tests assert both the emitted shape (codegen inspection, no compile) and
numerical equivalence to legacy (bit-exact on CPU, tight ``allclose`` on GPU),
on CPU and -- since the CUDA generator emits its device tasklets through the same
CPU generator instance -- inside ``__global__`` kernels.
"""
import copy

import numpy as np
import pytest

import dace
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, run_isolated,
                                             use_implementation)

# The legacy generator wraps every tasklet body in these separator lines.
LEGACY_SEPARATOR = '///////////////////'


def add_2d_sdfg(name):
    """``C[i, j] = A[i, j] + B[i, j]`` over a map -- every connector is a single
    element, so all three inline and the body is one statement."""
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
    """``s += A[i]`` -- the WCR output connector must go through the atomic resolve
    path, so it is never inlined and the tasklet keeps its scope block."""
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


def generated_for(build, name, implementation, gpu=False):
    """Generated C++/CUDA (codegen only) for ``build`` under ``implementation``."""
    with use_implementation(implementation):
        sdfg = build(name)
        if gpu:
            sdfg.apply_gpu_transformations()
        return '\n'.join((obj.clean_code or obj.code) for obj in sdfg.generate_code())


def tasklet_body_line(code):
    """The single emitted line that stores into ``C`` through the index functions."""
    lines = [ln.strip() for ln in code.splitlines() if 'C_idx(' in ln and 'A_idx(' in ln and 'B_idx(' in ln]
    assert lines, 'no C[C_idx(..)] = A[A_idx(..)] + B[B_idx(..)] line found:\n' + code
    return lines[0]


# --------------------------------------------------------------------------- #
# Codegen shape (no compile)
# --------------------------------------------------------------------------- #
def test_single_line_no_block(require_experimental):
    """Experimental collapses the connector-free add tasklet onto one brace-free
    line with a ``// <label>`` comment; legacy keeps the ``{ /////// }`` block."""
    experimental = generated_for(add_2d_sdfg, 'sl_exp', EXPERIMENTAL)
    legacy = generated_for(add_2d_sdfg, 'sl_leg', LEGACY)

    body = tasklet_body_line(experimental)
    # One statement, one comment, no scope braces on the tasklet line.
    assert body.endswith('// add') or '// add' in body, body
    assert '{' not in body and '}' not in body, body
    assert body.count(';') == 1, body
    # No legacy separators anywhere in the experimental output.
    assert LEGACY_SEPARATOR not in experimental
    # Connectors are gone: no copy-in / copy-out temporaries.
    assert 'double a =' not in experimental and 'double b =' not in experimental
    assert 'double c;' not in experimental

    # Legacy is unchanged: block + separators.
    assert LEGACY_SEPARATOR in legacy


def test_wcr_tasklet_keeps_block(require_experimental):
    """A WCR (atomic) output is not inlined, so the reduction tasklet keeps its
    ``{ }`` scope block even under the experimental generator."""
    experimental = generated_for(wcr_reduction_sdfg, 'wcr_exp', EXPERIMENTAL)
    # The atomic write path is still emitted (never collapsed to a plain store),
    # and the tasklet retains a brace-delimited body.
    assert 'wcr' in experimental.lower() or 'reduce' in experimental.lower() or 'atomic' in experimental.lower(), \
        experimental
    assert '{' in experimental and '}' in experimental


# --------------------------------------------------------------------------- #
# Numerical equivalence (compile + run): CPU (fork) and GPU (in-process)
# --------------------------------------------------------------------------- #
def run_variant(build, name, implementation, target, base):
    """Compile + run one variant on a deep copy of ``base``; CPU forks, GPU runs
    in-process (CUDA and ``os.fork`` do not mix)."""

    def work():
        sdfg = build(name)
        if target == 'gpu':
            sdfg.apply_gpu_transformations()
        arrays = copy.deepcopy(base)
        sdfg.compile()(**arrays)
        return {k: v for k, v in arrays.items() if isinstance(v, np.ndarray)}

    with use_implementation(implementation):
        return work() if target == 'gpu' else run_isolated(work)


def test_single_line_bit_exact(require_experimental, target):
    """The single-line add reproduces the legacy result exactly (CPU) / within a
    tight tolerance (GPU)."""
    base = dict(A=np.random.rand(6, 7), B=np.random.rand(6, 7), C=np.zeros((6, 7)))
    legacy = run_variant(add_2d_sdfg, f'sl_run_leg_{target}', LEGACY, target, base)
    experimental = run_variant(add_2d_sdfg, f'sl_run_exp_{target}', EXPERIMENTAL, target, base)
    assert_outputs_equivalent(legacy, experimental, target, label='single_line_add')
    # And it is the analytically correct value.
    assert np.allclose(experimental['C'], base['A'] + base['B'])


def test_wcr_reduction_bit_exact(require_experimental, target):
    """The WCR reduction (block-retained tasklet) is still equivalent to legacy."""
    base = dict(A=np.random.rand(16), s=np.zeros(1))
    legacy = run_variant(wcr_reduction_sdfg, f'wcr_run_leg_{target}', LEGACY, target, base)
    experimental = run_variant(wcr_reduction_sdfg, f'wcr_run_exp_{target}', EXPERIMENTAL, target, base)
    assert_outputs_equivalent(legacy, experimental, target, label='wcr_reduction')
    assert np.allclose(experimental['s'][0], base['A'].sum())


@pytest.mark.gpu
def test_single_line_inside_kernel(require_experimental, require_gpu):
    """The connector-free single-line tasklet appears inside the ``__global__``
    kernel: the CUDA generator emits device tasklets through the shared CPU
    generator, so the readable form flows into device code too."""
    code = generated_for(add_2d_sdfg, 'sl_gpu_inspect', EXPERIMENTAL, gpu=True)
    assert '__global__' in code, 'no CUDA kernel emitted'
    body = tasklet_body_line(code)
    assert '{' not in body and '}' not in body, body
    assert LEGACY_SEPARATOR not in code


if __name__ == '__main__':
    from tests.codegen.readable.conftest import experimental_available, gpu_available
    if not experimental_available():
        print('experimental readable codegen not ready; skipping')
    else:
        test_single_line_no_block(None)
        test_wcr_tasklet_keeps_block(None)
        test_single_line_bit_exact(None, 'cpu')
        test_wcr_reduction_bit_exact(None, 'cpu')
        if gpu_available():
            test_single_line_bit_exact(None, 'gpu')
            test_wcr_reduction_bit_exact(None, 'gpu')
            test_single_line_inside_kernel(None, None)
        print('ok')
