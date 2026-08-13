# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Code-generation tests for the inherited-GPU-error drain (``__dace_gpu_drain_error``).

These assert on generated source only, so they need neither a GPU nor a CUDA toolchain.
"""
import dace
import pytest

from dace.transformation.interstate import GPUTransformSDFG


def _gpu_sdfg(name: str = 'drain_probe') -> dace.SDFG:
    """A minimal SDFG that generates GPU code."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [8], dace.float64)
    sdfg.arrays['A'].optional = False
    state = sdfg.add_state()
    state.add_mapped_tasklet('inc',
                             dict(i='0:8'),
                             dict(inp=dace.Memlet('A[i]')),
                             'out = inp + 1.0',
                             dict(out=dace.Memlet('A[i]')),
                             external_edges=True)
    sdfg.apply_transformations(GPUTransformSDFG)
    return sdfg


def _sources(sdfg: dace.SDFG):
    """The generated CUDA file and the frame (host) file, as strings."""
    objs = sdfg.generate_code()
    cu = next(o.clean_code for o in objs if o.language == 'cu')
    frame = next(o.clean_code for o in objs if o.language == 'cpp' and o.name == sdfg.name)
    return cu, frame


def test_gpu_drain_emitted_in_init_and_per_call():
    """The drain is defined in the .cu, called from __dace_init_cuda, and called again at
    the top of every __program_<name> invocation.

    Initialization runs once per state, but a foreign GPU error can be left in the
    runtime's shared last-error slot between any two calls, so the per-call site is the
    one that actually protects a long-running process.
    """
    cu, frame = _sources(_gpu_sdfg())

    assert 'void __dace_gpu_drain_error(' in cu  # defined in the CUDA file
    assert '__dace_gpu_drain_error(__state);' in cu  # and called by the initializer

    # The host file cannot include the CUDA headers, so it declares the function and links
    # against the .cu definition - the same arrangement as __dace_init_cuda.
    assert 'DACE_EXPORTED void __dace_gpu_drain_error(' in frame
    decl = frame.index('DACE_EXPORTED void __dace_gpu_drain_error(')
    call = frame.index('__dace_gpu_drain_error(__state);')
    assert decl < call, 'the declaration must precede the call'
    assert call < frame.index('_internal(__state'), 'the drain must run before the program body'


def test_gpu_drain_absent_without_gpu_code():
    """A CPU-only SDFG must not reference the drain: the symbol only exists when the CUDA
    target emitted it, so an unconditional call would be a link error."""
    sdfg = dace.SDFG('drain_cpu_only_probe')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.arrays['A'].optional = False
    state = sdfg.add_state()
    state.add_mapped_tasklet('inc',
                             dict(i='0:8'),
                             dict(inp=dace.Memlet('A[i]')),
                             'out = inp + 1.0',
                             dict(out=dace.Memlet('A[i]')),
                             external_edges=True)
    frame = next(o.clean_code for o in sdfg.generate_code() if o.language == 'cpp' and o.name == 'drain_cpu_only_probe')
    assert '__dace_gpu_drain_error' not in frame


def test_gpu_init_template_substitution_does_not_break_comments():
    """Regression: a ``{placeholder}`` written inside a C++ comment in the init template.

    ``__dace_init_cuda`` is built with ``str.format``, so a placeholder inside a comment is
    substituted there like anywhere else. ``{initcode}`` expands to multiple statements, so
    writing it in a comment dumped the whole init body into the middle of that comment and
    left the rest of the sentence as a bare statement - nvcc reported
    ``identifier "often" is undefined``.

    Checked against the TEMPLATE SOURCE rather than against generated output. Reproducing
    the corruption through a generated file needs an SDFG whose ``{initcode}`` is non-empty
    (a CUB reduction, say) - so a test built on a simpler SDFG passes while the bug is
    present, which is worse than no test. Reading the source catches the whole class for
    every SDFG shape.

    ``{backend}`` is allowed: it expands to a single identifier (``cuda``/``hip``) and has
    always been used this way. Everything else is rejected, since a placeholder that
    expands to statements turns the rest of the comment line into code.
    """
    import re
    from pathlib import Path

    import dace.codegen.targets.cuda as cuda_target

    source = Path(cuda_target.__file__).read_text().split('\n')
    placeholder = re.compile(r'\{([a-z_][a-z_0-9]*)\}')
    offenders = []
    for lineno, line in enumerate(source, 1):
        if not line.strip().startswith('//'):
            continue
        for match in placeholder.finditer(line):
            if match.group(1) != 'backend':
                offenders.append(f'{cuda_target.__file__}:{lineno}: {{{match.group(1)}}} in {line.strip()!r}')

    assert not offenders, ('a format placeholder inside a generated C++ comment is substituted there, '
                           'so anything expanding to statements breaks the comment:\n  ' + '\n  '.join(offenders))


def test_gpu_mempool_setup_is_checked_and_follows_context_creation():
    """The memory-pool calls are wrapped, and run only after the GPU context exists.

    DACE_GPU_CHECK records into ``__state->gpu_context``, so a checked call placed before
    the context is constructed could not record - and, before ``gpu_context`` was given an
    initializer, could not even be read safely.
    """
    sdfg = dace.SDFG('drain_pool_probe')
    sdfg.add_array('A', [16], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.arrays['A'].optional = False
    sdfg.add_transient('t', [16], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.arrays['t'].pool = True
    state = sdfg.add_state()
    state.add_nedge(state.add_read('A'), state.add_write('t'), dace.Memlet('A[0:16]'))
    state.add_nedge(state.add_read('t'), state.add_write('A'), dace.Memlet('t[0:16]'))

    cu, _ = _sources(sdfg)
    if 'MemPool_t' not in cu:
        pytest.skip('this SDFG did not request pooled allocation')

    assert 'DACE_GPU_CHECK(cudaDeviceGetDefaultMemPool' in cu
    assert 'DACE_GPU_CHECK(cudaMemPoolSetAttribute' in cu
    assert cu.index('__state->gpu_context = new') < cu.index('MemPool_t')


if __name__ == '__main__':
    test_gpu_drain_emitted_in_init_and_per_call()
    test_gpu_drain_absent_without_gpu_code()
    test_gpu_init_template_substitution_does_not_break_comments()
    test_gpu_mempool_setup_is_checked_and_follows_context_creation()
