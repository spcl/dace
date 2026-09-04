# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CUB workspace protocol in the ``CUDA (device)`` reduction expansion.

CUB reads a null ``d_temp_storage`` as "only report the size and do nothing". An unchecked size
query, a failed allocation, or a zero-byte one (``cudaMalloc(&p, 0)`` hands back a null pointer WITH
``cudaSuccess``) each leave the pointer null, so the reduction performs no work and leaves the output
as it found it -- on fresh device memory, a clean array of zeros with no error raised anywhere.

These assert on emitted code, so they need a CUDA toolchain for neither compilation nor a run.
"""
import os
import re

import numpy as np
import pytest

import dace
import dace.libraries.standard as std

# (shape, reduced axes, output shape) for the two CUB entry points the expansion can reach:
# a trailing-axes reduction lowers to DeviceSegmentedReduce, an all-axes one to DeviceReduce.
CUB_CASES = [([111, 111, 111], (1, 2), [111]), ([1000], (0), [1])]


def generated_code_for(in_shape, axes, out_shape) -> str:
    """Every generated file for a ``CUDA (device)`` reduction over ``axes``, joined.

    The expansion splits across two: the workspace and the CUB calls land in the ``.cu``, the call
    to the reduce helper in the host ``.cpp``.
    """
    reduced = axes

    @dace.program
    def multidimred(a, b):
        b[:] = np.sum(a, axis=reduced)

    a = np.zeros(in_shape, np.float64)
    b = np.zeros(out_shape, np.float64)
    sdfg = multidimred.to_sdfg(a, b)
    sdfg.apply_gpu_transformations()
    rednode = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, std.Reduce))
    rednode.implementation = 'CUDA (device)'

    generated = sdfg.generate_code()
    # Keyed on the code object's TITLE, not its extension: the GPU object is emitted as `.cu` under
    # CUDA and `.cpp` under HIP, and asserting `cpp` would also match the host file and pass even
    # when nothing lowered to the GPU at all.
    assert any(code.title == 'CUDA' for code in generated), 'the reduction did not lower to a GPU file'
    return '\n'.join(code.clean_code for code in generated)


@pytest.mark.parametrize('in_shape,axes,out_shape', CUB_CASES)
def test_the_cub_workspace_size_query_is_checked(in_shape, axes, out_shape):
    """A failed query leaves the size at zero, which is the zero-byte allocation below."""
    code = generated_code_for(in_shape, axes, out_shape)
    query = re.search(r'^.*cub::Device\w*Reduce::\w+\(nullptr,.*$', code, re.MULTILINE)
    assert query, 'no CUB size query was emitted'
    assert 'DACE_GPU_CHECK' in query.group(0), f'the CUB size query is unchecked: {query.group(0).strip()}'


@pytest.mark.parametrize('in_shape,axes,out_shape', CUB_CASES)
def test_the_cub_workspace_is_never_allocated_zero_bytes_and_is_checked(in_shape, axes, out_shape):
    """The workspace now comes from the per-stream scratch pool (``get_scratch``), not a direct
    ``cudaMalloc`` -- the pool floors a zero-byte request and reports allocation failure itself (see
    ``test_the_scratch_pool_floors_zero_byte_requests_and_reports_allocation_failure`` below), so the
    call site only has to check the pointer it gets back.
    """
    code = generated_code_for(in_shape, axes, out_shape)
    fetch = re.search(r'^.*get_scratch<::dace::cub::ReduceTag>\(_cub_needed, stream, &_cub_status\);$', code,
                      re.MULTILINE)
    assert fetch, 'no CUB scratch-pool fetch was emitted'
    checked = re.search(r'^\s*if \(_cub_scratch == nullptr\) return.*$', code, re.MULTILINE)
    assert checked, 'the CUB scratch pointer is not checked for allocation failure'


@pytest.mark.parametrize('in_shape,axes,out_shape', CUB_CASES)
def test_the_reduction_itself_reports_its_status(in_shape, axes, out_shape):
    """The work call is the only site holding CUB's status for the reduction that produces the output."""
    code = generated_code_for(in_shape, axes, out_shape)
    assert re.search(
        r'gpuError_t __dace_reduce_\w+\(',
        code), ('the reduce helper returns void, so CUB\'s status is discarded at the only site that has it')
    # experimental_readable (the default CPU codegen) inlines the tasklet's _in/_out connectors to
    # the actual pointers they read/write; legacy keeps the literal connector names in the call. The
    # call site is the line that is not a declaration either way, and how the offloading pass spells
    # its device copy (``gpu_a``, ``a_gpu``) is that pass's business rather than this test's.
    call = re.search(r'^(?!.*gpuError_t).*__dace_reduce_\w+\(\w+, \w+,.*$', code, re.MULTILINE)
    assert call, 'no call to the reduce helper was emitted'
    assert 'DACE_GPU_CHECK' in call.group(0), f'the reduction call is unchecked: {call.group(0).strip()}'


def cudacommon_source() -> str:
    """The runtime header holding the check macros and the GPU context they record into."""
    path = os.path.join(os.path.dirname(dace.__file__), 'runtime', 'include', 'dace', 'cuda', 'cudacommon.cuh')
    with open(path) as fp:
        return fp.read()


def cub_scratch_source() -> str:
    """The runtime header implementing the per-stream CUB scratch pool (``get_scratch``)."""
    path = os.path.join(os.path.dirname(dace.__file__), 'runtime', 'include', 'dace', 'cub_scratch.cuh')
    with open(path) as fp:
        return fp.read()


def test_the_scratch_pool_floors_zero_byte_requests_and_reports_allocation_failure():
    """``cudaMalloc(&p, 0)`` returns a null pointer with ``cudaSuccess``, so ``get_scratch`` must
    never issue that call, and a real allocation failure must reach the caller through ``status``
    rather than a stale pointer.
    """
    source = cub_scratch_source()
    floor = re.search(r'gpuMalloc\(&e\.storage, bytes_needed \? bytes_needed : 1\);', source)
    assert floor, 'get_scratch no longer floors a zero-byte request to 1 byte'
    failure = re.search(r'if \(err != gpuSuccess\) \{(?:.|\n)*?return nullptr;\n\s*\}', source)
    assert failure, 'get_scratch does not handle a failed allocation'
    assert 'if (status) *status = err;' in failure.group(0), (
        'a failed allocation does not report its error through status')


def test_the_check_macro_evaluates_its_argument_once():
    """``DACE_GPU_CHECK`` wraps calls with side effects -- an allocation, and a whole reduction.

    Naming the argument again to format the message would perform the failing call a second time.
    """
    macro = re.search(r'#define DACE_GPU_CHECK\(err\)(?:.|\n)*?while \(0\)', cudacommon_source())
    assert macro, 'DACE_GPU_CHECK is no longer defined where this test looks for it'
    body = macro.group(0).split('gpuError_t errr = (err);', 1)[1]
    assert '(err)' not in body, ('DACE_GPU_CHECK evaluates its argument a second time in the error path; wrapping a '
                                 'call with side effects then performs it twice')


def test_the_recorded_error_is_the_first_one():
    """Only the first error names the call that actually broke; the rest describe consequences.

    A failed size query is the case this file is about: it leaves the workspace unsized, and the
    reduction that then reads it reports an error of its own, about the workspace rather than the
    query. That later one is what a caller is handed if recording overwrites.
    """
    source = cudacommon_source()
    recorder = re.search(r'void record_error\(gpuError_t err\)(?:.|\n)*?\n  \}', source)
    assert recorder, 'the GPU context no longer records errors through record_error'
    assert 'lasterror == (gpuError_t)0' in recorder.group(0), (
        'record_error records unconditionally, so a later error about a consequence replaces the first one')
    written = re.findall(r'^.*\blasterror = .*$', source, re.MULTILINE)
    assert len(written) == 1, (f'an error is recorded without going through record_error, which overwrites whatever '
                               f'was recorded before it: {[line.strip() for line in written]}')


if __name__ == '__main__':
    for shape, ax, out in CUB_CASES:
        test_the_cub_workspace_size_query_is_checked(shape, ax, out)
        test_the_cub_workspace_is_never_allocated_zero_bytes_and_is_checked(shape, ax, out)
        test_the_reduction_itself_reports_its_status(shape, ax, out)
    test_the_check_macro_evaluates_its_argument_once()
    test_the_recorded_error_is_the_first_one()
    test_the_scratch_pool_floors_zero_byte_requests_and_reports_allocation_failure()
