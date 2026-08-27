# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Scan(op=AFFINE)``: the first-order linear recurrence ``out[k] = c[k]*out[k-1] + d[k]``."""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import (COEF_CONNECTOR_NAME, INIT_CONNECTOR_NAME, INPUT_CONNECTOR_NAME,
                                                OUTPUT_CONNECTOR_NAME, Scan, ScanOp)

N = dace.symbol('N', dtype=dace.int64)


def affine_reference(coef: np.ndarray, delta: np.ndarray, seed: float) -> np.ndarray:
    """Sequential left-to-right recurrence -- the semantics the libnode must reproduce."""
    out = np.empty_like(delta)
    acc = seed
    for k in range(delta.shape[0]):
        acc = coef[k] * acc + delta[k]
        out[k] = acc
    return out


def build_affine_sdfg(dtype=dace.float64, with_init: bool = True, implementation: str = 'CPU') -> dace.SDFG:
    """One state holding a single ``Scan(AFFINE)`` reading ``coef``/``delta`` (+ ``seed``)."""
    sdfg = dace.SDFG(f'affine_scan_{implementation}_{"init" if with_init else "noinit"}')
    sdfg.add_array('coef', [N], dtype)
    sdfg.add_array('delta', [N], dtype)
    sdfg.add_array('out', [N], dtype)
    if with_init:
        sdfg.add_array('seed', [1], dtype)
    state = sdfg.add_state()

    node = Scan('affine', op=ScanOp.AFFINE)
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(state.add_read('delta'), None, node, INPUT_CONNECTOR_NAME, dace.Memlet('delta[0:N]'))
    state.add_edge(state.add_read('coef'), None, node, COEF_CONNECTOR_NAME, dace.Memlet('coef[0:N]'))
    if with_init:
        node.add_in_connector(INIT_CONNECTOR_NAME)
        state.add_edge(state.add_read('seed'), None, node, INIT_CONNECTOR_NAME, dace.Memlet('seed[0]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('out'), None, dace.Memlet('out[0:N]'))
    sdfg.validate()
    return sdfg


def contracting_inputs(n: int, seed_value: float = 0.25):
    """Coefficients strictly inside the unit circle, so the reference itself is well conditioned."""
    rng = np.random.default_rng(0)
    coef = (0.4 + 0.5 * rng.random(n)).astype(np.float64)
    delta = rng.standard_normal(n).astype(np.float64)
    return coef, delta, np.array([seed_value], dtype=np.float64)


@pytest.mark.parametrize('implementation', ['pure', 'CPU'])
@pytest.mark.parametrize('n', [1, 2, 17, 40001])
def test_affine_scan_matches_sequential_recurrence(implementation, n):
    """The libnode computes the recurrence, at every size the blocked lowering treats differently."""
    sdfg = build_affine_sdfg(implementation=implementation, with_init=True)
    coef, delta, seed = contracting_inputs(n)
    out = np.zeros(n, dtype=np.float64)
    sdfg(coef=coef, delta=delta, seed=seed, out=out, N=n)
    assert np.allclose(out, affine_reference(coef, delta, float(seed[0])), rtol=0, atol=1e-11)


def test_affine_scan_without_init_enters_at_zero():
    """No ``_scan_init`` means ``out[-1] == 0``, so ``out[0] == d[0]`` exactly."""
    n = 257
    sdfg = build_affine_sdfg(with_init=False)
    coef, delta, _ = contracting_inputs(n)
    out = np.zeros(n, dtype=np.float64)
    sdfg(coef=coef, delta=delta, out=out, N=n)
    assert out[0] == delta[0]
    assert np.allclose(out, affine_reference(coef, delta, 0.0), rtol=0, atol=1e-11)


def test_affine_scan_is_thread_count_stable_on_contracting_coefficients():
    """The blocked carry must not move the answer: same SDFG, 1 thread vs 8."""
    import os
    n = 65537
    coef, delta, seed = contracting_inputs(n)
    results = []
    for threads in ('1', '8'):
        old = os.environ.get('OMP_NUM_THREADS')
        os.environ['OMP_NUM_THREADS'] = threads
        try:
            out = np.zeros(n, dtype=np.float64)
            build_affine_sdfg()(coef=coef, delta=delta, seed=seed, out=out, N=n)
            results.append(out)
        finally:
            if old is None:
                del os.environ['OMP_NUM_THREADS']
            else:
                os.environ['OMP_NUM_THREADS'] = old
    assert np.allclose(results[0], results[1], rtol=0, atol=1e-11)


def test_affine_scan_of_one_static_element():
    """A statically length-1 subset is scalar-typed by the codegen, so it takes its own shape."""
    sdfg = dace.SDFG('affine_one')
    for name in ('coef', 'delta', 'out'):
        sdfg.add_array(name, [1], dace.float64)
    sdfg.add_array('seed', [1], dace.float64)
    state = sdfg.add_state()
    node = Scan('affine', op=ScanOp.AFFINE)
    node.add_in_connector(INIT_CONNECTOR_NAME)
    state.add_node(node)
    state.add_edge(state.add_read('delta'), None, node, INPUT_CONNECTOR_NAME, dace.Memlet('delta[0]'))
    state.add_edge(state.add_read('coef'), None, node, COEF_CONNECTOR_NAME, dace.Memlet('coef[0]'))
    state.add_edge(state.add_read('seed'), None, node, INIT_CONNECTOR_NAME, dace.Memlet('seed[0]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()

    out = np.zeros(1, dtype=np.float64)
    sdfg(coef=np.array([3.0]), delta=np.array([0.5]), seed=np.array([2.0]), out=out)
    assert out[0] == 3.0 * 2.0 + 0.5


@pytest.mark.parametrize('dtype,nptype', [(dace.float32, np.float32), (dace.float64, np.float64),
                                          (dace.int64, np.int64)])
def test_affine_scan_over_dtypes(dtype, nptype):
    """The recurrence is a multiply-add, so it must hold at every arithmetic element type.

    Integers make the check absolute rather than approximate: with integral coefficients the
    blocked lowering has to reproduce the sequential recurrence EXACTLY, so any drift in the
    block-boundary composition shows up as a hard inequality instead of a tolerance question.
    """
    n = 1201
    if np.issubdtype(nptype, np.integer):
        # A ZERO coefficient every fourth element, deliberately: it resets the recurrence, so
        # the composed map's ``a`` collapses to zero and the prefix before it must stop
        # contributing entirely. It also bounds the reference -- a coefficient of 2 compounding
        # over the whole range overflows int64 long before the end.
        coef = np.resize(np.array([2, 1, -1, 0], dtype=nptype), n).copy()
        delta = (np.arange(n) % 7 - 3).astype(nptype)
        seed = np.array([3], dtype=nptype)
    else:
        rng = np.random.default_rng(2)
        coef = (0.4 + 0.5 * rng.random(n)).astype(nptype)
        delta = rng.standard_normal(n).astype(nptype)
        seed = np.array([0.25], dtype=nptype)

    out = np.zeros(n, dtype=nptype)
    build_affine_sdfg(dtype=dtype)(coef=coef, delta=delta, seed=seed, out=out, N=n)

    want = np.empty(n, dtype=nptype)
    acc = seed[0]
    for k in range(n):
        acc = nptype(coef[k] * acc + delta[k])
        want[k] = acc
    if np.issubdtype(nptype, np.integer):
        assert np.array_equal(out, want)
        return
    # Measured in ULP of the largest value in the range, not as a relative tolerance: the
    # recurrence passes close to zero, where a relative bound says nothing, and the difference
    # against this reference is the numpy scalar chain's own rounding rather than the blocking
    # (it is the same at 1, 4 and 8 threads). Four ULP is headroom over the 1 measured.
    assert np.abs(out - want).max() <= 4 * np.spacing(np.abs(want).max())


@pytest.mark.parametrize('n', [255, 256, 257, 2047, 2048, 2049])
def test_affine_scan_across_block_boundaries(n):
    """Sizes straddling the blocking. The carry crosses a block exactly at these counts, and a
    composition that is off by one block reads as a wrong prefix from that point on -- which a
    single mid-sized ``n`` would not distinguish from a correct one."""
    coef, delta, seed = contracting_inputs(n)
    out = np.zeros(n, dtype=np.float64)
    build_affine_sdfg()(coef=coef, delta=delta, seed=seed, out=out, N=n)
    assert np.allclose(out, affine_reference(coef, delta, float(seed[0])), rtol=0, atol=1e-11)


def test_affine_scan_with_growing_coefficients_is_thread_stable():
    """Coefficients ABOVE one, where the block product actually grows.

    Contracting coefficients hide the one numerical risk this lowering has: ``fold_affine``
    forms ``prod(c)`` over a block, and only a coefficient bigger than one makes that product
    large. Checked relatively, since the values themselves span many orders of magnitude by the
    end of the range.
    """
    import os
    n = 20011
    coef = np.full(n, 1.0009, dtype=np.float64)
    delta = np.full(n, 0.001, dtype=np.float64)
    seed = np.array([1.0], dtype=np.float64)
    want = affine_reference(coef, delta, 1.0)

    for threads in ('1', '8'):
        old_value = os.environ.get('OMP_NUM_THREADS')
        os.environ['OMP_NUM_THREADS'] = threads
        try:
            out = np.zeros(n, dtype=np.float64)
            build_affine_sdfg()(coef=coef, delta=delta, seed=seed, out=out, N=n)
        finally:
            if old_value is None:
                del os.environ['OMP_NUM_THREADS']
            else:
                os.environ['OMP_NUM_THREADS'] = old_value
        assert np.allclose(out, want, rtol=1e-12, atol=0), f'drifted at OMP_NUM_THREADS={threads}'


def test_affine_node_wires_the_coefficient_connector():
    """``_scan_coef`` is part of the node's shape for AFFINE and absent for every other op."""
    assert COEF_CONNECTOR_NAME in Scan('a', op=ScanOp.AFFINE).in_connectors
    assert COEF_CONNECTOR_NAME not in Scan('s', op=ScanOp.SUM).in_connectors


def test_affine_scan_refuses_a_coefficient_of_the_wrong_length():
    """A shorter coefficient array is a wiring bug, not a broadcast."""
    sdfg = dace.SDFG('affine_bad_coef')
    sdfg.add_array('coef', [N], dace.float64)
    sdfg.add_array('delta', [N], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    state = sdfg.add_state()
    node = Scan('affine', op=ScanOp.AFFINE)
    state.add_node(node)
    state.add_edge(state.add_read('delta'), None, node, INPUT_CONNECTOR_NAME, dace.Memlet('delta[0:N]'))
    state.add_edge(state.add_read('coef'), None, node, COEF_CONNECTOR_NAME, dace.Memlet('coef[0:N-1]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('out'), None, dace.Memlet('out[0:N]'))
    with pytest.raises(ValueError, match='_scan_coef'):
        node.validate(sdfg, state)


def test_affine_scan_refuses_shapes_without_a_lowering():
    """Exclusive / multi-chain / strided affine scans refuse instead of falling back to a scalar op."""
    for attr, value in (('exclusive', True), ('chains', 2), ('stride', 2)):
        sdfg = build_affine_sdfg()
        node = next(n for n in sdfg.states()[0].nodes() if isinstance(n, Scan))
        setattr(node, attr, value)
        with pytest.raises(NotImplementedError, match='AFFINE'):
            sdfg.expand_library_nodes()


def build_affine_cuda_sdfg(seed_on_device: bool) -> dace.SDFG:
    """The affine scan over device-global buffers, with the seed on whichever side is asked for.

    The seed's side is the branch that matters: the wrapper takes it as a pointer and as a value and
    uses exactly one, because host code issuing the launch must not dereference a device address.
    """
    from dace import dtypes

    sdfg = dace.SDFG(f'affine_scan_cuda_{int(seed_on_device)}')
    for name in ('coef', 'delta', 'out'):
        sdfg.add_array(name, [N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    seed_storage = dtypes.StorageType.GPU_Global if seed_on_device else dtypes.StorageType.Default
    sdfg.add_array('seed', [1], dace.float64, storage=seed_storage)
    state = sdfg.add_state()

    node = Scan('affine', op=ScanOp.AFFINE)
    node.implementation = 'CUDA'
    node.schedule = dtypes.ScheduleType.GPU_Device
    node.add_in_connector(INIT_CONNECTOR_NAME)
    state.add_node(node)
    state.add_edge(state.add_read('delta'), None, node, INPUT_CONNECTOR_NAME, dace.Memlet('delta[0:N]'))
    state.add_edge(state.add_read('coef'), None, node, COEF_CONNECTOR_NAME, dace.Memlet('coef[0:N]'))
    state.add_edge(state.add_read('seed'), None, node, INIT_CONNECTOR_NAME, dace.Memlet('seed[0]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('out'), None, dace.Memlet('out[0:N]'))
    sdfg.validate()
    return sdfg


def test_the_cuda_affine_call_is_emitted_into_the_cuda_unit():
    """``cub/cub.cuh`` is a CUDA header, so the host translation unit cannot hold the call.

    The host tasklet gets a wrapper call and the wrapper's body goes to the CUDA unit -- the shape
    ``ExpandFindFirstCUDA`` established. Asserted on the emitted text because the text IS what the
    host compiler has to parse.
    """
    sdfg = build_affine_cuda_sdfg(seed_on_device=False)
    sdfg.expand_library_nodes()
    tasklet = next(n for n in sdfg.states()[0].nodes() if isinstance(n, dace.sdfg.nodes.Tasklet))
    assert 'inclusive_affine' not in tasklet.code.as_string, 'the device call reached the host tasklet'
    assert '__dace_scan_affine_' in tasklet.code.as_string
    assert 'inclusive_affine' in sdfg.global_code['cuda'].as_string


@pytest.mark.gpu
@pytest.mark.parametrize('seed_on_device', [False, True])
@pytest.mark.parametrize('n', [17, 4096])
def test_the_cuda_affine_scan_computes_the_recurrence(seed_on_device, n):
    """Both seed shapes reproduce the sequential recurrence the CPU lowering computes."""
    import cupy as cp

    sdfg = build_affine_cuda_sdfg(seed_on_device)
    coef, delta, seed = contracting_inputs(n)
    want = affine_reference(coef, delta, seed[0])
    out = cp.zeros(n, dtype=np.float64)
    sdfg(coef=cp.asarray(coef),
         delta=cp.asarray(delta),
         out=out,
         seed=cp.asarray(seed) if seed_on_device else seed,
         N=n)
    assert np.allclose(cp.asnumpy(out), want, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    test_affine_scan_matches_sequential_recurrence('CPU', 40001)
    test_affine_scan_without_init_enters_at_zero()
    test_affine_scan_of_one_static_element()
    test_affine_scan_is_thread_count_stable_on_contracting_coefficients()
    test_affine_node_wires_the_coefficient_connector()
    test_affine_scan_refuses_a_coefficient_of_the_wrong_length()
    test_affine_scan_refuses_shapes_without_a_lowering()
    test_the_cuda_affine_call_is_emitted_into_the_cuda_unit()
    print('ok')
