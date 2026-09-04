# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Standalone tests for the :class:`~dace.libraries.standard.nodes.ArgReduce`
libnode (argmax / argmin -> value + index, two scalar outputs).

The node reads its own operand -- a strided slice, and a unary ``transform`` applied per element
as it reads -- so that a caller wanting ``argmax |a[inc*j]|`` never has to stage the transformed
sequence first. Both are exercised here against a SEQUENTIAL reference rather than ``np.argmax``,
because the property under test is not "finds the maximum" but "finds the same occurrence of the
maximum the sequential scan does": the guard is strict, so the FIRST one wins, and the parallel
expansion has to reproduce that however OpenMP split the range across threads.
"""
import pathlib

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import ArgReduce

N = dace.symbol('N')
M = dace.symbol('M')
S = dace.symbol('S')


def _build(op: str):
    """SDFG with a single ArgReduce over ``a[0:N]`` -> ``val`` (float64) +
    ``idx`` (int64)."""
    sdfg = dace.SDFG(f'argreduce_{op}')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('val', [1], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    r = state.add_read('a')
    wv = state.add_write('val')
    wi = state.add_write('idx')
    node = ArgReduce('argreduce', op=op)
    state.add_node(node)
    state.add_edge(r, None, node, '_in', dace.Memlet('a[0:N]'))
    state.add_edge(node, '_out_val', wv, None, dace.Memlet('val[0]'))
    state.add_edge(node, '_out_idx', wi, None, dace.Memlet('idx[0]'))
    return sdfg


@pytest.mark.parametrize('op', ['max', 'min'])
def test_arg_reduce_value_and_index(op):
    sdfg = _build(op)
    sdfg.validate()
    sdfg.expand_library_nodes()
    n = 64
    rng = np.random.default_rng(0xA76 + (op == 'min'))
    a = rng.standard_normal(n)
    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a, val=val, idx=idx, N=n)
    if op == 'max':
        assert np.isclose(val[0], a.max())
        assert idx[0] == int(np.argmax(a))
    else:
        assert np.isclose(val[0], a.min())
        assert idx[0] == int(np.argmin(a))


@pytest.mark.parametrize('op', ['max', 'min'])
@pytest.mark.parametrize('stride', [2, 3])
def test_arg_reduce_strided_input(op, stride):
    """Strided input slice ``a[0:N*stride:stride]`` -- the expansion reads
    element ``j`` at ``_in[j*stride]`` (non-unit-stride code path) and returns
    the SLICE-LOCAL index ``j`` of the extreme strided element."""
    sdfg = dace.SDFG(f'argreduce_{op}_s{stride}')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('val', [1], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    r = state.add_read('a')
    wv = state.add_write('val')
    wi = state.add_write('idx')
    node = ArgReduce('argreduce', op=op)
    state.add_node(node)
    # Reduce over the strided slice a[0 : N : stride].
    state.add_edge(r, None, node, '_in', dace.Memlet(f'a[0:N:{stride}]'))
    state.add_edge(node, '_out_val', wv, None, dace.Memlet('val[0]'))
    state.add_edge(node, '_out_idx', wi, None, dace.Memlet('idx[0]'))
    sdfg.validate()
    sdfg.expand_library_nodes()

    m = 16
    n = m * stride
    rng = np.random.default_rng(700 + stride + (op == 'min'))
    a = rng.standard_normal(n)
    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a, val=val, idx=idx, N=n)
    strided = a[0:n:stride]
    expected_j = int(np.argmax(strided)) if op == 'max' else int(np.argmin(strided))
    assert np.isclose(val[0], strided[expected_j]), f"value: got {val[0]}, expected {strided[expected_j]}"
    assert idx[0] == expected_j, f"slice-local index: got {idx[0]}, expected {expected_j}"


def sequential_arg_extreme(seq, op):
    """The answer the loop being lifted computes: a strict comparison, so the FIRST extreme wins."""
    best, best_j = seq[0], 0
    for j in range(1, len(seq)):
        if (seq[j] > best) if op == 'max' else (seq[j] < best):
            best, best_j = seq[j], j
    return best, best_j


@pytest.mark.parametrize('op', ['max', 'min'])
@pytest.mark.parametrize('impl', ['pure', 'OpenMP'])
@pytest.mark.parametrize('transform', ['', 'abs'])
@pytest.mark.parametrize('stride', [1, 3])
def test_arg_reduce_reads_a_strided_transformed_gather_in_one_pass(op, impl, transform, stride):
    """``argmax f(a[lo::stride])`` with no staging buffer, on both CPU expansions.

    Sizes run from a single element up past the point where OpenMP splits the range across
    threads, and half the draws are rounded onto the integers so equal extremes actually occur --
    a tie is where a thread split can disagree with the sequential scan without being "wrong"
    about the value.
    """
    n_list = [1, 2, 7, 8, 9, 4095, 4096, 4097]
    sdfg = dace.SDFG(f'ar_{op}_{impl}_{transform or "id"}_{stride}')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('val', [1], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    node = ArgReduce('argreduce', op=op, transform=transform)
    node.implementation = impl
    state.add_node(node)
    sub = 'a[0:N]' if stride == 1 else f'a[0:N:{stride}]'
    state.add_edge(state.add_read('a'), None, node, '_in', dace.Memlet(sub))
    state.add_edge(node, '_out_val', state.add_write('val'), None, dace.Memlet('val[0]'))
    state.add_edge(node, '_out_idx', state.add_write('idx'), None, dace.Memlet('idx[0]'))
    sdfg.validate()
    sdfg.expand_library_nodes()
    csdfg = sdfg.compile()

    for n_elems in n_list:
        for ties in (False, True):
            total = n_elems * stride
            drawn = np.random.default_rng(0x318 + n_elems + 7 * ties).standard_normal(total)
            a = np.round(drawn) if ties else drawn
            seq = a[0:total:stride]
            if transform:
                seq = np.abs(seq)
            exp_v, exp_j = sequential_arg_extreme(list(seq), op)
            val = np.zeros(1)
            idx = np.zeros(1, dtype=np.int64)
            csdfg(a=a.copy(), val=val, idx=idx, N=total)
            assert val[0] == exp_v, f'n={n_elems} ties={ties}: value {val[0]} != {exp_v}'
            assert idx[0] == exp_j, f'n={n_elems} ties={ties}: index {idx[0]} != {exp_j} (first extreme)'


@pytest.mark.parametrize('impl', ['pure', 'OpenMP'])
@pytest.mark.parametrize('lo,step', [(3, 1), (3, 2), (5, 3)])
def test_arg_reduce_reads_from_the_slice_base_not_the_array_base(impl, lo, step):
    """A slice that does not start at element 0 must be read from ITS start.

    The lifted gather ``a[base + coeff*i]`` puts a non-zero base under the arg-reduction whenever
    the seed iteration sits above the array's first element, and an off-by-base read still
    produces a plausible value and index -- so it is checked, not assumed. The returned index
    stays SLICE-LOCAL either way.
    """
    m_elems = 37
    hi = lo + step * (m_elems - 1)
    sdfg = dace.SDFG(f'argreduce_base_{impl}_{lo}_{step}')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('val', [1], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    node = ArgReduce('argreduce', op='max', transform='abs')
    node.implementation = impl
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_in', dace.Memlet(f'a[{lo}:{hi + 1}:{step}]'))
    state.add_edge(node, '_out_val', state.add_write('val'), None, dace.Memlet('val[0]'))
    state.add_edge(node, '_out_idx', state.add_write('idx'), None, dace.Memlet('idx[0]'))
    sdfg.expand_library_nodes()

    n = hi + 4
    a = np.round(np.random.default_rng(lo * 31 + step).standard_normal(n))
    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a.copy(), val=val, idx=idx, N=n)
    seq = np.abs(a[lo:hi + 1:step])
    exp_v, exp_j = sequential_arg_extreme(list(seq), 'max')
    assert val[0] == exp_v and idx[0] == exp_j, f'got ({val[0]}, {idx[0]}), expected ({exp_v}, {exp_j})'


def test_arg_reduce_counts_a_symbolic_stride_from_the_memlet_volume():
    """A slice whose STRIDE is a runtime symbol still scans exactly its own length.

    A subset states its element count as ``ceiling((hi - lo + 1) / step)``, which sympy cannot
    resolve when ``step`` is a symbol -- it leaves ``n - 1 + ceiling(1/step)`` standing. The node
    therefore counts from the memlet's ``volume``, which the caller states. Getting this wrong is
    silent: the scan simply visits the wrong number of elements.
    """
    stride = dace.symbol('S')
    sdfg = dace.SDFG('argreduce_symbolic_stride')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('val', [1], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_symbol('S', dace.int64)
    state = sdfg.add_state()
    node = ArgReduce('argreduce', op='max')
    state.add_node(node)
    m = dace.Memlet(data='a', subset=dace.subsets.Range([(0, stride * (M - 1), stride)]), volume=M)
    state.add_edge(state.add_read('a'), None, node, '_in', m)
    state.add_edge(node, '_out_val', state.add_write('val'), None, dace.Memlet('val[0]'))
    state.add_edge(node, '_out_idx', state.add_write('idx'), None, dace.Memlet('idx[0]'))
    sdfg.validate()
    sdfg.expand_library_nodes()

    s, m_elems = 3, 40
    a = np.random.default_rng(0x5713).standard_normal(s * m_elems)
    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a.copy(), val=val, idx=idx, N=a.size, M=m_elems, S=s)
    seq = a[0:s * m_elems:s]
    exp_v, exp_j = sequential_arg_extreme(list(seq), 'max')
    assert val[0] == exp_v and idx[0] == exp_j, (
        f'symbolic stride: got ({val[0]}, {idx[0]}), expected ({exp_v}, {exp_j}) over {m_elems} elements')


def test_arg_reduce_refuses_a_dynamic_input_memlet():
    """A dynamic memlet's volume is a bound, not a count, so it cannot say how far to scan."""
    sdfg = _build('max')
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ArgReduce))
    state = next(st for st in sdfg.states() if node in st.nodes())
    next(e for e in state.in_edges(node) if e.dst_conn == '_in').data.dynamic = True
    with pytest.raises(ValueError, match='dynamic memlet'):
        node.validate(sdfg, state)


@pytest.mark.parametrize('op', ['max', 'min'])
def test_arg_reduce_tie_breaks_to_first(op):
    """Strict comparison -> the FIRST occurrence of the extreme wins (matches
    ``np.argmax``/``np.argmin``, which also return the first)."""
    sdfg = _build(op)
    sdfg.expand_library_nodes()
    # Two equal extremes; the earlier index must win.
    a = np.array([1.0, 5.0, 2.0, 5.0, 0.0, 0.0]) if op == 'max' else np.array([3.0, 0.0, 1.0, 0.0, 2.0])
    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a, val=val, idx=idx, N=a.shape[0])
    expected = int(np.argmax(a)) if op == 'max' else int(np.argmin(a))
    assert idx[0] == expected, f"{op}: got {idx[0]}, expected first extreme at {expected}"


def gpu_argreduce(op: str) -> dace.SDFG:
    """The same node with device operands, so the CUDA expansion is the one that applies."""
    sdfg = _build(op)
    sdfg.name = f'argreduce_gpu_{op}'
    # Input on the device, answers on the host: that is what ``host_connectors`` declares, and the
    # expansion writes both from the host code that issues the launch.
    sdfg.arrays['a'].storage = dace.StorageType.GPU_Global
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ArgReduce))
    node.implementation = 'CUDA'
    return sdfg


@pytest.mark.parametrize('op', ['max', 'min'])
def test_the_cuda_expansion_calls_cub_and_brings_the_answer_back(op):
    """CUB answers in DEVICE memory; both outputs are host scalars, so the answer has to be copied.

    The CUB call, the scratch block and the copy back live in ``dace::cub::arg_reduce`` rather than
    in the emitted string -- that is where the toolkit picks between CUB's deprecated
    ``KeyValuePair`` output and the two-iterator one that replaced it. So the emitted code is
    checked for the right op tag, and the header is checked for the three things it now owns.
    """
    sdfg = gpu_argreduce(op)
    sdfg.expand_library_nodes()
    code = '\n'.join(c.clean_code for c in sdfg.generate_code())

    tag = 'ArgMaxOp' if op == 'max' else 'ArgMinOp'
    assert f'::dace::cub::arg_reduce<::dace::cub::{tag}>' in code, (
        f'the CUDA expansion did not dispatch the {op} arg-reduction to ::dace::cub::{tag}')

    compat = (pathlib.Path(dace.__file__).parent / 'runtime' / 'include' / 'dace' / 'cub_compat.cuh').read_text()
    kind = 'ArgMax' if op == 'max' else 'ArgMin'
    # The gpu* spellings are the backend-neutral aliases, so one expansion serves CUDA and HIP.
    assert f'gpucub::DeviceReduce::{kind}' in compat, (
        f'::dace::cub::{tag} does not reach gpucub::DeviceReduce::{kind}')
    assert 'get_scratch<ReduceTag>' in compat, 'the workspace does not come from the scratch pool'
    assert 'gpuMemcpyAsync' in compat and 'gpuMemcpyDeviceToHost' in compat, \
        'the answer is never copied back, so the outputs read device memory'
    assert 'gpuStreamSynchronize' in compat, \
        'the copy back is asynchronous, so without a sync the host reads the answer before it lands'
    assert ArgReduce.host_connectors == frozenset(
        {'_out_val',
         '_out_idx'}), ('both answers are written by host code, so an offloader must be told to leave them there')


def gpu_gather_sdfg(name, subset, transform='', op='max', size=64):
    """A CUDA ArgReduce reading ``subset`` of a device array, optionally through ``transform``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [size], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('val', [1], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    node = ArgReduce('argreduce', op=op, transform=transform)
    node.implementation = 'CUDA'
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_in', dace.Memlet(subset))
    state.add_edge(node, '_out_val', state.add_write('val'), None, dace.Memlet('val[0]'))
    state.add_edge(node, '_out_idx', state.add_write('idx'), None, dace.Memlet('idx[0]'))
    return sdfg


#: ``(subset, transform, stride_the_wrapper_must_be_handed)``. ``None`` means the read is provably
#: contiguous and untransformed, so it must keep the raw-pointer fast path instead.
CUDA_GATHER_CASES = [
    ('a[0:64]', '', None),
    ('a[0:64]', 'abs', '1'),
    ('a[0:64:2]', '', '2'),
    ('a[0:64:2]', 'abs', '2'),
    ('a[0:64:S]', 'abs', 'S'),
]


@pytest.mark.parametrize('subset,transform,stride',
                         CUDA_GATHER_CASES,
                         ids=[c[0] + '/' + (c[1] or 'id') for c in CUDA_GATHER_CASES])
def test_the_cuda_expansion_gathers_a_strided_or_transformed_operand(subset, transform, stride):
    """A stride or a transform goes to CUB as an input iterator, not as a refusal.

    ``DeviceReduce`` walks an iterator, so ``xf(base[j * stride])`` is one streaming pass with no
    staged copy. The contiguous untransformed read must NOT pick up the iterator: a raw pointer is
    what lets CUB issue vectorised loads, so keeping that path is the point of branching at all.
    A symbolic stride (TSVC ``s318``'s ``inc``) is the case that has to reach the wrapper as a
    runtime argument -- the wrapper is a free function in the CUDA unit, where the symbol is not
    in scope.
    """
    sdfg = gpu_gather_sdfg(f'argreduce_gpu_gather_{abs(hash((subset, transform))) % 10**8}', subset, transform)
    sdfg.expand_library_nodes()
    code = '\n'.join(c.clean_code for c in sdfg.generate_code())

    if stride is None:
        assert 'gather_iterator' not in code, 'a contiguous untransformed read lost CUB\'s raw-pointer fast path'
        assert '__ar_stride' not in code, 'the fast path does not need a stride and must not carry one'
        return

    functor = 'AbsXf' if transform else 'IdentityXf'
    assert f'::dace::cub::gather_iterator<::dace::cub::{functor}>(__ar_in, __ar_stride)' in code, (
        f'the {subset!r}/{transform or "identity"} read did not reach CUB through a gather iterator')
    assert f'long long __ar_stride' in code, 'the wrapper does not take the stride as an argument'
    assert f'(long long)({stride})' in code, (
        f'the host tasklet does not hand the wrapper the stride {stride!r}, which is the only place '
        f'the symbol it is written in is in scope')


def test_the_cub_gather_iterator_is_built_on_the_supported_iterators():
    """cub's own iterators warn from CCCL 2.8 and are gone in CCCL 3, and warnings are errors here,
    so the gather has to be able to fall back to thrust's -- the same choice ``reduction.h`` makes."""
    compat = (pathlib.Path(dace.__file__).parent / 'runtime' / 'include' / 'dace' / 'cub_compat.cuh').read_text()
    assert 'thrust::transform_iterator' in compat and 'thrust::counting_iterator' in compat, (
        'the gather iterator has no thrust spelling, so it cannot build on CCCL 3')
    assert 'gpucub::TransformInputIterator' in compat and 'gpucub::CountingInputIterator' in compat, (
        'the gather iterator has no cub spelling, so it cannot build where thrust is absent')
    for functor in ('IdentityXf', 'AbsXf', 'StridedGather'):
        assert f'struct {functor}' in compat, f'{functor} is named by the expansion but not defined'


def test_arg_reduce_rejects_an_unknown_transform():
    """The transform name reaches the generated source verbatim, so it is a closed set."""
    with pytest.raises(ValueError, match='transform must be one of'):
        ArgReduce('argreduce', op='max', transform='sqrt')


@pytest.mark.gpu
@pytest.mark.parametrize('op', ['max', 'min'])
@pytest.mark.parametrize('transform', ['', 'abs'])
@pytest.mark.parametrize('stride', [1, 3])
def test_the_cuda_expansion_matches_the_sequential_scan_through_the_gather(op, transform, stride):
    """The device answer for a strided/transformed operand is the SEQUENTIAL one, on real hardware.

    Emitting a gather iterator only proves the code compiles; what matters is that CUB reduces over
    the sequence the node means and reports the index in THAT sequence -- slice-local, first
    occurrence on a tie, exactly as the CPU expansions answer. Half the draws are rounded onto the
    integers so ties actually occur, which is where a parallel combine can disagree with the
    sequential scan without being wrong about the value.
    """
    import cupy

    n_elems, total = 1000, 1000 * stride
    subset = 'a[0:N]' if stride == 1 else f'a[0:N:{stride}]'
    sdfg = dace.SDFG(f'argreduce_gpu_run_{op}_{transform or "id"}_{stride}')
    sdfg.add_array('a', [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('val', [1], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    node = ArgReduce('argreduce', op=op, transform=transform)
    node.implementation = 'CUDA'
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_in', dace.Memlet(subset))
    state.add_edge(node, '_out_val', state.add_write('val'), None, dace.Memlet('val[0]'))
    state.add_edge(node, '_out_idx', state.add_write('idx'), None, dace.Memlet('idx[0]'))
    sdfg.validate()
    csdfg = sdfg.compile()

    for ties in (False, True):
        drawn = np.random.default_rng(0x318 + stride + 7 * ties).standard_normal(total)
        a = np.round(drawn) if ties else drawn
        seq = a[0:total:stride]
        if transform:
            seq = np.abs(seq)
        exp_v, exp_j = sequential_arg_extreme(list(seq), op)
        val = np.zeros(1)
        idx = np.zeros(1, dtype=np.int64)
        csdfg(a=cupy.asarray(a), val=val, idx=idx, N=total)
        assert val[0] == exp_v, f'ties={ties}: device value {val[0]} != sequential {exp_v}'
        assert idx[0] == exp_j, (f'ties={ties}: device index {idx[0]} != {exp_j}; the index must be '
                                 f'slice-local and break the tie toward the FIRST occurrence')
    assert n_elems == len(seq), 'the gather did not present one element per strided position'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
