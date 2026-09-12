# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF's DEVICE dialect: an SDFG with kernels in it, rendered as ONE self-contained ``.hip`` unit.

The host dialects render a program the compiler that builds them can already express. The device
one has to hold two code objects in one file and reach two machines from one text, and that is
where its failures live -- so every test here is about one of the four:

* a definition emitted into BOTH code objects, which is a redefinition once they are concatenated
  (:func:`~dace.codegen.cpf.merged_object`), while a repeated PROTOTYPE must survive,
* a write-conflict resolution reached on the DEVICE, where an ``omp atomic`` compiles and does
  nothing and the GPU thread-block tree reduction is spelled with runtime templates,
* a library node whose operands are device-resident, which must not be re-pointed at a HOST
  implementation just because that one renders,
* the runtime the DEVICE implementation then calls, which CPF has to spell for itself.

Rendering, not running: these assert on the emitted text. What a GPU actually computes is the
subject of the library nodes' own suites, and CPF changes neither the algorithm nor the schedule.
"""
import re

import numpy as np
import pytest

import dace
from dace import cpf_lowering
from dace.codegen import cpf
from dace.codegen.codeobject import CodeObject
from dace.codegen.cpf import render as render_sdfg
from dace.libraries.standard.nodes import FindFirst, Scan
from dace.libraries.standard.nodes.scan import ScanOp
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.codegen.cpf.conftest import assert_standalone_device

N = dace.symbol('N')


def render_gpu(program, name: str):
    """``(sdfg, code)`` for a ``@dace.program`` taken through the GPU pipeline and rendered as HIP.

    The pipeline is the documented order and all three steps matter here: ``canonicalize`` leaves
    every choice parallel, ``offload_to_gpu`` moves the data and the maps onto the device, and
    ``finalize_for_target`` is what SELECTS the device library implementations this module is
    about. Rendering a graph that skipped the last step would test the host selection instead.
    """
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, validate=True, validate_all=False, target='gpu')
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, 'gpu', validate=True)
    rendering = render_sdfg(sdfg, language='hip')
    return rendering.sdfg, rendering.code


def duplicable_definitions(code: str):
    """The namespace-scope definitions a repeat of would be a redefinition error.

    Read with the emitter's own patterns rather than a second spelling of them: a test table that
    drifted from :data:`~dace.codegen.cpf.DUPLICABLE_DEFINITIONS` would pass while the unit did not
    compile. What is asserted from outside is the CONSEQUENCE -- each one appears once.
    """
    return [line for line in code.splitlines() if any(p.match(line) for p in cpf.DUPLICABLE_DEFINITIONS)]


def test_shared_definitions_are_emitted_once_and_prototypes_are_not_dropped():
    """A GPU SDFG generates two code objects, and separate compilation gives each one its own copy
    of what it needs: the ``<array>_idx`` and ``<array>_size`` helpers, and the SDFG's constants.
    Concatenated into one unit a second copy is a redefinition, so it is dropped.

    The other half is the trap. A repeated DECLARATION must NOT be dropped: the frame calls
    ``__dace_runkernel_*`` and the device object defines it, so the prototype appears in both, and
    removing the frame's copy would remove the only declaration before the call."""

    @dace.program
    def blend(x: dace.float64[N], y: dace.float64[N]):
        y[:] = x * 2.0 + 1.0

    _, code = render_gpu(blend, 'cpf_hip_dedup')
    assert_standalone_device(code, 'cpf_hip_dedup')

    definitions = duplicable_definitions(code)
    assert definitions, 'the device rendering emitted no namespace-scope definitions to check'
    repeated = sorted({line for line in definitions if definitions.count(line) > 1})
    assert not repeated, f'definition emitted more than once: {repeated}'

    prototypes = re.findall(r'^.*\b__dace_runkernel_\w+\([^;]*\);\s*$', code, re.MULTILINE)
    assert prototypes, 'the device rendering declared no kernel launcher'
    assert len(prototypes) > len(set(prototypes)), ('a launcher prototype must be allowed to repeat -- the frame '
                                                    'declares what the device object defines')


def test_device_reduction_folds_without_a_runtime_functor():
    """A reduction under a GPU map is a ``gpucub::BlockReduce`` over per-thread register partials
    plus one atomic per block, and the runtime spells all three of its pieces ``dace::_wcr_fixed``.

    CPF spells them itself: the per-thread accumulation is a plain read-modify-write, because the
    partial is that thread's own register slot and cannot conflict; the fold's binary operator is a
    lambda; and the one atomic per block is CPF's own ``cpf_gpu_atomic``."""

    @dace.program
    def total(a: dace.float64[N], out: dace.float64[1]):
        out[0] = np.sum(a)

    _, code = render_gpu(total, 'cpf_hip_reduce')
    assert_standalone_device(code, 'cpf_hip_reduce')

    assert 'gpucub::BlockReduce' in code, 'the block fold must survive -- CPF renders the tree reduction, not a loop'
    assert '_wcr_fixed' not in code, 'the reduction functor must not be the runtime template'
    partial = re.search(r'^\s*(__bpart_\w+)\[0\] = \1\[0\] \+ \(.*\);$', code, re.MULTILINE)
    assert partial is not None, f'the per-thread partial must accumulate with a plain +\n{code}'
    assert re.search(r'cpf_gpu_atomic\(.*\[\] \(const double &__cpf_acc, const double &__cpf_val\)', code), \
        'the block result must commit through one device atomic taking the operator as a functor'


def test_conflicting_device_wcr_is_an_atomic_and_never_an_omp_pragma():
    """A scatter whose index array may repeat is the WCR that stays conflicting. On the host CPF
    writes ``_Pragma("omp atomic update")`` for it; inside a ``__global__`` function that pragma is
    ignored by the device compiler, so the rendering would compile, run, and race. The device
    spelling has to be an actual device atomic."""

    @dace.program
    def scatter(idx: dace.int32[N], src: dace.float64[N], bins: dace.float64[N]):
        for i in dace.map[0:N]:
            bins[idx[i]] += src[i]

    _, code = render_gpu(scatter, 'cpf_hip_scatter')
    assert_standalone_device(code, 'cpf_hip_scatter')

    assert 'omp atomic' not in code and 'omp critical' not in code, \
        'an OpenMP atomic in device code is ignored by the device compiler, so the accumulation would race'
    assert re.search(r'cpf_gpu_atomic\(.*\[\] \(const double &__cpf_acc, const double &__cpf_val\) '
                     r'\{ return __cpf_acc \+ \(__cpf_val\); \}\)', code), \
        'the conflicting accumulation must take a device atomic carrying the resolution as a functor'


def device_find_first_sdfg(name: str) -> dace.SDFG:
    """A ``FindFirst`` over DEVICE memory at host level -- the shape a lifted early-exit loop has
    on a GPU graph. Built by hand because what is under test is the IMPLEMENTATION CHOICE, and a
    frontend program would reach it through the whole lift."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array('out', [1], dace.int64)
    state = sdfg.add_state()
    node = FindFirst('ff', predicate='_a[__i] > 0.5', begin=0, end=N)
    node.implementation = 'OpenMP'
    node.schedule = dace.dtypes.ScheduleType.GPU_Device
    node.add_in_connector('_a', dace.pointer(dace.float64))
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_a', dace.Memlet.from_array('a', sdfg.arrays['a']))
    state.add_edge(node, '_out_idx', state.add_write('out'), None, dace.Memlet('out[0]'))
    return sdfg


def test_a_library_node_over_device_memory_takes_its_device_implementation():
    """``FindFirst``'s host expansions refuse to lower over device memory -- correctly, a host
    search would dereference a device pointer -- so CPF must not re-point the node at one just
    because ``pure`` renders. The device implementation is renderable too: CPF spells the search
    (``find_first_index_device``) the same way it spells the host one."""
    sdfg = device_find_first_sdfg('cpf_hip_findfirst')
    code = cpf.cpf(sdfg, language='hip')
    assert_standalone_device(code, 'cpf_hip_findfirst')
    assert 'find_first_index_device' in code, 'the device search must be rendered, not the host one'
    assert '__global__ void find_first_kernel' in code, 'the search kernel is part of the unit'


def test_the_host_implementation_still_wins_over_host_memory():
    """The other side of the same rule, so it is a CHOICE and not a device-dialect override: the
    same node over HOST memory keeps the host implementation, which is the parallel search."""
    sdfg = device_find_first_sdfg('cpf_hip_findfirst_host')
    sdfg.arrays['a'].storage = dace.dtypes.StorageType.CPU_Heap
    node, state = next((n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, FindFirst))
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_HIP):
        assert cpf.renderable_implementations(node, state)[0] != 'CUDA'


def test_a_device_library_node_inside_a_kernel_keeps_the_device_code_implementation():
    """A node ALREADY inside a kernel has no launch to issue, so the device library call -- which
    is host code -- is not available to it. Both halves of
    :func:`~dace.codegen.cpf.on_device_at_host_level` have to hold, not just the storage one."""
    sdfg = dace.SDFG('cpf_hip_inner_scan')
    sdfg.add_array('a', [N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array('b', [N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Scan('scan')
    entry, exit_node = state.add_map('outer', {'i': '0:N'}, schedule=dace.dtypes.ScheduleType.GPU_Device)
    read, write = state.add_read('a'), state.add_write('b')
    state.add_memlet_path(read, entry, node, dst_conn='_scan_in', memlet=dace.Memlet.from_array('a', sdfg.arrays['a']))
    state.add_memlet_path(node,
                          exit_node,
                          write,
                          src_conn='_scan_out',
                          memlet=dace.Memlet.from_array('b', sdfg.arrays['b']))
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_HIP):
        assert not cpf.on_device_at_host_level(node, state)
        assert cpf.renderable_implementations(node, state)[0] != 'CUDA'


def device_scan_sdfg(name: str, op: ScanOp, coefficients: bool = False, seed: bool = False) -> dace.SDFG:
    """A ``Scan`` over DEVICE memory at host level, in the shape ``LoopToScan`` leaves on a GPU
    graph. ``seed`` puts the entry value in DEVICE memory, which is the case that used to break: a
    host expansion reads it, and a device-resident scalar cannot be read on the host."""
    sdfg = dace.SDFG(name)
    storage = dace.dtypes.StorageType.GPU_Global
    for array in ('a', 'b') + (('c', ) if coefficients else ()):
        sdfg.add_array(array, [N], dace.float64, storage=storage)
    state = sdfg.add_state()
    node = Scan('sc', op=op, exclusive=not coefficients and not seed)
    node.schedule = dace.dtypes.ScheduleType.GPU_Device
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_scan_in', dace.Memlet.from_array('a', sdfg.arrays['a']))
    state.add_edge(node, '_scan_out', state.add_write('b'), None, dace.Memlet.from_array('b', sdfg.arrays['b']))
    if coefficients:
        state.add_edge(state.add_read('c'), None, node, '_scan_coef', dace.Memlet.from_array('c', sdfg.arrays['c']))
    if seed:
        sdfg.add_array('s', [1], dace.float64, storage=storage)
        node.add_in_connector('_scan_init')
        state.add_edge(state.add_read('s'), None, node, '_scan_init', dace.Memlet('s[0]'))
    return sdfg


def test_a_device_scan_renders_the_device_scan_and_its_scratch():
    """The scan's device expansion issues ``gpucub::DeviceScan`` over a workspace it takes from the
    CUB scratch pool -- and the pool is normally an ENVIRONMENT, allocated by the init entry point
    and released by the exit one, neither of which a single self-contained call has. CPF defines
    the pool itself, over a buffer that allocates on first use and frees at static destruction, so
    the environment has nothing left to run and the rendering is admitted rather than refused."""
    code = cpf.cpf(device_scan_sdfg('cpf_hip_scan', ScanOp.SUM), language='hip')
    assert_standalone_device(code, 'cpf_hip_scan')
    assert '::gpucub::DeviceScan::ExclusiveScan' in code, 'the device scan must survive as the device scan'
    assert 'get_scratch<ScanTag>' in code, 'the workspace must come from CPF\'s own pool'
    assert code.count('static inline void *get_scratch(') == 1, 'the pool is defined once'


def test_a_device_resident_scan_seed_is_read_where_it_lives():
    """The seed of an affine scan is an ARRAY of one element per residue class, and on a GPU graph
    it is device-resident like everything else. Re-pointing the node at a host expansion made host
    code dereference it, which validation rejects outright -- so the device expansion is the one to
    keep, and it reads the seed on the device where it lives."""
    sdfg = device_scan_sdfg('cpf_hip_affine', ScanOp.AFFINE, coefficients=True, seed=True)
    code = cpf.cpf(sdfg, language='hip')
    assert_standalone_device(code, 'cpf_hip_affine')
    assert 'inclusive_affine' in code, 'the affine recurrence must render as the device scan over its affine maps'
    assert '__global__ void cpf_affine_pack_kernel' in code, 'the map-packing kernel is part of the unit'


@pytest.mark.parametrize('language', ('c++', 'c'))
def test_the_host_dialects_do_not_see_the_device_selection(language):
    """The device rule is gated on the dialect, so a host rendering of the same graph is unchanged
    -- which is what says this cannot regress the two dialects that were already complete."""
    sdfg = device_find_first_sdfg(f'cpf_hip_host_{language.replace("+", "p")}')
    sdfg.arrays['a'].storage = dace.dtypes.StorageType.CPU_Heap
    node, state = next((n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, FindFirst))
    with cpf_lowering.dialect_scope(cpf.LANGUAGES[language]):
        assert not cpf.on_device_at_host_level(node, state)
        assert cpf.renderable_implementations(node, state) == cpf.RENDERABLE_IMPLEMENTATIONS


def code_object(name: str, language: str, target_type: str, linkable: bool = True) -> CodeObject:
    """A CodeObject carrying only the labels :func:`~dace.codegen.cpf.frame_object` reads."""
    return CodeObject(name, '', language, None, 'title', target_type=target_type, linkable=linkable)


@pytest.mark.parametrize(
    'device_language,device_target_type',
    (
        ('cpp', 'hip'),  # MEASURED on gfx942: hipcc compiles .cpp, so the device unit is not 'hip'
        ('cu', 'cuda'),
        ('cu', ''),  # a target that labels only the language
        ('cpp', 'cuda'),
    ),
)
def test_the_device_object_is_found_however_the_backend_labels_it(device_language, device_target_type):
    """A GPU rendering arrives as frame + device object, and the two are merged rather than refused.

    Selecting the device object by LANGUAGE alone refused every HIP rendering: hipcc compiles
    ``.cpp``, so both objects came back ``language='cpp'``, both looked like the frame, and no
    single frame was found. Selecting by ``target_type`` alone fails the mirrored way on a target
    that leaves it at the default. Either label has to be enough.
    """
    frame = code_object('kern', 'cpp', '')
    device = code_object('kern_cuda', device_language, device_target_type)
    with cpf_lowering.dialect_scope(cpf.LANGUAGES['hip']):
        assert cpf.frame_object([frame, device], 'kern') is not None


def test_a_split_that_is_not_a_device_object_is_still_refused():
    """The merge is for the device pair only. Two host units are the split-translation-unit case
    the single-file contract cannot express, and it must keep saying so."""
    units = [code_object('kern', 'cpp', ''), code_object('kern_part2', 'cpp', '')]
    with cpf_lowering.dialect_scope(cpf.LANGUAGES['hip']):
        with pytest.raises(NotImplementedError, match='one translation unit'):
            cpf.frame_object(units, 'kern')


def test_non_linkable_objects_do_not_count_as_a_split():
    """The header and the sample ``main`` are generated for every SDFG and are not built, so a
    lone frame beside them is one translation unit, not three."""
    frame = code_object('kern', 'cpp', '')
    extras = [
        code_object('kern', 'h', '../../include', linkable=False),
        code_object('kern_main', 'cpp', '../../sample', linkable=False)
    ]
    assert cpf.frame_object([frame] + extras, 'kern') is frame
