import dace
from dace.transformation.interstate import StateFusion
from dace.sdfg import infer_types


def test_schedule_inference_simple():

    @dace.program
    def nested_call(A: dace.float64[3, 3]):
        return A + 1

    @dace.program
    def simple_schedule_inference(A: dace.float64[3, 3]):
        return nested_call(A)

    sdfg: dace.SDFG = simple_schedule_inference.to_sdfg(simplify=False)

    infer_types.infer_connector_types(sdfg)

    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    sdfg.apply_transformations_repeated(StateFusion)

    entry = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)][0]
    assert entry.schedule == dace.ScheduleType.CPU_Multicore


def accumulate_sdfg():
    """A map whose body reads and writes one scalar every iteration -- a loop-carried dependency."""
    import numpy as np

    @dace.program
    def accumulate(A: dace.float64[8], out: dace.float64[1]):
        s = np.float64(0.0)
        for i in dace.map[0:8]:
            s = s + A[i]
        out[0] = s

    return accumulate.to_sdfg(simplify=True)


def elementwise_sdfg():
    """The control: every iteration writes its own element, so the map is data-parallel."""

    @dace.program
    def elementwise(A: dace.float64[8], B: dace.float64[8]):
        for i in dace.map[0:8]:
            B[i] = A[i] + 1.0

    return elementwise.to_sdfg(simplify=True)


def map_entry_with_param(sdfg, param):
    entries = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and n.map.params == [param]
    ]
    assert len(entries) == 1, [
        n.map.params for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)
    ]
    return entries[0]


def test_loop_carried_map_is_not_scheduled_multicore():
    """An OpenMP team over a loop-carried dependency is a race; inference must not choose it."""
    sdfg = accumulate_sdfg()
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    assert map_entry_with_param(sdfg, 'i').schedule == dace.ScheduleType.Sequential


def test_elementwise_map_still_scheduled_multicore():
    """The gate is narrow: a map that writes its own element per iteration stays parallel."""
    sdfg = elementwise_sdfg()
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    assert map_entry_with_param(sdfg, 'i').schedule == dace.ScheduleType.CPU_Multicore


def generated_cpu_code(sdfg):
    return '\n'.join(obj.code for obj in sdfg.generate_code() if obj.language == 'cpp')


def test_loop_carried_map_emits_no_openmp_directive():
    """Neither ``parallel for`` nor ``simd``: both assert the iterations are independent."""
    code = generated_cpu_code(accumulate_sdfg())
    assert '#pragma omp parallel for' not in code, code
    assert '#pragma omp simd' not in code, code


def test_elementwise_map_still_emits_the_parallel_for():
    code = generated_cpu_code(elementwise_sdfg())
    assert '#pragma omp parallel for' in code, code


def host_answered_gpu_reduction():
    """A CUDA ``ArgReduce``: device operand, both answers written by host code after the launch."""
    from dace.libraries.standard.nodes.arg_reduce import ArgReduce
    sdfg = dace.SDFG('host_answered_gpu_reduction')
    sdfg.add_array('a', [64], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('val', [1], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    node = ArgReduce('argreduce', op='max')
    node.implementation = 'CUDA'
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_in', dace.Memlet('a[0:64]'))
    state.add_edge(node, '_out_val', state.add_write('val'), None, dace.Memlet('val[0]'))
    state.add_edge(node, '_out_idx', state.add_write('idx'), None, dace.Memlet('idx[0]'))
    return sdfg, node


def test_host_connectors_do_not_constrain_the_schedule():
    """``host_connectors`` names the operands that stay on the host, so they cast no schedule vote.

    Counting them makes the node read as both GPU_Device and CPU_Multicore, and inference raises on
    two constraints rather than picking one -- which used to make a CUDA ArgReduce impossible to
    compile at all, even though the storage split is exactly the contract the node declares.
    """
    sdfg, node = host_answered_gpu_reduction()
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    assert node.schedule == dace.ScheduleType.GPU_Device, (
        f'the device operand is the only one that votes, so the node runs on the device, not {node.schedule}')


def test_a_host_connector_still_votes_on_a_node_that_does_not_declare_one():
    """The exemption is per node: a library node with no ``host_connectors`` is unaffected."""
    from dace.libraries.standard.nodes.arg_reduce import ArgReduce
    sdfg, node = host_answered_gpu_reduction()
    assert ArgReduce.host_connectors == frozenset({'_out_val', '_out_idx'})
    node.host_connectors = frozenset()
    infer_types.infer_connector_types(sdfg)
    try:
        infer_types.set_default_schedule_and_storage_types(sdfg, None)
    except dace.sdfg.validation.InvalidSDFGNodeError as err:
        assert 'Multiple arrays' in str(err), err
        return
    raise AssertionError('a node that declares no host connector must still see both storages conflict')
