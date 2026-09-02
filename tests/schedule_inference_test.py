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
