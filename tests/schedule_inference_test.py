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

    sdfg: dace.SDFG = simple_schedule_inference.to_sdfg(coarsen=False)

    infer_types.infer_connector_types(sdfg)

    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    sdfg.apply_transformations_repeated(StateFusion)

    entry = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry)
    ][0]
    assert entry.schedule is dace.ScheduleType.CPU_Multicore
