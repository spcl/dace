import numpy as np
import dace
from dace.transformation.interstate import GPUTransformSDFG

from typing import Dict, Tuple


def my_add_mapped_tasklet(
        state,
        inpdict,
        outdict,
        name: str,
        map_ranges: Dict[str, dace.subsets.Subset],
        inputs: Dict[str, dace.memlet.Memlet],
        code: str,
        outputs: Dict[str, dace.memlet.Memlet],
        schedule=dace.dtypes.ScheduleType.Default,
        unroll_map=False,
        code_global="",
        code_init="",
        code_exit="",
        location="-1",
        language=dace.dtypes.Language.Python,
        debuginfo=None,
        external_edges=True,
) -> Tuple[dace.graph.nodes.Node]:
    """ Convenience function that adds a map entry, tasklet, map exit,
        and the respective edges to external arrays.
        :param name:       Tasklet (and wrapping map) name
        :param map_ranges: Mapping between variable names and their
                           subsets
        :param inputs:     Mapping between input local variable names and
                           their memlets
        :param code:       Code (written in `language`)
        :param outputs:    Mapping between output local variable names and
                           their memlets
        :param schedule:   Map schedule
        :param unroll_map: True if map should be unrolled in code
                           generation
        :param code_global: (optional) Global code (outside functions)
        :param language:   Programming language in which the code is
                           written
        :param debuginfo:  Debugging information (mostly for DIODE)
        :param external_edges: Create external access nodes and connect
                               them with memlets automatically

        :return: tuple of (tasklet, map_entry, map_exit)
    """
    import dace.graph.nodes as nd
    from dace.sdfg import getdebuginfo
    from dace.graph.labeling import propagate_memlet
    map_name = name + "_map"
    debuginfo = getdebuginfo(debuginfo)
    tasklet = nd.Tasklet(
        name,
        set(inputs.keys()),
        set(outputs.keys()),
        code,
        language=language,
        code_global=code_global,
        code_init=code_init,
        code_exit=code_exit,
        location=location,
        debuginfo=debuginfo,
    )
    map = state._map_from_ndrange(map_name,
                                  schedule,
                                  unroll_map,
                                  map_ranges,
                                  debuginfo=debuginfo)
    map_entry = nd.MapEntry(map)
    map_exit = nd.MapExit(map)
    state.add_nodes_from([map_entry, tasklet, map_exit])

    tomemlet = {}
    for name, memlet in inputs.items():

        memlet.name = name

        state.add_edge(map_entry, None, tasklet, name, memlet)
        tomemlet[memlet.data] = memlet

    if len(inputs) == 0:
        state.add_edge(map_entry, None, tasklet, None,
                       dace.memlet.EmptyMemlet())

    if external_edges:
        for inp, inpnode in inpdict.items():

            outer_memlet = propagate_memlet(state, tomemlet[inp], map_entry,
                                            True)
            state.add_edge(inpnode, None, map_entry, "IN_" + inp, outer_memlet)

            for e in state.out_edges(map_entry):
                if e.data.data == inp:
                    e._src_conn = "OUT_" + inp

            map_entry.add_in_connector("IN_" + inp)
            map_entry.add_out_connector("OUT_" + inp)

    tomemlet = {}
    for name, memlet in outputs.items():

        memlet.name = name

        state.add_edge(tasklet, name, map_exit, None, memlet)
        tomemlet[memlet.data] = memlet

    if len(outputs) == 0:
        state.add_edge(tasklet, None, map_exit, None, mm.EmptyMemlet())

    if external_edges:
        for out, outnode in outdict.items():

            outer_memlet = propagate_memlet(state, tomemlet[out], map_exit,
                                            True)
            state.add_edge(map_exit, "OUT_" + out, outnode, None, outer_memlet)

            for e in state.in_edges(map_exit):
                if e.data.data == out:
                    e._dst_conn = "IN_" + out

            map_exit.add_in_connector("IN_" + out)
            map_exit.add_out_connector("OUT_" + out)

    return tasklet, map_entry, map_exit


def create_zero_initialization(init_state, array_name):
    sdfg = init_state.parent
    array_shape = sdfg.arrays[array_name].shape

    array_access_node = init_state.add_write(array_name)

    indices = ["i" + str(k) for k, _ in enumerate(array_shape)]

    my_add_mapped_tasklet(
        init_state,
        inpdict={},
        outdict={array_name: array_access_node},
        name=(array_name + "_init_tasklet"),
        map_ranges={k: "0:" + str(v)
                    for k, v in zip(indices, array_shape)},
        inputs={},
        code='val = 0',
        outputs=dict(
            val=dace.Memlet.simple(array_access_node.data, ",".join(indices))),
        external_edges=True)


def create_test_sdfg():
    sdfg = dace.SDFG('test_sdfg')

    sdfg.add_array('BETA', shape=[10], dtype=dace.float32)
    sdfg.add_array('BETA_MAX', shape=[1], dtype=dace.float32)

    init_state = sdfg.add_state("init")
    state = sdfg.add_state("compute")

    sdfg.add_edge(init_state, state, dace.InterstateEdge())

    for arr in ['BETA_MAX']:
        create_zero_initialization(init_state, arr)

    BETA_MAX = state.add_access('BETA_MAX')
    BETA = state.add_access('BETA')

    beta_max_reduce = state.add_reduce(wcr="lambda a, b: max(a, b)",
                                       axes=(0, ),
                                       wcr_identity=-999999)
    state.add_edge(BETA, None, beta_max_reduce, None,
                   dace.memlet.Memlet.simple(BETA.data, '0:10'))
    state.add_edge(beta_max_reduce, None, BETA_MAX, None,
                   dace.memlet.Memlet.simple(BETA_MAX.data, '0:1'))

    return sdfg


if __name__ == '__main__':
    my_max_sdfg = create_test_sdfg()
    my_max_sdfg.validate()
    my_max_sdfg.apply_transformations(GPUTransformSDFG)

    BETA = np.random.rand(10).astype(np.float32)
    BETA_MAX = np.zeros(1).astype(np.float32)

    my_max_sdfg(BETA=BETA, BETA_MAX=BETA_MAX)

    assert (np.max(BETA) == BETA_MAX[0])
