# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import dtypes, nodes
from typing import Any, Dict, List, Union

N = dace.symbol("N")


@dace.program
def arrayop(inp: dace.float32[N], out: dace.float32[N]):
    for i in dace.map[0:N]:
        out[i] = 2 * inp[i]


def key_exists(d: Union[List[Any], Dict[str, Any]], key: str):
    if isinstance(d, list):
        for item in d:
            if key_exists(item, key):
                return True
    elif isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                return True
            if key_exists(v, key):
                return True

    return False


def test_lack_of_omp_props():

    sdfg = arrayop.to_sdfg(simplify=True)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.EntryNode):
            assert (isinstance(node, nodes.MapEntry))
            node.map.schedule = dtypes.ScheduleType.Sequential
            break

    json = sdfg.to_json()
    assert (not key_exists(json, 'omp_num_threads'))
    assert (not key_exists(json, 'omp_schedule'))
    assert (not key_exists(json, 'omp_chunk_size'))


def test_omp_props():

    sdfg = arrayop.to_sdfg(simplify=True)
    mapnode = None
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.EntryNode):
            assert (isinstance(node, nodes.MapEntry))
            mapnode = node.map
            break

    mapnode.schedule = dtypes.ScheduleType.CPU_Multicore
    json = sdfg.to_json()
    assert (key_exists(json, 'omp_num_threads'))
    assert (key_exists(json, 'omp_schedule'))
    assert (key_exists(json, 'omp_chunk_size'))
    code = sdfg.generate_code()[0].clean_code
    assert ("#pragma omp parallel for" in code)

    mapnode.omp_num_threads = 10
    code = sdfg.generate_code()[0].clean_code
    assert ("#pragma omp parallel for num_threads(10)" in code)

    mapnode.omp_schedule = dtypes.OMPScheduleType.Guided
    code = sdfg.generate_code()[0].clean_code
    assert ("#pragma omp parallel for schedule(guided) num_threads(10)" in code)

    mapnode.omp_chunk_size = 5
    code = sdfg.generate_code()[0].clean_code
    assert ("#pragma omp parallel for schedule(guided, 5) num_threads(10)" in code)


if __name__ == "__main__":
    test_lack_of_omp_props()
    test_omp_props()
