# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import re
import dace
from typing import Any, Dict
from dace import SDFG, InterstateEdge, properties
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation


def sort_tasklets_by_number(tasklets):
    """
    Sort tasklets with labels of the form 'assign_<number>' by the numeric part.
    """
    pattern = re.compile(r"^a_(\d+)$")

    def get_number(tasklet):
        m = pattern.match(tasklet.label)
        if m is None:
            raise ValueError(f"Tasklet label {tasklet.label} does not match pattern 'assign_<number>'")
        return int(m.group(1))

    return sorted(tasklets, key=get_number)


@properties.make_properties
@transformation.explicit_cf_compatible
class DetectScatter(ppl.Pass):
    # This pass is testes as part of the vectorization pipeline
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    scatter_template = """
{{
int64_t idx[{vector_length}] = {{ {initializer_values} }};
scatter_double(_in, idx, _out, {vector_length});
}}
"""

    def _apply(self, sdfg: SDFG) -> int:
        found_gathers = 0
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data.endswith("_packed"):
                    # If all inputs are tasklets of "_assign_X then we have a scatter store"
                    oes = {oe for oe in state.out_edges(node)}
                    dsts = {oe.dst for oe in oes}

                    # Only consider edges from Tasklets
                    all_tasklet_dsts = all({isinstance(s, nodes.Tasklet) for s in dsts})
                    tasklet_dsts = {s for s in dsts if isinstance(s, nodes.Tasklet)}
                    if not all_tasklet_dsts:
                        continue

                    # dtypes should be float64
                    if state.sdfg.arrays[node.data].dtype != dace.float64:
                        continue

                    # Extract numbers from labels of the form "assign_X"
                    numbers = []
                    pattern = re.compile(r"^a_(\d+)$")
                    all_match = True
                    for t in tasklet_dsts:
                        m = pattern.match(t.label)
                        if m is None:
                            all_match = False
                            break
                        numbers.append(int(m.group(1)))

                    if not all_match:
                        continue

                    # Check numbers form a contiguous sequence 0..N-1
                    if set(numbers) == set(range(len(numbers))):
                        print(f"Scatter store detected for node {node.data}, tasklets: {numbers}")
                    else:
                        # Numbers are not contiguous 0..N-1
                        continue

                    vector_length = len(numbers)

                    # All assign tasklets need to have 1 in edge
                    idx_data = set()
                    idx_data_and_subset = list()
                    tasklet_dsts_sorted = sort_tasklets_by_number(tasklet_dsts)
                    print(tasklet_dsts_sorted)
                    for dst in tasklet_dsts_sorted:
                        dst_in_edges = state.out_edges(dst)
                        if len(dst_in_edges) != 1:
                            continue
                        dst_in_edge = dst_in_edges[0]
                        data = dst_in_edge.data.data
                        subset = dst_in_edge.data.subset
                        idx_data_and_subset.append((data, subset))
                        idx_data.add(data)

                    if len(idx_data) != 1:
                        continue
                    # dtypes should be float64
                    if state.sdfg.arrays[next(iter(idx_data))].dtype != dace.float64:
                        continue

                    initializer_values = ", ".join([str(s) for d, s in idx_data_and_subset])
                    scatter_code = DetectScatter.scatter_template.format(initializer_values=initializer_values,
                                                                      vector_length=vector_length)

                    # Get the array we are gathering from
                    tasklet_dsts = set()
                    for dst in tasklet_dsts_sorted:
                        tasklet_dsts = tasklet_dsts.union({oe.dst for oe in state.out_edges(dst)})
                    assert len(tasklet_dsts) == 1
                    indirect_dst = tasklet_dsts.pop()

                    # Remove scalar assignment tasklets
                    for dst in tasklet_dsts_sorted:
                        state.remove_node(dst)

                    t1 = state.add_tasklet("scatter_store", {"_in"}, {"_out"}, scatter_code, dace.dtypes.Language.CPP)
                    state.add_edge(
                        t1, "_out", indirect_dst, None,
                        dace.memlet.Memlet.from_array(indirect_dst.data, state.sdfg.arrays[indirect_dst.data]))
                    state.add_edge(node, None, t1, "_in", 
                                   dace.memlet.Memlet.from_array(node.data, state.sdfg.arrays[node.data]))

                    found_gathers += 1

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    found_gathers += self._apply(node.sdfg)

        return found_gathers

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        found_gathers = self._apply(sdfg)
        if found_gathers > 0:
            sdfg.append_global_code('#include <stdint.h>\n#include "dace/vector_intrinsics/scatter.h"')
        sdfg.validate()
