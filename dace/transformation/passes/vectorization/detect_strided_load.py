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
    pattern = re.compile(r"^assign_(\d+)$")

    def get_number(tasklet):
        m = pattern.match(tasklet.label)
        if m is None:
            raise ValueError(f"Tasklet label {tasklet.label} does not match pattern 'assign_<number>'")
        return int(m.group(1))

    return sorted(tasklets, key=get_number)


import sympy as sp


def detect_fixed_increment(expr_strings):
    """
    Detect whether a list of expressions has a fixed increment.

    Parameters
    ----------
    expr_strings : list[str]
        Expressions like ["2*tile_i", "2*tile_i + 2", ...]

    Returns
    -------
    (increment, smallest_expr) : (sympy.Expr or None, sympy.Expr or None)
        increment      : fixed increment if detected, otherwise None
        smallest_expr  : expression with smallest offset, otherwise None
    """
    if len(expr_strings) < 2:
        return None, None

    # Parse expressions safely
    try:
        exprs = [dace.symbolic.SymExpr(s.strip()) for s in expr_strings]
    except Exception:
        return None, None

    # Find the base symbol (e.g., tile_i)
    symbols = set().union(*(e.free_symbols for e in exprs))
    if len(symbols) != 1:
        return None, None
    base = symbols.pop()

    coeffs = []
    offsets = []

    for e in exprs:
        e = sp.expand(e)
        a = e.coeff(base)
        b = sp.expand(e - a * base)
        coeffs.append(a)
        offsets.append(b)

    # All coefficients must match
    if not all(c == coeffs[0] for c in coeffs):
        return None, None

    # Check fixed increment
    deltas = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    if not all(d == deltas[0] for d in deltas):
        return None, None

    # Smallest expression = smallest offset
    min_idx = offsets.index(min(offsets))
    smallest_expr = exprs[min_idx]

    return deltas[0], smallest_expr


@properties.make_properties
@transformation.explicit_cf_compatible
class DetectStridedLoad(ppl.Pass):
    # This pass is testes as part of the vectorization pipeline
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    gather_template = """
{{
strided_load_double(_in, _out, {vector_length}, {stride});
}}
"""

    def _apply(self, sdfg: SDFG) -> int:
        found_gathers = 0
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data.endswith("_packed"):
                    # If all inputs are tasklets of "_assign_X then we have a gather load"
                    ies = {ie for ie in state.in_edges(node)}
                    srcs = {ie.src for ie in ies}

                    # Only consider edges from Tasklets
                    all_tasklet_srcs = all({isinstance(s, nodes.Tasklet) for s in srcs})
                    tasklet_srcs = {s for s in srcs if isinstance(s, nodes.Tasklet)}
                    if not all_tasklet_srcs:
                        continue

                    # dtypes should be float64
                    if state.sdfg.arrays[node.data].dtype != dace.float64:
                        continue

                    # Extract numbers from labels of the form "assign_X"
                    numbers = []
                    pattern = re.compile(r"^assign_(\d+)$")
                    all_match = True
                    for t in tasklet_srcs:
                        m = pattern.match(t.label)
                        if m is None:
                            all_match = False
                            break
                        numbers.append(int(m.group(1)))

                    if not all_match:
                        continue

                    # Check numbers form a contiguous sequence 0..N-1
                    if set(numbers) == set(range(len(numbers))):
                        print(f"Gather load detected for node {node.data}, tasklets: {numbers}")
                    else:
                        # Numbers are not contiguous 0..N-1
                        continue

                    vector_length = len(numbers)

                    # All assign tasklets need to have 1 in edge
                    idx_data = set()
                    idx_data_and_subset = list()
                    tasklet_srcs_sorted = sort_tasklets_by_number(tasklet_srcs)
                    print(tasklet_srcs_sorted)
                    for src in tasklet_srcs_sorted:
                        src_in_edges = state.in_edges(src)
                        if len(src_in_edges) != 1:
                            continue
                        src_in_edge = src_in_edges[0]
                        data = src_in_edge.data.data
                        subset = src_in_edge.data.subset
                        idx_data_and_subset.append((data, subset))
                        idx_data.add(data)

                    if len(idx_data) != 1:
                        continue
                    # dtypes should be float64
                    if state.sdfg.arrays[next(iter(idx_data))].dtype != dace.float64:
                        continue

                    initializer_values = ", ".join([str(s) for d, s in idx_data_and_subset])
                    initializers = [str(s) for d, s in idx_data_and_subset]
                    print(initializers)
                    fixed_increment, base_expr = detect_fixed_increment(initializers)
                    if fixed_increment is None:
                        continue
                    else:
                        print(f"Found fixed increment {fixed_increment}")

                    gather_code = DetectStridedLoad.gather_template.format(initializer_values=initializer_values,
                                                                           vector_length=vector_length,
                                                                           stride=fixed_increment)

                    # Get the array we are gathering from
                    tasklet_srcs = set()
                    tasklet_ies = set()
                    for src in tasklet_srcs_sorted:
                        tasklet_srcs = tasklet_srcs.union({ie.src for ie in state.in_edges(src)})
                        tasklet_ies = tasklet_ies.union({ie for ie in state.in_edges(src)})
                    assert len(tasklet_srcs) == 1
                    indirect_src = tasklet_srcs.pop()
                    indirect_ie = tasklet_ies.pop()

                    # Remove scalar assignment tasklets
                    for src in tasklet_srcs_sorted:
                        state.remove_node(src)

                    # Handle connectors
                    t1 = state.add_tasklet("gather_load", {"_in"}, {"_out"}, gather_code, dace.dtypes.Language.CPP)
                    print(indirect_src, indirect_ie.src_conn)
                    # Does not need to be an access node
                    base = base_expr
                    end = base + vector_length * fixed_increment
                    # TODO: support for multi dimensional
                    state.add_edge(
                        indirect_src, indirect_ie.src_conn, t1, "_in",
                        dace.memlet.Memlet(data=indirect_ie.data.data, subset=dace.subsets.Range([(base, end - 1, 1)])))

                    assert indirect_ie not in tasklet_ies
                    for ie in tasklet_ies:
                        if ie in state.edges():
                            state.remove_edge(ie)
                        ie.src.remove_out_connector(ie.src_conn)
                    indirect_src.add_out_connector(indirect_ie.src_conn)
                    state.add_edge(t1, "_out", node, None,
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
            sdfg.append_global_code('#include <stdint.h>\n#include "dace/vector_intrinsics/strided_load.h"')
        sdfg.validate()
