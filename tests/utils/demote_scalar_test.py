from typing import Tuple
import pytest
import dace
import numpy as np
import dace.sdfg.utils as sdutil

N = dace.symbol("N")

input_sets = [
    (0, "_if_cond_44", "((za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] < rcldtopcf) and (za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 1)] >= rcldtopcf))", 3),
    (1, "_if_cond_44", "((za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] < rcldtopcf) and (za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] >= rcldtopcf))", 2),
    (2, "_if_cond_44", "((za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] < rcldtopcf) and (za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 1)] >= rcldtopcf) and (za[((_for_it_47 + 1) - 2), ((_for_it_23 + 1) - 0)] >= rcldtopcf))", 4),
    (3, "_if_cond_44", "0", 0),
    (4, "_if_cond_44", "rcldtopcf", 1),
    (4, "_if_cond_44", "sym_rcldtopcf", 1),
]

# Create SDFG
def make_sdfg(assignment_key, assignment_val):
    sdfg = dace.SDFG('cond_edge_sdfg')

    # Symbols
    sdfg.add_symbol('_for_it_47', dace.int64)
    sdfg.add_symbol('_for_it_23', dace.int64)
    sdfg.add_scalar('rcldtopcf', dace.float64)
    sdfg.add_scalar('sym_rcldtopcf', dace.float64)

    # Arrays
    sdfg.add_array('za', [N, N], dace.float64)  # sizes arbitrary

    # States
    s0 = sdfg.add_state('start')
    s1 = sdfg.add_state('end')

    # Add defined _if_cond_44 as a symbol
    sdfg.add_symbol('_if_cond_44', dace.float64)

    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={assignment_key: assignment_val}))

    return sdfg


# Pytest-style runner
def run_two_sdfgs(sdfg1, sdfg2, inputs: dict):
    sdfg1(**inputs)
    sdfg2(**inputs)

# Pytest test function
@pytest.mark.parametrize("input", input_sets)
def test_conditional_edge(input: Tuple[int, str, str, int]):
    sdfgA = make_sdfg(input[1], input[2])
    sdfgB = make_sdfg(input[1], input[2])
    sdfgA.name = f"pattern_{input[0]}_original"
    sdfgA.save(f"pattern_{input[0]}_original.sdfgz", compress=True)
    sdutil.demote_symbol_to_scalar(sdfgB, "_if_cond_44", dace.float64)
    sdfgB.name = f"pattern_{input[0]}_demoted"
    sdfgB.save(f"pattern_{input[0]}_demoted.sdfgz", compress=True)

    tasklets = set()
    for state in sdfgB.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                tasklets.add((node, state))
    assert len(tasklets) == 1

    tasklet, state = tasklets.pop()
    assert len(state.in_edges(tasklet)) == input[3]

    _N = 10

    inputs = {
        'za': np.random.rand(_N, _N),
        '_for_it_47': np.int64(5),
        '_for_it_23': np.int64(7),
        'rcldtopcf': np.float64(0.3),
        'sym_rcldtopcf': np.float64(0.3),
        'N': _N,
    }

    run_two_sdfgs(sdfgA, sdfgB, inputs)

if __name__ == "__main__":
    test_conditional_edge()