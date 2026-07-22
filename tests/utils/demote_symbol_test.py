from typing import Tuple
import pytest
import sympy
import dace
import numpy as np
import dace.sdfg.utils as sdutil

N = dace.symbol("N")

input_sets = [
    (0, "_if_cond_44",
     "((za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] < rcldtopcf) and (za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 1)] >= rcldtopcf))",
     3),
    (1, "_if_cond_44",
     "((za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] < rcldtopcf) and (za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] >= rcldtopcf))",
     2),
    (2, "_if_cond_44",
     "((za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] < rcldtopcf) and (za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 1)] >= rcldtopcf) and (za[((_for_it_47 + 1) - 2), ((_for_it_23 + 1) - 0)] >= rcldtopcf))",
     4),
    (3, "_if_cond_44", "0", 0),
    (4, "_if_cond_44", "rcldtopcf", 1),
    (5, "_if_cond_44", "sym_rcldtopcf", 1),
]


def make_type2_sdfg():
    assignment_key = "_if_cond_1"
    assignment_val = "(((zqx[((_for_it_12 + 1) - 1), ((_for_it_11 + 1) - 1), (1 - 1)] + zqx[((_for_it_12 + 1) - 1), ((_for_it_11 + 1) - 1), (2 - 1)]) < rlmin) or (za[((_for_it_12 + 1) - 1), ((_for_it_11 + 1) - 1)] < ramin))"
    sdfg = dace.SDFG(f"single_complex_expression_sdfg_type_2")
    state1 = sdfg.add_state("complex_tasklet_state")

    sdfg.add_symbol("_for_it_11", dace.int64)
    sdfg.add_symbol("_for_it_12", dace.int64)

    for inm in ["ramin", "rlmin", "os"]:
        sdfg.add_scalar(inm, dace.float64, dace.dtypes.StorageType.Default, transient=False)

    sdfg.add_array("za", (
        5,
        5,
    ), dace.float64, dace.dtypes.StorageType.Default, transient=False)
    sdfg.add_array("zqx", (5, 5, 2), dace.float64, dace.dtypes.StorageType.Default, transient=False)

    state2 = sdfg.add_state("complex_tasklet_state2")

    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments={assignment_key: assignment_val}))

    sdfg.validate()
    return sdfg


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
def test_single_edge(input: Tuple[int, str, str, int]):
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


# Pytest test function
def test_complex_expr_and_connector_names():
    sdfgA = make_type2_sdfg()
    sdfgB = make_type2_sdfg()
    sdfgA.name = f"pattern_type2_original"
    sdfgA.save(f"pattern_type2_original.sdfgz", compress=True)
    sdutil.demote_symbol_to_scalar(sdfgB, "_if_cond_1", dace.float64)
    sdfgB.name = f"pattern_type2_demoted"

    tasklets = set()
    for state in sdfgB.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                tasklets.add((node, state))
    assert len(tasklets) == 1

    sdfgB.save(f"pattern_type2_demoted.sdfgz", compress=True)

    tasklet, state = tasklets.pop()

    sym_expr = dace.symbolic.SymExpr(tasklet.code.as_string.split(" = ")[1].strip())
    func_and_sym_names = {str(s)
                          for s in sym_expr.free_symbols}.union({str(f.func)
                                                                 for f in sym_expr.atoms(sympy.Function)})
    for arr_name in ["rlmin", "rlmax", "za", "zqx"]:
        assert arr_name not in func_and_sym_names


if __name__ == "__main__":
    for iset in input_sets:
        test_single_edge(iset)
    test_complex_expr_and_connector_names()
