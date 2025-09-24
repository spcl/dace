# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import typing
import dace
import re
import dace.sdfg.utils as sdutil
import copy
from dace.transformation.passes.split_tasklets import SplitTasklets
import numpy
import pytest
import ast

example_expressions = [
    "cfl_w_limit_out = (0.85 / dtime_0_in)",
    "z_w_con_c_out_0 = 0.0",
    "p_diag_out_w_concorr_c_0 = ((p_metrics_0_in_wgtfac_c_0 * z_w_concorr_mc_0_in_0) + ((1.0 - p_metrics_1_in_wgtfac_c_0) * z_w_concorr_mc_1_in_0))",
    "rot_vec_out_0 = ((((((vec_e_0_in_0 * ptr_int_0_in_geofac_rot_0) + (vec_e_1_in_0 * ptr_int_1_in_geofac_rot_0)) + (vec_e_2_in_0 * ptr_int_2_in_geofac_rot_0)) + (vec_e_3_in_0 * ptr_int_3_in_geofac_rot_0)) + (vec_e_4_in_0 * ptr_int_4_in_geofac_rot_0)) + (vec_e_5_in_0 * ptr_int_5_in_geofac_rot_0))",
    "p_diag_out_ddt_w_adv_pc_0 = (- (z_w_con_c_0_in_0 * (((p_prog_0_in_w_0 * p_metrics_0_in_coeff1_dwdz_0) - (p_prog_1_in_w_0 * p_metrics_1_in_coeff2_dwdz_0)) + (p_prog_2_in_w_0 * (p_metrics_2_in_coeff2_dwdz_0 - p_metrics_3_in_coeff1_dwdz_0)))))",
    "z_w_con_c_out_0 = ((0.85 * p_metrics_0_in_ddqz_z_half_0) / dtime_0_in)",
    "p_diag_out_max_vcfl_dyn = max_vcfl_dyn_var_152_0_in",
    "tmp_call_2_out = (p_diag_0_in_vt_0 ** 2)",
    "z_w_concorr_me_out_0 = ((p_prog_0_in_vn_0 * p_metrics_0_in_ddxn_z_full_0) + (p_diag_0_in_vt_0 * p_metrics_1_in_ddxt_z_full_0))",
    "_if_cond_23_out = global_data_0_in_lextra_diffu",
    "tmp_arg_18_out = (0.85 - (cfl_w_limit_0_in * dtime_0_in))",
    "levmask_out_0 = 0",
    "z_v_grad_w_out_0 = (((z_v_grad_w_0_in_0 * p_metrics_0_in_deepatmo_gradh_ifc_0) + (p_diag_0_in_vn_ie_0 * ((p_diag_1_in_vn_ie_0 * p_metrics_1_in_deepatmo_invr_ifc_0) - p_patch_0_in_edges_ft_e_0))) + (z_vt_ie_0_in_0 * ((z_vt_ie_1_in_0 * p_metrics_2_in_deepatmo_invr_ifc_0) + p_patch_1_in_edges_fn_e_0)))",
    "p_diag_out_ddt_w_adv_pc_0 = (p_diag_0_in_ddt_w_adv_pc_0 + ((difcoef_0_in * p_patch_0_in_cells_area_0) * ((((p_prog_0_in_w_0 * p_int_0_in_geofac_n2s_0) + (p_prog_1_in_w_0 * p_int_1_in_geofac_n2s_0)) + (p_prog_2_in_w_0 * p_int_2_in_geofac_n2s_0)) + (p_prog_3_in_w_0 * p_int_3_in_geofac_n2s_0))))",
    #"tmp_call_17_out = abs(w_con_e_0_in) * 2.0", # abs tasklets have issues in main branch currently, TODO: reenable after that bug is fixed
    #"tmp_call_15_out = abs(w_con_e_0_in)", # abs tasklets have issues in main branch currently, TODO: reenable after that bug is fixed
]

# Double-split tasklet test case
example_double_expressions = [("out1 = in1 * in2 * in3", "out2 = in4 * in5 * tmp")]


def _get_vars(ssa_line):
    lhs, rhs = ssa_line.split('=', 1)
    lhs_var = lhs.strip()
    rhs_vars = re.findall(r'\b[a-zA-Z_]\w*\b', rhs)
    return [lhs_var], rhs_vars


_single_tasklet_sdfg_counter = 0


def _generate_single_tasklet_sdfg(expression_str: str) -> dace.SDFG:
    global _single_tasklet_sdfg_counter
    _single_tasklet_sdfg_counter += 1

    sdfg = dace.SDFG(f"single_tasklet_sdfg_{_single_tasklet_sdfg_counter}")

    lhs_vars, rhs_vars = _get_vars(expression_str)

    assert len(lhs_vars) == 1, f"{lhs_vars} = {rhs_vars}"
    for var in lhs_vars + rhs_vars:
        sdfg.add_array(name=var + "_ARR", shape=(1, ), dtype=dace.float64)

    state = sdfg.add_state(label="main")
    state.add_mapped_tasklet(
        name="wrapper_map",
        map_ranges={"i": dace.subsets.Range([(0, 0, 1)])},
        inputs={rhs_var: dace.memlet.Memlet(expr=f"{rhs_var}_ARR[i]")
                for rhs_var in rhs_vars},
        code=expression_str,
        outputs={lhs_var: dace.memlet.Memlet(expr=f"{lhs_var}_ARR[i]")
                 for lhs_var in lhs_vars},
        external_edges=True,
        input_nodes={rhs_var: state.add_access(f"{rhs_var}_ARR")
                     for rhs_var in rhs_vars},
        output_nodes={lhs_var: state.add_access(f"{lhs_var}_ARR")
                      for lhs_var in lhs_vars},
    )

    for n in state.nodes():
        if state.degree(n) == 0:
            state.remove_node(n)

    sdfg.validate()
    return sdfg


_double_tasklet_sdfg_counter = 0


def _generate_double_tasklet_sdfg(expression_strs: typing.Tuple[str, str],
                                  direct_connection_between_tasklets: bool = False) -> dace.SDFG:
    global _double_tasklet_sdfg_counter
    _double_tasklet_sdfg_counter += 1

    sdfg = dace.SDFG(f"double_tasklet_sdfg_{_double_tasklet_sdfg_counter}")
    sdfg.add_scalar(name="tmp_Scalar", dtype=dace.float64, transient=True)
    state = sdfg.add_state(label="main")

    in_accesses = set()
    out_accesses = set()
    for i, expression_str in enumerate(expression_strs):
        lhs_vars, rhs_vars = _get_vars(expression_str)
        for var in lhs_vars:
            assert var != "tmp"
            sdfg.add_array(name=var + "_ARR", shape=(1, ), dtype=dace.float64)
        if i == len(expression_strs) - 1:
            for var in lhs_vars:
                out_accesses.add(state.add_access(var + "_ARR"))

        for var in rhs_vars:
            if var == "tmp":
                continue
            sdfg.add_array(name=var + "_ARR", shape=(1, ), dtype=dace.float64)
            in_accesses.add(state.add_access(var + "_ARR"))

    if not direct_connection_between_tasklets:
        tmp_access = state.add_access("tmp_Scalar")
    map_entry, map_exit = state.add_map(
        name="double_taskelt_map",
        ndrange={"i": dace.subsets.Range([(0, 0, 1)])},
    )

    for in_access in in_accesses:
        state.add_edge(in_access, None, map_entry, f"IN_{in_access.data}",
                       dace.memlet.Memlet.from_array(in_access.data, sdfg.arrays[in_access.data]))
        map_entry.add_in_connector(f"IN_{in_access.data}")
    for out_access in out_accesses:
        state.add_edge(map_exit, f"OUT_{out_access.data}", out_access, None,
                       dace.memlet.Memlet.from_array(out_access.data, sdfg.arrays[out_access.data]))
        map_exit.add_out_connector(f"OUT_{out_access.data}")

    added_tasklets = list()
    for i, expression_str in enumerate(expression_strs):
        lhs_vars, rhs_vars = _get_vars(expression_str)
        lhs_access = {f"{lhs_var}" for lhs_var in lhs_vars}
        rhs_access = {f"{rhs_var}" for rhs_var in rhs_vars}

        t = state.add_tasklet(name=f"t{i}", inputs=rhs_access, outputs=lhs_access, code=expression_str)
        added_tasklets.append(t)

    for i, expression_str in enumerate(expression_strs):
        lhs_vars, rhs_vars = _get_vars(expression_str)
        lhs_access = {f"{lhs_var}" for lhs_var in lhs_vars}
        rhs_access = {f"{rhs_var}" for rhs_var in rhs_vars}

        t = added_tasklets[i]

        if i == 0:
            for rhs_var in rhs_vars:
                state.add_edge(map_entry, f"OUT_{rhs_var}_ARR", t, rhs_var,
                               dace.memlet.Memlet.from_array(f"{rhs_var}_ARR", sdfg.arrays[f"{rhs_var}_ARR"]))
                map_entry.add_out_connector(f"OUT_{rhs_var}_ARR")
                t.add_in_connector(rhs_var)
            if not direct_connection_between_tasklets:
                for lhs_var in lhs_vars:
                    state.add_edge(t, lhs_var, tmp_access, None, dace.memlet.Memlet(expr=f"tmp_Scalar[0]"))
                    t.add_out_connector(lhs_var)
            else:
                for lhs_var in lhs_vars:
                    state.add_edge(t, lhs_var, added_tasklets[i + 1], "tmp", dace.memlet.Memlet(expr=f"tmp_Scalar[0]"))
                    t.add_out_connector(lhs_var)
        elif i == 1:
            for rhs_var in rhs_vars:
                if rhs_var == "tmp":
                    if not direct_connection_between_tasklets:
                        state.add_edge(tmp_access, None, t, rhs_var, dace.memlet.Memlet(expr=f"tmp_Scalar[0]"))
                        t.add_in_connector(rhs_var)
                    else:
                        # Handled already on the out connection
                        pass
                else:
                    state.add_edge(map_entry, f"OUT_{rhs_var}_ARR", t, rhs_var,
                                   dace.memlet.Memlet.from_array(f"{rhs_var}_ARR", sdfg.arrays[f"{rhs_var}_ARR"]))
                    map_entry.add_out_connector(f"OUT_{rhs_var}_ARR")
                    t.add_in_connector(rhs_var)
            for lhs_var in lhs_vars:
                state.add_edge(t, lhs_var, map_exit, f"IN_{lhs_var}_ARR",
                               dace.memlet.Memlet.from_array(f"{lhs_var}_ARR", sdfg.arrays[f"{lhs_var}_ARR"]))
                t.add_out_connector(lhs_var)
                map_exit.add_in_connector(f"IN_{lhs_var}_ARR")
        else:
            raise Exception("Tasklet length >2")

    sdfg.validate()
    return sdfg


def _one_assign_one_op(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    assigns = sum(isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)) for n in ast.walk(tree))
    ops = sum(
        isinstance(n, (ast.Call, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Lambda)) for n in ast.walk(tree))
    return (assigns, ops)


def _check_tasklet_properties(sdfg: dace.SDFG):
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.Tasklet):
            assert n.language == dace.dtypes.Language.Python
            assert _one_assign_one_op(n.code.as_string) == (1, 1) or _one_assign_one_op(n.code.as_string) == (
                1, 0), f"{n.code.as_string} has (assigns, ops): {_one_assign_one_op(n.code.as_string)}"


def _run_compile_and_comparison_test(sdfg: dace.SDFG):
    sdfg.compile()
    original_sdfg = copy.deepcopy(sdfg)
    original_sdfg.save("original.sdfgz", compress=True)
    SplitTasklets().apply_pass(sdfg=sdfg, pipeline_results={})
    sdfg.validate()
    sdfg.compile()

    array_names = {array_name for array_name, arr in original_sdfg.arrays.items() if arr.transient is False}
    arr_dict = {arr_name: numpy.random.rand(1) for arr_name in array_names}
    cp_arr_dict = copy.deepcopy(arr_dict)

    original_sdfg(**arr_dict)
    sdfg(**cp_arr_dict)

    # Assert that all tasklets have a single op inside
    _check_tasklet_properties(sdfg)

    for name in arr_dict:
        a = arr_dict[name]
        b = cp_arr_dict[name]
        assert numpy.allclose(a, b), f"Arrays for '{name}' differ:\n{a}\nvs\n{b}"


@pytest.mark.parametrize("expression_str", example_expressions)
def test_single_tasklet_split(expression_str: str):
    sdfg = _generate_single_tasklet_sdfg(expression_str)
    _run_compile_and_comparison_test(sdfg)


@pytest.mark.parametrize("expression_strs", example_double_expressions)
def test_double_tasklet_split(expression_strs: typing.Tuple[str, str]):
    sdfg = _generate_double_tasklet_sdfg(expression_strs, False)
    _run_compile_and_comparison_test(sdfg)


@pytest.mark.parametrize("expression_strs", example_double_expressions)
def test_double_tasklet_split_direct_tasklet_connection(expression_strs: typing.Tuple[str, str]):
    sdfg = _generate_double_tasklet_sdfg(expression_strs, True)
    _run_compile_and_comparison_test(sdfg)


S1 = dace.symbol("S1")
S2 = dace.symbol("S2")
S = dace.symbol("S")


@dace.program
def tasklet_in_nested_sdfg(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    offset1: dace.int64,
    offset2: dace.int64,
):
    for i, j in dace.map[S1:S2:1, S1:S2:1] @ dace.dtypes.ScheduleType.Sequential:
        a[i + offset1, j + offset1] = ((1.5 * b[i + offset1, j + offset2]) + (2.0 * a[i + offset1, j + offset2])) / 3.5


@dace.program
def cast_tasklet_first_in_a_map(a: dace.float64[S, S], ):
    for i, j in dace.map[S1:S2:1, S1:S2:1] @ dace.dtypes.ScheduleType.Sequential:
        a[i, j] = dace.float64(i) + ((dace.float64(j) + 5.2) * 2.7)


def test_expressions_with_nested_sdfg_and_explicit_typecast():
    _S1 = 1
    _S2 = 65
    _S = _S2 - _S1
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    # Create copies for comparison
    A_orig = A.copy()
    B_orig = B.copy()
    A_vec = A.copy()
    B_vec = B.copy()

    # Original SDFG
    sdfg = tasklet_in_nested_sdfg.to_sdfg()
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    SplitTasklets().apply_pass(copy_sdfg, {})
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(a=A_orig, b=B_orig, S=_S, S1=_S1, S2=_S2, offset1=-1, offset2=-1)
    c_copy_sdfg(a=A_vec, b=B_vec, S=_S, S1=_S1, S2=_S2, offset1=-1, offset2=-1)

    # Compare results
    assert numpy.allclose(A_orig, A_vec)
    assert numpy.allclose(B_orig, B_vec)


def test_expressions_with_typecast_first_in_map():
    _S1 = 0
    _S2 = 32
    _S = _S2 - _S1
    A = numpy.random.random((_S, _S))

    # Create copies for comparison
    A_orig = A.copy()
    A_vec = A.copy()

    # Original SDFG
    sdfg = cast_tasklet_first_in_a_map.to_sdfg()
    sdfg.validate()
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    SplitTasklets().apply_pass(copy_sdfg, {})
    c_copy_sdfg = copy_sdfg.compile()
    copy_sdfg.validate()

    c_sdfg(a=A_orig, S1=_S1, S2=_S2, S=_S)
    c_copy_sdfg(a=A_vec, S1=_S1, S2=_S2, S=_S)

    # Compare results
    assert numpy.allclose(A_orig, A_vec)


if __name__ == "__main__":
    test_expressions_with_nested_sdfg_and_explicit_typecast()
    test_expressions_with_typecast_first_in_map()
    for expression_str in example_expressions:
        test_single_tasklet_split(expression_str)
    for expression_strs in example_double_expressions:
        test_double_tasklet_split(expression_strs)
    for expression_strs in example_double_expressions:
        test_double_tasklet_split_direct_tasklet_connection(expression_strs)
