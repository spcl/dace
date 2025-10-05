# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import numpy
from dace.properties import CodeBlock
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.offset_loop_and_maps import OffsetLoopsAndMaps
from dace.transformation.passes.symbol_propagation import SymbolPropagation
import gc

# klev, kidia, kfdia : Symbols
# z1, z2. rmin: scalar
# za, zli, zliqfrac, zlicefrac : Array[nlev, kfdia]
# zqx: Array[klev, kfidia, 5]

klev = dace.symbol("klev")
kidia = dace.symbol("kidia")
kfdia = dace.symbol("kfdia")
N = dace.symbol("N")


@dace.program
def _simple_element_wise(za: dace.float64[klev, klev]):
    for i in range(1, klev + 1):
        for j in range(1, klev + 1):
            za[i - 1, j - 1] = 2.0


@dace.program
def _cloudsc_snippet_one(za: dace.float64[klev, kfdia], zliqfrac: dace.float64[klev, kfdia],
                         zicefrac: dace.float64[klev, kfdia], zqx: dace.float64[klev, kfdia,
                                                                                5], zli: dace.float64[klev, kfdia],
                         zy: dace.float64[klev, kfdia, 5], zx: dace.float64[klev, kfdia, 4], rlmin: dace.float64,
                         z1: dace.int64, z2: dace.int64, cond_int: dace.int64):
    for i in range(1, klev + 1):
        for j in range(kidia + 1, kfdia + 1):
            za[i - 1, j - 1] = 2.0 * za[i - 1, j - 1] - 4.37
            cond1 = rlmin > (0.5 * (zqx[i - 1, j - 1, z1] + zqx[i, j, z2]))
            if cond1:
                zliqfrac[i - 1, j - 1] = zqx[i - 1, j - 1, z1] * zli[i - 1, j - 1]
                zicefrac[i - 1, j - 1] = 1.0 - zliqfrac[i - 1, j - 1]
            else:
                zliqfrac[i - 1, j - 1] = 0.0
                zicefrac[i - 1, j - 1] = 0.0
            for m in dace.map[1:5:1] @ dace.dtypes.ScheduleType.Sequential:
                zx[i - 1, j - 1, m - 1] = zy[i - 1, z1, z2]


@dace.program
def _cloudsc_snippet_one_within_if(za: dace.float64[klev, kfdia], zliqfrac: dace.float64[klev, kfdia],
                                   zicefrac: dace.float64[klev, kfdia], zqx: dace.float64[klev, kfdia, 5],
                                   zli: dace.float64[klev, kfdia], zy: dace.float64[klev, kfdia,
                                                                                    5], zx: dace.float64[klev, kfdia,
                                                                                                         4],
                                   rlmin: dace.float64, z1: dace.int64, z2: dace.int64, cond_int: dace.int64):
    if cond_int > 2:
        for i in range(1, klev + 1):
            for j in range(kidia + 1, kfdia + 1):
                za[i - 1, j - 1] = 2.0 * za[i - 1, j - 1] - 4.37
                cond1 = rlmin > (0.5 * (zqx[i - 1, j - 1, z1] + zqx[i, j, z2]))
                if cond1:
                    zliqfrac[i - 1, j - 1] = zqx[i - 1, j - 1, z1] * zli[i - 1, j - 1]
                    zicefrac[i - 1, j - 1] = 1.0 - zliqfrac[i - 1, j - 1]
                else:
                    zliqfrac[i - 1, j - 1] = 0.0
                    zicefrac[i - 1, j - 1] = 0.0
                for m in dace.map[1:5:1] @ dace.dtypes.ScheduleType.Sequential:
                    zx[i - 1, j - 1, m - 1] = zy[i - 1, z1, z2]


def _get_symbol_use_in_tasklet_sdfg():

    @dace.program
    def _symbol_use_in_tasklet(A: dace.float64[N]):
        for i in range(1, N):
            scl = i + 1
            A[scl] = A[i]

    sdfg = _symbol_use_in_tasklet.to_sdfg()
    states = list(sdfg.all_states())
    assert len(states) == 1
    state: dace.SDFGState = states[0]

    # Remove dummy nodes I used to get an SDFG with a Loop region fast
    for node in state.nodes():
        state.remove_node(node)

    t_offset = state.add_tasklet(name="create_scl", inputs={}, outputs={"_out"}, code="_out = i - 1")
    t_assign = state.add_tasklet(name="assign", inputs={"_in"}, outputs={"_out"}, code="_out = _in")
    state.sdfg.add_scalar(name="tmp", dtype=dace.int64, storage=dace.dtypes.StorageType.Register, transient=True)
    tmp_read_write = state.add_access("tmp")
    a_write = state.add_access("A")

    state.add_edge(t_offset, "_out", tmp_read_write, None, dace.memlet.Memlet("tmp[0]"))
    state.add_edge(tmp_read_write, None, t_assign, "_in", dace.memlet.Memlet("tmp[0]"))
    state.add_edge(t_assign, "_out", a_write, None, dace.memlet.Memlet("A[i]"))
    sdfg.validate()
    return sdfg


def _for_regions_and_beings(sdfg: dace.SDFG):
    d = dict()
    for cfg in sdfg.all_control_flow_regions():
        if isinstance(cfg, LoopRegion):
            d[cfg.loop_variable] = cfg.init_statement.as_string.split("=")[-1].strip() if isinstance(
                cfg.init_statement, CodeBlock) else str(cfg.init_statement)
    return d


def _make_args():
    # small sizes that exercise indexing safely
    klev_v = 5
    kidia_v = 4
    kfdia_v = 3

    numpy.random.seed(42)
    za = numpy.random.randn(klev_v, kfdia_v).astype(numpy.float64)
    zliqfrac = numpy.zeros((klev_v, kfdia_v)).astype(numpy.float64)
    zicefrac = numpy.zeros((klev_v, kfdia_v)).astype(numpy.float64)
    zqx = numpy.random.randn(klev_v, kfdia_v, 5).astype(numpy.float64)
    zli = numpy.random.randn(klev_v, kfdia_v).astype(numpy.float64)
    zy = numpy.random.randn(klev_v, kfdia_v, 5).astype(numpy.float64)
    zx = numpy.zeros((klev_v, kfdia_v, 4), dtype=numpy.float64)
    rlmin = numpy.float64(0.1 + numpy.abs(numpy.random.randn()))
    z1 = numpy.int64(0)  # valid index into third dimension (0..4)
    z2 = numpy.int64(1)
    cond_int = numpy.int64(3)

    return dict(za=za,
                zliqfrac=zliqfrac,
                zicefrac=zicefrac,
                zqx=zqx,
                zli=zli,
                zy=zy,
                zx=zx,
                rlmin=rlmin,
                z1=z1,
                z2=z2,
                klev=klev_v,
                kidia=kidia_v,
                kfdia=kfdia_v,
                cond_int=cond_int)


def _run_and_compare(original_sdfg, transformed_sdfg, arg_names_to_compare):
    """
    Run original_sdfg and transformed_sdfg on the same inputs and compare all mutable
    arrays in arg_names (list of names) using numpy.allclose.
    """
    # Produce fresh inputs
    args = _make_args()
    # Make deep copies for running both SDFGs
    orig_args = {k: (numpy.copy(v) if isinstance(v, numpy.ndarray) else v) for k, v in args.items()}
    trans_args = {k: (numpy.copy(v) if isinstance(v, numpy.ndarray) else v) for k, v in args.items()}

    # Run: SDFG call expects keyword args matching parameter names
    original_sdfg(**orig_args)
    transformed_sdfg(**trans_args)

    # Compare all arrays named in arg_names (these are outputs mutated in-place)
    af = False
    for name in arg_names_to_compare:
        a = orig_args[name]
        b = trans_args[name]
        if not numpy.allclose(a, b, atol=1e-8, rtol=1e-6):
            print(f"Mismatch for '{name}', the diff is: {b - a}")
            af = True
    if af is True:
        assert False

    for k, v in orig_args.items():
        del k
        del v
    for k, v in trans_args.items():
        del k
        del v
    gc.collect()


def test_loop_offsetting():
    sdfg = _cloudsc_snippet_one.to_sdfg()
    sdfg.validate()
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_offset"
    OffsetLoopsAndMaps(begin_expr=None, offset_expr="-1").apply_pass(copy_sdfg, {})
    # Begin expressions should be:
    # 0 and kidia + 1
    regions = _for_regions_and_beings(copy_sdfg)
    assert regions["i"] == "0", f"Expected 0 but got {regions['i']}"
    assert regions["j"] == "kidia", f"Expected kidia but got {regions['j']}"
    copy_sdfg.validate()
    sdfg.save("before.sdfg")
    copy_sdfg.save("after.sdfg")
    _run_and_compare(sdfg, copy_sdfg, ["za", "zliqfrac", "zicefrac", "zx"])


def test_loop_offsetting_w_begin_expr():
    sdfg = _cloudsc_snippet_one.to_sdfg()
    sdfg.validate()
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_offset"
    _pass = OffsetLoopsAndMaps(begin_expr="1", offset_expr="-1")
    assert _pass.begin_expr == "1"
    assert _pass.offset_expr == "-1"
    _pass.apply_pass(copy_sdfg, {})
    # Begin expressions should be:
    # 0 and kidia + 1
    regions = _for_regions_and_beings(copy_sdfg)
    assert regions["i"] == "0", f"Expected 0 but got {regions['i']}"
    assert regions["j"] == "(kidia + 1)", f"Expected kidia but got {regions['j']}"
    copy_sdfg.validate()
    copy_sdfg.simplify()
    _run_and_compare(sdfg, copy_sdfg, ["za", "zliqfrac", "zicefrac", "zx"])


def test_symbol_use_in_tasklet():
    sdfg = _get_symbol_use_in_tasklet_sdfg()
    sdfg.validate()
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_offset"
    OffsetLoopsAndMaps(begin_expr=None, offset_expr="-1").apply_pass(copy_sdfg, {})
    # 1 taskelt should be left
    num_tasklets = 0
    for state in copy_sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                num_tasklets += 1
    assert num_tasklets == 1
    copy_sdfg.validate()
    copy_sdfg.simplify()

    orig_args = {"N": 5, "A": numpy.zeros((5, ))}
    trans_args = {"N": 5, "A": numpy.zeros((5, ))}
    original_sdfg = sdfg
    transformed_sdfg = copy_sdfg

    original_sdfg(**orig_args)
    transformed_sdfg(**trans_args)

    a = orig_args["A"]
    b = trans_args["A"]
    assert numpy.allclose(a, b, atol=1e-8, rtol=1e-6), f"Mismatch for 'A', the diff is: {b - a}"


def test_simple_element_wise():
    sdfg = _simple_element_wise.to_sdfg()
    sdfg.validate()
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_offset"
    OffsetLoopsAndMaps(begin_expr="1", offset_expr="-1").apply_pass(copy_sdfg, {})
    # 1 taskelt should be left
    num_tasklets = 0
    for state in copy_sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                num_tasklets += 1
    assert num_tasklets == 1
    copy_sdfg.validate()
    copy_sdfg.simplify()

    orig_args = {"klev": 5, "za": numpy.zeros((5, 5))}
    trans_args = {"klev": 5, "za": numpy.zeros((5, 5))}
    original_sdfg = sdfg
    transformed_sdfg = copy_sdfg

    original_sdfg(**orig_args)
    transformed_sdfg(**trans_args)

    a = orig_args["za"]
    b = trans_args["za"]
    assert numpy.allclose(a, b, atol=1e-18, rtol=1e-18), f"Mismatch for 'za', the diff is: {b - a}"


def test_begin_expr_condition():
    sdfg = _cloudsc_snippet_one.to_sdfg()
    sdfg.validate()
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_offset"
    OffsetLoopsAndMaps(begin_expr="1", offset_expr="-1").apply_pass(copy_sdfg, {})
    # Begin expressions should be:
    # 0 and kidia + 1
    regions = _for_regions_and_beings(copy_sdfg)
    assert regions["i"] == "0", f"Expected 0 but got {regions['i']}"
    assert regions["j"] == "(kidia + 1)", f"Expected kidia + 1 but got {regions['j']}"
    copy_sdfg.validate()
    copy_sdfg.simplify()
    _run_and_compare(sdfg, copy_sdfg, ["za", "zliqfrac", "zicefrac", "zx"])


def test_with_conditional():
    sdfg = _cloudsc_snippet_one_within_if.to_sdfg()
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_offset"
    OffsetLoopsAndMaps(begin_expr="1", offset_expr="-1").apply_pass(copy_sdfg, {})
    regions = _for_regions_and_beings(copy_sdfg)
    assert regions["i"] == "0", f"Expected 0 but got {regions['i']}"
    assert regions["j"] == "(kidia + 1)", f"Expected kidia + 1 but got {regions['j']}"
    _run_and_compare(sdfg, copy_sdfg, ["za", "zliqfrac", "zicefrac", "zx"])


if __name__ == "__main__":
    test_simple_element_wise()
    test_symbol_use_in_tasklet()
    test_loop_offsetting()
    test_loop_offsetting_w_begin_expr()
    test_begin_expr_condition()
    test_with_conditional()
