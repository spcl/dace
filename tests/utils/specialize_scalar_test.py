# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace.codegen.control_flow import LoopRegion
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock
import dace.sdfg.utils as sdutil

N = dace.symbolic.symbol("N")


def _get_sdfg_for_dynamic_map_input():
    sdfg = dace.SDFG("dynamic_input")
    sdfg.add_scalar("nlev", dtype=dace.int32, transient=False)
    sdfg.add_symbol("nlev_sym", stype=dace.int32)
    s0 = sdfg.add_state("s0")
    sdfg.add_array("A", ["nlev_sym"], dtype=dace.float64, transient=False)
    s1 = sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"nlev_sym": "nlev"}))
    an = s1.add_access("A")
    _, _, _ = s1.add_mapped_tasklet(name="assign_map",
                                    map_ranges={"i": dace.subsets.Range(ranges=[
                                        ("0", "nlev-1", "1"),
                                    ])},
                                    inputs={},
                                    outputs={"_out": dace.memlet.Memlet("A[i]")},
                                    code="_out = 0",
                                    input_nodes=None,
                                    external_edges=True,
                                    output_nodes={"A": an})
    return sdfg


def _get_sdfg_with_symbol_use_in_if():
    sdfg = dace.SDFG("basic1")
    _, A = sdfg.add_array(name="A", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    _, B = sdfg.add_array(name="B", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    cb = ConditionalBlock(label="cfb1", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb, is_start_block=True)
    cfg = dace.ControlFlowRegion(label="cfg1", sdfg=cb.sdfg, parent=cb)
    cb.add_branch(condition=CodeBlock("nlev < 100"), branch=cfg)
    s1 = cfg.add_state(label="s1")
    aA = s1.add_access("A")
    aB = s1.add_access("B")
    s1.add_edge(aA, None, aB, None, dace.memlet.Memlet.from_array("A", A))

    return sdfg


def _get_sdfg_with_symbol_use_in_for_cfg():
    sdfg = dace.SDFG("basic1")
    _, A = sdfg.add_array(name="A", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    _, B = sdfg.add_array(name="B", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    sdfg.add_scalar(name="nlev", dtype=dace.int64)
    loop = LoopRegion(
        label="cfb1",
        condition_expr="i < nlev",
        loop_var="i",
        initialize_expr="i = 0",
        update_expr="i = (i + 1)",
        sdfg=sdfg,
    )
    sdfg.add_node(loop, is_start_block=True)
    s1 = loop.add_state(label="s1")
    aA = s1.add_access("A")
    aB = s1.add_access("B")
    s1.add_edge(aA, None, aB, None, dace.memlet.Memlet.from_array("A", A))

    return sdfg


def test_specialize_with_dynamic_input():
    sdfg = _get_sdfg_for_dynamic_map_input()
    sdfg.validate()
    sdutil.specialize_scalar(sdfg=sdfg, scalar_name="nlev", scalar_val="90")
    sdfg.validate()
    sdfg.compile()
    map_entries = set()
    for s in sdfg.all_states():
        for n in s.nodes():
            if isinstance(n, dace.nodes.MapEntry):
                map_entries.add(n)
    assert len(map_entries) == 1
    map_entry: dace.nodes.MapEntry = next(iter(map_entries))
    range: dace.subsets.Range = map_entry.map.range
    assert range == dace.subsets.Range(ranges=[("0", "89", "1")])


def test_use_in_if_block():
    sdfg = _get_sdfg_with_symbol_use_in_if()
    sdfg.validate()
    sdutil.specialize_scalar(sdfg=sdfg, scalar_name="nlev", scalar_val="90")
    sdfg.validate()
    sdfg.compile()
    cbs = set()
    for n in sdfg.all_control_flow_regions():
        if isinstance(n, ConditionalBlock):
            cbs.add(n)
    assert len(cbs) == 1
    cb: ConditionalBlock = next(iter(cbs))
    assert len(cb.branches) == 1
    assert "nlev" not in cb.branches[0][0].as_string


def test_use_in_for_cfg():
    sdfg = _get_sdfg_with_symbol_use_in_for_cfg()
    sdfg.validate()
    sdfg.compile()
    sdutil.specialize_scalar(sdfg=sdfg, scalar_name="nlev", scalar_val="90")
    sdfg.validate()
    sdfg.compile()


def test_interstate_assignment():
    sdfg = dace.SDFG("dynamic_input")
    sdfg.add_scalar("nlev", dtype=dace.int32, transient=False)
    sdfg.add_symbol("nlev_sym", stype=dace.int32)
    s0 = sdfg.add_state("s0")
    sdfg.add_array("A", ["nlev_sym"], dtype=dace.float64, transient=False)
    s1 = sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"nlev_sym": "nlev"}))
    sdutil.specialize_scalar(sdfg=sdfg, scalar_name="nlev", scalar_val="90")
    for e in sdfg.all_interstate_edges():
        assert e.data.assignments == {"nlev_sym": "90"}


def test_interstate_read():
    sdfg = dace.SDFG("interstate_read")
    _, nlev = sdfg.add_scalar("nlev", dtype=dace.int32, transient=True)
    sdfg.add_symbol("nlev_sym", stype=dace.int32)
    s1 = sdfg.add_state("s1")
    nlev_an = s1.add_access("nlev")
    tasklet = s1.add_tasklet(name="assign", inputs={}, outputs={"_out"}, code="_out = nlev_sym")
    tasklet.add_out_connector("_out")
    s1.add_edge(tasklet, "_out", nlev_an, None, dace.memlet.Memlet.from_array("nlev", nlev))

    sdfg.validate()

    sdutil.specialize_scalar(sdfg=sdfg, scalar_name="nlev", scalar_val="90")

    sdfg.validate()

    # Should be no nodes in the state anymore
    assert len(s1.nodes()) == 0


@dace.program
def scale_by_scalar(A: dace.float64[N], C: dace.float64[N], scale: dace.float64):
    for i in dace.map[0:N]:
        C[i] = A[i] * scale


@dace.program
def divide_by_scalar(A: dace.float64[N], C: dace.float64[N], n: dace.int64):
    for i in dace.map[0:N]:
        C[i] = A[i] / n


def get_sdfg_with_uncast_division(language: dace.dtypes.Language):
    """Map tasklet dividing an array element by a scalar, with no type cast around the divisor."""
    code = {
        dace.dtypes.Language.Python: "__out = (__in1 / __in2)",
        dace.dtypes.Language.CPP: "__out = (__in1 / __in2);",
    }[language]
    sdfg = dace.SDFG(f"uncast_division_{language.name.lower()}")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    sdfg.add_scalar("n", dace.float64 if language == dace.dtypes.Language.CPP else dace.int64)
    state = sdfg.add_state("s0")
    state.add_mapped_tasklet(name="divide",
                             map_ranges={"i": "0:N"},
                             inputs={
                                 "__in1": dace.Memlet("A[i]"),
                                 "__in2": dace.Memlet("n[0]")
                             },
                             code=code,
                             language=language,
                             outputs={"__out": dace.Memlet("C[i]")},
                             external_edges=True)
    return sdfg


def test_float64_value_survives_bit_exactly():
    """A baked float64 must reach the generated code bit-exactly.

    The previous sympy round-trip (``SymExpr(rhs).subs(...)`` -> ``sympy.pycode``) printed
    ``sympy.Float`` at 15 significant digits, which drops the low bits of a float64.
    """
    scale = 3.141592653589793  # needs 16 significant digits to round-trip
    sdfg = scale_by_scalar.to_sdfg(simplify=True)
    sdutil.specialize_scalar(sdfg=sdfg, scalar_name="scale", scalar_val=scale)
    sdfg.validate()

    A = np.full(4, 1.0, dtype=np.float64)
    C = np.zeros(4, dtype=np.float64)
    sdfg(A=A, C=C, N=4, scale=scale)

    # Compare bit patterns, not printed strings.
    assert C.tobytes() == np.full(4, scale, dtype=np.float64).tobytes()


def test_baked_integer_does_not_become_integer_division():
    """Baking an integer into a division must stay true division.

    ``sympy.pycode`` folded ``__in1 / __in2`` with ``__in2 = 2`` into ``(1 / 2) * __in1``.
    That is true division in Python but *integer* division once the C++ backend emits it
    verbatim, so the whole term collapsed to zero.
    """
    sdfg = get_sdfg_with_uncast_division(dace.dtypes.Language.Python)
    sdutil.specialize_scalar(sdfg=sdfg, scalar_name="n", scalar_val=2)
    sdfg.validate()

    A = np.full(4, 3.0, dtype=np.float64)
    C = np.zeros(4, dtype=np.float64)
    sdfg(A=A, C=C, N=4, n=2)

    # 3.0 / 2 is exact in binary floating point, so this is an equality check, not a tolerance.
    assert C.tobytes() == np.full(4, 1.5, dtype=np.float64).tobytes()


def test_non_python_tasklet_accepts_numeric_value():
    """The non-Python branch used to hand a raw float to a string join and raise ``TypeError``."""
    sdfg = get_sdfg_with_uncast_division(dace.dtypes.Language.CPP)
    sdutil.specialize_scalar(sdfg=sdfg, scalar_name="n", scalar_val=0.5)
    sdfg.validate()

    A = np.full(4, 3.0, dtype=np.float64)
    C = np.zeros(4, dtype=np.float64)
    sdfg(A=A, C=C, N=4, n=0.5)

    assert C.tobytes() == np.full(4, 6.0, dtype=np.float64).tobytes()


def test_tasklet_with_type_cast():
    """Frontend tasklets wrap divisors in ``dace.float64(...)``; ``sympy.pycode`` could not print it."""
    sdfg = divide_by_scalar.to_sdfg(simplify=True)
    sdutil.specialize_scalar(sdfg=sdfg, scalar_name="n", scalar_val=2)
    sdfg.validate()

    A = np.full(4, 3.0, dtype=np.float64)
    C = np.zeros(4, dtype=np.float64)
    sdfg(A=A, C=C, N=4, n=2)

    assert C.tobytes() == np.full(4, 1.5, dtype=np.float64).tobytes()


if __name__ == "__main__":
    test_specialize_with_dynamic_input()
    test_use_in_if_block()
    test_interstate_assignment()
    test_interstate_read()
    test_use_in_for_cfg()
    test_float64_value_survives_bit_exactly()
    test_baked_integer_does_not_become_integer_division()
    test_non_python_tasklet_accepts_numeric_value()
    test_tasklet_with_type_cast()
