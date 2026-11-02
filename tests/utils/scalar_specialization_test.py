import copy
import json
import re

import numpy as np

import dace
import dace.libraries.comm.nodes
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.sdfg.utils import specialize_scalar, specialize_scalars, specialize_symbols

Y = dace.symbolic.symbol("Y")
X = dace.symbolic.symbol("X")


@dace.program
def gather_load(A: dace.float64[Y, X], B: dace.int64[Y, X], C: dace.float64[Y, X], scale: dace.float64):
    for i, j in dace.map[0:Y:1, 0:X:1]:
        C[i, j] = A[i, B[i, j]] * scale


# Test function
def test_nested_sdfg():
    # Create SDFG
    sdfg = gather_load.to_sdfg()
    sdfg.validate()
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.validate()
    # Specialize the scalar 'scale' in the copy
    specialize_scalar(copy_sdfg, "scale", 2.0)

    # Create inputs
    y_val, x_val = 32, 16
    A = np.random.rand(y_val, x_val)
    B = np.random.randint(0, x_val, size=(y_val, x_val), dtype=np.int64)
    C_orig = np.zeros((y_val, x_val))
    C_spec = np.zeros((y_val, x_val))

    # Run original SDFG
    sdfg(A=A, B=B, C=C_orig, scale=2.0, Y=y_val, X=x_val)

    # Run specialized SDFG (scale is fixed to 2.0)
    # scale arg is unused but need to pass
    copy_sdfg(A=A, B=B, C=C_spec, Y=y_val, X=x_val, scale=2.0)

    # Verify correctness for scale=2.0
    np.testing.assert_allclose(C_spec, A[np.arange(y_val)[:, None], B] * 2.0)
    np.testing.assert_allclose(C_spec, C_orig)


def _scalar_into_tasklet(code, language, value):
    """One float64 output driven by a tasklet reading float64 scalar 's', specialized to `value`."""
    sdfg = dace.SDFG("spec_" + str(abs(hash((code, language, value))))[:8])
    sdfg.add_scalar("s", dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("compute", {"_in"}, {"o"}, code, language=language)
    state.add_edge(state.add_read("s"), None, tasklet, "_in", dace.Memlet("s[0]"))
    state.add_edge(tasklet, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    specialize_scalar(sdfg, "s", value)
    result = np.zeros([1])
    sdfg(out=result, s=np.float64(value))
    return result[0], tasklet.code.as_string


def test_integer_value_keeps_floating_point_division():
    """An int replacing a float scalar must stay a float literal, or `/` becomes integer division."""
    for language, code in ((dace.dtypes.Language.Python, "o = _in / 2"), (dace.dtypes.Language.CPP, "o = _in / 2;")):
        value, emitted = _scalar_into_tasklet(code, language, 5)
        assert "5.0" in emitted, emitted
        assert value == 2.5, (language, value)


def test_cpp_tasklet_substitution_matches_unspaced_operators():
    """Token matching split only on whitespace and brackets, so `_in/2` was never substituted."""
    value, code = _scalar_into_tasklet("o = _in/2;", dace.dtypes.Language.CPP, 5.0)
    assert "_in" not in code, code
    assert value == 2.5


def test_float_value_round_trips_exactly():
    """repr, not a 15-significant-digit rendering, so a float64 survives substitution."""
    value, _ = _scalar_into_tasklet("o = _in", dace.dtypes.Language.Python, 1 / 3)
    assert value == 1 / 3


@dace.program
def affine(A: dace.float64[X], B: dace.float64[X], a: dace.float64, b: dace.float64, c: dace.float64):
    for i in dace.map[0:X:1]:
        B[i] = A[i] * a + b - c


def _canonical_json(sdfg):
    """Serialized SDFG with guids blanked, so two independently built graphs compare equal."""
    return re.sub(r'"guid"\s*:\s*"[^"]*"', '"guid": ""', json.dumps(sdfg.to_json(), sort_keys=True))


def test_batched_specialization_matches_one_at_a_time():
    """Baking the set in one walk must land the same graph as one call per scalar, and still run."""
    values = {"a": 2.0, "b": 3.0, "c": 4.0}

    looped = affine.to_sdfg()
    for name, value in values.items():
        specialize_scalar(looped, name, value)
    batched = affine.to_sdfg()
    specialize_scalars(batched, values)
    assert _canonical_json(batched) == _canonical_json(looped)

    x_val = 16
    A = np.random.rand(x_val)
    B = np.zeros(x_val)
    batched(A=A, B=B, X=x_val, **values)
    np.testing.assert_allclose(B, A * 2.0 + 3.0 - 4.0)


def test_interacting_values_keep_sequential_meaning():
    """A value naming another key cannot be applied simultaneously -- it must fall back to the loop."""
    values = {"a": "b", "b": "2.0", "c": "1.0"}

    looped = affine.to_sdfg()
    for name, value in values.items():
        specialize_scalar(looped, name, value)
    batched = affine.to_sdfg()
    specialize_scalars(batched, values)
    assert _canonical_json(batched) == _canonical_json(looped)


M = dace.symbolic.symbol("M")
K = dace.symbolic.symbol("K")


@dace.program
def sym_scale(A: dace.float64[M], B: dace.float64[M]):
    for i in dace.map[0:M:1]:
        B[i] = A[i] * K


def test_specialize_symbols_bakes_and_removes():
    """A baked free symbol is gone from symbols/free_symbols and runs without being passed."""
    sdfg = sym_scale.to_sdfg()
    specialize_symbols(sdfg, {"K": 3})
    assert "K" not in sdfg.symbols and "K" not in {str(s) for s in sdfg.free_symbols}

    m_val = 16
    A = np.random.rand(m_val)
    B = np.zeros(m_val)
    sdfg(A=A, B=B, M=m_val)  # K no longer an argument
    np.testing.assert_allclose(B, A * 3.0)


def test_specialize_symbols_batched():
    """Baking several symbols in one call bakes each and drops all of them."""
    sdfg = sym_scale.to_sdfg()
    specialize_symbols(sdfg, {"M": 8, "K": 5})
    assert not ({"M", "K"} & (set(sdfg.symbols) | {str(s) for s in sdfg.free_symbols}))

    A = np.random.rand(8)
    B = np.zeros(8)
    sdfg(A=A, B=B)  # neither M nor K passed
    np.testing.assert_allclose(B, A * 5.0)


def test_copy_out_of_the_scalar_keeps_the_value():
    """A copy from the baked scalar into another container still writes the value into that container."""
    sdfg = dace.SDFG("copy_out_of_specialized_scalar")
    sdfg.add_scalar("flag", dace.int32)
    sdfg.add_scalar("staged", dace.int32, transient=True)
    sdfg.add_array("out", [1], dace.int32)
    state = sdfg.add_state()
    staged = state.add_access("staged")
    state.add_nedge(state.add_read("flag"), staged, dace.Memlet("flag[0]"))
    read = state.add_tasklet("read", {"__in"}, {"__out"}, "__out = __in + 1")
    state.add_edge(staged, None, read, "__in", dace.Memlet("staged[0]"))
    state.add_edge(read, "__out", state.add_write("out"), None, dace.Memlet("out[0]"))

    specialize_scalars(sdfg, {"flag": 7})
    sdfg.validate()
    assert "staged" in {n.data for n in state.data_nodes()}

    out = np.zeros(1, dtype=np.int32)
    sdfg(out=out, flag=np.int32(0))
    assert out[0] == 8


def test_while_loop_keeps_its_missing_statements():
    sdfg = dace.SDFG("specialize_while")
    sdfg.add_scalar("limit", dace.int32)
    sdfg.add_symbol("i", dace.int32)
    loop = LoopRegion("while_loop", "i < limit", None, None, None)
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)

    specialize_scalars(sdfg, {"limit": 3})

    assert loop.loop_condition.as_string == "(i < 3)"
    assert loop.init_statement is None and loop.update_statement is None


def test_one_scalar_feeding_two_nested_connectors():
    inner = dace.SDFG("two_connectors_inner")
    for name in ("a", "b", "r"):
        inner.add_scalar(name, dace.int32)
    inner_state = inner.add_state(is_start_block=True)
    tasklet = inner_state.add_tasklet("add", {"x": None, "y": None}, {"o": None}, "o = x + 2 * y")
    inner_state.add_edge(inner_state.add_read("a"), None, tasklet, "x", dace.Memlet("a"))
    inner_state.add_edge(inner_state.add_read("b"), None, tasklet, "y", dace.Memlet("b"))
    inner_state.add_edge(tasklet, "o", inner_state.add_write("r"), None, dace.Memlet("r"))

    outer = dace.SDFG("two_connectors_outer")
    outer.add_scalar("v", dace.int32)
    outer.add_array("res", [1], dace.int32)
    state = outer.add_state(is_start_block=True)
    nested = state.add_nested_sdfg(inner, {"a": None, "b": None}, {"r": None})
    read = state.add_read("v")
    state.add_edge(read, None, nested, "a", dace.Memlet("v"))
    state.add_edge(read, None, nested, "b", dace.Memlet("v"))
    state.add_edge(nested, "r", state.add_write("res"), None, dace.Memlet("res[0]"))

    specialize_scalars(outer, {"v": 5})

    outer.validate()
    assert "a" not in inner.arrays and "b" not in inner.arrays
    res = np.zeros(1, dtype=np.int32)
    outer(res=res, v=5)
    assert res[0] == 15


def nested_reading_connector(name: str):
    """An SDFG with scalars ``n`` and ``m`` and a nest whose connector ``n`` reads ``m``, plus ``x = n + 1`` inside."""
    inner = dace.SDFG("connector_inner")
    inner.add_scalar("n", dace.int32)
    inner.add_scalar("o", dace.int32)
    inner.add_symbol("x", dace.int32)
    first = inner.add_state(is_start_block=True)
    second = inner.add_state_after(first, assignments={"x": "n + 1"})
    tasklet = second.add_tasklet("use", {}, {"out": None}, "out = x")
    second.add_edge(tasklet, "out", second.add_write("o"), None, dace.Memlet("o"))

    outer = dace.SDFG(name)
    outer.add_scalar("n", dace.int32)
    outer.add_scalar("m", dace.int32)
    outer.add_array("res", [1], dace.int32)
    state = outer.add_state(is_start_block=True)
    nested = state.add_nested_sdfg(inner, {"n": None}, {"o": None})
    state.add_edge(state.add_read("m"), None, nested, "n", dace.Memlet("m"))
    state.add_edge(nested, "o", state.add_write("res"), None, dace.Memlet("res[0]"))
    return outer, inner


def test_nested_interstate_edge_reads_its_own_binding():
    outer, inner = nested_reading_connector("interstate_binding")

    specialize_scalars(outer, {"n": 5})

    assigned = [edge.data.assignments for edge in inner.all_interstate_edges()]
    assert assigned == [{"x": "n + 1"}]


def nest_mapping_n(mapping: str, name: str):
    inner = dace.SDFG(f"{name}_inner")
    inner.add_symbol("n", dace.int32)
    inner.add_array("o", [1], dace.int32)
    inner_state = inner.add_state(is_start_block=True)
    tasklet = inner_state.add_tasklet("use", {}, {"out": None}, "out = n")
    inner_state.add_edge(tasklet, "out", inner_state.add_write("o"), None, dace.Memlet("o[0]"))

    outer = dace.SDFG(name)
    outer.add_symbol("n", dace.int32)
    outer.add_array("res", [1], dace.int32)
    state = outer.add_state(is_start_block=True)
    nested = state.add_nested_sdfg(inner, {}, {"o": None}, symbol_mapping={"n": mapping})
    state.add_edge(nested, "o", state.add_write("res"), None, dace.Memlet("res[0]"))
    return outer, nested


def test_nested_symbol_takes_its_mapped_value():
    outer, nested = nest_mapping_n("n + 1", "mapped_value")

    specialize_symbols(outer, {"n": 3})

    assert not nested.symbol_mapping
    res = np.zeros(1, dtype=np.int32)
    outer(res=res)
    assert res[0] == 4


def test_nested_loop_keeps_its_iterator():
    inner = dace.SDFG("loop_iterator_inner")
    inner.add_symbol("k", dace.int32)
    loop = LoopRegion("loop", "k < 2", "k", "k = 0", "k = k + 1")
    inner.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)
    outer = dace.SDFG("loop_iterator")
    outer.add_symbol("k", dace.int32)
    state = outer.add_state(is_start_block=True)
    state.add_nested_sdfg(inner, {}, {})

    specialize_symbols(outer, {"k": 3})

    assert loop.loop_variable == "k" and loop.init_statement.as_string == "k = 0"


def map_with_scalar_body(name: str):
    sdfg = dace.SDFG(name)
    sdfg.add_scalar("v", dace.float64)
    sdfg.add_scalar("t", dace.float64, transient=True)
    sdfg.add_array("out", [3], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    entry, exit_node = state.add_map("copy", {"i": "0:3"})
    held = state.add_access("t")
    tasklet = state.add_tasklet("pass_on", {"x": None}, {"o": None}, "o = x")
    state.add_memlet_path(state.add_read("v"), entry, held, memlet=dace.Memlet("v[0] -> [0]"))
    state.add_edge(held, None, tasklet, "x", dace.Memlet("t"))
    state.add_memlet_path(tasklet, exit_node, state.add_write("out"), src_conn="o", memlet=dace.Memlet("out[i]"))
    return sdfg


def test_copy_through_a_map_keeps_the_value():
    sdfg = map_with_scalar_body("copy_through_map")

    specialize_scalars(sdfg, {"v": 2.0})

    sdfg.validate()
    out = np.zeros(3)
    sdfg(out=out, v=9.0)
    assert out.tolist() == [2.0, 2.0, 2.0]


def copy_to_out(name: str, memlet: dace.Memlet):
    sdfg = dace.SDFG(name)
    sdfg.add_scalar("v", dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    state.add_edge(state.add_read("v"), None, state.add_write("out"), None, memlet)
    return sdfg, state


def test_copy_keeps_its_reduction():
    sdfg, state = copy_to_out("copy_wcr", dace.Memlet("v[0] -> [0]", wcr="lambda a, b: a + b"))

    specialize_scalars(sdfg, {"v": 2.0})

    assert [e.data.wcr is not None for e in state.edges()] == [True]


def test_copy_with_a_memlet_on_the_target():
    sdfg, state = copy_to_out("copy_target_memlet", dace.Memlet("out[0]"))

    specialize_scalars(sdfg, {"v": 2.0})

    assert not [n for n in state.data_nodes() if n.data == "v"]
    out = np.zeros(1)
    sdfg(out=out, v=9.0)
    assert out[0] == 2.0


def test_side_effect_tasklet_survives_losing_its_input():
    sdfg = dace.SDFG("side_effect")
    sdfg.add_scalar("v", dace.int32)
    state = sdfg.add_state(is_start_block=True)
    tasklet = state.add_tasklet(
        "report", {"x": None}, {}, 'printf("%d", x);', language=dace.Language.CPP, side_effects=True
    )
    state.add_edge(state.add_read("v"), None, tasklet, "x", dace.Memlet("v"))

    specialize_scalars(sdfg, {"v": 2})

    assert tasklet in state.nodes() and '"%d", 2' in tasklet.code.as_string


def test_library_node_keeps_its_scalar_connector():
    sdfg = dace.SDFG("library_input")
    sdfg.add_scalar("root", dace.int32)
    sdfg.add_array("buf", [4], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    node = dace.libraries.comm.nodes.Bcast("bcast")
    state.add_node(node)
    state.add_edge(state.add_read("buf"), None, node, "_inbuffer", dace.Memlet("buf"))
    state.add_edge(state.add_read("root"), None, node, "_root", dace.Memlet("root"))
    state.add_edge(node, "_outbuffer", state.add_write("buf"), None, dace.Memlet("buf"))

    specialize_scalars(sdfg, {"root": 0})

    feeding = [e for e in state.in_edges(node) if e.dst_conn == "_root"]
    assert len(feeding) == 1 and sdfg.arrays[feeding[0].data.data].transient


def test_tasklet_in_a_map_keeps_its_scope():
    sdfg = dace.SDFG("scope_after_fold")
    sdfg.add_scalar("v", dace.float64)
    for name in ("a", "b", "c"):
        sdfg.add_array(name, [2], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    entry, exit_node = state.add_map("both", {"i": "0:2"})
    folded = state.add_tasklet("folded", {"x": None}, {"o": None}, "o = x + 1")
    kept = state.add_tasklet("kept", {"x": None}, {"o": None}, "o = x")
    state.add_memlet_path(state.add_read("v"), entry, folded, dst_conn="x", memlet=dace.Memlet("v[0]"))
    state.add_memlet_path(state.add_read("a"), entry, kept, dst_conn="x", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(folded, exit_node, state.add_write("b"), src_conn="o", memlet=dace.Memlet("b[i]"))
    state.add_memlet_path(kept, exit_node, state.add_write("c"), src_conn="o", memlet=dace.Memlet("c[i]"))

    specialize_scalars(sdfg, {"v": 2.0})

    sdfg.validate()
    a, b, c = np.array([5.0, 6.0]), np.zeros(2), np.zeros(2)
    sdfg(a=a, b=b, c=c, v=9.0)
    assert b.tolist() == [3.0, 3.0] and c.tolist() == [5.0, 6.0]


def branch_on(condition: str, name: str, language=dace.Language.Python):
    sdfg = dace.SDFG(name)
    block = ConditionalBlock("branch")
    sdfg.add_node(block, is_start_block=True)
    region = ControlFlowRegion("taken", sdfg=sdfg)
    region.add_state("body", is_start_block=True)
    block.add_branch(CodeBlock(condition, language), region)
    return sdfg, block


def test_branch_value_keeps_operator_precedence():
    sdfg, block = branch_on("n ** 2 > 3", "precedence")
    sdfg.add_scalar("n", dace.int32)

    specialize_scalars(sdfg, {"n": -2})

    assert eval(block.branches[0][0].as_string)


def test_cpp_branch_keeps_its_language():
    sdfg, block = branch_on("n > 0 && n < 9", "cpp_branch", dace.Language.CPP)
    sdfg.add_scalar("n", dace.int32)

    specialize_scalars(sdfg, {"n": 4})

    assert block.branches[0][0].language == dace.Language.CPP
    assert block.branches[0][0].as_string == "4 > 0 && 4 < 9"


def test_negative_value_in_cpp_tasklet():
    sdfg = dace.SDFG("negative_cpp")
    sdfg.add_scalar("v", dace.float64)
    sdfg.add_array("y", [1], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    tasklet = state.add_tasklet("sub", {"x": None}, {"o": None}, "o = 1.0-x;", language=dace.Language.CPP)
    state.add_edge(state.add_read("v"), None, tasklet, "x", dace.Memlet("v"))
    state.add_edge(tasklet, "o", state.add_write("y"), None, dace.Memlet("y[0]"))

    specialize_scalars(sdfg, {"v": -1.5})

    y = np.zeros(1)
    sdfg(y=y, v=9.0)
    assert y[0] == 2.5


def test_integer_value_on_a_float_branch_stays_float():
    sdfg, block = branch_on("dt / 2 > 2.2", "float_branch")
    sdfg.add_scalar("dt", dace.float64)

    specialize_scalars(sdfg, {"dt": 5})

    assert "5.0" in block.branches[0][0].as_string


def test_float32_value_keeps_its_width():
    sdfg = dace.SDFG("float32_value")
    sdfg.add_scalar("s", dace.float32)
    sdfg.add_array("x", [1], dace.float32)
    sdfg.add_array("y", [1], dace.float32)
    state = sdfg.add_state(is_start_block=True)
    tasklet = state.add_tasklet("mul", {"a": None, "b": None}, {"o": None}, "o = a * b")
    state.add_edge(state.add_read("x"), None, tasklet, "a", dace.Memlet("x[0]"))
    state.add_edge(state.add_read("s"), None, tasklet, "b", dace.Memlet("s"))
    state.add_edge(tasklet, "o", state.add_write("y"), None, dace.Memlet("y[0]"))

    specialize_scalars(sdfg, {"s": 0.1})

    assert repr(float(np.float32(0.1))) in tasklet.code.as_string


def test_scalar_used_as_a_subscript_inside_a_nest():
    inner = dace.SDFG("subscript_inner")
    inner.add_scalar("n", dace.int32)
    inner.add_array("r", [4], dace.float64)
    inner_state = inner.add_state(is_start_block=True)
    write = inner_state.add_tasklet("put", {}, {"o": None}, "o = 1.0")
    inner_state.add_edge(write, "o", inner_state.add_write("r"), None, dace.Memlet("r[n]"))
    outer = dace.SDFG("subscript_outer")
    outer.add_scalar("v", dace.int32)
    outer.add_array("r", [4], dace.float64)
    state = outer.add_state(is_start_block=True)
    nested = state.add_nested_sdfg(inner, {"n": None}, {"r": None})
    state.add_edge(state.add_read("v"), None, nested, "n", dace.Memlet("v"))
    state.add_edge(nested, "r", state.add_write("r"), None, dace.Memlet("r"))

    specialize_scalars(outer, {"v": 2})

    outer.validate()
    r = np.zeros(4)
    outer(r=r, v=0)
    assert r.tolist() == [0.0, 0.0, 1.0, 0.0]


def test_interacting_values_reach_a_folded_tasklet():
    sdfg = dace.SDFG("interacting_tasklet")
    sdfg.add_scalar("a", dace.float64)
    sdfg.add_scalar("b", dace.float64)
    sdfg.add_array("y", [1], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    tasklet = state.add_tasklet("use", {"x": None}, {"o": None}, "o = x")
    state.add_edge(state.add_read("a"), None, tasklet, "x", dace.Memlet("a"))
    state.add_edge(tasklet, "o", state.add_write("y"), None, dace.Memlet("y[0]"))

    specialize_scalars(sdfg, {"a": "b", "b": "2.0"})

    assert tasklet.code.as_string.strip() == "o = 2.0"


if __name__ == "__main__":
    test_nested_sdfg()
    test_integer_value_keeps_floating_point_division()
    test_cpp_tasklet_substitution_matches_unspaced_operators()
    test_float_value_round_trips_exactly()
    test_batched_specialization_matches_one_at_a_time()
    test_interacting_values_keep_sequential_meaning()
    test_specialize_symbols_bakes_and_removes()
    test_specialize_symbols_batched()
    test_copy_out_of_the_scalar_keeps_the_value()
    test_while_loop_keeps_its_missing_statements()
    test_one_scalar_feeding_two_nested_connectors()
    test_nested_interstate_edge_reads_its_own_binding()
    test_nested_symbol_takes_its_mapped_value()
    test_nested_loop_keeps_its_iterator()
    test_copy_through_a_map_keeps_the_value()
    test_copy_keeps_its_reduction()
    test_copy_with_a_memlet_on_the_target()
    test_side_effect_tasklet_survives_losing_its_input()
    test_library_node_keeps_its_scalar_connector()
    test_tasklet_in_a_map_keeps_its_scope()
    test_branch_value_keeps_operator_precedence()
    test_cpp_branch_keeps_its_language()
    test_negative_value_in_cpp_tasklet()
    test_integer_value_on_a_float_branch_stays_float()
    test_float32_value_keeps_its_width()
    test_scalar_used_as_a_subscript_inside_a_nest()
    test_interacting_values_reach_a_folded_tasklet()
