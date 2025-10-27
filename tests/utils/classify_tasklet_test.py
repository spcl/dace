import pytest
import dace
import typing
import dace.sdfg.tasklet_utils as tutil

tasklet_infos = [
    # === ARRAY + SYMBOL ===
    ("out = in_a + sym_b", "array", {"a"}, {}, {"sym_b"},
     {"type": tutil.TaskletType.ARRAY_SYMBOL, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "+", "constant1": "sym_b", "constant2": None}),
    ("out = in_a - sym_b", "array", {"a"}, {}, {"sym_b"},
     {"type": tutil.TaskletType.ARRAY_SYMBOL, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "-", "constant1": "sym_b", "constant2": None}),
    ("out = in_a * sym_b", "array", {"a"}, {}, {"sym_b"},
     {"type": tutil.TaskletType.ARRAY_SYMBOL, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "*", "constant1": "sym_b", "constant2": None}),
    ("out = in_a / sym_b", "array", {"a"}, {}, {"sym_b"},
     {"type": tutil.TaskletType.ARRAY_SYMBOL, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "/", "constant1": "sym_b", "constant2": None}),

    # === ARRAY + CONSTANT ===
    ("out = in_a + 2", "array", {"a"}, {}, {},
     {"type": tutil.TaskletType.ARRAY_SYMBOL, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "+", "constant1": "2", "constant2": None}),
    ("out = in_a * 3", "array", {"a"}, {}, {},
     {"type": tutil.TaskletType.ARRAY_SYMBOL, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "*", "constant1": "3", "constant2": None}),
    ("out = in_a / 2.5", "array", {"a"}, {}, {},
     {"type": tutil.TaskletType.ARRAY_SYMBOL, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "/", "constant1": "2.5", "constant2": None}),
    ("out = in_a - 5", "array", {"a"}, {}, {},
     {"type": tutil.TaskletType.ARRAY_SYMBOL, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "-", "constant1": "5", "constant2": None}),

    # === ARRAY + ARRAY ===
    ("out = in_a + in_b", "array", {"a", "b"}, {}, {},
     {"type": tutil.TaskletType.ARRAY_ARRAY, "lhs": "out", "rhs1": "in_a",
      "rhs2": "in_b", "op": "+", "constant1": None, "constant2": None}),
    ("out = in_a - in_b", "array", {"a", "b"}, {}, {},
     {"type": tutil.TaskletType.ARRAY_ARRAY, "lhs": "out", "rhs1": "in_a",
      "rhs2": "in_b", "op": "-", "constant1": None, "constant2": None}),
    ("out = in_a * in_b", "array", {"a", "b"}, {}, {},
     {"type": tutil.TaskletType.ARRAY_ARRAY, "lhs": "out", "rhs1": "in_a",
      "rhs2": "in_b", "op": "*", "constant1": None, "constant2": None}),
    ("out = in_a / in_b", "array", {"a", "b"}, {}, {},
     {"type": tutil.TaskletType.ARRAY_ARRAY, "lhs": "out", "rhs1": "in_a",
      "rhs2": "in_b", "op": "/", "constant1": None, "constant2": None}),

    # === SCALAR + SYMBOL ===
    ("out = in_x + sym_y", "scalar", {}, {"x"}, {"sym_y"},
     {"type": tutil.TaskletType.SCALAR_SYMBOL, "lhs": "out", "rhs1": "in_x",
      "rhs2": None, "op": "+", "constant1": "sym_y", "constant2": None}),
    ("out = in_x * sym_y", "scalar", {}, {"x"}, {"sym_y"},
     {"type": tutil.TaskletType.SCALAR_SYMBOL, "lhs": "out", "rhs1": "in_x",
      "rhs2": None, "op": "*", "constant1": "sym_y", "constant2": None}),
    ("out = in_x - sym_y", "scalar", {}, {"x"}, {"sym_y"},
     {"type": tutil.TaskletType.SCALAR_SYMBOL, "lhs": "out", "rhs1": "in_x",
      "rhs2": None, "op": "-", "constant1": "sym_y", "constant2": None}),

    # === SCALAR + SCALAR ===
    ("out = in_x + in_y", "scalar", {}, {"x", "y"}, {},
     {"type": tutil.TaskletType.SCALAR_SCALAR, "lhs": "out", "rhs1": "in_x",
      "rhs2": "in_y", "op": "+", "constant1": None, "constant2": None}),
    ("out = in_x * in_y", "scalar", {}, {"x", "y"}, {},
     {"type": tutil.TaskletType.SCALAR_SCALAR, "lhs": "out", "rhs1": "in_x",
      "rhs2": "in_y", "op": "*", "constant1": None, "constant2": None}),
    ("out = in_x / in_y", "scalar", {}, {"x", "y"}, {},
     {"type": tutil.TaskletType.SCALAR_SCALAR, "lhs": "out", "rhs1": "in_x",
      "rhs2": "in_y", "op": "/", "constant1": None, "constant2": None}),

    # === SYMBOL + SYMBOL ===
    ("out = sym_a + sym_b", "scalar", {}, {}, {"sym_a", "sym_b"},
     {"type": tutil.TaskletType.SYMBOL_SYMBOL, "lhs": "out", "rhs1": None,
      "rhs2": None, "op": "+", "constant1": "sym_a", "constant2": "sym_b"}),
    ("out = sym_a * sym_b", "scalar", {}, {}, {"sym_a", "sym_b"},
     {"type": tutil.TaskletType.SYMBOL_SYMBOL, "lhs": "out", "rhs1": None,
      "rhs2": None, "op": "*", "constant1": "sym_a", "constant2": "sym_b"}),
    ("out = sym_a / sym_b", "scalar", {}, {}, {"sym_a", "sym_b"},
     {"type": tutil.TaskletType.SYMBOL_SYMBOL, "lhs": "out", "rhs1": None,
      "rhs2": None, "op": "/", "constant1": "sym_a", "constant2": "sym_b"}),

    # === FUNCTIONAL / SUPPORTED OPS ===
    ("out = abs(in_a)", "array", {"a"}, {}, {},
     {"type": tutil.TaskletType.UNARY_ARRAY, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "abs", "constant1": None, "constant2": None}),
    ("out = exp(in_a)", "array", {"a"}, {}, {},
     {"type": tutil.TaskletType.UNARY_ARRAY, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "exp", "constant1": None, "constant2": None}),
    ("out = sqrt(in_a)", "array", {"a"}, {}, {},
     {"type": tutil.TaskletType.UNARY_ARRAY, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "sqrt", "constant1": None, "constant2": None}),
    ("out = log(in_a)", "array", {"a"}, {}, {},
     {"type": tutil.TaskletType.UNARY_ARRAY, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "log", "constant1": None, "constant2": None}),
    ("out = pow(in_a, 2)", "array", {"a"}, {}, {},
     {"type": tutil.TaskletType.ARRAY_SYMBOL, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "pow", "constant1": "2", "constant2": None}),
    ("out = min(in_a, in_b)", "array", {"a", "b"}, {}, {},
     {"type": tutil.TaskletType.ARRAY_ARRAY, "lhs": "out", "rhs1": "in_a",
      "rhs2": "in_b", "op": "min", "constant1": None, "constant2": None}),
    ("out = max(in_a, in_b)", "array", {"a", "b"}, {}, {},
     {"type": tutil.TaskletType.ARRAY_ARRAY, "lhs": "out", "rhs1": "in_a",
      "rhs2": "in_b", "op": "max", "constant1": None, "constant2": None}),
    ("out = abs(sym_a)", "array", {}, {}, {"sym_a"},
     {"type": tutil.TaskletType.UNARY_SYMBOL, "lhs": "out", "rhs1": None,
      "rhs2": None, "op": "abs", "constant1": "sym_a", "constant2": None}),
    ("out = exp(in_a)", "array", {"a"}, {}, {},
     {"type": tutil.TaskletType.UNARY_ARRAY, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "exp", "constant1": None, "constant2": None}),
    ("out = sqrt(in_a)", "scalar", {}, {"a"}, {},
     {"type": tutil.TaskletType.UNARY_SCALAR, "lhs": "out", "rhs1": "in_a",
      "rhs2": None, "op": "sqrt", "constant1": None, "constant2": None}),
    
    # === ASSIGNMENTS ===
    ("out = in_a", "array", {"a"}, {}, {},
    {"type": tutil.TaskletType.ARRAY_ARRAY_ASSIGNMENT, "lhs": "out",
    "rhs1": "in_a", "rhs2": None, "op": "=", "constant1": None, "constant2": None}),
    ("out = in_b", "array", {"b"}, {}, {},
    {"type": tutil.TaskletType.ARRAY_ARRAY_ASSIGNMENT, "lhs": "out",
    "rhs1": "in_b", "rhs2": None, "op": "=", "constant1": None, "constant2": None}),
    ("out = in_b", "array", {}, {"b"}, {},
    {"type": tutil.TaskletType.ARRAY_SCALAR_ASSIGNMENT, "lhs": "out",
    "rhs1": "in_b", "rhs2": None, "op": "=", "constant1": None, "constant2": None}),
    ("out = in_b", "scalar", {"b"}, {}, {},
    {"type": tutil.TaskletType.SCALAR_ARRAY_ASSIGNMENT, "lhs": "out",
    "rhs1": "in_b", "rhs2": None, "op": "=", "constant1": None, "constant2": None}),
    ("out = in_b", "scalar", {}, {"b"}, {},
    {"type": tutil.TaskletType.SCALAR_SCALAR_ASSIGNMENT, "lhs": "out",
    "rhs1": "in_b", "rhs2": None, "op": "=", "constant1": None, "constant2": None}),
    ("out = sym_a", "array", {}, {}, {"sym_a"},
    {"type": tutil.TaskletType.ARRAY_SYMBOL_ASSIGNMENT, "lhs": "out",
    "rhs1": None, "rhs2": None, "op": "=", "constant1": "sym_a", "constant2": None}),

    # === SINGLE-INPUT TWO RHS CASE ===
    ("out = in_a * in_a", "array", {"a"}, {}, {},
    {"type": tutil.TaskletType.ARRAY_ARRAY, "lhs": "out",
    "rhs1": "in_a", "rhs2": "in_a", "op": "*", "constant1": None, "constant2": None}),
    ("out = in_a + in_a", "array", {"a"}, {}, {},
    {"type": tutil.TaskletType.ARRAY_ARRAY, "lhs": "out",
    "rhs1": "in_a", "rhs2": "in_a", "op": "+", "constant1": None, "constant2": None}),
    ("out = in_a + in_a", "array", {}, {"a"}, {},
    {"type": tutil.TaskletType.SCALAR_SCALAR, "lhs": "out",
    "rhs1": "in_a", "rhs2": "in_a", "op": "+", "constant1": None, "constant2": None}),
]

i = 0
def _gen_sdfg(tasklet_info: typing.Tuple[str, str, typing.Set[str], typing.Set[str], typing.Set[str], tutil.TaskletType]) -> dace.SDFG:
    global i
    i += 1
    sdfg = dace.SDFG(f"sd{i}")
    state = sdfg.add_state("s0", is_start_block=True)

    expr_str, out_type, in_arrays, in_scalars, in_symbols, _ = tasklet_info

    t1 = state.add_tasklet(
        name="t1",
        inputs={f"in_{a}" for a in in_arrays}.union({f"in_{a}" for a in in_scalars}),
        outputs={"out"},
        code=expr_str
    )

    for in_array in in_arrays:
        sdfg.add_array(in_array, (1,), dace.float64)
        state.add_edge(
            state.add_access(in_array), None,
            t1, f"in_{in_array}",
            dace.memlet.Memlet(f"{in_array}[0]")
        )
    for in_scalar in in_scalars:
        sdfg.add_scalar(in_scalar, dace.float64)
        state.add_edge(
            state.add_access(in_scalar), None,
            t1, f"in_{in_scalar}",
            dace.memlet.Memlet(f"{in_scalar}[0]")
        )
    for in_symbol in in_symbols:
        sdfg.add_symbol(in_symbol, dace.float64)
    
    if out_type == "array":
        sdfg.add_array("O", (1,), dace.float64)
    else:
        sdfg.add_scalar("O", dace.float64)

    state.add_edge(t1, "out",
                   state.add_access("O"), None,
                   dace.memlet.Memlet("O[0]" if out_type == "array" else "O"))

    sdfg.validate()
    return sdfg

@pytest.mark.parametrize("tasklet_info",tasklet_infos)
def test_single_tasklet_split(tasklet_info):
    sdfg = _gen_sdfg(tasklet_info)
    sdfg.validate()
    sdfg.compile()

    _, _, _, _, _, desired_tasklet_info = tasklet_info

    tasklets = {(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)}
    assert len(tasklets) == 1
    tasklet, state = tasklets.pop()

    tasklet_info_dict = tutil.classify_tasklet(state=state, node=tasklet)
    print(desired_tasklet_info)
    print(tasklet_info_dict)


    assert desired_tasklet_info == tasklet_info_dict, f"Expected: {desired_tasklet_info}, Got: {tasklet_info_dict}"


if __name__ == "__main__":
    for config_tuple in tasklet_infos:
        test_single_tasklet_split(config_tuple)