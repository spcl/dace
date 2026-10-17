# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the compile check and the repair loop around it."""

import shutil

import numpy as np
import pytest

import dace
from dace import dtypes, nodes
from dace.libraries.ai import verify
from dace.libraries.ai.backend import TaskletSpec
from dace.libraries.ai.context import collect_context
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.expansion import ExpandAI
from dace.libraries.ai.nodes import AINode
from dace.sdfg import infer_types
from tests.library.ai.ai_test_utils import offline, stub_provider, stub_provider_sequence

pytestmark = pytest.mark.skipif(
    shutil.which("c++") is None and shutil.which("g++") is None, reason="needs a host C++ compiler"
)

GOOD = "_out = 2.0 * _in;"
BAD = "_out = this_symbol_does_not_exist(_in);"


def _sdfg_and_node(name: str = "ai_verify"):
    """
    Builds a one-element SDFG around an :class:`AINode`.

    :param name: Name of the SDFG.
    :return: A tuple of (SDFG, state, node).
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    node = AINode("double", "Double the input.", inputs={"_in"}, outputs={"_out"})
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_in", dace.Memlet("A[0]"))
    state.add_edge(node, "_out", state.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg, state, node


def _check(spec: TaskletSpec, sdfg: dace.SDFG, state: dace.SDFGState, node: AINode, folder) -> verify.CheckResult:
    """
    Runs the compile check on a specification for a node.

    :return: The check result.
    """
    ctx = collect_context(node, state, sdfg)
    tasklet = ExpandAI._make_tasklet(node, spec, ctx)
    return verify.compile_check(tasklet, node, state, sdfg, ctx, [], str(folder))


def test_check_accepts_valid_code(tmp_path):
    sdfg, state, node = _sdfg_and_node()
    result = _check(TaskletSpec(code=GOOD), sdfg, state, node, tmp_path)
    assert result.ok and not result.skipped
    assert any(GOOD in source for source in result.sources.values())


def test_check_rejects_invalid_code_with_diagnostics(tmp_path):
    sdfg, state, node = _sdfg_and_node()
    result = _check(TaskletSpec(code=BAD), sdfg, state, node, tmp_path)
    assert not result.ok
    assert "this_symbol_does_not_exist" in result.diagnostics
    assert result.sources, "the generated sources are kept for the repair prompt"


def test_check_builds_global_code_and_state_fields(tmp_path):
    sdfg, state, node = _sdfg_and_node()
    spec = TaskletSpec(
        code="__state->counter += 1;\n_out = std::sqrt(_in);",
        code_global="#include <cmath>",
        state_fields=["int counter;"],
    )
    result = _check(spec, sdfg, state, node, tmp_path)
    assert result.ok and not result.skipped, result.skipped


def test_a_state_field_can_point_to_a_type_from_global_code():
    # The state struct is emitted before code_global, so a field of a type defined there is a
    # pointer declared with the struct keyword
    sdfg, state, node = _sdfg_and_node("ai_state_field_pointer")
    spec = TaskletSpec(
        code="__state->scale_0->calls += 1;\n_out = __state->scale_0->factor * _in;",
        code_global="struct ai_scale_0 {\n    double factor;\n    int calls;\n};",
        code_init="__state->scale_0 = new ai_scale_0{2.0, 0};",
        code_exit="delete __state->scale_0;",
        state_fields=["struct ai_scale_0 *scale_0;"],
    )
    with stub_provider(spec) as provider:
        node.expand(state, "ai")
    assert len(provider.calls) == 1, "the compile check rejected the code"

    A = np.array([3.0])
    B = np.zeros(1)
    sdfg(A=A, B=B)
    assert B[0] == 6.0


def test_check_defines_map_symbols(tmp_path):
    sdfg = dace.SDFG("ai_check_map")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("B", [8], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("m", {"i": "0:8"})
    node = AINode("double", "Double the input.", inputs={"_in"}, outputs={"_out"})
    state.add_memlet_path(state.add_read("A"), entry, node, dst_conn="_in", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(node, exit_node, state.add_write("B"), src_conn="_out", memlet=dace.Memlet("B[i]"))

    result = _check(TaskletSpec(code="_out = _in * i;"), sdfg, state, node, tmp_path)
    assert result.ok and not result.skipped, result.skipped


def test_check_covers_python_tasklets(tmp_path):
    sdfg, state, node = _sdfg_and_node()
    result = _check(TaskletSpec(code="_out = _in * 2", language="Python"), sdfg, state, node, tmp_path)
    assert result.ok and not result.skipped, result.skipped


def test_repair_loop_recovers_from_a_compile_error():
    sdfg, state, node = _sdfg_and_node()

    with stub_provider_sequence([TaskletSpec(code=BAD), TaskletSpec(code=GOOD)]) as provider:
        node.expand(state, "ai")

    assert len(provider.calls) == 2, "the model was not asked to repair the broken code"
    repair_conversation = provider.calls[1]
    assert [m["role"] for m in repair_conversation] == ["user", "assistant", "user"]
    assert "this_symbol_does_not_exist" in repair_conversation[-1]["content"]
    assert "does not compile" in repair_conversation[-1]["content"]

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert tasklet.code.as_string.strip() == GOOD


def test_repair_loop_gives_up_after_the_configured_attempts():
    sdfg, state, node = _sdfg_and_node()

    with dace.config.set_temporary("ai", "max_repair_attempts", value=1):
        with stub_provider(TaskletSpec(code=BAD)) as provider:
            with pytest.raises(AIExpansionError, match="does not compile"):
                node.expand(state, "ai")

    # One initial generation plus one repair
    assert len(provider.calls) == 2
    # The library node is left in place, so the failure can be diagnosed
    assert any(isinstance(n, AINode) for n in state.nodes())


def test_the_compile_check_can_be_disabled():
    sdfg, state, node = _sdfg_and_node()

    with dace.config.set_temporary("ai", "compile_check", value=False):
        with stub_provider(TaskletSpec(code=BAD)) as provider:
            node.expand(state, "ai")

    assert len(provider.calls) == 1, "the code was checked even though the check is off"


#: Storage, memlet subset, dynamic, and the expected C++ types of the input and the output. A single
#: element of GPU memory is a pointer, and so is a dynamic output.
CONNECTOR_SHAPES = {
    "cpu_element": (dtypes.StorageType.CPU_Heap, "0, 0", False, "float", "float"),
    "cpu_tile": (dtypes.StorageType.CPU_Heap, "0:4, 0:4", False, "float*", "float*"),
    "gpu_element": (dtypes.StorageType.GPU_Global, "0, 0", False, "float*", "float*"),
    "gpu_tile": (dtypes.StorageType.GPU_Global, "0:4, 0:4", False, "float*", "float*"),
    "dynamic_element": (dtypes.StorageType.CPU_Heap, "0, 0", True, "float", "float*"),
}


@pytest.mark.parametrize("shape", sorted(CONNECTOR_SHAPES))
def test_connector_types_match_codegen(shape):
    """The connector types told to the model must be the ones ``infer_types`` later resolves."""
    storage, subset, dynamic, expected_in, expected_out = CONNECTOR_SHAPES[shape]

    sdfg = dace.SDFG(f"ai_conn_{shape}")
    sdfg.add_array("A", [8, 8], dace.float32, storage=storage)
    sdfg.add_array("B", [8, 8], dace.float32, storage=storage)
    state = sdfg.add_state()
    node = AINode("n", "Do something.", inputs={"_in"}, outputs={"_out"})
    state.add_node(node)
    read = dace.Memlet(f"A[{subset}]")
    write = dace.Memlet(f"B[{subset}]")
    read.dynamic = write.dynamic = dynamic
    state.add_edge(state.add_read("A"), None, node, "_in", read)
    state.add_edge(node, "_out", state.add_write("B"), None, write)

    predicted = {c.name: c.ctype for c in collect_context(node, state, sdfg).connectors}

    with dace.config.set_temporary("ai", "compile_check", value=False):
        with stub_provider(TaskletSpec(code="// nothing")):
            node.expand(state, "ai")
    infer_types.infer_connector_types(sdfg)

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    actual = {
        name: str(conntype.ctype)
        for name, conntype in list(tasklet.in_connectors.items()) + list(tasklet.out_connectors.items())
    }
    assert predicted == {"_in": expected_in, "_out": expected_out}
    assert actual == predicted


if __name__ == "__main__":
    with offline():
        test_a_state_field_can_point_to_a_type_from_global_code()
        test_repair_loop_recovers_from_a_compile_error()
        test_repair_loop_gives_up_after_the_configured_attempts()
        test_the_compile_check_can_be_disabled()
        for shape in sorted(CONNECTOR_SHAPES):
            test_connector_types_match_codegen(shape)
