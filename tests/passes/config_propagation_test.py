# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ConfigPropagation: one external variant per configuration combination, behind a dispatcher with the signature."""

import os

import numpy as np
import pytest

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.config_propagation import ConfigPropagation, specialize_config, staged_copies
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars

N = dace.symbol("N")
LDEEP = dace.symbol("ldeep", dace.int32)

CONFIG_NAMES = ("lvn_only", "istep", "ldeep")  # substrings: also catches staged copies


@dace.program
def tendencies(a: dace.float64[N], b: dace.float64[N], lvn_only: dace.int32[1], istep: dace.int32):
    if istep == 1:
        b[:] = a * 2.0
    else:
        b[:] = a * 3.0
    if lvn_only[0] == 0:
        b[:] = b + 1.0
    if LDEEP == 1:
        b[:] = -b


def reference(a: np.ndarray, lvn_only: int, istep: int, ldeep: int) -> np.ndarray:
    b = a * (2.0 if istep == 1 else 3.0)
    if lvn_only == 0:
        b = b + 1.0
    return -b if ldeep == 1 else b


def config_conditions(sdfg: dace.SDFG) -> list[str]:
    """Branch conditions anywhere in ``sdfg`` that still read a configuration name."""
    found = []
    for block in sdfg.all_control_flow_blocks(recursive=True):
        if isinstance(block, ConditionalBlock):
            for cond, _ in block.branches:
                if cond is not None and any(name in cond.as_string for name in CONFIG_NAMES):
                    found.append(cond.as_string)
    return found


def test_variants_and_dispatcher(tmp_path):
    sdfg = tendencies.to_sdfg(simplify=True)
    assert config_conditions(sdfg), "the unspecialized program must branch on the configuration"
    domains = {"lvn_only": [0, 1], "istep": [1, 2], "ldeep": [0]}
    root = ConfigPropagation(domains, str(tmp_path)).apply_pass(sdfg, {})

    generic = f"{sdfg.name}_generic.sdfgz"
    files = sorted(os.listdir(tmp_path))
    variants = sorted(f"{sdfg.name}_lvn_only_{l}_istep_{s}_ldeep_0.sdfgz" for l in (0, 1) for s in (1, 2))
    assert files == sorted(variants + [generic])
    for f in variants:
        variant = dace.SDFG.from_file(str(tmp_path / f))
        assert config_conditions(variant) == [], f
        args = {n for n, d in variant.arrays.items() if not d.transient}
        assert "istep" not in args, f  # a baked scalar argument leaves the signature
        assert "lvn_only" in args, f  # a length-1 array argument stays: callers pass its buffer
        assert "ldeep" not in variant.free_symbols, f

    dispatch = [b for b in root.nodes() if isinstance(b, ConditionalBlock)]
    assert len(dispatch) == 1 and len(dispatch[0].branches) == 5
    assert dispatch[0].branches[-1][0] is None, "the last branch is the generic fallback"

    a = np.random.default_rng(0).random(16)
    for lvn_only in (0, 1):
        for istep in (1, 2):
            b = np.zeros_like(a)
            root(a=a, b=b, lvn_only=np.array([lvn_only], dtype=np.int32), istep=np.int32(istep), ldeep=0, N=16)
            np.testing.assert_allclose(b, reference(a, lvn_only, istep, 0), rtol=0, atol=0)
    list(root.all_control_flow_regions(recursive=True, load_ext=True))  # signature analysis needs the bodies
    assert set(root.arglist()) == set(sdfg.arglist())  # the dispatcher reads every configuration value


def test_values_outside_the_domains_run_the_generic_sdfg(tmp_path):
    """A value no variant was built for, a singleton included, takes the unspecialized SDFG instead of a wrong one."""
    sdfg = tendencies.to_sdfg(simplify=True)
    root = ConfigPropagation({"lvn_only": [0, 1], "istep": [1], "ldeep": [0]}, str(tmp_path)).apply_pass(sdfg, {})

    a = np.random.default_rng(3).random(16)
    for lvn_only, istep, ldeep in ((0, 2, 0), (1, 1, 1), (2, 1, 0)):
        b = np.zeros_like(a)
        root(a=a, b=b, lvn_only=np.array([lvn_only], dtype=np.int32), istep=np.int32(istep), ldeep=ldeep, N=16)
        np.testing.assert_allclose(b, reference(a, lvn_only, istep, ldeep), rtol=0, atol=0)


def test_singleton_domains_call_one_variant(tmp_path):
    """With nothing left to choose, the dispatcher has one variant and the generic fallback."""
    sdfg = tendencies.to_sdfg(simplify=True)
    root = ConfigPropagation({"lvn_only": [1], "istep": [2], "ldeep": [1]}, str(tmp_path)).apply_pass(sdfg, {})
    only = f"{sdfg.name}_lvn_only_1_istep_2_ldeep_1.sdfgz"
    assert sorted(os.listdir(tmp_path)) == sorted([only, f"{sdfg.name}_generic.sdfgz"])
    (dispatch,) = [b for b in root.nodes() if isinstance(b, ConditionalBlock)]
    assert len(dispatch.branches) == 2 and dispatch.branches[-1][0] is None

    a = np.random.default_rng(1).random(16)
    b = np.zeros_like(a)
    root(a=a, b=b, lvn_only=np.array([1], dtype=np.int32), istep=np.int32(2), ldeep=1, N=16)
    np.testing.assert_allclose(b, reference(a, 1, 2, 1), rtol=0, atol=0)


def test_staged_argument_takes_the_value(tmp_path):
    """A length-1 argument the frontend staged into a scalar copy: the copy, not the argument, guards the branch."""
    sdfg = tendencies.to_sdfg(simplify=True)
    ConvertLengthOneArraysToScalars(preserve_abi=True, filter={"lvn_only"}).apply_pass(sdfg, {})
    staged = [name for name in sdfg.arrays if name.startswith("scal_lvn_only")]
    assert staged, "the argument must be staged for this test to mean anything"
    root = ConfigPropagation({"lvn_only": [0, 1], "istep": [1], "ldeep": [0]}, str(tmp_path)).apply_pass(sdfg, {})
    for f in os.listdir(tmp_path):
        if f.endswith("_generic.sdfgz"):
            continue
        variant = dace.SDFG.from_file(str(tmp_path / f))
        assert config_conditions(variant) == [], f
        accessed = {n.data for n, _ in variant.all_nodes_recursive() if isinstance(n, dace.nodes.AccessNode)}
        assert not set(staged) & accessed, f

    a = np.random.default_rng(2).random(16)
    for lvn_only in (0, 1):
        b = np.zeros_like(a)
        root(a=a, b=b, lvn_only=np.array([lvn_only], dtype=np.int32), istep=np.int32(1), ldeep=0, N=16)
        np.testing.assert_allclose(b, reference(a, lvn_only, 1, 0), rtol=0, atol=0)


def test_written_configuration_is_refused():
    sdfg = dace.SDFG("written_flag")
    sdfg.add_scalar("flag", dace.int32)
    state = sdfg.add_state(is_start_block=True)
    tasklet = state.add_tasklet("bump", {}, {"o": None}, "o = 1")
    state.add_edge(tasklet, "o", state.add_write("flag"), None, dace.Memlet("flag"))

    with pytest.raises(ValueError, match="flag"):
        specialize_config(sdfg, {"flag": 0})


def flag_copied_to_transient(two_sources: bool) -> dace.SDFG:
    sdfg = dace.SDFG("staged_mixed")
    for name in ("x", "y", "res"):
        sdfg.add_array(name, [1], dace.int32)
    sdfg.add_scalar("t", dace.int32, transient=True)
    state = sdfg.add_state(is_start_block=True)
    held = state.add_access("t")
    state.add_edge(state.add_read("x"), None, held, None, dace.Memlet("x[0] -> [0]"))
    if two_sources:
        state.add_edge(state.add_read("y"), None, held, None, dace.Memlet("y[0] -> [0]"))
    tasklet = state.add_tasklet("use", {"a": None}, {"o": None}, "o = a")
    state.add_edge(held, None, tasklet, "a", dace.Memlet("t"))
    state.add_edge(tasklet, "o", state.add_write("res"), None, dace.Memlet("res[0]"))
    return sdfg


def test_staged_copy_of_two_values_is_not_baked():
    assert staged_copies(flag_copied_to_transient(True), {"x": 0, "y": 1}) == {}
    assert staged_copies(flag_copied_to_transient(False), {"x": 0}) == {"t": 0}


def test_staged_reduction_is_not_baked():
    sdfg = flag_copied_to_transient(False)
    for edge in sdfg.start_block.edges():
        if edge.data.data == "x":
            edge.data.wcr = "lambda a, b: a + b"

    assert staged_copies(sdfg, {"x": 0}) == {}


def test_multi_element_configuration_is_refused():
    sdfg = dace.SDFG("element_flag")
    sdfg.add_array("cfg", [3], dace.int32)
    sdfg.add_state(is_start_block=True)

    with pytest.raises(ValueError, match="cfg"):
        specialize_config(sdfg, {"cfg": 7})


def test_reassigned_symbol_is_refused():
    sdfg = dace.SDFG("reassigned_symbol")
    sdfg.add_symbol("k", dace.int32)
    first = sdfg.add_state(is_start_block=True)
    sdfg.add_state_after(first, assignments={"k": "k + 1"})

    with pytest.raises(ValueError, match="k"):
        specialize_config(sdfg, {"k": 1})


def test_singleton_values_name_the_variant(tmp_path):
    sdfg = tendencies.to_sdfg()
    ConfigPropagation({"ldeep": [0]}, str(tmp_path)).apply_pass(sdfg, {})
    ConfigPropagation({"ldeep": [1]}, str(tmp_path)).apply_pass(sdfg, {})

    names = sorted(os.listdir(tmp_path))
    assert f"{sdfg.name}_ldeep_0.sdfgz" in names and f"{sdfg.name}_ldeep_1.sdfgz" in names


def test_the_dispatcher_keeps_the_argument_order_constants_and_codegen_flags_of_its_program(tmp_path):
    sdfg = tendencies.to_sdfg(simplify=True)
    sdfg.add_constant("DEPTH", 4)
    sdfg.openmp_array_reductions = True  # what canonicalize sets on its output

    root = ConfigPropagation({"ldeep": [1]}, str(tmp_path)).apply_pass(sdfg, {})

    assert root.arg_names == sdfg.arg_names
    assert root.constants == sdfg.constants
    assert root.openmp_array_reductions and root.openmp_sections == sdfg.openmp_sections
