# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ConfigPropagation: one external variant per configuration combination, behind a dispatcher with the signature."""

import os

import numpy as np
import pytest

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.config_propagation import ConfigPropagation, apply_invariants, resolve_invariants
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
    arguments = set(sdfg.arglist())
    root = ConfigPropagation(domains, str(tmp_path)).apply_pass(sdfg, {})
    assert root is sdfg, "the SDFG itself becomes the caller"

    generic = f"{sdfg.name}_generic.sdfgz"
    files = sorted(os.listdir(tmp_path))
    variants = sorted(f"{sdfg.name}_lvn_only_{l}_istep_{s}.sdfgz" for l in (0, 1) for s in (1, 2))
    assert files == sorted(variants + [generic])
    for f in variants:
        variant = dace.SDFG.from_file(str(tmp_path / f))
        assert config_conditions(variant) == [], f
        args = {n for n, d in variant.arrays.items() if not d.transient}
        assert "istep" not in args, f  # a baked scalar argument leaves the signature
        assert "lvn_only" in args, f  # a length-1 array argument stays: callers pass its buffer
        assert "ldeep" not in variant.free_symbols, f

    (dispatch,) = root.nodes()  # if / else if / else, and nothing else
    assert isinstance(dispatch, ConditionalBlock) and len(dispatch.branches) == 5
    assert dispatch.branches[-1][0] is None, "the last branch is the generic fallback"
    assert not [n for n, d in root.arrays.items() if d.transient]

    a = np.random.default_rng(0).random(16)
    for lvn_only in (0, 1):
        for istep in (1, 2):
            b = np.zeros_like(a)
            root(a=a, b=b, lvn_only=np.array([lvn_only], dtype=np.int32), istep=np.int32(istep), ldeep=0, N=16)
            np.testing.assert_allclose(b, reference(a, lvn_only, istep, 0), rtol=0, atol=0)
    list(root.all_control_flow_regions(recursive=True, load_ext=True))  # signature analysis needs the bodies
    assert set(root.arglist()) == arguments  # the dispatcher reads every configuration value


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
    assert sorted(os.listdir(tmp_path)) == sorted([f"{sdfg.name}.sdfgz", f"{sdfg.name}_generic.sdfgz"])
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


def test_invariants_resolve_through_chains_and_identities():
    """A chain resolves to the root symbol; a symbol equal to itself is no cycle, two symbols equal to each other are."""
    resolved = resolve_invariants({"w_d1": "nlevp1", "nlevp1": "nlev + 1", "nlev": "nlev"})
    want = {"w_d1": "nlev + 1", "nlevp1": "nlev + 1", "nlev": "nlev"}
    assert {name: dace.symbolic.pystr_to_symbolic(expr) for name, expr in resolved.items()} == {
        name: dace.symbolic.pystr_to_symbolic(expr) for name, expr in want.items()
    }
    with pytest.raises(ValueError, match="Cyclic"):
        resolve_invariants({"a": "b + 1", "b": "a + 1"})


def nested_fill(depth: int) -> dace.SDFG:
    """``A[i] = i`` over ``A[w_d1]``, the map ``depth`` nested SDFGs below the returned root."""
    sdfg = dace.SDFG(f"fill_{depth}")
    sdfg.add_symbol("w_d1", dace.int32)
    sdfg.add_array("A", ["w_d1"], dace.float64)
    state = sdfg.add_state()
    if depth == 0:
        state.add_mapped_tasklet("fill", {"i": "0:w_d1"}, {}, "a = i", {"a": dace.Memlet("A[i]")}, external_edges=True)
        return sdfg
    node = state.add_nested_sdfg(nested_fill(depth - 1), {}, {"A"}, {"w_d1": "w_d1"})
    state.add_edge(node, "A", state.add_write("A"), None, dace.Memlet("A[0:w_d1]"))
    return sdfg


def test_a_symbol_two_nested_sdfgs_down_reads_the_root_symbol_of_its_invariant():
    """``w_d1`` becomes ``nlev + 1`` at every level, and each nested SDFG maps and declares ``nlev``."""
    sdfg = nested_fill(2)
    sdfg.add_symbol("nlev", dace.int32)
    apply_invariants(sdfg, {"w_d1": "nlev + 1"})
    sdfg.validate()
    calls = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert len(calls) == 2
    for node in calls:
        assert node.symbol_mapping["nlev"] == "nlev" and "w_d1" not in node.symbol_mapping
        assert node.sdfg.symbols["nlev"] == dace.int32 and "w_d1" not in node.sdfg.symbols

    a = np.zeros(5)
    sdfg(A=a, nlev=4)
    np.testing.assert_array_equal(a, np.arange(5.0))


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    for test in (
        test_variants_and_dispatcher,
        test_values_outside_the_domains_run_the_generic_sdfg,
        test_singleton_domains_call_one_variant,
        test_staged_argument_takes_the_value,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
    test_invariants_resolve_through_chains_and_identities()
    test_a_symbol_two_nested_sdfgs_down_reads_the_root_symbol_of_its_invariant()
