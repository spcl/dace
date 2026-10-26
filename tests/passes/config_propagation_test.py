# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ConfigPropagation: one external variant per configuration combination, behind a dispatcher with the signature."""

import os

import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.config_propagation import ConfigPropagation
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

    files = sorted(os.listdir(tmp_path))
    assert files == sorted(f"{sdfg.name}_lvn_only_{l}_istep_{s}.sdfgz" for l in (0, 1) for s in (1, 2))
    for f in files:
        variant = dace.SDFG.from_file(str(tmp_path / f))
        assert config_conditions(variant) == [], f
        args = {n for n, d in variant.arrays.items() if not d.transient}
        assert "istep" not in args, f  # a baked scalar argument leaves the signature
        assert "lvn_only" in args, f  # a length-1 array argument stays: callers pass its buffer
        assert "ldeep" not in variant.free_symbols, f

    dispatch = [b for b in root.nodes() if isinstance(b, ConditionalBlock)]
    assert len(dispatch) == 1 and len(dispatch[0].branches) == 4

    a = np.random.default_rng(0).random(16)
    for lvn_only in (0, 1):
        for istep in (1, 2):
            b = np.zeros_like(a)
            root(a=a, b=b, lvn_only=np.array([lvn_only], dtype=np.int32), istep=np.int32(istep), ldeep=0, N=16)
            np.testing.assert_allclose(b, reference(a, lvn_only, istep, 0), rtol=0, atol=0)
    list(root.all_control_flow_regions(recursive=True, load_ext=True))  # signature analysis needs the bodies
    assert set(root.arglist()) == set(sdfg.arglist()) - {"ldeep"}  # a baked singleton symbol is no parameter


def test_singleton_domains_call_one_variant(tmp_path):
    """With nothing left to choose, the dispatcher calls its one variant from a plain state, without a branch."""
    sdfg = tendencies.to_sdfg(simplify=True)
    root = ConfigPropagation({"lvn_only": [1], "istep": [2], "ldeep": [1]}, str(tmp_path)).apply_pass(sdfg, {})
    assert os.listdir(tmp_path) == [f"{sdfg.name}.sdfgz"]
    assert not any(isinstance(b, ConditionalBlock) for b in root.nodes())

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
        variant = dace.SDFG.from_file(str(tmp_path / f))
        assert config_conditions(variant) == [], f
        accessed = {n.data for n, _ in variant.all_nodes_recursive() if isinstance(n, dace.nodes.AccessNode)}
        assert not set(staged) & accessed, f

    a = np.random.default_rng(2).random(16)
    for lvn_only in (0, 1):
        b = np.zeros_like(a)
        root(a=a, b=b, lvn_only=np.array([lvn_only], dtype=np.int32), istep=np.int32(1), ldeep=0, N=16)
        np.testing.assert_allclose(b, reference(a, lvn_only, 1, 0), rtol=0, atol=0)
