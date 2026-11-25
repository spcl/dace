# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``FuseIdenticalConnectors`` gives a nested SDFG one connector per outer container it binds as a whole."""

import json
import re

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import FuseIdenticalConnectors

N = dace.symbol("N")


def nested_sdfg(sdfg: dace.SDFG) -> nodes.NestedSDFG:
    return next(n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))


def reader_of_two_views(
    subsets: tuple[str, str] = ("0:N", "0:N"),
    dtype: dace.typeclass = dace.float64,
    strides: tuple | None = None,
    second: str = "A",
) -> dace.SDFG:
    """``B[i] = x[i] + y[i]`` in a nested SDFG whose ``x`` and ``y`` both bind ``A``."""
    sdfg = dace.SDFG("fuse_two_reads")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    if second != "A":
        sdfg.add_array(second, [N], dace.float64)
    inner = dace.SDFG("reads")
    inner.add_array("x", [N], dace.float64)
    inner.add_array("y", [N], dtype, strides=strides)
    inner.add_array("b", [N], dace.float64)
    inner.add_state().add_mapped_tasklet(
        "add",
        {"i": "0:N"},
        {"p": dace.Memlet("x[i]"), "q": dace.Memlet("y[i]")},
        "o = p + q",
        {"o": dace.Memlet("b[i]")},
        external_edges=True,
    )
    state = sdfg.add_state()
    node = state.add_nested_sdfg(inner, dict.fromkeys(["x", "y"]), dict.fromkeys(["b"]))
    a = state.add_read("A")
    state.add_edge(a, None, node, "x", dace.Memlet(f"A[{subsets[0]}]"))
    other = a if second == "A" else state.add_read(second)
    state.add_edge(other, None, node, "y", dace.Memlet(f"{second}[{subsets[1]}]"))
    state.add_edge(node, "b", state.add_write("B"), None, dace.Memlet("B[0:N]"))
    return sdfg


def in_place_doubler() -> dace.SDFG:
    """``y[i] = 2 * x[i]`` in a nested SDFG reading ``A`` through ``x`` and writing it through ``y``."""
    sdfg = dace.SDFG("fuse_read_write")
    sdfg.add_array("A", [N], dace.float64)
    inner = dace.SDFG("doubles")
    inner.add_array("x", [N], dace.float64)
    inner.add_array("y", [N], dace.float64)
    inner.add_state().add_mapped_tasklet(
        "double", {"i": "0:N"}, {"p": dace.Memlet("x[i]")}, "o = 2 * p", {"o": dace.Memlet("y[i]")}, external_edges=True
    )
    state = sdfg.add_state()
    node = state.add_nested_sdfg(inner, dict.fromkeys(["x"]), dict.fromkeys(["y"]))
    state.add_edge(state.add_read("A"), None, node, "x", dace.Memlet("A[0:N]"))
    state.add_edge(node, "y", state.add_write("A"), None, dace.Memlet("A[0:N]"))
    return sdfg


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_two_reads_of_one_array_become_one_connector():
    sdfg = reader_of_two_views()
    assert sdfg.apply_transformations_repeated(FuseIdenticalConnectors) == 1
    node = nested_sdfg(sdfg)
    assert list(node.in_connectors) == ["x"] and "y" not in node.sdfg.arrays
    sdfg.validate()
    assert re.search(r"\by\b", json.dumps(node.sdfg.to_json())) is None
    a = np.random.default_rng(1).random(64)
    b = np.zeros(64)
    sdfg(A=a, B=b, N=64)
    assert np.allclose(b, 2 * a)


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_a_read_and_a_write_of_one_array_become_one_in_out_connector():
    sdfg = in_place_doubler()
    assert sdfg.apply_transformations_repeated(FuseIdenticalConnectors) == 1
    node = nested_sdfg(sdfg)
    assert list(node.in_connectors) == ["x"] and list(node.out_connectors) == ["x"]
    sdfg.validate()
    assert re.search(r"\by\b", json.dumps(node.sdfg.to_json())) is None
    a = np.random.default_rng(2).random(64)
    expected = 2 * a
    sdfg(A=a, N=64)
    assert np.allclose(a, expected)


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_two_reads_of_different_parts_of_the_whole_array_join_their_subsets():
    sdfg = reader_of_two_views(("0:32", "32:64"))
    assert sdfg.apply_transformations_repeated(FuseIdenticalConnectors) == 1
    node = nested_sdfg(sdfg)
    (edge,) = sdfg.start_block.in_edges(node)
    assert str(edge.data.subset) == "0:64"
    sdfg.validate()
    a = np.random.default_rng(3).random(64)
    b = np.zeros(64)
    sdfg(A=a, B=b, N=64)
    assert np.allclose(b, 2 * a)


@pytest.mark.parametrize(
    "variant",
    [{"dtype": dace.float32}, {"strides": (2,)}, {"second": "C"}],
    ids=["other-dtype", "other-strides", "other-container"],
)
# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_connectors_that_do_not_describe_the_same_array_stay_apart(variant):
    sdfg = reader_of_two_views(**variant)
    assert sdfg.apply_transformations_repeated(FuseIdenticalConnectors) == 0
    assert list(nested_sdfg(sdfg).in_connectors) == ["x", "y"]


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_windows_of_one_array_stay_apart():
    sdfg = reader_of_two_views(("0:N/2", "N/2:N"))
    inner = nested_sdfg(sdfg).sdfg
    for name in ("x", "y"):
        inner.arrays[name].shape = (N // 2,)
        inner.arrays[name].total_size = N // 2
    assert sdfg.apply_transformations_repeated(FuseIdenticalConnectors) == 0
    assert list(nested_sdfg(sdfg).in_connectors) == ["x", "y"]


if __name__ == "__main__":
    test_two_reads_of_one_array_become_one_connector()
    test_a_read_and_a_write_of_one_array_become_one_in_out_connector()
    test_two_reads_of_different_parts_of_the_whole_array_join_their_subsets()
    test_connectors_that_do_not_describe_the_same_array_stay_apart({"dtype": dace.float32})
    test_connectors_that_do_not_describe_the_same_array_stay_apart({"strides": (2,)})
    test_connectors_that_do_not_describe_the_same_array_stay_apart({"second": "C"})
    test_windows_of_one_array_stay_apart()
