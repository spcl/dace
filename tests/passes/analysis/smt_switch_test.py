# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``optimizer.canonicalization.smt``: False runs canonicalization as if z3 were not installed, so the loops only
the solver can decide stay sequential. TSVC ``s118`` is one: its inner loop ``for j < i: a[i] += bb[j, i] *
a[i - j - 1]`` reads only elements below ``i``, and only z3 proves that the reads never meet the write ``a[i]``, which
makes the inner loop a parallel reduction."""

import pytest

from dace.config import set_temporary
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import smt_dependence
from dace.transformation.passes.canonicalize import canonicalize
from tests.corpus.measure_parallelization import cpu_params
from tests.corpus.tsvc import tsvc

pytestmark = pytest.mark.skipif(not smt_dependence.has_z3(), reason="needs z3")


def canonicalized_s118(tag: str):
    kernel = tsvc.collect(name="s118_d_single")[0]
    sdfg = tsvc.to_sdfg(kernel, tag=tag, simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, **cpu_params(4))
    return sdfg


def loops_and_maps(sdfg) -> tuple[int, int]:
    loops = sum(1 for region in sdfg.all_control_flow_regions() if isinstance(region, LoopRegion))
    return loops, sum(1 for node, parent in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))


def test_the_switch_turns_the_solver_off_and_back_on():
    with set_temporary("optimizer", "canonicalization", "smt", value=False):
        assert not smt_dependence.has_z3()
    assert smt_dependence.has_z3()


def test_s118_inner_loop_is_parallel_with_the_solver_and_sequential_without_it():
    assert loops_and_maps(canonicalized_s118("smt_on")) == (1, 1), "the outer loop, and the inner loop as a map"
    with set_temporary("optimizer", "canonicalization", "smt", value=False):
        without = canonicalized_s118("smt_off")
    assert loops_and_maps(without) == (2, 0), "both loops stay sequential"


if __name__ == "__main__":
    test_the_switch_turns_the_solver_off_and_back_on()
    test_s118_inner_loop_is_parallel_with_the_solver_and_sequential_without_it()
