# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import dace
from dace import nodes
from dace.transformation.interstate import ContinueToCondition
from dace.sdfg.state import ContinueBlock, LoopRegion, ConditionalBlock, ControlFlowRegion
from dace.properties import CodeBlock


def test_regular_loop():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            if a[i] > 10:
                continue
            a[i] = a[i] + 1

    sdfg = tester.to_sdfg()
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    sdfg.apply_transformations_repeated(ContinueToCondition)
    sdfg.validate()

    # Check that the continue node was removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 0


def test_no_condition():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            continue
            a[i] = a[i] + 1

    sdfg = tester.to_sdfg()
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    sdfg.apply_transformations_repeated(ContinueToCondition)
    sdfg.validate()

    # Check that the continue node was not removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1


def test_nested_loop():

    @dace.program
    def tester(a: dace.float64[20, 10]):
        for i in range(20):
            for j in range(10):
                if i > j:
                    continue
                a[i, j] = a[i, j] + 1

    sdfg = tester.to_sdfg()
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    sdfg.apply_transformations_repeated(ContinueToCondition)
    sdfg.validate()

    # Check that the continue node was removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 0

    # Check correctness
    a = dace.ndarray([20, 10], dtype=dace.float64)
    a[:] = np.random.rand(20, 10).astype(dace.float64.type)
    a_copy = a.copy()
    sdfg(a=a)

    for i in range(20):
        for j in range(10):
            if i > j:
                assert a[i, j] == a_copy[i, j]
            else:
                assert a[i, j] == a_copy[i, j] + 1


def test_loop_in_nested_sdfg():
    sdfg = dace.SDFG("nested1")
    s11 = sdfg.add_state(is_start_block=True)

    sdfg2 = dace.SDFG("nested2")
    s11.add_node(nodes.NestedSDFG("n2", sdfg2, {}, {}))

    loop = LoopRegion("loop", "i < 64", "i", "i = 0", "i = i + 1")
    sdfg2.add_node(loop)
    cond = ConditionalBlock("cond", sdfg2)
    cfr = ControlFlowRegion("cfr", sdfg2)
    cont = ContinueBlock("cont", sdfg2, cfr)
    cfr.add_node(cont)
    cond.add_branch(CodeBlock("i == 0"), cfr)
    loop.add_node(cond)
    s21 = loop.add_state_after(cond)

    sdfg3 = dace.SDFG("nested3")
    s21.add_node(nodes.NestedSDFG("n3", sdfg3, {}, {}))
    sdfg3.add_state(is_start_block=True)
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    xform = ContinueToCondition()
    xform.cb = cont
    if xform.can_be_applied(cfr, 0, sdfg2):
        xform.apply(cfr, sdfg2)

    # sdfg.apply_transformations_repeated(ContinueToCondition)
    sdfg.validate()

    # Check that the continue node was removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 0


if __name__ == '__main__':
    test_regular_loop()
    test_no_condition()
    test_nested_loop()
    test_loop_in_nested_sdfg()
