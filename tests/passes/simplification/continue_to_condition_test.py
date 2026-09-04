# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace import nodes
from dace.transformation.passes.simplification.continue_to_condition import ContinueToCondition
from dace.sdfg.state import ContinueBlock, LoopRegion, ConditionalBlock, ControlFlowRegion
from dace.properties import CodeBlock
from dace.transformation.pass_pipeline import FixedPointPipeline


def test_regular_loop():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            if a[i] > 10:
                continue
            a[i] = a[i] + 1

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ContinueToCondition"])
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the continue node was removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 0


def test_flipped():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            if a[i] <= 10:
                a[i] = a[i] + 1
            else:
                continue

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ContinueToCondition"])
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the continue node was not removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1


def test_no_condition():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            continue
            a[i] = a[i] + 1

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ContinueToCondition", "EmptyLoopElimination"])
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the continue node was not removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1


def test_prev_operation():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            if a[i] > 10:
                a[i] = a[i] + 10
                continue
            a[i] = a[i] + 1

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ContinueToCondition"])
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the continue node was not removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1


def test_succ_operation():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            if a[i] > 10:
                continue
                a[i] = a[i] + 10
            a[i] = a[i] + 1

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ContinueToCondition"])
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the continue node was removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 0


def test_nested_loop():

    @dace.program
    def tester(a: dace.float64[20, 10]):
        for i in range(20):
            for j in range(10):
                if i > j:
                    continue
                a[i, j] = a[i, j] + 1

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ContinueToCondition"])
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
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
    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the continue node was removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 0


def test_multiple_branches():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            if a[i] > 10:
                continue
            else:
                a[i] = a[i] + 1

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ContinueToCondition"])
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the continue node was not removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1


def test_multiple_branches2():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            if a[i] > 10:
                continue
            elif a[i] < 5:
                continue
            elif a[i] < 8:
                a[i] = a[i] + 2
            else:
                a[i] = a[i] + 1

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ContinueToCondition"])
    sdfg.validate()

    # Should have exactly two continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 2

    # Apply the transformation
    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the continue nodes were not removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 2


def test_nested_conditions():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            if a[i] > 10:
                if a[i] < 20:
                    continue
            a[i] = a[i] + 1

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ContinueToCondition"])
    sdfg.validate()

    # Should have exactly one continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1

    # Apply the transformation
    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the continue node was removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1


def test_nested_loops():

    @dace.program
    def tester(a: dace.float64[20, 20]):
        for i in range(20):
            if a[i, 0] > 10:
                continue

            for j in range(20):
                if a[i, j] > 10:
                    if a[i, j] < 20:
                        continue
                a[i, j] = a[i, j] + 1

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ContinueToCondition"])
    sdfg.validate()

    # Should have exactly two continue node
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 2

    # Apply the transformation
    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the continue node were removed
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 1


def test_consecutive_continues():
    """Four single-branch continues in a row -- the shape subset_sum's backtracking search has.

    Converting the first one MOVES every block after it into the conditional's branch and removes
    them from the loop region, so a candidate collected before that apply names a region that no
    longer holds it: ``outer_cfg.successors(pg)`` raised ``KeyError: ConditionalBlock`` rather than
    answering, and the whole ``to_sdfg(simplify=True)`` died with it.
    """

    @dace.program
    def tester(a: dace.int64[20], out: dace.int64[1]):
        total = 0
        for i in range(20):
            v = a[i]
            if v == 0:
                continue
            if v == 1:
                continue
            if v == 2:
                continue
            if v == 3:
                continue
            total += v
        out[0] = total

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ContinueToCondition"])
    sdfg.validate()

    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 4

    ppl = FixedPointPipeline([ContinueToCondition()])
    ppl.apply_pass(sdfg, {})
    sdfg.validate()

    # Only the FIRST converts: converting it moves the other three inside the new branch region,
    # whose parent is a ControlFlowRegion rather than the LoopRegion this pass requires. That is the
    # pass declining, which is its job -- the defect was that it DIED here instead of declining.
    cont_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ContinueBlock)]
    assert len(cont_nodes) == 3

    a = np.arange(20, dtype=np.int64) % 7
    out = np.zeros(1, dtype=np.int64)
    sdfg(a=a, out=out)
    assert out[0] == int(a[a > 3].sum())


if __name__ == '__main__':
    test_regular_loop()
    test_flipped()
    test_no_condition()
    test_prev_operation()
    test_succ_operation()
    test_nested_loop()
    test_loop_in_nested_sdfg()
    test_multiple_branches()
    test_multiple_branches2()
    test_nested_conditions()
    test_nested_loops()
    test_consecutive_continues()
