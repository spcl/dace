# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import dace
from dace.transformation.passes.empty_loop_elimination import EmptyLoopElimination
from dace.sdfg.state import LoopRegion


def test_regular_loop():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            pass

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["EmptyLoopElimination"])
    sdfg.validate()

    # Should have exactly one loop region
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 1

    # Apply the transformation
    EmptyLoopElimination().apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the loop region is gone
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 0


def test_negative():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            a[i] = i

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["EmptyLoopElimination"])
    sdfg.validate()

    # Should have exactly one loop region
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 1

    # Apply the transformation
    EmptyLoopElimination().apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the loop region is not eliminated
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 1


def test_symbolic_loop():

    N = dace.symbol('N')

    @dace.program
    def tester(a: dace.float64[20]):
        N = 0
        for i in range(20):
            N += 1
        a[0] = N

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["EmptyLoopElimination"])
    sdfg.validate()

    # Should have exactly one loop region
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 1

    # Apply the transformation
    EmptyLoopElimination().apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the loop region is not eliminated
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 1


def test_nested_empty_loops():

    @dace.program
    def tester(a: dace.float64[20, 20]):
        for i in range(20):
            for j in range(20):
                pass

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["EmptyLoopElimination"])
    sdfg.validate()

    # Should have two loop regions (nested)
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 2

    # Apply the transformation
    EmptyLoopElimination().apply_pass(sdfg, {})
    sdfg.validate()

    # Check that both loop regions are eliminated
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 0


def test_loop_with_break():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            break

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["EmptyLoopElimination"])
    sdfg.validate()

    # Should have exactly one loop region
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 1

    # Apply the transformation
    EmptyLoopElimination().apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the loop region is eliminated
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 0


def test_loop_with_continue():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            continue

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["EmptyLoopElimination"])
    sdfg.validate()

    # Should have exactly one loop region
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 1

    # Apply the transformation
    EmptyLoopElimination().apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the loop region is eliminated
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 0


def test_loop_with_return():

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(20):
            return

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["EmptyLoopElimination"])
    sdfg.validate()

    # Should have exactly one loop region
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 1

    # Apply the transformation
    EmptyLoopElimination().apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the loop region is not eliminated
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 1


def test_partially_nested_empty_loops():

    @dace.program
    def tester(a: dace.float64[20, 20]):
        for i in range(10):
            for j in range(5):
                pass
            a[i, 0] = i

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["EmptyLoopElimination"])
    sdfg.validate()

    # Should have two loop regions
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 2

    # Apply the transformation
    EmptyLoopElimination().apply_pass(sdfg, {})
    sdfg.validate()

    # Check that only the inner empty loop is eliminated
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 1


def test_empty_loop_with_symbolic_bounds():

    N = dace.symbol('N')

    @dace.program
    def tester(a: dace.float64[20], N: dace.int32):
        for i in range(N):
            pass

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.simplify(skip=["EmptyLoopElimination"])
    sdfg.validate()

    # Should have exactly one loop region
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 1

    # Apply the transformation
    EmptyLoopElimination().apply_pass(sdfg, {})
    sdfg.validate()

    # Check that the loop region is eliminated even with symbolic bounds
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == 0


if __name__ == '__main__':
    test_regular_loop()
    test_negative()
    test_symbolic_loop()
    test_nested_empty_loops()
    test_loop_with_break()
    test_loop_with_continue()
    test_loop_with_return()
    test_partially_nested_empty_loops()
    test_empty_loop_with_symbolic_bounds()
