# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests loop normalization trainsformations. """

import numpy as np
import pytest
import dace
from dace.memlet import Memlet
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis


@pytest.mark.parametrize(
    "start, step",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (4, 2),
        (8, 4),
        (16, 4),
        (0, -1),
        (1, -1),
        (2, -2),
        (4, -2),
        (8, -4),
        (16, -4),
    ],
)
def test_normalize(start, step):
    """
    Tests if loop normalization works correctly in the general case.
    """

    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [32], dace.float32)
    sdfg.add_array("B", [32], dace.float32)

    cmp = "< 32" if step >= 0 else ">= 1"
    loop = LoopRegion("loop", f"i {cmp}", "i", f"i = {start}", f"i = i + {step}")
    sdfg.add_node(loop)
    s = loop.add_state("loop_body", is_start_block=True)
    a = s.add_access("A")
    b = s.add_access("B")
    s.add_edge(a, None, b, None, Memlet("A[i] -> B[i]"))

    sdfg.validate()
    assert loop_analysis.get_init_assignment(loop) == start
    assert loop_analysis.get_loop_stride(loop) == step

    if start == 0 and step == 1:
        assert loop.normalize() == False
    else:
        assert loop.normalize() == True

    # Check if loop normalization was successful
    assert loop_analysis.get_init_assignment(loop) == 0
    assert loop_analysis.get_loop_stride(loop) == 1

    # Validate correctness
    A = dace.ndarray([32], dtype=dace.float32)
    A[:] = np.random.rand(32).astype(dace.float32.type)
    B = dace.ndarray([32], dtype=dace.float32)
    B[:] = np.random.rand(32).astype(dace.float32.type)
    sdfg(A=A, B=B)

    if step >= 0:
        for i in range(start, 32, step):
            assert B[i] == A[i]
    else:
        for i in range(start, 0, step):
            assert B[i] == A[i]


def test_normalize_altered_iter():
    """
    Tests if loop normalization works correctly with an altered iteration variable.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [32], dace.float32)
    sdfg.add_array("B", [32], dace.float32)

    loop = LoopRegion("loop", f"i < 32", "i", f"i = 2", f"i = i + 1")
    sdfg.add_node(loop)
    s = loop.add_state("loop_body", is_start_block=True)
    a = s.add_access("A")
    b = s.add_access("B")
    s.add_edge(a, None, b, None, Memlet("A[i] -> B[i]"))
    loop.add_state_after(s, assignments={"i": "i // 2"})
    sdfg.validate()

    # Should not apply
    assert loop.normalize() == False


def test_normalize_nonlin_step():
    """
    Tests if loop normalization works correctly with a non-linear step.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [32], dace.float32)
    sdfg.add_array("B", [32], dace.float32)

    loop = LoopRegion("loop", f"i < 32", "i", f"i = 0", f"i = i * i")
    sdfg.add_node(loop)
    s = loop.add_state("loop_body", is_start_block=True)
    a = s.add_access("A")
    b = s.add_access("B")
    s.add_edge(a, None, b, None, Memlet("A[i] -> B[i]"))
    sdfg.validate()

    # Should not apply
    assert loop.normalize() == False


def test_normalize_altered_step():
    """
    Tests if loop normalization works correctly with an altered step.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [32], dace.float32)
    sdfg.add_array("B", [32], dace.float32)
    sdfg.add_symbol("step", dace.int32)

    loop = LoopRegion("loop", f"i < 32", "i", f"i = 0", f"i = i + step")
    sdfg.add_node(loop)
    s = loop.add_state("loop_body", is_start_block=True)
    a = s.add_access("A")
    b = s.add_access("B")
    s.add_edge(a, None, b, None, Memlet("A[i] -> B[i]"))
    loop.add_state_after(s, assignments={"step": "step + 1"})
    sdfg.validate()

    # Should not apply
    assert loop.normalize() == False


def test_inequality():
    """
    Tests if loop normalization works correctly with an inequality condition.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [32], dace.float32)
    sdfg.add_array("B", [32], dace.float32)
    sdfg.add_symbol("step", dace.int32)

    loop = LoopRegion("loop", f"i != 32", "i", f"i = 0", f"i = i + step")
    sdfg.add_node(loop)
    s = loop.add_state("loop_body", is_start_block=True)
    a = s.add_access("A")
    b = s.add_access("B")
    s.add_edge(a, None, b, None, Memlet("A[i] -> B[i]"))
    sdfg.validate()

    # Should not apply
    assert loop.normalize() == False


if __name__ == "__main__":
    for b in [0, 1, 2, 4, 8, 16]:
        for s in [1, 2, 4, -1, -2, -4]:
            test_normalize(b, s)
    test_normalize_altered_iter()
    test_normalize_nonlin_step()
    test_normalize_altered_step()
    test_inequality()
