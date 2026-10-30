# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import unittest

import numpy as np

import dace
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation.interstate import TrivialLoopElimination

I = dace.symbol("I")
J = dace.symbol("J")


@dace.program
def trivial_loop(data: dace.float64[I, J]):
    for i in range(1, 2):
        for j in dace.map[0:J]:
            data[i, j] = data[i, j] + data[i - 1, j]


class TrivialLoopEliminationTest(unittest.TestCase):
    def test_zero_trip_loop_is_deleted(self):
        """A provably zero-trip loop is removed outright -- splicing its body in would fabricate an
        iteration the loop never runs (the body writes ``data[0]``, which must stay untouched)."""
        from dace.sdfg.state import LoopRegion

        sdfg = dace.SDFG("zero_trip")
        sdfg.add_array("data", [4], dace.float64)
        loop = LoopRegion("l", "i < 0", "i", "i = 0", "i = i + 1")
        sdfg.add_node(loop, is_start_block=True)
        body = loop.add_state("body")
        tasklet = body.add_tasklet("w", set(), {"o"}, "o = 1.0")
        body.add_edge(tasklet, "o", body.add_write("data"), None, dace.Memlet("data[0]"))
        sdfg.validate()

        count = sdfg.apply_transformations(TrivialLoopElimination)
        self.assertEqual(count, 1)
        sdfg.validate()
        self.assertEqual(len([r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]), 0)

        arr = np.zeros(4)
        sdfg(data=arr)
        self.assertTrue(np.array_equal(arr, np.zeros(4)))

    def test_semantic_eq(self):
        A1 = np.random.rand(16, 16)
        A2 = np.copy(A1)

        sdfg = trivial_loop.to_sdfg(simplify=False)
        sdfg(A1, I=A1.shape[0], J=A1.shape[1])

        count = sdfg.apply_transformations(TrivialLoopElimination)
        self.assertGreater(count, 0)
        sdfg(A2, I=A1.shape[0], J=A1.shape[1])

        self.assertTrue(np.allclose(A1, A2))

    def test_splicing_a_body_that_holds_a_region(self):
        """apply must bind the matched loop once up front; reset_cfg_list renumbers regions on splice."""
        n = 8
        sdfg = dace.SDFG("splice_body_region")
        sdfg.add_array("a", [n], dace.float64)
        trivial = LoopRegion(
            "L", initialize_expr="i = 0", condition_expr="i < 1", update_expr="i = i + 1", loop_var="i"
        )
        sdfg.add_node(trivial, is_start_block=True)
        head = trivial.add_state("head", is_start_block=True)
        head.add_edge(head.add_read("a"), None, head.add_write("a"), None, dace.Memlet("a[0]"))
        body_region = ControlFlowRegion("body_region", sdfg=sdfg)
        trivial.add_node(body_region)
        trivial.add_edge(head, body_region, dace.InterstateEdge())
        inner = body_region.add_state("inner", is_start_block=True)
        inner.add_edge(inner.add_read("a"), None, inner.add_write("a"), None, dace.Memlet("a[1]"))
        sdfg.validate()

        self.assertEqual(sdfg.apply_transformations(TrivialLoopElimination), 1)
        sdfg.validate()
        self.assertNotIn(trivial, sdfg.nodes(), "the spliced loop is gone from its parent")
        self.assertIn(body_region, sdfg.nodes(), "its body region was reparented into the parent")
        self.assertIs(body_region.parent_graph, sdfg, "and knows the parent it moved to")

    def test_a_transient_sized_by_the_iterator_loses_it(self):
        """A conv2d's one-group loop ``for g in range(1)`` sizes its per-group slice ``N*(g + 1) - N*g``. The
        substitution must reach that descriptor too, or ``g`` stays a free symbol nothing binds."""
        sdfg = dace.SDFG("group_slice")
        sdfg.add_symbol("N", dace.int64)
        sdfg.add_array("a", ["N"], dace.float64)
        sdfg.add_transient("tmp", ["N*(g + 1) - N*g"], dace.float64)
        loop = LoopRegion("G", initialize_expr="g = 0", condition_expr="g < 1", update_expr="g = g + 1", loop_var="g")
        sdfg.add_node(loop, is_start_block=True)
        body = loop.add_state("body", is_start_block=True)
        body.add_nedge(body.add_read("a"), body.add_access("tmp"), dace.Memlet("a[N*g:N*(g + 1)]"))
        body.add_nedge(body.add_access("tmp"), body.add_write("a"), dace.Memlet("tmp[0:N]"))

        self.assertEqual(sdfg.apply_transformations(TrivialLoopElimination), 1)

        self.assertNotIn("g", {str(s) for s in sdfg.free_symbols})
        self.assertEqual(str(sdfg.arrays["tmp"].shape[0]), "N")
        sdfg.validate()

    def test_a_name_another_loop_still_binds_is_left_in_the_descriptors(self):
        """Two sibling loops over ``i`` (before unique iterators): eliminating the trivial one must not
        rewrite a descriptor the other loop's ``i`` sizes."""
        sdfg = dace.SDFG("shared_iterator")
        sdfg.add_array("a", [4], dace.float64)
        sdfg.add_transient("tmp", ["i + 1"], dace.float64)
        trivial = LoopRegion(
            "T", initialize_expr="i = 0", condition_expr="i < 1", update_expr="i = i + 1", loop_var="i"
        )
        other = LoopRegion("O", initialize_expr="i = 0", condition_expr="i < 3", update_expr="i = i + 1", loop_var="i")
        sdfg.add_node(trivial, is_start_block=True)
        sdfg.add_node(other)
        sdfg.add_edge(trivial, other, dace.InterstateEdge())
        for region in (trivial, other):
            state = region.add_state("s", is_start_block=True)
            state.add_nedge(state.add_read("a"), state.add_write("tmp"), dace.Memlet("a[i]"))

        self.assertEqual(sdfg.apply_transformations(TrivialLoopElimination), 1)

        self.assertEqual(str(sdfg.arrays["tmp"].shape[0]), "i + 1")


if __name__ == "__main__":
    unittest.main()
