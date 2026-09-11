# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation.interstate import TrivialLoopElimination
import unittest
import numpy as np

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

        sdfg = dace.SDFG('zero_trip')
        sdfg.add_array('data', [4], dace.float64)
        loop = LoopRegion('l', 'i < 0', 'i', 'i = 0', 'i = i + 1')
        sdfg.add_node(loop, is_start_block=True)
        body = loop.add_state('body')
        tasklet = body.add_tasklet('w', set(), {'o'}, 'o = 1.0')
        body.add_edge(tasklet, 'o', body.add_write('data'), None, dace.Memlet('data[0]'))
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
        """A loop body can hold a control-flow region, and the splice reparents it into the parent.

        That reparenting calls ``reset_cfg_list``, which renumbers every region -- so ``apply`` must
        bind the matched loop ONCE up front instead of re-resolving ``self.loop`` as
        ``cfg_list[cfg_id].node(node_id)`` after it. Re-resolving landed in a different region and
        raised ``NodeNotFoundError`` on warpx_boris_push and bfs, whose single-trip loops sit inside
        an outer sweep and carry a region in the body.
        """
        n = 8
        sdfg = dace.SDFG('splice_body_region')
        sdfg.add_array('a', [n], dace.float64)
        trivial = LoopRegion('L',
                             initialize_expr='i = 0',
                             condition_expr='i < 1',
                             update_expr='i = i + 1',
                             loop_var='i')
        sdfg.add_node(trivial, is_start_block=True)
        head = trivial.add_state('head', is_start_block=True)
        head.add_edge(head.add_read('a'), None, head.add_write('a'), None, dace.Memlet('a[0]'))
        body_region = ControlFlowRegion('body_region', sdfg=sdfg)
        trivial.add_node(body_region)
        trivial.add_edge(head, body_region, dace.InterstateEdge())
        inner = body_region.add_state('inner', is_start_block=True)
        inner.add_edge(inner.add_read('a'), None, inner.add_write('a'), None, dace.Memlet('a[1]'))
        sdfg.validate()

        self.assertEqual(sdfg.apply_transformations(TrivialLoopElimination), 1)
        sdfg.validate()
        self.assertNotIn(trivial, sdfg.nodes(), 'the spliced loop is gone from its parent')
        self.assertIn(body_region, sdfg.nodes(), 'its body region was reparented into the parent')
        self.assertIs(body_region.parent_graph, sdfg, 'and knows the parent it moved to')


if __name__ == '__main__':
    unittest.main()
