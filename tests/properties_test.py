"""Unit tests for dace.graph.properties module."""

import unittest
import sympy as sp
import dace
from collections import OrderedDict


class PropertyTests(unittest.TestCase):
    """Implements unit tests for dace.graph.properties.Property class."""
    def test_indirect_properties(self):

        m = dace.graph.nodes.Map(
            "test_map", ['i', 'j', 'k'],
            dace.subsets.Range([(0, 10, 1), (0, sp.Symbol("N"), 4),
                                (0, sp.Symbol("M"), None)]))

        entry = dace.graph.nodes.MapEntry(m)

        to_string = dace.graph.nodes.Map.__properties__["params"].to_string
        from_string = dace.graph.nodes.Map.__properties__["params"].from_string

        self.assertTrue(to_string(m.params) == "['i', 'j', 'k']")
        self.assertTrue(to_string(entry.params) == "['i', 'j', 'k']")

        entry.params = from_string("[k, j, i]")

        self.assertTrue(to_string(m.params) == "['k', 'j', 'i']")
        self.assertTrue(to_string(entry.params) == "['k', 'j', 'i']")

    def test_range_property(self):

        m = dace.graph.nodes.Map(
            "test_map",
            [sp.Symbol("i"), sp.Symbol("j"),
             sp.Symbol("k")],
            dace.subsets.Range([(0, 9, 1), (0, sp.Symbol("N") - 1, 4),
                                (1, sp.Symbol("M") - 1, None)]))

        to_string = dace.graph.nodes.Map.__properties__["range"].to_string
        from_string = dace.graph.nodes.Map.__properties__["range"].from_string

        self.assertTrue("0:10, 0:N:4, 1:M" in to_string(m.range))

        m.range = from_string("5:105:5, 0:2*N, 0:10*N:N")

        self.assertTrue("5:105:5, 0:2*N, 0:10*N:N" in to_string(m.range))

    def test_reference_property(self):

        from_string = dace.memlet.Memlet.__properties__["data"].from_string

        sdfg = dace.SDFG("test_sdfg",
                         OrderedDict([("foo", dace.dtypes.float32)]))

        state0 = dace.SDFGState("s0", sdfg)
        state1 = dace.SDFGState("s1", sdfg)
        sdfg.add_node(state0)
        sdfg.add_node(state1)

        _, arr0 = sdfg.add_array("arr0", (16, 16), dace.dtypes.float32)
        data0 = dace.graph.nodes.AccessNode('arr0')

        state0.add_node(data0)
        sdfg.add_array("arr1", (16, 16), dace.dtypes.float32)
        state0.add_node(dace.graph.nodes.AccessNode('arr1'))
        sdfg.add_array("arr2", (16, 16), dace.dtypes.float32)
        state1.add_node(dace.graph.nodes.AccessNode('arr2'))

        memlet = dace.memlet.Memlet('arr2', 1, "0:N", 1)

        with self.assertRaises(TypeError):
            # Must pass SDFG as second argument
            memlet = dace.memlet.Memlet(
                dace.memlet.Memlet.__properties__["data"].from_string("arr0"),
                None, 1, "i", 1)

        memlet.data = 'arr0'

        self.assertEqual(sdfg.arrays[memlet.data], arr0)


if __name__ == '__main__':
    unittest.main()
