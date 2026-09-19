# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for dace.sdfg.properties module."""

import copy
import unittest
import sympy as sp
import dace


class PropertyTests(unittest.TestCase):
    """Implements unit tests for dace.sdfg.properties.Property class."""

    def test_deep_copied_code_block_keeps_code_and_language_but_owns_its_ast(self):
        python_block = dace.properties.CodeBlock('b = a + 1')
        cpp_block = dace.properties.CodeBlock('b = a + 1;', dace.dtypes.Language.CPP)
        original_text = python_block.as_string

        python_clone, cpp_clone, python_clone_again = copy.deepcopy([python_block, cpp_block, python_block])
        python_clone.code[0].targets[0].id = 'c'

        self.assertEqual(python_block.as_string, original_text)
        self.assertNotEqual(python_clone.as_string, original_text)
        self.assertIs(python_clone_again, python_clone)
        self.assertIsNot(cpp_clone, cpp_block)
        self.assertEqual(cpp_clone.as_string, 'b = a + 1;')
        self.assertEqual(cpp_clone.language, dace.dtypes.Language.CPP)
        self.assertEqual(python_clone.language, dace.dtypes.Language.Python)

    def test_code_blocks_sharing_one_statement_list_still_share_it_after_a_deep_copy(self):
        first = dace.properties.CodeBlock('b = a * 2')
        second = dace.properties.CodeBlock(first)

        first_clone, second_clone = copy.deepcopy([first, second])

        self.assertIs(first_clone.code, second_clone.code)
        self.assertIsNot(first_clone.code, first.code)
        self.assertIsNot(first_clone.code[0], first.code[0])
        self.assertEqual(first_clone.as_string, first.as_string)

    def test_indirect_properties(self):

        m = dace.sdfg.nodes.Map("test_map", ['i', 'j', 'k'],
                                dace.subsets.Range([(0, 10, 1), (0, sp.Symbol("N"), 4), (0, sp.Symbol("M"), None)]))

        entry = dace.sdfg.nodes.MapEntry(m)

        to_string = dace.sdfg.nodes.Map.__properties__["params"].to_string
        from_string = dace.sdfg.nodes.Map.__properties__["params"].from_string

        self.assertTrue(to_string(m.params) == "['i', 'j', 'k']")
        self.assertTrue(to_string(entry.params) == "['i', 'j', 'k']")

        entry.params = from_string("[k, j, i]")

        self.assertTrue(to_string(m.params) == "['k', 'j', 'i']")
        self.assertTrue(to_string(entry.params) == "['k', 'j', 'i']")

    def test_range_property(self):

        m = dace.sdfg.nodes.Map(
            "test_map", [sp.Symbol("i"), sp.Symbol("j"), sp.Symbol("k")],
            dace.subsets.Range([(0, 9, 1), (0, sp.Symbol("N") - 1, 4), (1, sp.Symbol("M") - 1, None)]))

        to_string = dace.sdfg.nodes.Map.__properties__["range"].to_string
        from_string = dace.sdfg.nodes.Map.__properties__["range"].from_string

        self.assertTrue("0:10, 0:N:4, 1:M" in to_string(m.range))

        m.range = from_string("5:105:5, 0:2*N, 0:10*N:N")

        self.assertTrue("5:105:5, 0:2*N, 0:10*N:N" in to_string(m.range))

    def test_reference_property(self):

        from_string = dace.memlet.Memlet.__properties__["data"].from_string

        sdfg = dace.SDFG("test_sdfg")

        state0 = dace.SDFGState("s0", sdfg)
        state1 = dace.SDFGState("s1", sdfg)
        sdfg.add_node(state0)
        sdfg.add_node(state1)

        _, arr0 = sdfg.add_array("arr0", (16, 16), dace.dtypes.float32)
        data0 = dace.sdfg.nodes.AccessNode('arr0')

        state0.add_node(data0)
        sdfg.add_array("arr1", (16, 16), dace.dtypes.float32)
        state0.add_node(dace.sdfg.nodes.AccessNode('arr1'))
        sdfg.add_array("arr2", (16, 16), dace.dtypes.float32)
        state1.add_node(dace.sdfg.nodes.AccessNode('arr2'))

        memlet = dace.memlet.Memlet.simple('arr2', '0:N', num_accesses=1)
        memlet.data = 'arr0'

        self.assertEqual(sdfg.arrays[memlet.data], arr0)


if __name__ == '__main__':
    unittest.main()
