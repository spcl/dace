import unittest
from functools import reduce
from operator import mul

import dace
from dace import SDFG, Memlet
from dace.codegen.targets import cpp
from dace.subsets import Range
from dace.transformation.dataflow import RedundantArray


class ReshapeStrides(unittest.TestCase):
    def test_multidim_array_all_dims_unit(self):
        r = Range([(0, 0, 1), (0, 0, 1)])

        # To smaller-sized shape
        target_dims = [1]
        self.assertEqual(r.num_elements_exact(), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, [1])

        # To equal-sized shape
        target_dims = [1, 1]
        self.assertEqual(r.num_elements_exact(), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, [1, 1])

        # To larger-sized shape
        target_dims = [1, 1, 1]
        self.assertEqual(r.num_elements_exact(), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, [1, 1, 1])

    def test_multidim_array_some_dims_unit(self):
        r = Range([(0, 1, 1), (0, 0, 1)])

        # To smaller-sized shape
        target_dims = [2]
        self.assertEqual(r.num_elements_exact(), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)

        # To equal-sized shape
        target_dims = [2, 1]
        self.assertEqual(r.num_elements_exact(), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)

        # To equal-sized shape
        target_dims = [2, 1, 1]
        self.assertEqual(r.num_elements_exact(), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)

    def test_multidim_array_different_shape(self):
        r = Range([(0, 4, 1), (0, 5, 1)])

        # To smaller-sized shape
        target_dims = [30]
        self.assertEqual(r.num_elements_exact(), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)

        # To equal-sized shape
        target_dims = [15, 2]
        self.assertEqual(r.num_elements_exact(), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)

        # To equal-sized shape
        target_dims = [3, 5, 2]
        self.assertEqual(r.num_elements_exact(), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)


class RedundantArrayCrashesCodegenTest(unittest.TestCase):
    """
    This test demonstrates the bug in CPP Codegen that the [PR](https://github.com/spcl/dace/pull/1692) fixes.
    """
    @staticmethod
    def original_graph_with_redundant_array():
        g = SDFG('prog')
        g.add_array('A', (5, 5), dace.float32)
        g.add_array('b', (1,), dace.float32, transient=True)
        g.add_array('c', (5, 5), dace.float32, transient=True)

        st0 = g.add_state('st0', is_start_block=True)
        st = st0

        # Make a single map that copies A[i, j] to a transient "scalar" b, then copies that out to a transient array
        # c[i, j], then finally back to A[i, j] again.
        A = st.add_access('A')
        en, ex = st.add_map('m0', {'i': '0:1', 'j': '0:1'})
        st.add_memlet_path(A, en, dst_conn='IN_A', memlet=Memlet(expr='A[0:1, 0:1]'))
        b = st.add_access('b')
        st.add_memlet_path(en, b, src_conn='OUT_A', memlet=Memlet(expr='A[i, j] -> b[0]'))
        c = st.add_access('c')
        st.add_memlet_path(b, c, memlet=Memlet(expr='b[0] -> c[i, j]'))
        st.add_memlet_path(c, ex, dst_conn='IN_A', memlet=Memlet(expr='c[i, j] -> A[i, j]'))
        A = st.add_access('A')
        st.add_memlet_path(ex, A, src_conn='OUT_A', memlet=Memlet(expr='A[0:1, 0:1]'))
        st0.fill_scope_connectors()

        g.validate()
        g.compile()
        return g

    def test_removal(self):
        g = self.original_graph_with_redundant_array()
        g.apply_transformations(RedundantArray)
        g.validate()
        g.compile()

        # TODO: The produced graph still has bug. So, let's test for its existence.
        assert len(g.states()) == 1
        st = g.states()[0]
        assert len(st.source_nodes()) == 1
        src = st.source_nodes()[0]
        assert len(st.out_edges(src)) == 1
        e = st.out_edges(src)[0]
        # This is the wrong part. These symbols are not available in this scope.
        assert e.data.free_symbols == {'i', 'j'}


if __name__ == '__main__':
    unittest.main()
