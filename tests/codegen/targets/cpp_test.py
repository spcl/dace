import unittest
from functools import reduce
from operator import mul

from dace.codegen.targets import cpp
from dace.subsets import Range


class ReshapeStrides(unittest.TestCase):
    def test_multidim_array_all_dims_unit(self):
        r = Range([(0, 0, 1), (0, 0, 1)])

        # To smaller-sized shape
        target_dims = [1]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, [1])
        self.assertEqual(strides, [1])

        # To equal-sized shape
        target_dims = [1, 1]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, [1, 1])
        self.assertEqual(strides, [1, 1])

        # To larger-sized shape
        target_dims = [1, 1, 1]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, [1, 1, 1])
        self.assertEqual(strides, [1, 1, 1])

    def test_multidim_array_some_dims_unit(self):
        r = Range([(0, 1, 1), (0, 0, 1)])

        # To smaller-sized shape
        target_dims = [2]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [1])

        # To equal-sized shape
        target_dims = [2, 1]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [1, 1])
        # To equal-sized shape, but units first.
        target_dims = [1, 2]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [2, 1])

        # To larger-sized shape.
        target_dims = [2, 1, 1]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [1, 1, 1])
        # To larger-sized shape, but units first.
        target_dims = [1, 1, 2]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [2, 2, 1])

    def test_multidim_array_different_shape(self):
        r = Range([(0, 4, 1), (0, 5, 1)])

        # To smaller-sized shape
        target_dims = [30]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [1])

        # To equal-sized shape
        target_dims = [15, 2]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [2, 1])

        # To larger-sized shape
        target_dims = [3, 5, 2]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [10, 2, 1])

    def test_from_strided_range(self):
        r = Range([(0, 4, 2), (0, 6, 2)])

        # To smaller-sized shape
        target_dims = [12]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [1])

        # To equal-sized shape
        target_dims = [4, 3]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [3, 1])

        # To larger-sized shape
        target_dims = [2, 3, 2]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [6, 2, 1])

    def test_from_strided_and_offset_range(self):
        r = Range([(10, 14, 2), (10, 16, 2)])

        # To smaller-sized shape
        target_dims = [12]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [1])

        # To equal-sized shape
        target_dims = [4, 3]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [3, 1])

        # To larger-sized shape
        target_dims = [2, 3, 2]
        self.assertEqual(reduce(mul, r.size_exact()), reduce(mul, target_dims))
        reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
        self.assertEqual(reshaped, target_dims)
        self.assertEqual(strides, [6, 2, 1])


if __name__ == '__main__':
    unittest.main()
