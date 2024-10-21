import unittest
from typing import Collection

import dace
from dace import subsets


def make_a_range_with_min_elements(min_elems: Collection):
    return subsets.Range([(e, e, 1) for e in min_elems])


class TestOffsetNew(unittest.TestCase):
    def test_range_offset_same_shape(self):
        n, m = dace.symbol('n', dtype=dace.int32, positive=True), dace.symbol('m', dtype=dace.int32, positive=True)
        r0 = subsets.Range([(5, 5 + n - 1, 1), (5, 5 + m - 1, 1)])

        # No offset
        off = [0, 0]
        rExpect = r0
        self.assertEqual(rExpect, r0.offset_by(off, False, [0, 1]))
        self.assertEqual(rExpect, r0.offset_new(make_a_range_with_min_elements(off), False, [0, 1]))
        self.assertEqual(rExpect, r0.offset_by(off, True, [0, 1]))
        self.assertEqual(rExpect, r0.offset_new(make_a_range_with_min_elements(off), True, [0, 1]))

        # Positive offset
        off = [5, 4]
        negative = False
        rExpect = subsets.Range([(10, 10 + n - 1, 1), (9, 9 + m - 1, 1)])
        self.assertEqual(rExpect, r0.offset_by(off, negative, [0, 1]))
        self.assertEqual(rExpect, r0.offset_new(make_a_range_with_min_elements(off), negative, [0, 1]))
        # Only partially
        rExpect = subsets.Range([(9, 9 + m - 1, 1)])
        partInds = [1]
        self.assertEqual(rExpect, r0.offset_by(off, negative, partInds))
        self.assertEqual(rExpect, r0.offset_new(make_a_range_with_min_elements(off), negative, partInds))

        # Negative offset
        off = [5, 4]
        negative = True
        rExpect = subsets.Range([(0, n - 1, 1), (1, 1 + m - 1, 1)])
        self.assertEqual(rExpect, r0.offset_by(off, negative, [0, 1]))
        self.assertEqual(rExpect, r0.offset_new(make_a_range_with_min_elements(off), negative, [0, 1]))

    def test_range_offset_partial_indices(self):
        n, m = dace.symbol('n', dtype=dace.int32, positive=True), dace.symbol('m', dtype=dace.int32, positive=True)
        r0 = subsets.Range([(5, 5 + n - 1, 1), (5, 5 + m - 1, 1)])
        off = [5, 4]

        partInds = [0]
        rExpect = subsets.Range([(10, 10 + n - 1, 1)])
        self.assertEqual(rExpect, r0.offset_by(off, False, partInds))
        self.assertEqual(rExpect, r0.offset_new(make_a_range_with_min_elements(off), False, partInds))

        partInds = [1]
        rExpect = subsets.Range([(9, 9 + m - 1, 1)])
        self.assertEqual(rExpect, r0.offset_by(off, False, partInds))
        self.assertEqual(rExpect, r0.offset_new(make_a_range_with_min_elements(off), False, partInds))

    def test_range_offset_bad_input(self):
        n, m = dace.symbol('n', dtype=dace.int32, positive=True), dace.symbol('m', dtype=dace.int32, positive=True)
        r0 = subsets.Range([(5, 5 + n - 1, 1), (5, 5 + m - 1, 1)])

        # Offset list is too short
        off = [5]
        inds = [1]
        with self.assertRaises(AssertionError):
            r0.offset_by(off, False, inds)
        with self.assertRaises(AssertionError):
            r0.offset_new(make_a_range_with_min_elements(off), False, inds)

        # Index out of bounds.
        off = [5, 4]
        inds = [0, 1001]
        with self.assertRaises(AssertionError):
            r0.offset_by(off, False, inds)
        with self.assertRaises(AssertionError):
            r0.offset_new(make_a_range_with_min_elements(off), False, inds)

    def test_indices_offset_same_shape(self):
        n, m = dace.symbol('n', dtype=dace.int32, positive=True), dace.symbol('m', dtype=dace.int32, positive=True)
        ind0 = subsets.Indices([n, m])

        # No offset
        off = [0, 0]
        indExpect = ind0
        self.assertEqual(indExpect, ind0.offset_by(off, False))
        self.assertEqual(indExpect, ind0.offset_new(make_a_range_with_min_elements(off), False))
        self.assertEqual(indExpect, ind0.offset_by(off, True))
        self.assertEqual(indExpect, ind0.offset_new(make_a_range_with_min_elements(off), True))

        # Positive offset
        off = [5, 4]
        negative = False
        indExpect = subsets.Indices([n + 5, m + 4])
        self.assertEqual(indExpect, ind0.offset_by(off, negative))
        self.assertEqual(indExpect, ind0.offset_new(make_a_range_with_min_elements(off), negative))

        # Negative offset
        off = [5, 4]
        negative = True
        indExpect = subsets.Indices([n - 5, m - 4])
        self.assertEqual(indExpect, ind0.offset_by(off, negative))
        self.assertEqual(indExpect, ind0.offset_new(make_a_range_with_min_elements(off), negative))

    def test_indices_offset_smaller_dims(self):
        n, m = dace.symbol('n', dtype=dace.int32, positive=True), dace.symbol('m', dtype=dace.int32, positive=True)
        ind0 = subsets.Indices([n, m])

        # Offset size too small
        off = [5]
        indExpect = subsets.Indices([n + 5])
        self.assertEqual(indExpect, ind0.offset_by(off, False))
        self.assertEqual(indExpect, ind0.offset_new(make_a_range_with_min_elements(off), False))

    def test_indices_offset_bad_input(self):
        n, m = dace.symbol('n', dtype=dace.int32, positive=True), dace.symbol('m', dtype=dace.int32, positive=True)
        ind0 = subsets.Indices([n, m])

        # Offset size too big
        off = [5, 4, 3]
        with self.assertRaises(AssertionError):
            ind0.offset_by(off, False)
        with self.assertRaises(AssertionError):
            ind0.offset_new(make_a_range_with_min_elements(off), False)


if __name__ == '__main__':
    unittest.main()
