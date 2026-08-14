# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the AccessRanges analysis pass. """
import dace
from dace.transformation.passes.analysis import AccessRanges

N = dace.symbol('N')


def test_simple():

    @dace.program
    def tester(A: dace.float64[N, N], B: dace.float64[20, 20]):
        for i, j in dace.map[0:20, 0:N]:
            A[i, j] = 1

    sdfg = tester.to_sdfg(simplify=True)
    ranges = AccessRanges().apply_pass(sdfg, {})
    assert len(ranges) == 1  # Only one SDFG
    ranges = ranges[0]
    assert len(ranges) == 1  # Only one array is accessed

    # Construct write memlet
    memlet = dace.Memlet('A[0:20, 0:N]')
    memlet._is_data_src = False

    assert ranges['A'] == {memlet}


def test_simple_ranges():

    @dace.program
    def tester(A: dace.float64[N, N], B: dace.float64[20, 20]):
        A[:, :] = 0
        A[1:21, 1:21] = B
        A[0, 0] += 1

    sdfg = tester.to_sdfg(simplify=True)
    ranges = AccessRanges().apply_pass(sdfg, {})
    assert len(ranges) == 1  # Only one SDFG
    ranges = ranges[0]
    assert len(ranges) == 2  # Two arrays are accessed

    assert len(ranges['B']) == 1
    assert next(iter(ranges['B'])).src_subset == dace.subsets.Range([(0, 19, 1), (0, 19, 1)])

    # A copy memlet may name EITHER endpoint as its ``data``, putting the other
    # side in ``other_subset`` -- both spellings describe the same copy, and
    # which one a given frontend emits is not what this pass is about. Compare
    # what each access says about A: the range touched, and in which direction.
    def side_of(memlet: dace.Memlet, name: str):
        if memlet.data == name:
            return str(memlet.subset), not memlet._is_data_src
        return str(memlet.other_subset), memlet._is_data_src

    assert {side_of(memlet, 'A')
            for memlet in ranges['A']} == {
                ('0:N, 0:N', True),
                ('1:21, 1:21', True),
                ('0, 0', False),
                ('0, 0', True),
            }


if __name__ == '__main__':
    test_simple()
    test_simple_ranges()
