# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.subsets import Range, Indices


def test_compose():
    a1 = Range.from_string('0, 0:N, 10:20')
    a2 = Range.from_string('0, 0:N, 5:10')

    a_res = Range.from_string('0, 0:N, 15:20')
    assert a_res == a1.compose(a2)

    b1 = Range.from_string('0,0,0:M,0:N')
    b2 = Range.from_string('0,0,0,0:N')

    b_res = Range.from_string('0,0,0,0:N')
    assert b_res == b1.compose(b2)

    c1 = Range.from_string('0, 0:N, 0:M, 50:100')
    c2 = Range.from_string('0, 0, 0, 20:40')
    c3 = Indices.from_string('0 , 0 , 0 , 0')

    c_res1 = Range.from_string('0, 0, 0, 70:90')
    c_res2 = Indices.from_string('0, 0, 0, 50')
    assert c_res1 == c1.compose(c2)
    assert c_res2 == c1.compose(c3)

    d1 = Range.from_string('i,j,0:N')
    d2 = Indices.from_string('0,0,k')

    d_res = Indices.from_string('i,j,k')
    assert d_res == d1.compose(d2)


if __name__ == '__main__':
    test_compose()
