# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.


def test_ndloop():
    from dace.frontend.python import ndloop

    f1dres = []

    def f1d(ind):
        f1dres.append(ind)

    f1dparamres = []

    def f1dparam(i, p1, p2):
        f1dparamres.append((p1, p2, i))

    f2dres = []

    def f2d(y, x):
        f2dres.append((y, x))

    expected_result = [0, 1]
    ndloop.NDLoop(slice(0, 2, None), f1d)
    assert f1dres == expected_result

    # Using args
    expected_result = [(5, 6, 0), (5, 6, 1), (5, 6, 2)]
    ndloop.NDLoop(slice(0, 3, None), f1dparam, 5, 6)
    assert f1dparamres == expected_result

    # Using kwargs
    f1dparamres = []
    expected_result = [(7, 8, 0), (7, 8, 1), (7, 8, 2)]
    ndloop.NDLoop(slice(0, 3, None), f1dparam, p2=8, p1=7)
    assert f1dparamres == expected_result

    expected_result = [(0, 4), (0, 6), (0, 8), (1, 4), (1, 6), (1, 8), (2, 4), (2, 6), (2, 8)]
    ndloop.NDLoop((slice(0, 3, None), slice(4, 9, 2)), f2d)
    assert f2dres == expected_result


if __name__ == "__main__":
    test_ndloop()
