# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Test that the simplify argument to to_sdfg is propagated when parsing calls to other dace programs """

import dace


@dace
def nested_prog1(X: dace.int32[2, 2]):
    return X + 1


@dace
def nested_prog2(X: dace.int32[2, 2]):
    return nested_prog1(X + 1)


@dace
def propagate_strict(X: dace.int32[2, 2]):
    return nested_prog2(X + 1)


def test_propagate_strict():
    strict_sdfg = propagate_strict.to_sdfg(simplify=True)
    assert len(list(strict_sdfg.all_sdfgs_recursive())) == 1

    non_strict_sdfg = propagate_strict.to_sdfg(simplify=False)
    assert len(list(non_strict_sdfg.all_sdfgs_recursive())) > 1


if __name__ == "__main__":
    test_propagate_strict()
