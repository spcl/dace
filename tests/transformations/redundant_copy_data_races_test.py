# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from dace import nodes


@dace.program
def rw_data_race(A: dace.float32[10, 10]):
    A2 = A.copy()
    A[1:-1, 1:-1] = A2[:-2, :-2] + A2[2:, 2:]


def test_rw_data_race():
    sdfg = rw_data_race.to_sdfg(simplify=True)
    access_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.AccessNode)]
    assert (len(access_nodes) > 2)


if __name__ == "__main__":
    test_rw_data_race()
