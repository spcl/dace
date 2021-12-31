# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import os


@dace.program
def strict_after_load(A: dace.float32[10, 20], B: dace.float32[10, 20]):
    for i, j in dace.map[0:10, 0:20]:
        B[i, j] = A[i, j] + 1


def test():
    sdfg = strict_after_load.to_sdfg(strict=False)
    sdfg.save(os.path.join('_dacegraphs', 'before.sdfg'))
    sdfg = dace.SDFG.from_file(os.path.join('_dacegraphs', 'before.sdfg'))
    sdfg.coarsen_dataflow()
    sdfg.compile()


if __name__ == "__main__":
    test()
