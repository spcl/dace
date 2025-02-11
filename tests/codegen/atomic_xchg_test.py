# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def xchg(locked: dace.int32[1], output: dace.int32[20]):
    for i in dace.map[0:20]:
        with dace.tasklet:
            l >> locked(-1, lambda old, new: new)
            out >> output[i]

            # Will exchange "locked" with 4, storing the result in "l"
            l = 4

            # Will write out the old value of "locked" into output[i]
            out = l


def test_xchg():
    locked = np.ones([1], dtype=np.int32)
    A = np.zeros([20], dtype=np.int32)

    xchg(locked, A)

    # Validate result
    winner = -1
    for i in range(20):
        if A[i] == 1:
            if winner != -1:
                raise ValueError('More than one thread read 1')
            winner = i
        elif A[i] != 4:
            raise ValueError('Values can be either 1 or 4')
    assert locked[0] == 4
    print('PASS. Winner:', winner)


if __name__ == '__main__':
    test_xchg()
