import dace
import numpy as np


def global_list_test():
    axes = [0, 2, 1]
    @dace
    def global_list(A: dace.int32[3, 2, 4]):
        return np.transpose(A, axes=axes)

    inp = np.random.randint(0, 10, (3, 2, 4)).astype(np.int32)
    result = global_list(A=inp.copy())
    assert np.allclose(result, np.transpose(inp.copy(), axes=axes))


if __name__ == "__main__":
    global_list_test()

