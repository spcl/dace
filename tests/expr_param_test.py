import dace
import numpy as np

N = dace.symbol('N')
M = N + 1


@dace.program
def exprparam():
    return np.eye(M)


def test_exprparam():
    result = exprparam(N=5)
    assert result.shape[0] == 6
    assert np.allclose(result, np.eye(6))


if __name__ == '__main__':
    test_exprparam()
