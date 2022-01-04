# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace


def test():
    s = dace.define_stream()
    S = dace.define_streamarray([2, 2])

    for i in range(6):
        s[0].append(i)
        for j in range(2):
            S[0, j].append(i + j)
            S[1, j].append(i + j * 10)

    results = []
    while len(s[0]):
        results.append(s[0].popleft())
    while len(S[1, 1]):
        results.append(S[1, 1].popleft())

    assert results == [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]


def test_consume_python():
    inputs = [1, 2, 3, 5, 1]
    S = dace.stream(inputs)
    result = []
    for s in dace.consume(S):
        result.append(s)

    assert inputs == list(reversed(result))


if __name__ == "__main__":
    test()
    test_consume_python()
