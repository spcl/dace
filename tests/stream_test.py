# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace


def test():
    s = dace.define_stream()
    S = dace.define_streamarray([2, 2])

    for i in range(6):
        s.append(i)
        for j in range(2):
            S[0, j].append(i + j)
            S[1, j].append(i + j * 10)

    results = []
    while len(s):
        results.append(s.popleft())
    while len(S[1, 1]):
        results.append(S[1, 1].popleft())

    assert results == [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]


def test_consume_python():
    inputs = [1, 2, 3, 5, 1]
    S = dace.stream(inputs)
    result = []
    for s, _ in dace.consume(S):
        result.append(s)  # noqa: PERF402

    assert inputs == list(reversed(result))


def test_define_stream_python():
    """Outside of programs, streams are unbounded, whatever their buffer size in programs."""
    for buffer_size in (0, 1, 2):
        s = dace.define_stream(dace.int32, buffer_size)
        s.extend(range(5))
        assert list(s) == [0, 1, 2, 3, 4]
    assert dace.define_streamarray([2, 3], dace.int32).shape == (2, 3)


def test_consume_python_processing_elements():
    """Elements go to the processing elements in turn, and pushes during the loop are consumed."""
    S = dace.define_stream(dace.int32)
    S.append(3)
    seen = []
    for value, pe in dace.consume(S, 2):
        seen.append((value, pe))
        if value > 0:
            S.append(value - 1)
    assert seen == [(3, 0), (2, 1), (1, 0), (0, 1)]


def test_consume_python_condition():
    """A condition stops the loop early; an empty stream stops it even if the condition never holds."""
    S = dace.stream(range(10))
    count = [0]
    for _, _ in dace.consume(S, 4, lambda: count[0] == 3):
        count[0] += 1
    assert count[0] == 3 and len(S) == 7

    S = dace.stream([1, 2])
    assert [v for v, _ in dace.consume(S, 1, lambda: False)] == [2, 1]


if __name__ == "__main__":
    test()
    test_consume_python()
    test_define_stream_python()
    test_consume_python_processing_elements()
    test_consume_python_condition()
