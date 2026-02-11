# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
import pytest


def callback_inhibitor(f):
    return f


def test_string_literal_in_callback():
    success = False

    @callback_inhibitor
    def cb(a):
        nonlocal success
        if a == 'a':
            success = True

    @dace
    def tester(a):
        cb('a')

    a = np.random.rand(1)
    with pytest.warns(match="Automatically creating callback"):
        tester(a)

    assert success is True


def test_bytes_literal_in_callback():
    success = False

    @callback_inhibitor
    def cb(a):
        nonlocal success
        if a == b'Hello World!':
            success = True

    @dace
    def tester(a):
        cb(b'Hello World!')

    a = np.random.rand(1)
    with pytest.warns(match="Automatically creating callback"):
        tester(a)

    assert success is True


def test_string_literal_in_callback_2():
    success = False

    @callback_inhibitor
    def cb(a):
        nonlocal success
        if a == "b'Hello World!'":
            success = True

    @dace
    def tester(a):
        cb("b'Hello World!'")

    a = np.random.rand(1)
    with pytest.warns(match="Automatically creating callback"):
        tester(a)

    assert success is True


def test_string_literal_comparison():

    @dace
    def tester():
        return "foo" < "bar"

    assert np.allclose(tester(), False)


@pytest.mark.skip('Syntax is not yet supported')
def test_string_literal():

    @dace
    def tester():
        return 'Hello World!'

    assert tester()[0] == 'Hello World!'


@pytest.mark.skip('Syntax is not yet supported')
def test_bytes_literal():

    @dace
    def tester():
        return b'Hello World!'

    assert tester()[0] == b'Hello World!'


def test_string_literal_in_complex_object():
    success = False

    class HashableObject:

        def __init__(self, q) -> None:
            self.q = q

        def __hash__(self) -> int:
            return hash(('a', self.q))

        def __eq__(self, other: 'HashableObject') -> bool:
            return self.q == other.q

    @callback_inhibitor
    def cb(a, b, c):
        nonlocal success
        if set(a.keys()) == {'hello', 2}:
            if a['hello'] == {'w': 'orld'} and a[2] == 3:
                if b == 4 and c == {'something', HashableObject(9)}:
                    success = True

    @dace
    def tester(a: int):
        cb(a={'hello': {'w': 'orld'}, 2: 3}, b=a, c={'something', HashableObject(9)})

    with pytest.warns(match="Automatically creating callback"):
        tester(4)
    assert success is True


if __name__ == '__main__':
    test_string_literal_in_callback()
    test_bytes_literal_in_callback()
    test_string_literal_in_callback_2()
    # test_string_literal()
    # test_bytes_literal()
    test_string_literal_comparison()
    test_string_literal_in_complex_object()
