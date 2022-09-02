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
    tester(a)

    assert success is True


def test_string_literal_comparison():
    @dace
    def tester():
        return "foo" < "bar"

    assert np.allclose(tester(), False)


@pytest.mark.skip
def test_string_literal():

    @dace
    def tester():
        return 'Hello World!'

    assert tester()[0] == 'Hello World!'

def test_string_callback():
    success = False
    @callback_inhibitor
    def callback(val):
        nonlocal success
        if isinstance(val, str):
            success = True

    @dace
    def tester():
        callback('hi')

    tester()


@pytest.mark.skip
def test_bytes_literal():

    @dace
    def tester():
        return b'Hello World!'

    assert tester()[0] == b'Hello World!'


if __name__ == '__main__':
    test_string_literal_in_callback()
    test_bytes_literal_in_callback()
    test_string_literal_in_callback_2()
    # test_string_literal()
    # test_bytes_literal()
    test_string_literal_comparison()
