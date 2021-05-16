# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


def assert_exists(name):
    dace.library.get_library(name)


def assert_not_exists(name):
    raised = False
    try:
        dace.library.get_library(name)
    except:
        raised = True
        pass
    if not raised:
        raise RuntimeError("Library " + name + " exists.")


def test_include():
    assert_not_exists("FooLib")
    assert_not_exists("BarLib")

    import foolib  # foolib depends on barlib

    assert_exists("FooLib")
    assert_exists("BarLib")


if __name__ == '__main__':
    test_include()
