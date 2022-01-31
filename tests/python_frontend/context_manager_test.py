# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import contextlib
import dace
import numpy as np
import pytest


def dace_inhibitor(f):
    return f


# Reason: Context manager object is not yet parsed successfully by dace (object requires type hints)
@pytest.mark.skip
def test_context_manager_decorator():
    class Ctx:
        def __init__(self) -> None:
            self.did_start = False
            self.should_pass = False

        @contextlib.contextmanager
        def mgr(self, name: str):
            self.start(name)
            yield
            self.stop()

        @dace_inhibitor
        def start(self, name: str):
            if name == 'pass':
                self.did_start = True

        @dace_inhibitor
        def stop(self):
            if self.did_start:
                self.should_pass = True

    ctx = Ctx()

    @dace.program
    def prog(A: dace.float64[20]):
        with ctx.mgr('pass'):
            A[:] = 0

    A = np.random.rand(20)
    prog(A)
    assert ctx.should_pass


if __name__ == '__main__':
    test_context_manager_decorator()
