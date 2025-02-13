# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import contextlib
import dace
import numpy as np


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

        def start(self, name: str):
            if name == 'pass':
                self.did_start = True

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


def test_ctxmgr_name_clash():
    
    from context_managers.context_a import my_dace_ctxmgr_program as prog_a
    from context_managers.context_b import my_dace_ctxmgr_program as prog_b

    rng = np.random.default_rng(42)

    def dace_blocker(f):
        return f
    
    @dace_blocker
    def randint():
        return rng.integers(0, 2)

    @dace.program
    def ctxmgr_name_clashing_0():
        i: dace.int64 = randint()
        if i == 0:
            prog_a()
        else:
            prog_b()
        return i

    @dace.program
    def ctxmgr_name_clashing_1():
        i: dace.int64 = randint()
        if i == 0:
            prog_a()
        else:
            prog_b()
        return i
    
    sdfg = ctxmgr_name_clashing_0.to_sdfg()
    
    for i, f in enumerate([ctxmgr_name_clashing_0, ctxmgr_name_clashing_1]):

        if i > 0:
            f.load_precompiled_sdfg(sdfg.build_folder)

        a_count = 0
        b_count = 0
        for _ in range(100):
            res = f()
            if res[0] == 0:
                a_count += 1
            else:
                b_count += 1
        assert a_count > 0 and b_count > 0


if __name__ == '__main__':
    test_context_manager_decorator()
    test_ctxmgr_name_clash()
