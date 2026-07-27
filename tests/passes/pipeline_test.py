# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation import pass_pipeline as ppl


@dace.program
def empty():
    pass


class MyPass(ppl.Pass):

    def __init__(self):
        self.applied = 0

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _) -> bool:
        return True

    def apply_pass(self, sdfg, _):
        self.applied += 1
        return 1


def test_simple_pipeline():
    p = MyPass()
    pipe = ppl.Pipeline([p])
    sdfg = empty.to_sdfg()

    pipe.apply_pass(sdfg, {})
    assert p.applied == 1
    result = pipe.apply_pass(sdfg, {})
    assert p.applied == 2
    assert result == {'MyPass': 1}


def test_pipeline_with_dependencies():

    class PassA(MyPass):

        def depends_on(self):
            return [MyPass]

        def apply_pass(self, sdfg, pipeline_results):
            res = super().apply_pass(sdfg, pipeline_results)
            return pipeline_results['MyPass'] + res

    p = PassA()
    pipe = ppl.Pipeline([p])
    sdfg = empty.to_sdfg()

    result = pipe.apply_pass(sdfg, {})
    assert p.applied == 1
    assert result == {'MyPass': 1, 'PassA': 2}


def test_pipeline_modification_rerun():

    class MyAnalysis(MyPass):

        def should_reapply(self, modified: ppl.Modifies) -> bool:
            return modified & ppl.Modifies.Symbols

        def modifies(self) -> ppl.Modifies:
            return ppl.Modifies.Nothing

    class PassA(MyPass):

        def depends_on(self):
            return [MyAnalysis]

        def modifies(self) -> ppl.Modifies:
            return ppl.Modifies.Descriptors

    class PassB(MyPass):

        def depends_on(self):
            return [MyAnalysis]

        def modifies(self) -> ppl.Modifies:
            return ppl.Modifies.Symbols

    class PassC(MyPass):

        def depends_on(self):
            return [MyAnalysis]

        def modifies(self) -> ppl.Modifies:
            return ppl.Modifies.Everything

    an, pa, pb, pc = MyAnalysis(), PassA(), PassB(), PassC()

    pipe = ppl.Pipeline([an, pa, pb, pc])
    sdfg = empty.to_sdfg()

    result = pipe.apply_pass(sdfg, {})
    assert an.applied == 2  # Three dependencies but only need to rerun once
    assert pa.applied == 1
    assert pb.applied == 1
    assert pc.applied == 1
    assert result == {'MyAnalysis': 1, 'PassA': 1, 'PassB': 1, 'PassC': 1}


def test_pipeline_dependency_order():
    # Regression test for deterministic dependency ordering: when a pass returns multiple
    # independent dependencies, they must be applied in the exact order they are listed.
    order = []

    class RecordingPass(ppl.Pass):

        def modifies(self) -> ppl.Modifies:
            return ppl.Modifies.Nothing

        def should_reapply(self, _) -> bool:
            return False

        def apply_pass(self, sdfg, _):
            order.append(type(self).__name__)
            return 1

    class DepA(RecordingPass):
        pass

    class DepB(RecordingPass):
        pass

    class DepC(RecordingPass):
        pass

    class Dependent(RecordingPass):

        def depends_on(self):
            return [DepB, DepC, DepA]

    pipe = ppl.Pipeline([Dependent()])
    sdfg = empty.to_sdfg()

    pipe.apply_pass(sdfg, {})

    # Dependencies run in the listed order, before the dependent pass itself.
    assert order == ['DepB', 'DepC', 'DepA', 'Dependent']


if __name__ == '__main__':
    test_simple_pipeline()
    test_pipeline_with_dependencies()
    test_pipeline_modification_rerun()
    test_pipeline_dependency_order()
