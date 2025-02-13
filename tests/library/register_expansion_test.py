# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.library
from dace.transformation import transformation as xf
import pytest


@dace.library.node
class MyLibNode(dace.nodes.LibraryNode):
    implementations = {}
    default_implementation = 'pure'

    def __init__(self, name='MyLibNode', **kwargs):
        super().__init__(name=name, **kwargs)


def test_register_expansion():
    sdfg = dace.SDFG('libtest')
    state = sdfg.add_state()
    n = state.add_node(MyLibNode())

    # Expect KeyError as pure expansion not given
    with pytest.raises(KeyError):
        sdfg()

    @dace.library.register_expansion(MyLibNode, 'pure')
    class ExpandMyLibNode(xf.ExpandTransformation):
        environments = []

        @staticmethod
        def expansion(node: MyLibNode, state: dace.SDFGState, sdfg: dace.SDFG, **kwargs):
            return dace.nodes.Tasklet('donothing', code='pass')

    # After registering the expansion, the code should work
    sdfg()


if __name__ == '__main__':
    test_register_expansion()
