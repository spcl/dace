# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A library node with no edges (a barrier between two states, say) expands into a tasklet that stays in the state."""

import dace
from dace import SDFG, SDFGState, dtypes, library, nodes
from dace.transformation.transformation import ExpandTransformation

MARKER = "volatile int isolated_library_node_ran = 1; (void)isolated_library_node_ran;"


@library.expansion
class ExpandIsolated(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: "Isolated", parent_state: SDFGState, parent_sdfg: SDFG) -> nodes.Tasklet:
        return nodes.Tasklet(node.label, {}, {}, MARKER, language=dtypes.Language.CPP, side_effects=True)


@library.node
class Isolated(nodes.LibraryNode):
    implementations = {"tasklet": ExpandIsolated}
    default_implementation = "tasklet"


def test_isolated_library_node_expands_in_place():
    sdfg = dace.SDFG("expand_isolated_library_node")
    state = sdfg.add_state()
    state.add_node(Isolated("isolated"))

    sdfg.expand_library_nodes()
    tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]
    assert len(tasklets) == 1 and tasklets[0].code.as_string == MARKER
    assert MARKER in sdfg.generate_code()[0].clean_code
    sdfg.compile()
