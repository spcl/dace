# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.properties import ListProperty
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from .. import environments


@dace.library.expansion
class ExpandDummyMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, *args, **kwargs):
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            inputs={},
            outputs={'__out'},
            code='',
            state_fields=node._fields)
        return tasklet


@dace.library.node
class Dummy(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandDummyMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, fields, *args, **kwargs):
        super().__init__(name,
                         *args,
                         outputs={'__out'},
                         **kwargs)
        self._fields = fields

    def validate(self, sdfg, state):
        return
