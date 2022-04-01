# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
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
            state_fields=node.fields)
        return tasklet


@dace.library.node
class Dummy(dace.sdfg.nodes.LibraryNode):
    """
    The Dummy library node is used to add fields to the DaCe program's state (see code generation) that are strictly
    related to MPI communication (see distributed descriptors `dace/distr_types`: ProcessGrid, Subarray, RedistrArray).
    NOTE: This is a temporary solution until the SDFG API has a method that adds fields to the DaCe program's state
    directly, without a Tasklet or CodeNode.
    """

    # Global properties
    implementations = {
        "MPI": ExpandDummyMPI,
    }
    default_implementation = "MPI"

    fields = dace.properties.ListProperty(default=[], element_type=str)

    def __init__(self, name, fields=[], *args, **kwargs):
        super().__init__(name,
                         *args,
                         outputs={'__out'},
                         **kwargs)
        self.fields = fields

    def validate(self, sdfg, state):
        return
