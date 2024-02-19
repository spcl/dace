# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode


@dace.library.expansion
class ExpandWinFenceMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        window_name = node.window_name
        code = f"""
            MPI_Win_fence(_assertion, __state->{window_name}_window);
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)
        return tasklet


@dace.library.node
class Win_fence(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandWinFenceMPI,
    }
    default_implementation = "MPI"

    window_name = dace.properties.Property(dtype=str, default=None)

    def __init__(self, name, window_name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_assertion"}, outputs={"_out"}, **kwargs)
        self.window_name = window_name
