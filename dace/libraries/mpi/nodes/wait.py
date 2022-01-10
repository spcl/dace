# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import dtypes


@dace.library.expansion
class ExpandWaitMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        req, status = node.validate(parent_sdfg, parent_state)
        code = f"""
            MPI_Status _s;
            MPI_Wait(_request, &_s);
            _stat_tag = _s.MPI_TAG;
            _stat_source = _s.MPI_SOURCE;
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.in_connectors
        conn = {c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == '_request' else t) for c, t in conn.items()}
        tasklet.in_connectors = conn
        return tasklet


@dace.library.node
class Wait(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandWaitMPI,
    }
    default_implementation = "MPI"

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_request"}, outputs={"_stat_tag", "_stat_source"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: req, status
        """

        req, status = None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_request":
                req = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_status":
                status = sdfg.arrays[e.data.data]

        return req, status


@dace.library.expansion
class ExpandWaitallPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of Waitall.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise (NotImplementedError)


@dace.library.expansion
class ExpandWaitallMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        count = node.validate(parent_sdfg, parent_state)
        code = f"""
            MPI_Status _s[{count}];
            MPI_Waitall({count}, _request, _s);
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.in_connectors
        conn = {c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == '_request' else t) for c, t in conn.items()}
        tasklet.in_connectors = conn
        return tasklet


@dace.library.node
class Waitall(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandWaitallMPI,
    }
    default_implementation = "MPI"

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_request"}, outputs={}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: req, status
        """

        count = None
        for e in state.in_edges(self):
            if e.dst_conn == "_request":
                count = e.data.subset.num_elements()

        if not count:
            raise ValueError("At least 1 request object must be passed to Waitall")

        return count
