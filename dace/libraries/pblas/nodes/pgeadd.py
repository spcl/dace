# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.pblas import environments
from dace import data as dt, dtypes, memlet as mm, SDFG, SDFGState, symbolic
from dace.frontend.common import op_repository as oprepo


@dace.library.expansion
class ExpandBlockCyclicScatterPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of Block Cyclic Scatter.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise(NotImplementedError)
  
@dace.library.expansion
class ExpandBlockCyclicScatterMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKLScaLAPACK]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):

        in_shape, in_ld, out_shape, out_ld = node.validate(parent_sdfg, parent_state)


        code = (f"const MKL_INT i_zero = 0, i_one = 1, i_negone = -1;\n"
                f"const double  zero = 0.0E+0, one = 1.0E+0;\n"
                f"const char trans = 'N';\n"
                f"MKL_INT iam, nprocs, nprow, npcol, myrow, mycol, info;\n"
                f"nprow = Px; npcol = Px;\n"
                f"blacs_pinfo(&iam, &nprocs);\n"
                f"blacs_get(&i_negone, &i_zero, _context);\n"
                f"blacs_gridinit(_context, \"C\", &nprow, &npcol);\n"
                f"blacs_gridinfo(_context, &nprow, &npcol, &myrow, &mycol);\n"
                f"MKL_INT grows = {in_shape[0]};\n"
                f"MKL_INT gcols = {in_shape[1]};\n"
                f"MKL_INT gld = {in_ld};\n"
                # f"MKL_INT lrows = {out_shape[0]};\n"
                # f"MKL_INT lcols = {out_shape[1]};\n"
                f"MKL_INT lld = {out_ld};\n"
                f"descinit(_gdescriptor, &grows, &gcols, &grows, &gcols, &i_zero, &i_zero, _context, &gld, &info);\n"
                f"descinit(_ldescriptor, &grows, &gcols, &_block_sizes[0], &_block_sizes[1], &i_zero, &i_zero, _context, &lld, &info);\n"
                f"pdgeadd(&trans, &grows, &gcols, &one, _inbuffer, &i_one, &i_one, _gdescriptor, &zero, _outbuffer, &i_one, &i_one, _ldescriptor);")

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.out_connectors
        conn = {c: (dtypes.pointer(dace.int32) if c == '_context' else t) for c, t in conn.items()}
        tasklet.out_connectors = conn
        return tasklet


@dace.library.node
class BlockCyclicScatter(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MKL": ExpandBlockCyclicScatterMKL,
    }
    default_implementation = "MKL"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inbuffer", "_block_sizes"},
                         outputs={"_outbuffer", "_context", "_gdescriptor", "_ldescriptor"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """
        
        inbuffer, block_sizes, outbuffer  = None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
                out_shape = e.data.subset.size_exact()
                out_ld = outbuffer.strides[-2]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
                in_shape = e.data.subset.size_exact()
                in_ld = inbuffer.strides[-2]
            if e.dst_conn == "_block_sizes":
                block_sizes = sdfg.arrays[e.data.data]


        return in_shape, in_ld, out_shape, out_ld


@dace.library.expansion
class ExpandBlockCyclicGatherPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of Block Cyclic Gather.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise(NotImplementedError)
  
@dace.library.expansion
class ExpandBlockCyclicGatherMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKLScaLAPACK]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):

        in_shape, in_ld, out_shape, out_ld = node.validate(parent_sdfg, parent_state)


        code = (f"const MKL_INT i_one = 1;\n"
                f"const double  zero = 0.0E+0, one = 1.0E+0;\n"
                f"const char trans = 'N';\n"
                f"MKL_INT grows = {out_shape[0]};\n"
                f"MKL_INT gcols = {out_shape[1]};\n"
                f"pdgeadd(&trans, &grows, &gcols, &one, _inbuffer, &i_one, &i_one, _ldescriptor, &zero, _outbuffer, &i_one, &i_one, _gdescriptor);")

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        # conn = tasklet.out_connectors
        # conn = {c: (dtypes.pointer(dace.int32) if c == '_context' else t) for c, t in conn.items()}
        # tasklet.out_connectors = conn
        return tasklet


@dace.library.node
class BlockCyclicGather(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MKL": ExpandBlockCyclicGatherMKL,
    }
    default_implementation = "MKL"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inbuffer", "_gdescriptor", "_ldescriptor"},
                         outputs={"_outbuffer"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """
        
        inbuffer, block_sizes, outbuffer  = None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
                out_shape = e.data.subset.size_exact()
                out_ld = outbuffer.strides[-2]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
                in_shape = e.data.subset.size_exact()
                in_ld = inbuffer.strides[-2]


        return in_shape, in_ld, out_shape, out_ld