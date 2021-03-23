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

        rows, cols = node.validate(parent_sdfg, parent_state)


        code = (f"if (!__state->__mkl_scalapack_grid_init) {{\n"
                f"    __state->__mkl_scalapack_prows = Px;\n"
                f"    __state->__mkl_scalapack_pcols = Py;\n"
                f"    blacs_gridinit(&__state->__mkl_scalapack_context, \"R\", &__state->__mkl_scalapack_prows, &__state->__mkl_scalapack_pcols);\n"
                f"    blacs_gridinfo(&__state->__mkl_scalapack_context, &__state->__mkl_scalapack_prows, &__state->__mkl_scalapack_pcols, &__state->__mkl_scalapack_myprow, &__state->__mkl_scalapack_mypcol);\n"
                f"    __state->__mkl_scalapack_grid_init = true;\n"
                f"}}\n"
                f"const double  zero = 0.0E+0, one = 1.0E+0;\n"
                f"const char trans = 'N';\n"
                f"MKL_INT grows = {rows};\n"
                f"MKL_INT gcols = {cols};\n"
                f"MKL_INT brows = _block_sizes[0];\n"
                f"MKL_INT bcols = (gcols > 1 ? _block_sizes[1]: 1);\n"
                f"MKL_INT mloc = numroc( &grows, &brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);\n"
                f"MKL_INT nloc = numroc( &gcols, &bcols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);\n"
                f"MKL_INT gld = grows;\n"
                f"MKL_INT lld = mloc;\n"
                f"MKL_INT info;\n"
                f"descinit(_gdescriptor, &grows, &gcols, &grows, &gcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &gld, &info);\n"
                f"descinit(_ldescriptor, &grows, &gcols, &brows, &bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &lld, &info);\n"
                f"if (gcols > 1) {{ mkl_dimatcopy('R', 'T', grows, gcols, 1.0, _inbuffer, gcols, grows); }}\n"
                f"pdgeadd(&trans, &grows, &gcols, &one, _inbuffer, &__state->__mkl_int_one, &__state->__mkl_int_one, _gdescriptor, &zero, _outbuffer, &__state->__mkl_int_one, &__state->__mkl_int_one, _ldescriptor);")

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.in_connectors
        conn = {c: (dtypes.pointer(t)
                    if c == '_block_sizes' and not isinstance(t, dtypes.pointer)
                    else t) for c, t in conn.items()}
        tasklet.in_connectors = conn
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
                         outputs={"_outbuffer", "_gdescriptor", "_ldescriptor"},
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
                # out_ld = outbuffer.strides[-2]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
                in_shape = e.data.subset.size_exact()
                # in_ld = inbuffer.strides[-2]
            if e.dst_conn == "_block_sizes":
                block_sizes = sdfg.arrays[e.data.data]

        if len(in_shape) == 2:
            rows = in_shape[0]
            cols = in_shape[1]
        else:
            rows = in_shape[0]
            cols = 1

        return rows, cols


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

        in_shape, out_shape = node.validate(parent_sdfg, parent_state)


        code = (f"if (!__state->__mkl_scalapack_grid_init) {{\n"
                f"    __state->__mkl_scalapack_prows = Px;\n"
                f"    __state->__mkl_scalapack_pcols = Py;\n"
                f"    blacs_gridinit(&__state->__mkl_scalapack_context, \"R\", &__state->__mkl_scalapack_prows, &__state->__mkl_scalapack_pcols);\n"
                f"    blacs_gridinfo(&__state->__mkl_scalapack_context, &__state->__mkl_scalapack_prows, &__state->__mkl_scalapack_pcols, &__state->__mkl_scalapack_myprow, &__state->__mkl_scalapack_mypcol);\n"
                f"    __state->__mkl_scalapack_grid_init = true;\n"
                f"}}\n"
                f"const double  zero = 0.0E+0, one = 1.0E+0;\n"
                f"const char trans = 'N';\n"
                # f"const char trans = 'T';\n"
                f"MKL_INT grows = {out_shape[0]};\n"
                f"MKL_INT gcols = {out_shape[1]};\n"
                f"pdgeadd(&trans, &grows, &gcols, &one, _inbuffer, &__state->__mkl_int_one, &__state->__mkl_int_one, _ldescriptor, &zero, _outbuffer, &__state->__mkl_int_one, &__state->__mkl_int_one, _gdescriptor);\n"
                f"if (gcols > 1) {{ mkl_dimatcopy('R', 'T', gcols, grows, 1.0, _outbuffer, grows, gcols); }}")

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
                if len(out_shape) == 1:
                    out_shape = [out_shape[0], 1]
                # out_ld = outbuffer.strides[-2]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
                in_shape = e.data.subset.size_exact()
                # in_ld = inbuffer.strides[-2]
                if len(in_shape) == 1:
                    in_shape = [in_shape[0], 1]


        return in_shape, out_shape