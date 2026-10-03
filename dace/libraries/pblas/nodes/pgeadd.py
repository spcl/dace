# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.pblas import environments
from dace import dtypes
from dace.libraries.mpi.nodes.node import expanded_input_connectors
from dace.libraries.pblas.nodes.node import scalapack_grid_code
from dace.ordered import OrderedSet


@dace.library.expansion
class ExpandBlockCyclicScatterMKL(ExpandTransformation):

    environments = [environments.intel_mkl_mpich.IntelMKLScaLAPACKMPICH]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):

        rows, cols = node.validate(parent_sdfg, parent_state)

        code = f"""
        {scalapack_grid_code(node, parent_state, True)}
            const double  zero = 0.0E+0, one = 1.0E+0;
            const char trans = 'N';
            MKL_INT grows = {rows};
            MKL_INT gcols = {cols};
            MKL_INT brows = _block_sizes[0];
            MKL_INT bcols = (gcols > 1 ? _block_sizes[1]: 1);
            MKL_INT mloc = numroc( &grows, &brows, &__myprow, &__state->__mkl_int_zero, &__nprow);
            MKL_INT nloc = numroc( &gcols, &bcols, &__mypcol, &__state->__mkl_int_zero, &__npcol);
            MKL_INT gld = gcols;
            MKL_INT lld = mloc;
            MKL_INT info;
            descinit(_gdescriptor, &gcols, &grows, &gcols, &grows, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__ctxt, &gld, &info);
            descinit(_ldescriptor, &grows, &gcols, &brows, &bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__ctxt, &lld, &info);
            if (gcols == 1) {{ pdcopy(&grows, _inbuffer,  &__state->__mkl_int_one, &__state->__mkl_int_one, _gdescriptor, &__state->__mkl_int_one, _outbuffer, &__state->__mkl_int_one, &__state->__mkl_int_one, _ldescriptor, &__state->__mkl_int_one); }}
            else {{ pdtran(&grows, &gcols, &one, _inbuffer,  &__state->__mkl_int_one, &__state->__mkl_int_one, _gdescriptor, &zero, _outbuffer, &__state->__mkl_int_one, &__state->__mkl_int_one, _ldescriptor); }}
        """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        # NOTE: The commented out part does not work properly when expanding
        # from regular BLAS GEMV (somehow `_block_sizes` stays a scalar for the
        # vector input).
        conn = tasklet.in_connectors
        conn = {
            c: (
                dtypes.pointer(dace.int32) if c == '_block_sizes'  # and not isinstance(t, dtypes.pointer)
                else t)
            for c, t in conn.items()
        }
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
                         inputs=OrderedSet(('_inbuffer', '_block_sizes')),
                         outputs=OrderedSet(('_outbuffer', '_gdescriptor', '_ldescriptor')),
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        inbuffer, block_sizes, outbuffer = None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
                out_shape = e.data.subset.size_exact()
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
                in_shape = e.data.subset.size_exact()
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
        raise (NotImplementedError)


@dace.library.expansion
class ExpandBlockCyclicGatherMKL(ExpandTransformation):

    environments = [environments.intel_mkl_mpich.IntelMKLScaLAPACKMPICH]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):

        in_shape, out_shape = node.validate(parent_sdfg, parent_state)

        code = f"""
        {scalapack_grid_code(node, parent_state, True)}
            const double  zero = 0.0E+0, one = 1.0E+0;
            const char trans = 'N';
            MKL_INT grows = {out_shape[0]};
            MKL_INT gcols = {out_shape[1]};
            MKL_INT brows = _block_sizes[0];
            MKL_INT bcols = (gcols > 1 ? _block_sizes[1]: 1);
            MKL_INT mloc = numroc( &grows, &brows, &__myprow, &__state->__mkl_int_zero, &__nprow);
            MKL_INT nloc = numroc( &gcols, &bcols, &__mypcol, &__state->__mkl_int_zero, &__npcol);
            MKL_INT gld = gcols;
            MKL_INT lld = mloc;
            MKL_INT info;
            MKL_INT _gdescriptor[9], _ldescriptor[9];
            descinit(_gdescriptor, &gcols, &grows, &gcols, &grows, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__ctxt, &gld, &info);
            descinit(_ldescriptor, &grows, &gcols, &brows, &bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__ctxt, &lld, &info);
            if (gcols == 1) {{ pdcopy(&grows, _inbuffer,  &__state->__mkl_int_one, &__state->__mkl_int_one, _ldescriptor, &__state->__mkl_int_one, _outbuffer, &__state->__mkl_int_one, &__state->__mkl_int_one, _gdescriptor, &__state->__mkl_int_one); }}
            else {{ pdtran(&gcols, &grows, &one, _inbuffer,  &__state->__mkl_int_one, &__state->__mkl_int_one, _ldescriptor, &zero, _outbuffer, &__state->__mkl_int_one, &__state->__mkl_int_one, _gdescriptor); }}
        """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
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
                         inputs=OrderedSet(('_inbuffer', '_block_sizes')),
                         outputs={"_outbuffer"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        inbuffer, block_sizes, outbuffer = None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
                out_shape = e.data.subset.size_exact()
                if len(out_shape) == 1:
                    out_shape = [out_shape[0], 1]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
                in_shape = e.data.subset.size_exact()
                if len(in_shape) == 1:
                    in_shape = [in_shape[0], 1]

        return in_shape, out_shape
