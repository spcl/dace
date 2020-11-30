# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace.symbolic import symstr
from dace.properties import Property
import dace.library
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
from .. import environments
import numpy as np

from dace.libraries.blas.utility.fpga_helper import StreamWriteVector, StreamReadVector
from dace.libraries.blas.utility.fpga_helper import streamReadMatrixFull
from dace.libraries.blas.utility.reductions import fpga_make_matrixPartialReduction


@dace.library.expansion
class ExpandGemvPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_a, outer_array_a, shape_a, strides_a), (edge_x, outer_array_x,
                                                       shape_x, _),
         _) = _get_matmul_operands(node,
                                   parent_state,
                                   parent_sdfg,
                                   name_lhs="_a",
                                   name_rhs="_x",
                                   name_out="_y")

        dtype_a = outer_array_a.dtype.type
        dtype_x = outer_array_x.dtype.type
        dtype_y = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_a, dtype_x).type]

        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if trans_shape_a[1] != shape_x[0]:
            raise SyntaxError(
                "Matrix-vector product size mismatch: {} vs. {}".format(
                    trans_shape_a[1], shape_x[0]))

        N, M = trans_shape_a[0], trans_shape_a[1]
        shape_y = (N, )

        if outer_array_a.storage != outer_array_x.storage:
            raise ValueError("Input matrices must have same storage")
        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_a",
                                    shape_a,
                                    dtype_a,
                                    strides=strides_a,
                                    storage=storage)
        _, array_x = sdfg.add_array("_x", shape_x, dtype_x, storage=storage)
        _, array_y = sdfg.add_array("_y", shape_y, dtype_y, storage=storage)

        if node.alpha == 1.0:
            mul_program = "__out = __a * __x"
        else:
            mul_program = "__out = {} * __a * __x".format(
                _cast_to_dtype_str(node.alpha, dtype_a))

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        if node.beta == 0:
            mul_out, mul_out_array = "_y", array_y
            output_nodes = None
        else:
            mul_out, mul_out_array = tmp, array_tmp = sdfg.add_temp_transient(
                shape_y, dtype_y, storage=storage)

            access_tmp = state.add_read(tmp)
            output_nodes = {mul_out: access_tmp}

        # Initialization map
        init_state.add_mapped_tasklet(
            "gemv_init",
            {"_o%d" % i: "0:%s" % symstr(d)
             for i, d in enumerate(shape_y)}, {},
            "out = 0", {
                "out":
                dace.Memlet.simple(
                    mul_out, ",".join(["_o%d" % i
                                       for i in range(len(shape_y))]))
            },
            external_edges=True)

        # Multiplication map
        state.add_mapped_tasklet(
            "_GEMV_", {"__i%d" % i: "0:%s" % s
                       for i, s in enumerate([N, M])},
            {
                "__a":
                dace.Memlet.simple(
                    "_a", "__i1, __i0" if node.transA else "__i0, __i1"),
                "__x":
                dace.Memlet.simple("_x", "__i1")
            },
            mul_program, {
                "__out":
                dace.Memlet.simple(
                    mul_out, "__i0", wcr_str="lambda x, y: x + y")
            },
            external_edges=True,
            output_nodes=output_nodes)

        if node.beta != 0:
            add_program = "__y_out = ({} * __y_in) + __tmp".format(
                _cast_to_dtype_str(node.beta, dtype_a))

            memlet_idx = "__i"

            # addition map
            state.add_mapped_tasklet(
                "_Add_", {"__i": "0:{}".format(N)}, {
                    "__y_in": dace.Memlet.simple("_y", memlet_idx),
                    "__tmp": dace.Memlet.simple(mul_out, "__i"),
                },
                add_program, {"__y_out": dace.Memlet.simple("_y", "__i")},
                external_edges=True,
                input_nodes={mul_out: access_tmp})

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGemvPure.make_sdfg(node, state, sdfg)




@dace.library.expansion
class ExpandGemvFPGAStreamingRowTiles(ExpandTransformation):

    environments = []


    @staticmethod
    def make_sdfg(
            dtype, 
            nTile,
            mTile,
            partialWidth,
            n, m,
            vecWidthM,
            a, b
        ):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        # A: n rows, m columns, row-major (or transposed column-major)
        gemv_sdfg = dace.SDFG("gemv_fpga_stream_rowTiles")
        gemv_sdfg.add_symbol(a.name, a.dtype)

        if b != 0:
            gemv_sdfg.add_symbol(b.name, b.dtype)

        gemv_state = gemv_sdfg.add_state()

        A_in = gemv_state.add_stream(
            '_A',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = None
        if b != 0:
            y_in = gemv_state.add_stream(
                '_y',
                dtype,
                veclen=1,
                buffer_size=32,
                storage=dtypes.StorageType.FPGA_Local,
                transient=(True if b == 0 else False)
            )

        # Must be received n/nTile times
        x_in = gemv_state.add_stream(
            '_x',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        res = gemv_state.add_stream(
            '_res',
            dtype,
            veclen=1,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )


        # ---------- ----------
        # COMPUTE
        # ---------- ----------

        nMap_entry, nMap_exit = gemv_state.add_map(
            'nTile_map',
            dict(i = '0:{0}/{1}'.format(n, nTile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        yTile_sdfg = Expand_GEMV_FPGA_Streaming_RowTiles.make_yTile(
            dtype,
            nTile,
            mTile,
            tileRowStreamed,
            partialWidth,
            n, m,
            vecWidthM,
            a, b
        )
        nested_sdfg = gemv_state.add_nested_sdfg(
            yTile_sdfg,
            gemv_sdfg,
            {'_A_yTile', '_x_yTile', '_y_yTile'} if b != 0 else {'_A_yTile', '_x_yTile'},
            {'yTile'}
        )

        gemv_state.add_memlet_path(
            A_in, nMap_entry, nested_sdfg,
            dst_conn='_A_yTile',
            memlet=Memlet.simple(A_in.data, "0:{0}*{1}".format(n, m), veclen=vecWidthM)
        )

        gemv_state.add_memlet_path(
            x_in, nMap_entry, nested_sdfg,
            dst_conn='_x_yTile',
            memlet=Memlet.simple(x_in.data, "0:{}".format(m), veclen=vecWidthM)
        )

        if b != 0:
            gemv_state.add_memlet_path(
                y_in, nMap_entry, nested_sdfg,
                dst_conn='_y_yTile',
                memlet=Memlet.simple(y_in.data,"0:{}".format(n))
            ) 

        gemv_state.add_memlet_path(
            nested_sdfg, nMap_exit, res,
            src_conn='yTile',
            memlet=Memlet.simple(res.data, "0:{}".format(n)) #  num_accesses=-1) # num_accesses=int(nTile))
        )

        return gemv_sdfg

        



    @staticmethod
    def make_yTile(dtype, nTile, mTile, tileRowStreamed, partialWidth, n, m, vecWidthM, a, b):

        yTile_sdfg = dace.SDFG("yTile_sdfg")

        init_state = yTile_sdfg.add_state('yTile_init')
        compute_state = yTile_sdfg.add_state('yTile_compute')

        yTile_sdfg.add_symbol(a.name, a.dtype)
        if b != 0:
            yTile_sdfg.add_symbol(b.name, b.dtype)

        A_in = compute_state.add_stream(
            '_A_yTile',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = None
        if b != 0:
            y_in = compute_state.add_stream(
                '_y_yTile',
                dtype,
                veclen=1,
                buffer_size=32,
                storage=dtypes.StorageType.FPGA_Local
            )

        x_in = compute_state.add_stream(
            '_x_yTile',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        yTile_sdfg.add_array('y_tileRes', shape=[nTile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)

        data_out = compute_state.add_stream(
            'yTile',
            dtype,
            veclen=1,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )


        # ---------- ----------
        # INIT State
        # # ---------- ----------
        fpga_initArray(
            init_state,
            'y_tileRes',
            nTile,
            0,
            unroll=True
        )


        # ---------- ----------
        # Compute
        # ---------- ----------
        mMap_entry, mMap_exit = compute_state.add_map(
            'mTile_map',
            dict(j = '0:{0}/{1}'.format(m, mTile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        outerComputeMap_entry, outerComputeMap_exit = compute_state.add_map(
            'outerCompute_map',
            dict(ii = '0:{}'.format(nTile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        y_out = compute_state.add_write('y_tileRes')
        
        reducedTile_sdfg = fpga_make_matrixPartialReduction(
            dtype,
            nTile,
            mTile,
            partialWidth,
            n, m,
            vecWidthM,
            a, b
        )


        nested_sdfg = compute_state.add_nested_sdfg(
            reducedTile_sdfg,
            yTile_sdfg,
            {'_A_red', '_x_red', '_y_stream_red'} if b != 0 else {'_A_red', '_x_red'},
            {'_y_red', '_res_red'}
        )


        compute_state.add_memlet_path(
            A_in, mMap_entry, outerComputeMap_entry, nested_sdfg,
            dst_conn='_A_red',
            memlet=Memlet.simple(A_in.data, "0:{}*{}".format(n, m), veclen=vecWidthM)
        )

        compute_state.add_memlet_path(
            x_in, mMap_entry, outerComputeMap_entry, nested_sdfg,
            dst_conn='_x_red',
            memlet=Memlet.simple(x_in.data, "0:{}".format(m), veclen=vecWidthM)
        )

        if b != 0:
            compute_state.add_memlet_path(
                y_in, mMap_entry, outerComputeMap_entry, nested_sdfg,
                dst_conn='_y_stream_red',
                memlet=Memlet.simple(y_in.data, "0:{}".format(n))
            )

        compute_state.add_memlet_path(
            nested_sdfg, outerComputeMap_exit, mMap_exit, y_out,
            src_conn='_y_red',
            memlet=Memlet.simple(y_out.data, "0:{}".format(nTile))
        )

        compute_state.add_memlet_path(
            nested_sdfg, outerComputeMap_exit, mMap_exit, data_out,
            src_conn='_res_red',
            memlet=Memlet.simple(data_out.data, "0:{}".format(n))
        )



        yTile_sdfg.fill_scope_connectors()
        yTile_sdfg.add_edge(init_state, compute_state, dace.InterstateEdge(None))

        return yTile_sdfg


    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGemvFPGAStreamingRowTiles.make_sdfg(
            node.dtype,
            int(node.nTile),
            int(node.mTile),
            node.partialWidth,
            node.n,
            node.m,
            int(node.vecWidthM),
            node.alpha,
            node.beta
        )


@dace.library.node
class Gemv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGemvPure,,
        "fpga_stream": ExpandGemvFPGAStreamingRowTiles
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)
    transA = Property(dtype=bool,
                      desc="Whether to transpose A before multiplying")
    alpha = Property(
        dtype=tuple(dace.dtypes._CONSTANT_TYPES),
        default=1,
        desc="A scalar which will be multiplied with A @ x before adding y")
    beta = Property(dtype=tuple(dace.dtypes._CONSTANT_TYPES),
                    default=1,
                    desc="A scalar which will be multiplied with y")


    # FPGA
    nTile = dace.properties.SymbolicProperty(allow_none=False, default=1)
    mTile = dace.properties.SymbolicProperty(allow_none=False, default=1)
    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"))
    m = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("m"))

    vecWidthM = dace.properties.SymbolicProperty(allow_none=False, default=1)
    partialWidth = dace.properties.SymbolicProperty(default=1, allow_none=False)


    def __init__(self,
                 name,
                 dtype=None,
                 location=None,
                 transA=False,
                 alpha=1,
                 beta=0,
                 nTile=1,
                 mTile=1,
                 partialWidth=2,
                 n=dace.symbolic.symbol("n"),
                 m=dace.symbolic.symbol("m"),
                 vecWidthM=1
                 ):

        # FPGA ???
        # input_cons = {'_A', '_x'}
        # if b != 0:
        #     input_cons = {"_A", "_x", "_y"}

        super().__init__(name,
                         location=location,
                         inputs={"_a", "_x"},
                         outputs={"_y"})

        self.dtype = dtype
        self.transA = transA
        self.alpha = alpha
        self.beta = beta

        # FPGA
        self.n = n
        self.m = m
        self.nTile = nTile
        self.mTile = mTile
        self.vecWidthM = vecWidthM
        self.partialWidth = partialWidth

        

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2, 3]:
            raise ValueError("Expected 2 or 3 inputs to GEMV")
        size_y_in = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_a":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a = subset.size()
            if dst_conn == "_x":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_x = subset.size()
            if dst_conn == "_y":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_y_in = subset.size()

        if self.transA:
            size_a = list(reversed(size_a))

        if len(size_a) != 2 or len(size_x) != 1:
            raise ValueError(
                "Matrix-vector product only supported on matrix-vector input")

        if size_a[1] != size_x[0]:
            raise ValueError("Inputs to matrix-matrix product "
                             "must agree in the k-dimension")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from matrix-vector product")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_y_out = out_subset.size()
        if size_y_in is not None and size_y_in != size_y_out:
            raise ValueError("Input y-vector must match output y-vector.")
        if (len(size_y_out) != 1 or size_y_out[0] != size_a[0]):
            raise ValueError("Vector input to GEMV must match matrix rows.")

    def compare(self, other):

        if (self.dtype == other.dtype and self.vecWidthM == other.vecWidthM
            and self.implementation == other.implementation
            and self.nTile == other.nTile and self.mTile == other.mTile):

            return True
        else:
            return False

    # Implementation dependent, extend if more implementations added
    def streamProductionLatency(self):

        return (self.m / self.mTile - 1) * self.mTile * self.nTile + self.mTile

    def streamConsumptionLatency(self):

        return {
            "_x": self.nTile,
            "_y": (self.m / self.mTile - 1) * self.mTile * self.nTile + self.mTile,
            "_A": 0
        }


    def getStreamReader(self):
        
        return {
            "_x" : streamReadVector(
                '-',
                self.m,
                self.dtype,
                vecWidth=int(self.vecWidthM),
                repeat='{}/{}'.format(self.n, self.nTile)
            ),
            "_y" : streamReadVector(
                '-',
                self.n,
                self.dtype
            ),
            "_A" : streamReadMatrixFull(
                '-',
                self.n,
                self.m,
                self.nTile,
                self.mTile,
                self.dtype,
                tileByRow=True,
                vecWidth=int(self.vecWidthM)
            )

        }

    def getStreamWriter(self):
        
        return {
            "_res" : streamWriteVector(
                '-',
                self.n,
                self.dtype
            )
        }