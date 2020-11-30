import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.pattern_matching import ExpandTransformation
from ... import environments

from dace import dtypes
from dace.memlet import Memlet

from dace.libraries.blas.utility.initialization import fpga_init_array
from dace.libraries.blas.utility.memory_operations import fpga_streamToLocal

from dace.libraries.blas.utility.fpga_helper import StreamReadVector
from dace.libraries.blas.utility.fpga_helper import streamReadMatrixFull, streamWriteMatrixFull



@dace.library.expansion
class Expand_GER_Pure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, n, m, a):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        ger_sdfg = dace.SDFG("ger_sdfg")
        ger_state = ger_sdfg.add_state()

        ger_sdfg.add_symbol(a.name, a.dtype)

        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------
        ger_sdfg.add_array('_A', shape=[n*m], dtype=dtype) # Row-major, n: rows, m: cols
        ger_sdfg.add_array('_x', shape=[m], dtype=dtype)
        ger_sdfg.add_array('_y', shape=[n], dtype=dtype)

        ger_sdfg.add_array('_RES', shape=[n*m], dtype=dtype)
        
        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        A_in = ger_state.add_read('_A')
        x_in = ger_state.add_read('_x')
        y_in = ger_state.add_read('_y')

        Res_out = ger_state.add_write('_RES')


        computeMap_entry, computeMap_exit = ger_state.add_map(
            'compute_map',
            dict(i = '0:{}'.format(n), j = '0:{}'.format(m))
        )

        compute_task = ger_state.add_tasklet(
            'compute_task',
            ['A_con', 'x_con', 'y_con'],
            ['outCon'],
            'outCon = A_con + x_con * y_con * {}'.format(a)
        )

        ger_state.add_memlet_path(
            A_in, computeMap_entry, compute_task,
            dst_conn='A_con',
            memlet=Memlet.simple(
                A_in.data, 'i * {} + j'.format(m)
            )
        )

        ger_state.add_memlet_path(
            x_in, computeMap_entry, compute_task,
            dst_conn='x_con',
            memlet=Memlet.simple(
                x_in.data, 'j'
            )
        )

        ger_state.add_memlet_path(
            y_in, computeMap_entry, compute_task,
            dst_conn='y_con',
            memlet=Memlet.simple(
                y_in.data, 'i'
            )
        )


        ger_state.add_memlet_path(
            compute_task, computeMap_exit, Res_out,
            src_conn='outCon',
            memlet=Memlet.simple(
                Res_out.data, 'i * {} + j'.format(m)
            )
        )

        return ger_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) + ".")
        return Expand_GER_Simple.make_sdfg(node.dtype, node.n, node.m, node.a)





@dace.library.expansion
class Expand_GER_FPGA_Streaming_RowTiles(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, nTile, mTile, n, m, vecWidthM, a):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        # A: n rows, m columns, row-major (or transposed column-major)

        ger_sdfg = dace.SDFG("ger_fpga_stream_rowTiles")

        ger_sdfg.add_symbol(a.name, a.dtype)
        ger_state = ger_sdfg.add_state('ger_compute')

        A_in = ger_state.add_stream(
            '_A',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = ger_state.add_stream(
            '_y',
            dtype,
            veclen=1,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        # Must be received n/nTiles times
        x_in = ger_state.add_stream(
            '_x',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        res = ger_state.add_stream(
            '_RES',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )


        ger_sdfg.add_array('y_buf_row', shape=[nTile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)

        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        y_buf = ger_state.add_write('y_buf_row')

        nMap_entry, nMap_exit = ger_state.add_map(
            'nBlock_map',
            dict(i = '0:{}/{}'.format(n, nTile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        mMap_entry, mMap_exit = ger_state.add_map(
            'mTile_map',
            dict(j = '0:{}/{}'.format(m, mTile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        outerComputeMap_entry, outerComputeMap_exit = ger_state.add_map(
            'outerCompute_map',
            dict(ii = '0:{}'.format(nTile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        tile_sdfg = None
        if tileRowStreamed:
            tile_sdfg = Expand_GER_FPGA_Streaming_RowTiles.make_computeTileRowStreamed(
                dtype,
                nTile,
                mTile,
                n, m,
                vecWidthM,
                a
            )
        else:

            raise ValueError("Not supported a.t.m")
            
            tile_sdfg = Expand_GER_FPGA_Streaming_RowTiles.make_computeTileColStreamed(
                dtype, 
                nTile,
                mTile,
                n, m
            )

        nested_sdfg = ger_state.add_nested_sdfg(
            tile_sdfg,
            ger_sdfg,
            {'_A_tile', '_x_tile', '_y_tile'}, 
            {'_A_out_tile', '_y_buf_tile'}
        )

        ger_state.add_memlet_path(
            A_in, nMap_entry, mMap_entry, outerComputeMap_entry, nested_sdfg,
            dst_conn='_A_tile',
            memlet=Memlet.simple(A_in.data, "0:{}*{}".format(n, m), veclen=vecWidthM)
        )

        ger_state.add_memlet_path(
            x_in, nMap_entry, mMap_entry, outerComputeMap_entry, nested_sdfg,
            dst_conn='_x_tile',
            memlet=Memlet.simple(x_in.data, "0:{}".format(m), veclen=vecWidthM)
        )

        ger_state.add_memlet_path(
            y_in, nMap_entry, mMap_entry, outerComputeMap_entry, nested_sdfg,
            dst_conn='_y_tile',
            memlet=Memlet.simple(y_in.data, "0:{}".format(n))
        )

        ger_state.add_memlet_path(
            nested_sdfg, outerComputeMap_exit, mMap_exit, nMap_exit, res,
            src_conn='_A_out_tile',
            memlet=Memlet.simple(res.data, "0:{}*{}".format(n, m), veclen=vecWidthM)
        )

        ger_state.add_memlet_path(
            nested_sdfg, outerComputeMap_exit, mMap_exit, nMap_exit, y_buf,
            src_conn='_y_buf_tile',
            memlet=Memlet.simple(y_buf.data, "0:{}".format(nTile))
        )

        return ger_sdfg



    @staticmethod
    def make_computeTileRowStreamed(dtype, nTile, mTile, n, m, vecWidthM, a):

        tile_sdfg = dace.SDFG("tile_sdfg")
        tile_sdfg.add_symbol(a.name, a.dtype)


        init_state = tile_sdfg.add_state('init_state_tile')
        compute_state = tile_sdfg.add_state('copmute_state_tile')

        read_y_state = tile_sdfg.add_state('read_y_reduceTile')
        read_empty_state =  tile_sdfg.add_state('read_empty_reduceTile')

        A_in = compute_state.add_stream(
            '_A_tile',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        x_in = compute_state.add_stream(
            '_x_tile',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = read_y_state.add_stream(
            '_y_tile',
            dtype,
            veclen=1,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )
        
        tile_sdfg.add_array('_y_buf_tile', shape=[nTile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local)
        tile_sdfg.add_array('y_buf', shape=[1], dtype=dtype, storage=dtypes.StorageType.FPGA_Registers, transient=True)
        tile_sdfg.add_array('x_buf', shape=[mTile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)


        A_out = compute_state.add_stream(
            '_A_out_tile',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )


        # ---------- ----------
        # INIT State
        # ---------- ----------
        y_out = read_y_state.add_write("_y_buf_tile")
        y_buf = read_y_state.add_access("y_buf")

        read_y_state.add_memlet_path(
            y_in, y_buf,
            memlet=Memlet.simple(y_buf.data, "0")
        )

        read_y_state.add_memlet_path(
            y_buf, y_out,
            memlet=Memlet.simple(y_buf.data, "0", other_subset_str="ii")
        )

        tile_sdfg.add_edge(init_state, read_y_state, dace.InterstateEdge("j == 0"))
        tile_sdfg.add_edge(read_y_state, compute_state, dace.InterstateEdge(None))
        tile_sdfg.add_edge(init_state, read_empty_state, dace.InterstateEdge("j != 0"))
        tile_sdfg.add_edge(read_empty_state, compute_state, dace.InterstateEdge(None))



        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        y_in = compute_state.add_read('_y_buf_tile')
        x_buf = compute_state.add_write('x_buf')

        innerComputeMap_entry, innerComputeMap_exit = compute_state.add_map(
            'innerCompute_map',
            dict(jj = '0:{}/{}'.format(mTile, vecWidthM)),
            schedule=dtypes.ScheduleType.FPGA_Device,
        )

        red_sdfg = Expand_GER_FPGA_Streaming_RowTiles.make_unrolledCompute(
            dtype,
            nTile,
            mTile,
            n, m,
            vecWidthM,
            a
        )

        nested_sdfg = compute_state.add_nested_sdfg(
            red_sdfg,
            tile_sdfg,
            {'_A_unroll', '_x_unroll', '_y_unroll'},
            {'_A_out_unroll', '_x_buf_unroll'}
        )

        compute_state.add_memlet_path(
            A_in, innerComputeMap_entry, nested_sdfg,
            dst_conn='_A_unroll',
            memlet=Memlet.simple(A_in.data, "0:{}*{}".format(n, m), veclen=vecWidthM)
        )

        compute_state.add_memlet_path(
            x_in, innerComputeMap_entry, nested_sdfg,
            dst_conn='_x_unroll',
            memlet=Memlet.simple(x_in.data, "0:{}".format(m), veclen=vecWidthM)
        )

        compute_state.add_memlet_path(
            y_in, innerComputeMap_entry, nested_sdfg,
            dst_conn='_y_unroll',
            memlet=Memlet.simple(y_in.data, "0:{}".format(nTile))
        )

        compute_state.add_memlet_path(
            nested_sdfg, innerComputeMap_exit, A_out,
            src_conn='_A_out_unroll',
            memlet=Memlet.simple(A_out.data, "0:{}*{}".format(n ,m), veclen=vecWidthM)
        )

        compute_state.add_memlet_path(
            nested_sdfg, innerComputeMap_exit, x_buf,
            src_conn='_x_buf_unroll',
            memlet=Memlet.simple(x_buf.data, "0:{}".format(mTile), veclen=vecWidthM)
        )


        return tile_sdfg


    
    @staticmethod
    def make_unrolledCompute(dtype, nTile, mTile, n, m, vecWidthM, a):

        inner_sdfg = dace.SDFG("vectorize_inner_graph")
        inner_sdfg.add_symbol(a.name, a.dtype)

        init_state = inner_sdfg.add_state("init_state")
        compute_state = inner_sdfg.add_state('compute_state')

        read_x_state = inner_sdfg.add_state("readX_state")
        read_empty_state = inner_sdfg.add_state("readEmpty_state")

        stream_out_state = inner_sdfg.add_state('streamOut_state')


        A_in = compute_state.add_stream(
            '_A_unroll',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        x_in = read_x_state.add_stream(
            '_x_unroll',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        A_out = stream_out_state.add_stream(
            '_A_out_unroll',
            dtype,
            veclen=vecWidthM,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        inner_sdfg.add_array('_y_unroll', shape=[nTile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local)
        inner_sdfg.add_array('_x_buf_unroll', shape=[nTile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local)

        inner_sdfg.add_array('A_out_buf', shape=[vecWidthM], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
        inner_sdfg.add_array('A_vec_buf', shape=[vecWidthM], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
        inner_sdfg.add_array('A_buf', shape=[vecWidthM], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
        inner_sdfg.add_array('x_vecbuf', shape=[vecWidthM], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
        inner_sdfg.add_array('x_membuf', shape=[vecWidthM], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)


        data_out = read_x_state.add_write('_x_buf_unroll')

        copyX_task = read_x_state.add_tasklet(
            'streamToLocal_map',
            ['inCon'],
            ['outCon'],
            'outCon = inCon'
        )



        read_x_state.add_memlet_path(
            x_in, copyX_task,
            dst_conn="inCon",
            memlet=Memlet.simple(x_in.data, "0", veclen=vecWidthM)
        )

        read_x_state.add_memlet_path(
            copyX_task, data_out,
            src_conn="outCon",
            memlet=Memlet.simple(data_out.data, "jj * {}".format(vecWidthM), veclen=vecWidthM)
        )

        

        inner_sdfg.add_edge(init_state, read_x_state, dace.InterstateEdge("ii == 0"))
        inner_sdfg.add_edge(read_x_state, compute_state, dace.InterstateEdge(None))   

        inner_sdfg.add_edge(init_state, read_empty_state, dace.InterstateEdge("ii != 0"))
        inner_sdfg.add_edge(read_empty_state, compute_state, dace.InterstateEdge(None))     


        x_in = compute_state.add_read("_x_buf_unroll")
        y_in = compute_state.add_read("_y_unroll")
        A_out_buf = compute_state.add_write("A_out_buf")


        compute_task = compute_state.add_tasklet(
            'compute_task',
            ['A_con', 'x_con', 'y_con'],
            ['outCon'],
            'outCon = A_con + x_con * y_con * {}'.format(a)
        )

        compute_state.add_memlet_path(
            A_in, compute_task,
            dst_conn='A_con',
            memlet=Memlet.simple(A_in.data, "0", veclen=vecWidthM)
        )

        compute_state.add_memlet_path(
            x_in, compute_task,
            dst_conn='x_con',
            memlet=Memlet.simple(x_in.data, "jj * {}".format(vecWidthM), veclen=vecWidthM)
        )

        compute_state.add_memlet_path(
            y_in, compute_task,
            dst_conn='y_con',
            memlet=Memlet.simple(y_in.data, "ii")
        )

        compute_state.add_memlet_path(
            compute_task, A_out_buf,
            src_conn='outCon',
            memlet=Memlet.simple(A_out_buf.data, "0", veclen=vecWidthM)
        )

        # ---------- ----------
        # STREAM RESULT
        # ---------- ---------
        A_out_buf = stream_out_state.add_read('A_out_buf')
        
        stream_out_state.add_memlet_path(
            A_out_buf, A_out,
            memlet=Memlet.simple(A_out.data, "0", veclen=vecWidthM)
        )

        inner_sdfg.add_edge(compute_state, stream_out_state, dace.InterstateEdge(None))

        return inner_sdfg



    @staticmethod
    def make_computeTileColStreamed(dtype, nTile, mTile, n, m):

        tile_sdfg = dace.SDFG("tile_sdfg")

        init_state = tile_sdfg.add_state('init_state_tile')
        compute_state = tile_sdfg.add_state('copmute_state_tile')

        A_in = compute_state.add_stream(
            '_A_tile',
            dtype,
            veclen=1,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        x_in = init_state.add_stream(
            '_x_tile',
            dtype,
            veclen=1,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )
        
        tile_sdfg.add_array('_a_tile', shape=[1], dtype=dtype, storage=dtypes.StorageType.FPGA_Registers)
        tile_sdfg.add_array('_y_tile', shape=[nTile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local)

        tile_sdfg.add_array('x_buf', shape=[mTile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)

        A_out = compute_state.add_stream(
            '_A_out_tile',
            dtype,
            veclen=1,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )


        # ---------- ----------
        # INIT State
        # ---------- ----------
        fpga_streamToLocal(
            init_state,
            x_in,
            'x_buf',
            mTile
        )



        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        x_in = compute_state.add_read('x_buf')
        y_in = compute_state.add_read('_y_tile')
        a_in = compute_state.add_read("_a_tile")

        innerComputeMap_entry, innerComputeMap_exit = compute_state.add_map(
            'outerCompute_map',
            dict(ii = '0:{}'.format(nTile)),
            schedule=dtypes.ScheduleType.FPGA_Device,
            unroll=True
        )


        outerComputeMap_entry, outerComputeMap_exit = compute_state.add_map(
            'innerCompute_map',
            dict(jj = '0:{}'.format(mTile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        compute_task = compute_state.add_tasklet(
            'compute_task',
            ['A_con', 'a_con', 'x_con', 'y_con'],
            ['outCon'],
            'outCon = A_con + x_con * y_con * a_con'
        )

        compute_state.add_memlet_path(
            A_in, outerComputeMap_entry, innerComputeMap_entry, compute_task,
            dst_conn='A_con',
            memlet=Memlet.simple(A_in.data, "0")
        )

        compute_state.add_memlet_path(
            x_in, outerComputeMap_entry, innerComputeMap_entry, compute_task,
            dst_conn='x_con',
            memlet=Memlet.simple(x_in.data, "jj")
        )

        compute_state.add_memlet_path(
            y_in, outerComputeMap_entry, innerComputeMap_entry, compute_task,
            dst_conn='y_con',
            memlet=Memlet.simple(y_in.data, "ii")
        )

        compute_state.add_memlet_path(
            a_in, outerComputeMap_entry, innerComputeMap_entry, compute_task,
            dst_conn='a_con',
            memlet=Memlet.simple(a_in.data, "0")
        )

        compute_state.add_memlet_path(
            compute_task, innerComputeMap_exit, outerComputeMap_exit, A_out,
            src_conn='outCon',
            memlet=Memlet.simple(A_out.data, "0")
        )

        tile_sdfg.add_edge(init_state, compute_state, dace.InterstateEdge(None))

        return tile_sdfg




    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) + ".")
        return Expand_GER_FPGA_Streaming_RowTiles.make_sdfg(
            node.dtype,
            node.nTile,
            node.mTile,
            node.n,
            node.m,
            int(node.vecWidthM),
            node.a
        )







@dace.library.node
class Ger(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": Expand_GER_Pure,
        "fpga_stream_tiledRows": Expand_GER_FPGA_Streaming_RowTiles,
    }
    default_implementation = 'pure'

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    nTile = dace.properties.SymbolicProperty(allow_none=False, default=1)
    mTile = dace.properties.SymbolicProperty(allow_none=False, default=1)

    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"))
    m = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("m"))
    a = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("a"))

    vecWidthM = dace.properties.SymbolicProperty(allow_none=False, default=1)

    def __init__(self, name,
        dtype=dace.float32,
        nTile=1, mTile=1,
        n=dace.symbolic.symbol("n"),
        m=dace.symbolic.symbol("m"),
        a=dace.symbolic.symbol("a"),
        vecWidthM=1,
        *args, **kwargs
        ):
        super().__init__(
            # A: nxm, x: m, y: n
            name, *args, inputs={"_A", "_x", "_y"}, outputs={"_RES"}, **kwargs
        )
        self.dtype = dtype

        self.nTile = nTile
        self.mTile = mTile

        self.vecWidthM = vecWidthM

        self.n = n
        self.m = m
        self.a = a


    def compare(self, other):

        if (self.dtype == other.dtype and self.vecWidthM == other.vecWidthM
            and self.implementation == other.implementation
            and self.nTile == other.nTile and self.mTile == other.mTile):

            return True
        else:
            return False


        # Implementation dependent, extend if more implementations added
    def streamProductionLatency(self):

        return 0

    def streamConsumptionLatency(self):

        return {
            "_x": self.nTile,
            "_y": (self.m / self.mTile - 1) * self.mTile * self.nTile + self.mTile,
            "_A": 0
        }

    def validate(self, sdfg, state):
        
        # PENDING: implement validation
        return True


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
            "_RES" : streamWriteMatrixFull(
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