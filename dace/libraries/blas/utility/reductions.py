"""
Various helper functions for reduction operations
"""

from dace.memlet import Memlet
from dace import dtypes
from dace import config
from dace import InterstateEdge
from dace import SDFG
from dace import vector

from dace.libraries.blas.utility.initialization import fpga_init_array



def fpga_binary_compute_partial_reduction(
        sdfg,
        state,
        src_x,
        src_y,
        dest,
        dtype,
        srcSize,
        veclen,
        partial_width,
        computeTasklet,
        reductionLambda='lambda a, b: a + b',
        buffer_size_x=config.Config.get(
            "library", "blas", "fpga", "default_stream_depth"),
        buffer_size_y=config.Config.get(
            "library", "blas", "fpga", "default_stream_depth"),
        vec_type=None
    ):

    """
    Applies an element-wise binary operator and directly reduces
    #veclen elements into a single element by the given
    reduction function
    """

    if not vec_type:
        vec_type=dtype

    x_in = state.add_stream(
        src_x,
        vec_type,
        buffer_size=buffer_size_x,
        storage=dtypes.StorageType.FPGA_Local
    )
    y_in = state.add_stream(
        src_y,
        vec_type,
        buffer_size=buffer_size_y,
        storage=dtypes.StorageType.FPGA_Local
    )
    buf_in = state.add_write(dest)

    outerMap_entry, outerMap_exit = state.add_map(
        'compRed_map',
        dict(i="0:{0}/{1}".format(srcSize, veclen)),
        schedule=dtypes.ScheduleType.FPGA_Device
    )


    # Nested SDFG
    # -----------------------
    inner_sdfg = SDFG("partial_reduction_inner")

    init_state = inner_sdfg.add_state("init_state")
    compute_state = inner_sdfg.add_state('compute_state')
    store_state = inner_sdfg.add_state('store_state')

    inner_sdfg.add_array('buf_x', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
    inner_sdfg.add_array('buf_y', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)

    x_in_inner = init_state.add_stream(
        'inner_x',
        vec_type,
        buffer_size=buffer_size_x,
        storage=dtypes.StorageType.FPGA_Local
    )

    y_in_inner = init_state.add_stream(
        'inner_y',
        vec_type,
        buffer_size=buffer_size_y,
        storage=dtypes.StorageType.FPGA_Local
    )

    inner_sdfg.add_array(
        'partial_out',
        shape=[(max(partial_width, 2))],
        dtype=dtype,
        storage=dtypes.StorageType.FPGA_Local if partial_width > 8 else dtypes.StorageType.FPGA_Registers
    )


    inner_sdfg.add_scalar(
        'reg_buf',
        #shape=[1],
        dtype=dtype,
        storage=dtypes.StorageType.FPGA_Registers,
        transient=True
    )


    # INIT
    # -----------------------
    buf_x = init_state.add_write('buf_x')
    buf_y = init_state.add_write('buf_y')

    init_state.add_memlet_path(
        x_in_inner, buf_x,
        memlet=Memlet.simple(x_in_inner.data, "0")
    )

    init_state.add_memlet_path(
        y_in_inner, buf_y,
        memlet=Memlet.simple(y_in_inner.data, "0")
    )

    fpga_init_array(
        init_state,
        'reg_buf',
        1, 0
    )


    # COMPUTE
    # -----------------------
    x_buf = compute_state.add_read('buf_x')
    y_buf = compute_state.add_read('buf_y')
    reg_buf = compute_state.add_write('reg_buf')


    innerMap_entry, innerMap_exit = compute_state.add_map(
        'comp_red_inner_map',
        dict(j='0:{}'.format(veclen)),
        unroll=True,
        schedule=dtypes.ScheduleType.FPGA_Device
    )

    innerTasklet = compute_state.add_tasklet(
        'comp_red_task',
        ['inCon1', 'inCon2'],
        ['outCon'],
        computeTasklet
    )

    compute_state.add_memlet_path(
        x_buf, innerMap_entry, innerTasklet,
        dst_conn='inCon1',
        memlet=Memlet.simple(
            x_buf.data, 'j',
        )
    )

    compute_state.add_memlet_path(
        y_buf, innerMap_entry, innerTasklet,
        dst_conn='inCon2',
        memlet=Memlet.simple(
            y_buf.data, 'j',
        )
    )

    compute_state.add_memlet_path(
        innerTasklet, innerMap_exit, reg_buf,
        src_conn='outCon',
        memlet=Memlet.simple(
            reg_buf.data, '0',
            wcr_str=reductionLambda,
        )
    )

     # STORE
    # -----------------------
    reg_buf = store_state.add_read('reg_buf')
    partial_in = store_state.add_read('partial_out')
    partial_res = store_state.add_write('partial_out')

    store_task = store_state.add_tasklet(
        'store_out_task',
        ['inCon', 'prevCon'],
        ['outCon'],
        'outCon = inCon + prevCon'
    )

    store_state.add_memlet_path(
        reg_buf, store_task,
        dst_conn='inCon',
        memlet=Memlet.simple(
            reg_buf.data, '0',
        )
    )

    store_state.add_memlet_path(
        partial_in, store_task,
        dst_conn='prevCon',
        memlet=Memlet.simple(
            partial_in.data, '0' if partial_width == 1 else 'i % {0}'.format(partial_width),
        )
    )

    store_state.add_memlet_path(
        store_task, partial_res,
        src_conn='outCon',
        memlet=Memlet.simple(
            partial_res.data, '0' if partial_width == 1 else 'i % {0}'.format(partial_width),
            # wcr_str=reductionLambda,
        )
    )


    inner_sdfg.fill_scope_connectors()
    inner_sdfg.add_edge(init_state, compute_state, InterstateEdge(None))
    inner_sdfg.add_edge(compute_state, store_state, InterstateEdge(None))


    # CONNECT NESTED
    # -----------------------


    nested_sdfg = state.add_nested_sdfg(
        inner_sdfg,
        sdfg,
        {'inner_x', 'inner_y'},
        {'partial_out'}

    )

    state.add_memlet_path(
        x_in, outerMap_entry, nested_sdfg,
        dst_conn='inner_x',
        memlet=Memlet.simple(
            x_in.data, '0:{}'.format(srcSize/veclen)
        )
    )

    state.add_memlet_path(
        y_in, outerMap_entry, nested_sdfg,
        dst_conn='inner_y',
        memlet=Memlet.simple(
            y_in.data, '0:{}'.format(srcSize/veclen)
        )
    )

    state.add_memlet_path(
        nested_sdfg, outerMap_exit, buf_in,
        src_conn='partial_out',
        memlet=Memlet.simple(
            buf_in.data, '0:{}'.format((max(partial_width, 2))),
            # wcr_str=reductionLambda
        )
    )







def fpga_linear_result_reduction(
        state,
        src,
        dest,
        dtype,
        srcSize,
        reductionLambda='lambda a, b: a + b',
        toMem=False
    ):
    """
    Reduces an input array linearly into a scalar
    with the given reduction function
    """

    red_buf = state.add_read(src)
    buf_out = 0
    if toMem:
        buf_out = state.add_write(dest)
    else:
        buf_out = state.add_stream(
            dest,
            dtype,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

    outerMap_entry, outerMap_exit = state.add_map(
        'reduce_map',
        dict(i="0:{0}".format(srcSize)),
        schedule=dtypes.ScheduleType.FPGA_Device,
        unroll=True
    )

    innerTasklet = state.add_tasklet(
        'reduce_task',
        ['inCon'],
        ['outCon'],
        'outCon = inCon'
    )

    state.add_memlet_path(
        red_buf, outerMap_entry, innerTasklet,
        dst_conn='inCon',
        memlet=Memlet.simple(
            red_buf.data, 'i'
        )
    )

    state.add_memlet_path(
        innerTasklet, outerMap_exit, buf_out,
        src_conn='outCon',
        memlet=Memlet.simple(
            buf_out.data, '0',
            wcr_str=reductionLambda,
        )
    )





def fpga_make_matrixPartialReduction(dtype, nTile, mTile, partialWidth, n, m, veclen, a, b):
    """
    Inner reduction graph for GEMV on FPGA
    """

    # Assumes:
    # i: rowBlock index
    # j: colBlock index
    # ii: tileRow index
    # tile row streamed

    redTile_sdfg = SDFG("redTile_sdfg")


    redTile_sdfg.add_symbol(a.name, a.dtype)

    if b != 0:
        redTile_sdfg.add_symbol(b.name, b.dtype)

    init_state = redTile_sdfg.add_state('init_reduceTile')
    compute_state = redTile_sdfg.add_state('compute_reduceTile')
    red_state = redTile_sdfg.add_state('red_reduceTile')
    store_state = redTile_sdfg.add_state('store_reduceTile')

    read_y_state = redTile_sdfg.add_state('read_y_reduceTile')
    read_empty_state =  redTile_sdfg.add_state('read_empty_reduceTile')
    write_y_state = redTile_sdfg.add_state('write_y_reduceTile')
    write_empty_state =  redTile_sdfg.add_state('write_empty_reduceTile')
    end_state = redTile_sdfg.add_state('end_reduceTile')

    vec_type = vector(dtype, veclen)
    singleton_vec = vector(dtype, 1)
    A_in = compute_state.add_stream(
        '_A_red',
        vec_type,
        buffer_size=32,
        storage=dtypes.StorageType.FPGA_Local
    )

    x_in = compute_state.add_stream(
        '_x_red',
        vec_type,
        buffer_size=32,
        storage=dtypes.StorageType.FPGA_Local
    )

    y_in = None
    if b != 0:
        y_in = read_y_state.add_stream(
            '_y_stream_red',
            singleton_vec,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

    y_out_stream = write_y_state.add_stream(
        '_res_red',
        singleton_vec,
        buffer_size=32,
        storage=dtypes.StorageType.FPGA_Local
    )

    redTile_sdfg.add_array('_y_red', shape=[nTile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local)

    redTile_sdfg.add_array('red_buf',
        shape=[max(2, partialWidth)],
        dtype=dtype,
        storage=dtypes.StorageType.FPGA_Local if partialWidth > 8 else dtypes.StorageType.FPGA_Registers,
        transient=True
    )
    #redTile_sdfg.add_array('res_buf', shape=[1], dtype=dtype, storage=dtypes.StorageType.FPGA_Registers, transient=True)
    redTile_sdfg.add_scalar('res_buf', dtype=dtype, storage=dtypes.StorageType.FPGA_Registers, transient=True)
    redTile_sdfg.add_array('x_buf', shape=[mTile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)


    # ---------- ----------
    # Y READ
    # ---------- ----------
    y_out = read_y_state.add_write("_y_red")

    code = 'outCon = inCon * {}'.format(b)
    if b == 0:
        code = 'outCon = 0'

    y_read_tasklet = read_y_state.add_tasklet(
        'read_y_tasklet',
        (['inCon'] if b != 0 else []),
        ['outCon'],
        code
    )

    if b != 0:
        read_y_state.add_memlet_path(
            y_in, y_read_tasklet,
            dst_conn='inCon',
            memlet=Memlet.simple(y_in.data, "0", num_accesses=-1)
        )

    read_y_state.add_memlet_path(
        y_read_tasklet, y_out,
        src_conn='outCon',
        memlet=Memlet.simple(y_out.data, "ii")
    )


    redTile_sdfg.add_edge(init_state, read_y_state, InterstateEdge("j == 0"))
    redTile_sdfg.add_edge(read_y_state, compute_state, InterstateEdge(None))
    redTile_sdfg.add_edge(init_state, read_empty_state, InterstateEdge("j != 0"))
    redTile_sdfg.add_edge(read_empty_state, compute_state, InterstateEdge(None))


    # ---------- ----------
    # INIT
    # ---------- ----------
    fpga_init_array(
        init_state,
        'red_buf',
        partialWidth,
        0
    )

    fpga_init_array(
        init_state,
        'res_buf',
        1,
        0
    )


    # ---------- ----------
    # COMPUTE
    # ---------- ----------
    buf_out = compute_state.add_write('red_buf')
    buf_in = compute_state.add_read('red_buf')
    x_buf = compute_state.add_read('x_buf')

    innerComputeMap_entry, innerComputeMap_exit = compute_state.add_map(
        'innerCompute_map',
        dict(jj = '0:{0}/{1}'.format(mTile, veclen)),
        schedule=dtypes.ScheduleType.FPGA_Device,
    )


    red_sdfg = make_unrolledCompute(
        dtype,
        nTile,
        mTile,
        veclen,
        partialWidth,
        n, m, a
    )

    nested_sdfg = compute_state.add_nested_sdfg(
        red_sdfg,
        redTile_sdfg,
        {'_A_unroll', '_x_unroll', '_buf_in_unroll'},
        {'buf_out', '_x_buf'}
    )

    compute_state.add_memlet_path(
        A_in, innerComputeMap_entry, nested_sdfg,
        dst_conn='_A_unroll',
        memlet=Memlet.simple(A_in.data, "0:{}*{}".format(n, m))#, veclen=veclen)
    )

    compute_state.add_memlet_path(
        x_in, innerComputeMap_entry, nested_sdfg,
        dst_conn='_x_unroll',
        memlet=Memlet.simple(x_in.data, "0:{}".format(m))#, veclen=veclen)
    )

    compute_state.add_memlet_path(
        buf_in, innerComputeMap_entry, nested_sdfg,
        dst_conn='_buf_in_unroll',
        memlet=Memlet.simple(buf_in.data, "0:{}".format(max(2, partialWidth)))
    )

    compute_state.add_memlet_path(
        nested_sdfg, innerComputeMap_exit, buf_out,
        src_conn='buf_out',
        memlet=Memlet.simple(buf_out.data, "0:{}".format(max(2, partialWidth)))
    )

    compute_state.add_memlet_path(
        nested_sdfg, innerComputeMap_exit, x_buf,
        src_conn='_x_buf',
        memlet=Memlet.simple(x_buf.data, "0:{}".format(mTile))#, veclen=veclen)
    )



    # ---------- ----------
    # REDUCE
    # ---------- ----------
    buf_in = red_state.add_read('red_buf')
    buf_res = red_state.add_write('res_buf')

    task, mapEn, mapEx = red_state.add_mapped_tasklet(
        'finalReduction',
        dict(k='0:{}'.format(partialWidth)),
        dict(inCon=Memlet.simple(buf_in.data, 'k')),
        'outCon = inCon',
        dict(
            outCon=Memlet.simple(
                buf_res.data, '0',
                wcr_str='lambda a, b: a + b'
            )
        )
    )

    red_state.add_edge(
        buf_in, None,
        mapEn, None,
        memlet=Memlet.simple(buf_in.data, "0:{}".format(partialWidth))
    )

    red_state.add_edge(
        mapEx, None,
        buf_res, None,
        memlet=Memlet.simple(
            buf_res.data,
            "0",
            wcr_str='lambda a, b: a + b'
        )
    )


    # ---------- ----------
    # STORE
    # ---------- ----------
    res = store_state.add_read('res_buf')
    out = store_state.add_write('_y_red')

    store_task = store_state.add_tasklet(
        'storeRed_task',
        ['inCon'],
        ['outCon'],
        'outCon = inCon'
    )

    store_state.add_memlet_path(
        res, store_task,
        dst_conn='inCon',
        memlet=Memlet.simple(res.data, '0')
    )

    store_state.add_memlet_path(
        store_task, out,
        src_conn='outCon',
        memlet=Memlet.simple(
            out.data,
            "ii",
            wcr_str="lambda a, b: a + b"
        )
    )

    # ---------- ----------
    # Stream out
    # ---------- ----------
    y_in = write_y_state.add_read('_y_red')

    write_y_state.add_memlet_path(
        y_in, y_out_stream,
        memlet=Memlet.simple(y_in.data, "ii", other_subset_str='0')
    )


    redTile_sdfg.add_edge(store_state, write_y_state, InterstateEdge("j == ({0}/{1}) - 1".format(m, mTile)))
    redTile_sdfg.add_edge(write_y_state, end_state, InterstateEdge(None))
    redTile_sdfg.add_edge(store_state, write_empty_state, InterstateEdge("j != ({0}/{1}) - 1".format(m, mTile)))
    redTile_sdfg.add_edge(write_empty_state, end_state, InterstateEdge(None))

    redTile_sdfg.add_edge(compute_state, red_state, InterstateEdge(None))
    redTile_sdfg.add_edge(red_state, store_state, InterstateEdge(None))

    redTile_sdfg.fill_scope_connectors()

    return redTile_sdfg




def make_unrolledCompute(dtype, nTile, mTile, veclen, partialWidth, n, m, a):

    inner_sdfg = SDFG("partial_reduction_inner")

    inner_sdfg.add_symbol(a.name, a.dtype)

    init_state = inner_sdfg.add_state("init_state")
    compute_state = inner_sdfg.add_state('compute_state')
    write_state = inner_sdfg.add_state("write_state")

    read_x_state = inner_sdfg.add_state("readX_state")
    read_empty_state = inner_sdfg.add_state("readEmpty_state")

    vec_type = vector(dtype, veclen)
    singleton_vec = vector(dtype, 1)
    A_in = init_state.add_stream(
        '_A_unroll',
        vec_type,
        buffer_size=32,
        storage=dtypes.StorageType.FPGA_Local
    )

    x_in = read_x_state.add_stream(
        '_x_unroll',
        vec_type,
        buffer_size=32,
        storage=dtypes.StorageType.FPGA_Local
    )

    inner_sdfg.add_array('_buf_in_unroll', shape=[max(2, partialWidth)], dtype=dtype,
        storage=dtypes.StorageType.FPGA_Local if partialWidth > 8 else dtypes.StorageType.FPGA_Registers
    )
    inner_sdfg.add_array('buf_out', shape=[max(2, partialWidth)], dtype=dtype,
        storage=dtypes.StorageType.FPGA_Local if partialWidth > 8 else dtypes.StorageType.FPGA_Registers
    )

    inner_sdfg.add_array('_x_buf', shape=[mTile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local)
    inner_sdfg.add_array('vecBuf_x', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Registers, transient=True)
    inner_sdfg.add_array('memBuf_x', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Registers, transient=True)


    inner_sdfg.add_array('vecBuf_A', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Registers, transient=True)
    inner_sdfg.add_array('memBuf_A', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Registers, transient=True)



    #inner_sdfg.add_array(
    inner_sdfg.add_scalar(
        'buf_reg',
        #shape=[1],
        dtype=dtype,
        storage=dtypes.StorageType.FPGA_Registers,
        transient=True
    )


    # INIT
    # -----------------------
    fpga_init_array(
        init_state,
        'buf_reg',
        1,
        0
    )

    vecBuf_A = init_state.add_access("vecBuf_A")
    memBuf_A = init_state.add_write("memBuf_A")

    copyMap_entry, copyMap_exit = init_state.add_map(
        'streamToLocalA_map',
        dict(k_stream = '0:{0}'.format(veclen)),
        schedule=dtypes.ScheduleType.FPGA_Device,
        unroll=True
    )

    copy_task = init_state.add_tasklet(
        'streamToLocalA_map',
        ['inCon'],
        ['outCon'],
        'outCon = inCon'
    )

    init_state.add_memlet_path(
        A_in, vecBuf_A,
        memlet=Memlet.simple(vecBuf_A.data, "0")#, veclen=veclen)
    )

    init_state.add_memlet_path(
        vecBuf_A, copyMap_entry, copy_task,
        dst_conn='inCon',
        memlet=Memlet.simple(vecBuf_A.data, "k_stream")
    )

    init_state.add_memlet_path(
        copy_task, copyMap_exit, memBuf_A,
        src_conn='outCon',
        memlet=Memlet.simple(memBuf_A.data, "k_stream")
    )




    data_out = read_x_state.add_write('_x_buf')


    copyX_task = read_x_state.add_tasklet(
        'streamToLocal_map',
        ['inCon'],
        ['outCon'],
        'outCon = inCon'
    )


    read_x_state.add_memlet_path(
        x_in, copyX_task,
        dst_conn='inCon',
        memlet=Memlet.simple(x_in.data, "0")#, veclen=veclen)
    )

    read_x_state.add_memlet_path(
        copyX_task, data_out,
        src_conn='outCon',
        memlet=Memlet.simple(data_out.data, "jj * {}".format(veclen))#, veclen=veclen)
    )



    inner_sdfg.add_edge(init_state, read_x_state, InterstateEdge("ii == 0"))
    inner_sdfg.add_edge(read_x_state, compute_state, InterstateEdge(None))

    inner_sdfg.add_edge(init_state, read_empty_state, InterstateEdge("ii != 0"))
    inner_sdfg.add_edge(read_empty_state, compute_state, InterstateEdge(None))


    # COMPUTE
    # -----------------------
    inner_buf_reg = compute_state.add_write('buf_reg')
    x_in = compute_state.add_read('_x_buf')

    vecBuf = compute_state.add_access('vecBuf_x')
    memBuf = compute_state.add_access('memBuf_x')

    copyX_task = compute_state.add_tasklet(
        'streamToLocal_map',
        ['inCon'],
        ['outCon'],
        'outCon = inCon'
    )

    compute_state.add_memlet_path(
        x_in, copyX_task,
        dst_conn="inCon",
        memlet=Memlet.simple(x_in.data, "jj * {}".format(veclen))#, veclen=veclen)
    )

    compute_state.add_memlet_path(
        copyX_task, vecBuf,
        src_conn="outCon",
        memlet=Memlet.simple(vecBuf.data, "0")#, veclen=veclen)
    )

    compute_state.add_memlet_path(
        vecBuf, memBuf,
        memlet=Memlet.simple(memBuf.data, "0")#, veclen=veclen)
    )


    innerMap_entry, innerMap_exit = compute_state.add_map(
        'compRed_innerMap',
        dict(j_inner='0:{}'.format(veclen)),
        unroll=True,
        schedule=dtypes.ScheduleType.FPGA_Device
    )

    innerTasklet = compute_state.add_tasklet(
        'compute_task',
        ['A_con', 'x_con'],
        ['outCon'],
        'outCon = {} * A_con * x_con'.format(a)
    )

    compute_state.add_memlet_path(
        memBuf_A, innerMap_entry, innerTasklet,
        dst_conn='A_con',
        memlet=Memlet.simple(
            memBuf_A.data, 'j_inner'
        )
    )

    compute_state.add_memlet_path(
        memBuf, innerMap_entry, innerTasklet,
        dst_conn='x_con',
        memlet=Memlet.simple(
            memBuf.data, 'j_inner'
        )
    )

    compute_state.add_memlet_path(
        innerTasklet, innerMap_exit, inner_buf_reg,
        src_conn='outCon',
        memlet=Memlet.simple(
            inner_buf_reg.data, '0',
            wcr_str='lambda a, b: a + b',
        )
    )

    # WRITE
    # -----------------------
    inner_buf_reg = write_state.add_read('buf_reg')
    partial_out = write_state.add_write('buf_out')
    partial_in = write_state.add_read('_buf_in_unroll')

    write_task = write_state.add_tasklet(
        'write_out_task',
        ['inCon', 'prevCon'],
        ['outCon'],
        'outCon = prevCon + inCon'
    )

    write_state.add_memlet_path(
        inner_buf_reg, write_task,
        dst_conn='inCon',
        memlet=Memlet.simple(inner_buf_reg.data, '0')
    )

    write_state.add_memlet_path(
        partial_in, write_task,
        dst_conn='prevCon',
        memlet=Memlet.simple(partial_in.data, 'jj % {0}'.format(partialWidth))
    )

    write_state.add_memlet_path(
        write_task, partial_out,
        src_conn='outCon',
        memlet=Memlet.simple(partial_out.data, 'jj % {0}'.format(partialWidth))
    )

    inner_sdfg.fill_scope_connectors()
    inner_sdfg.add_edge(compute_state, write_state, InterstateEdge(None))

    return inner_sdfg
