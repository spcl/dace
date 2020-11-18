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


    inner_sdfg.add_array(
        'reg_buf',
        shape=[1],
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