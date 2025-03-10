import dace
from dace.properties import CodeBlock

def generate_systolic_BSP(i, j, gi, gj, gM, gN, tM, tN, tK, M, N, K):
    loop_param = dace.symbol("_c")

    BSP_stride = K

    BSP_init_code_block =  None

    BSP_loop_code_block = CodeBlock(
        code=f"""
{loop_param} = {gi}+{gj}
while {loop_param} <= {gi}+{gj} + {K}/{tK}:
    {loop_param} = {loop_param} + 1
            """,
        language=dace.dtypes.Language.Python
    )

    BSP_compute_code_block = CodeBlock(
        code=f"""
if (({loop_param} > {gi} + {gj}) and ({loop_param} <= {gi} + {gj} + {K}/{tK})):
    pass
""",
        language=dace.dtypes.Language.Python
    )

    BSP_communication_code_block = CodeBlock(
        code=f"""
if (({loop_param} >= {gi} + {gj}) and ({loop_param} < {gi} + {gj} + {K}/{tK})):
    if {gi} == 0:
        local_B [({loop_param+1})%2][:][:] = B[{tK}*{loop_param}-{tK}*({gi+gj}):{tK}*({loop_param+1})-{tK}*({gi+gj})][:]
        s_local_B [{gi}][{gj}][{loop_param}%2][:][:] = local_B [{loop_param}%2][:][:]
    elif {gi} < {gM} - 1:
        s_local_B [{gi}][{gj}][{loop_param}%2][:][:]   = local_B [{loop_param}%2][:][:]
        local_B [({loop_param+1})%2][:][:] = s_local_B [({gi+gM-1})%{gM}][{gj}][{loop_param}%2][:][:]
    elif {gi} == {gM} - 1:
        local_B [({loop_param+1})%2][:][:] = s_local_B [({gi+gM-1})%{gM}][{gj}][{loop_param}%2][:][:]
        
    if {gj} == 0:
        local_A [({loop_param+1})%2][:][:] = A[:][{tK}*{loop_param}-{tK}*({gi+gj}):{tK}*({loop_param+1})-{tK}*({gi+gj})]
        s_local_A [{gi}][{gj}][{loop_param}%2][:][:] = local_A [{loop_param}%2][:][:]
    elif {gj} < {gN} - 1:
        s_local_A [{gi}][{gj}][{loop_param}%2][:][:]   = local_A [{loop_param}%2][:][:]
        local_A [({loop_param+1})%2][:][:] = s_local_A [{gi}][({gj+gN-1})%{gN}][{loop_param}%2][:][:]
    elif {gj} == {gN} - 1:
        local_A [({loop_param+1})%2][:][:] = s_local_A [{gi}][({gj+gN-1})%{gN}][{loop_param}%2][:][:]
""",
        language=dace.dtypes.Language.Python
    )

    BSP_sync = True

    pre_shift_code_block = CodeBlock(
        code=f'''
if (({i} == 0) && ({j} == 0))
{{
    for (int sync_iter = 0; sync_iter < {gi} + {gj}; sync_iter++){{
        flex_global_barrier_xy();
    }}
    if (flex_is_dm_core()) {{
        flex_dma_async_wait_all();
    }}
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
}}                         
        ''',
        language=dace.dtypes.Language.CPP
    )


    post_shift_code_block = CodeBlock(
        code=f'''
if (({i} >= {M} - {gM}*{tM}) && ({j} >= {N} - {gN}*{tN}))
{{
    for (int sync_iter = 0; sync_iter < {gM+gN} - 1 - {gi} - {gj} - 1; sync_iter++){{
        flex_global_barrier_xy();
    }}
    if (flex_is_dm_core()) {{
        flex_dma_async_wait_all();
    }}
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
}}                         
        ''',
        language=dace.dtypes.Language.CPP
    )



    return (pre_shift_code_block, 
            BSP_stride,
            BSP_init_code_block, 
            BSP_loop_code_block, 
            BSP_compute_code_block, 
            BSP_communication_code_block, 
            BSP_sync, 
            post_shift_code_block)


def generate_cannon_BSP(i, j, gi, gj, gM, gN, tM, tN, tK, M, N, K):
    loop_param = dace.symbol("_c")

    if gM != gN:
        raise ValueError("Cannon's algorithm requires gM == gN")

    BSP_stride = tK * gM

    BSP_init_code_block =  CodeBlock(
        code=f"""
local_B[0][:][:] = B[(({gi+gj})%{gM})*{tK}:(({gi+gj})%{gM})*{tK}+{tK}][:] 
local_A[0][:][:] = A[:][(({gi+gj})%{gM})*{tK}:(({gi+gj})%{gM})*{tK}+{tK}]
""",
        language=dace.dtypes.Language.Python
    )

    BSP_loop_code_block = CodeBlock(
        code=f"""
{loop_param} = 0
while {loop_param} < {gM}:
    {loop_param} = {loop_param} + 1
            """,
        language=dace.dtypes.Language.Python
    )

    BSP_compute_code_block = None

    BSP_communication_code_block = CodeBlock(
        code=f"""
s_local_B [{gi}][{gj}][{loop_param}%2][:][:]   = local_B [{loop_param}%2][:][:]
local_B [({loop_param+1})%2][:][:] = s_local_B [({gi+gM-1})%{gM}][{gj}][{loop_param}%2][:][:] 
s_local_A [{gi}][{gj}][{loop_param}%2][:][:]   = local_A [{loop_param}%2][:][:]
local_A [({loop_param+1})%2][:][:] = s_local_A [{gi}][({gj+gN-1})%{gN}][{loop_param}%2][:][:]
""",
        language=dace.dtypes.Language.Python
    )

    BSP_sync = True

    pre_shift_code_block = None

    post_shift_code_block = None


    return (pre_shift_code_block, 
            BSP_stride,
            BSP_init_code_block, 
            BSP_loop_code_block, 
            BSP_compute_code_block, 
            BSP_communication_code_block, 
            BSP_sync, 
            post_shift_code_block)
     

def generate_summa_BSP(i, j, gi, gj, gM, gN, tM, tN, tK, M, N, K):
    loop_param = dace.symbol("_c")

    BSP_stride = K

    BSP_init_code_block =  None

    BSP_loop_code_block = CodeBlock(
        code=f"""
{loop_param} = 0
while {loop_param} <= {K}/{tK}:
    {loop_param} = {loop_param} + 1
            """,
        language=dace.dtypes.Language.Python
    )

    BSP_compute_code_block = CodeBlock(
        code=f"""
if (({loop_param} > 0) and ({loop_param} <= {K}/{tK})):
    pass
""",
        language=dace.dtypes.Language.Python
    )

    BSP_communication_code_block = CodeBlock(
        code=f"""
if (({loop_param} >= 0) and ({loop_param} < {K}/{tK})):
    if {gi} == 0:
        local_B[({loop_param+1})%2][:][:] = B[{tK}*{loop_param}:{tK}*({loop_param+1})][:]
        s_local_B[{gi}:{gi+gM}][{gj}][({loop_param+1})%2][:][:] = local_B[({loop_param+1})%2][:][:]
    elif {gi} <= {gM} - 1:
        local_B[({loop_param+1})%2][:][:] = s_local_B[{gi}][{gj}][({loop_param+1})%2][:][:]
    
        
    if {gj} == 0:
        local_A[({loop_param+1})%2][:][:] = A[:][{tK}*{loop_param}:{tK}*({loop_param+1})]
        s_local_A[{gi}][{gj}:{gj+gN}][({loop_param+1})%2][:][:] = local_A[({loop_param+1})%2][:][:]
    elif {gj} <= {gN} - 1:
        local_A[({loop_param+1})%2][:][:] = s_local_A[{gi}][{gj}][({loop_param+1})%2][:][:]

""",
        language=dace.dtypes.Language.Python
    )

    BSP_sync = True

    pre_shift_code_block = None

    post_shift_code_block = None


    return (pre_shift_code_block, 
            BSP_stride,
            BSP_init_code_block, 
            BSP_loop_code_block, 
            BSP_compute_code_block, 
            BSP_communication_code_block, 
            BSP_sync, 
            post_shift_code_block)

def generate_summa_systolic_BSP(i, j, gi, gj, gM, gN, tM, tN, tK, M, N, K, summa_range=(2,2)):
    (sr_m, sr_n) = summa_range
    if (gM % sr_m != 0) or (gN % sr_n != 0):
        raise ValueError("gM and gN should be divisible by summa_range")

    s_i = gi//sr_m
    s_j = gj//sr_n
    o_i = f"({gi} % {sr_m})"
    o_j = f"({gj} % {sr_n})"

    loop_param = dace.symbol("_c")

    BSP_stride = K

    BSP_init_code_block =  None

    BSP_loop_code_block = CodeBlock(
        code=f"""
{loop_param} = {s_i}+{s_j}
while {loop_param} <= {s_i}+{s_j}+ {K}/{tK}:
    {loop_param} = {loop_param} + 1
            """,
        language=dace.dtypes.Language.Python
    )

    BSP_compute_code_block = CodeBlock(
        code=f"""
if (({loop_param} > {s_i}+{s_j}) and ({loop_param} <= {s_i}+{s_j} + {K}/{tK})):
    pass
""",
        language=dace.dtypes.Language.Python
    )

    BSP_communication_code_block = CodeBlock(
        code=f"""
if (({loop_param} >= {s_i}+{s_j}) and ({loop_param} < {s_i}+{s_j} + {K}/{tK})):
    if {s_i} == 0 and {o_i} == 0:
        local_B[({loop_param+1})%2][:][:] = B[{tK}*{loop_param}-{tK}*({s_i}+{s_j}):{tK}*({loop_param+1})-{tK}*({s_i}+{s_j})][:]
        s_local_B[{gi}:{gi}+{sr_m}][{gj}][({loop_param+1})%2][:][:] = local_B[({loop_param+1})%2][:][:]
    elif {s_i} > 0 and {o_i} == 0:
        local_B [({loop_param+1})%2][:][:] = s_local_B [({gi+gM-1})%{gM}][{gj}][{loop_param}%2][:][:]
        s_local_B[{gi}:{gi}+{sr_m}][{gj}][({loop_param+1})%2][:][:] = local_B[({loop_param+1})%2][:][:]
    elif {o_i} > 0:
        local_B[({loop_param+1})%2][:][:] = s_local_B[{gi}][{gj}][({loop_param+1})%2][:][:]

    if {s_j} == 0 and {o_j} == 0:
        local_A[({loop_param+1})%2][:][:] = A[:][{tK}*{loop_param}-{tK}*({s_i}+{s_j}):{tK}*({loop_param+1})-{tK}*({s_i}+{s_j})]
        s_local_A[{gi}][{gj}:{gj}+{sr_n}][({loop_param+1})%2][:][:] = local_A[({loop_param+1})%2][:][:]
    elif {s_j} > 0 and {o_j} == 0:
        local_A [({loop_param+1})%2][:][:] = s_local_A [{gi}][({gj+gN-1})%{gN}][{loop_param}%2][:][:]
        s_local_A[{gi}][{gj}:{gj}+{sr_n}][({loop_param+1})%2][:][:] = local_A[({loop_param+1})%2][:][:]
    elif {o_j} > 0:
        local_A[({loop_param+1})%2][:][:] = s_local_A[{gi}][{gj}][({loop_param+1})%2][:][:]
""",
        language=dace.dtypes.Language.Python
    )

    BSP_sync = True

    pre_shift_code_block = CodeBlock(
        code=f'''
if (({i} == 0) && ({j} == 0))
{{
    for (int sync_iter = 0; sync_iter < {s_i}+{s_j}; sync_iter++){{
        flex_global_barrier_xy();
    }}
    if (flex_is_dm_core()) {{
        flex_dma_async_wait_all();
    }}
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
}}                         
        ''',
        language=dace.dtypes.Language.CPP
    )


    post_shift_code_block = CodeBlock(
        code=f'''
if (({i} >= {M} - {gM}*{tM}) && ({j} >= {N} - {gN}*{tN}))
{{
    for (int sync_iter = 0; sync_iter < {gM//sr_m+gN//sr_n} - 1 - ({s_i}+{s_j}) - 1; sync_iter++){{
        flex_global_barrier_xy();
    }}
    if (flex_is_dm_core()) {{
        flex_dma_async_wait_all();
    }}
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
}}                         
        ''',
        language=dace.dtypes.Language.CPP
    )



    return (pre_shift_code_block, 
            BSP_stride,
            BSP_init_code_block, 
            BSP_loop_code_block, 
            BSP_compute_code_block, 
            BSP_communication_code_block, 
            BSP_sync, 
            post_shift_code_block)
