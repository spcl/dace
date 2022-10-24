# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from typing import Any, Dict, Optional
from dace import dtypes, memlet as mm, properties, data as dt
from dace.symbolic import symstr
import dace.library
from dace import SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.blas_helpers import (to_blastype, get_gemm_opts, check_access, dtype_to_cudadatatype,
                                              to_cublas_computetype)
from dace.libraries.blas.nodes.matmul import _get_codegen_gemm_opts
from .. import environments
import numpy as np
from numbers import Number


@dace.library.expansion
class ExpandAHHTPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, state: SDFGState, sdfg: SDFG):
        nnz, arows, acols, hcols = node.validate(sdfg, state)
        hname = list(state.in_edges_by_connector(node, '_h1'))[0].data.data
        htype = sdfg.arrays[hname].dtype
        sname = state.out_edges(node)[0].data.data
        stype = sdfg.arrays[sname].dtype

        @dace.program
        def ahht_pure(_a_row: dace.int32[nnz], _a_col: dace.int32[nnz],
                      _h1: htype[arows, hcols], _h2: htype[acols, hcols],
                      _s_data: stype[nnz]):
            for i, k in dace.map[0:nnz, 0:hcols]:
                _s_data[i] = _h1[_a_row[i], k] * _h2[_a_col[i], k] + _s_data[i]
        
        return ahht_pure.to_sdfg()


@dace.library.expansion
class ExpandAHHTCUDA(ExpandTransformation):
    from dace.libraries.standard.environments import CUDA
    environments = [CUDA]

    @staticmethod
    def expansion(node, state: SDFGState, sdfg: SDFG):
        nnz, arows, acols, hcols = node.validate(sdfg, state)
        sname = state.out_edges(node)[0].data.data
        stype = sdfg.arrays[sname].dtype
        node_id = state.node_id(node)
        state_id = sdfg.node_id(state)
        idstr = '{sdfg}_{state}_{node}'.format(sdfg=sdfg.name, state=state_id, node=node_id)

        from dace.codegen.prettycode import CodeIOStream
        cuda_globalcode = CodeIOStream()
        cuda_globalcode.write(
            """
#define FULL_MASK_{id} 0xFFFFFFFF
#define MAX_GRID_Y_{id} 65535

__device__ {T} sumWarp_{id}({T} a) {{
    a += __shfl_xor_sync(FULL_MASK_{id}, a, 16);
    a += __shfl_xor_sync(FULL_MASK_{id}, a, 8);
    a += __shfl_xor_sync(FULL_MASK_{id}, a, 4);
    a += __shfl_xor_sync(FULL_MASK_{id}, a, 2);
    a += __shfl_xor_sync(FULL_MASK_{id}, a, 1);
    return a;
}}

//__device__ {T} dotBlock_{id}({T} a, {T} b) {{
__device__ {T} dotBlock_{id}({T} a) {{
    int idx = threadIdx.x;
    __shared__ {T} warp_sums[32];

    //{T} warp_sum = sumWarp_{id}(a * b);
    {T} warp_sum = sumWarp_{id}(a);
    if (idx < 32) {{
        warp_sums[idx] = 0;
    }}
    __syncthreads();

    if ((idx & 31) == 0) {{
        warp_sums[idx >> 5] = warp_sum;
    }}
    __syncthreads();

    if (idx < 32) {{
        a = sumWarp_{id}(warp_sums[idx]);
    }}
    return a;
}}
            """.format(id=idstr, T=stype)
        )

        nsdfg = SDFG('nested_sdfg')
        nstate = nsdfg.add_state('nested_state')

        tasklet_code = """
for (int idx = i; idx < {nnz}; idx += gridDim.x) {{
    if (j == 0) {{
        __s_data[idx] = 0;
    }}
    __syncthreads();
    int r = __a_row[idx];
    int c = __a_col[idx];
    {T} sum = {T}(0);
    for (int k = j; k < {cols}; k += blockDim.x) {{
        {T} a = __h1[r * {cols} + k];
        {T} b = __h2[c * {cols} + k];
        sum += a * b;
    }}
    for (int offset = warpSize/2; offset > 0; offset /= 2) {{
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }}
    if (j % warpSize == 0) {{
        atomicAdd(__s_data + idx, sum);
    }}
}}
    """.format(nnz=nnz, cols=hcols, id=idstr, T=stype)

        datadict = {}

        for e in state.all_edges(node):
            if e.src is node:
                cname = e.src_conn
            else:
                cname = e.dst_conn
            desc = sdfg.arrays[e.data.data]
            nname, ndesc = nsdfg.add_array(cname, desc.shape, desc.dtype)
            nnode = nstate.add_access(nname)
            datadict[cname] = nnode
            if desc.storage not in (dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned):
                nname, ndesc = nsdfg.add_array(nname, desc.shape, desc.dtype, storage=dtypes.StorageType.GPU_Global,
                                               find_new_name=True)
                gnode = nstate.add_access(nname)
                datadict[cname] = gnode
                if cname == '_s_data':
                    nstate.add_nedge(gnode, nnode)
                else:
                    nstate.add_nedge(nnode, gnode)
                # nnode = gnode
            # if cname == '_s_data':
            #     nstate.add_edge(tasklet, '_s_data', nnode, None, dace.Memlet.from_array(nname, ndesc))
            # else:
            #     nstate.add_edge(nnode, None, tasklet, cname, dace.Memlet.from_array(nname, ndesc))

        tasklet, me, mx = nstate.add_mapped_tasklet(
            name='callingKernel',
            map_ranges={'i': f'0:min({nnz}, 65536)', 'j': f'0:64'},
            inputs={
                '__a_row': dace.Memlet(f'{datadict["_a_row"].data}[0:{nnz}]'),
                '__a_col': dace.Memlet(f'{datadict["_a_col"].data}[0:{nnz}]'),
                '__h1': dace.Memlet(f'{datadict["_h1"].data}[0:{arows}, 0:{hcols}]'),
                '__h2': dace.Memlet(f'{datadict["_h2"].data}[0:{acols}, 0:{hcols}]')},
            outputs={'__s_data': dace.Memlet(f'{datadict["_s_data"].data}[0:{nnz}]')},
            code=tasklet_code,
            language=dace.dtypes.Language.CPP,
            external_edges=False
        )

        for k, v in datadict.items():
            if k == '_s_data':
                nstate.add_nedge(mx, v, dace.Memlet.from_array(v.data, nsdfg.arrays[v.data]))
            else:
                nstate.add_nedge(v, me, dace.Memlet.from_array(v.data, nsdfg.arrays[v.data]))
        nstate.fill_scope_connectors()

        me.map.schedule = dace.dtypes.ScheduleType.GPU_Device

        from dace.transformation.dataflow import MapExpansion
        from dace.sdfg import nodes
        nsdfg.apply_transformations_repeated(MapExpansion)
        for n in nstate.nodes():
            if isinstance(n, nodes.MapEntry) and "j" in n.map.params:
                n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock
        
        return nsdfg


@dace.library.node
class AHHT(dace.sdfg.nodes.LibraryNode):
    """
    Executes A ⊙ (H x HT). A is a sparse adjacency matrix in COO format, while H is dense.
    """

    # Global properties
    implementations = {
        "pure": ExpandAHHTPure,
        "CUDA": ExpandAHHTCUDA
    }
    default_implementation = None

    def __init__(self, name, location=None):
        super().__init__(name,
                         location=location,
                         inputs=({"_a_row", "_a_col", "_h1", "_h2"}),
                         outputs={"_s_data"})

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 4:
            raise ValueError("Expected 4 inputs to AHHT")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a_row':
                subset = dc(memlet.subset)
                subset.squeeze()
                size0 = subset.size()
            if dst_conn == '_a_col':
                subset = dc(memlet.subset)
                subset.squeeze()
                size1 = subset.size()
            if dst_conn == '_h1':
                subset = dc(memlet.subset)
                subset.squeeze()
                size2 = subset.size()
            if dst_conn == '_h2':
                subset = dc(memlet.subset)
                subset.squeeze()
                size3 = subset.size()

        assert size0[0] == size1[0]
        assert size2[1] == size3[1]
        
        nnz = size0[0]
        arows = size2[0]
        acols = size3[0]
        hcols = size2[1]

        return nnz, arows, acols, hcols


@dace.library.expansion
class ExpandAHHTNormPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, state: SDFGState, sdfg: SDFG):
        nnz, arows, acols, hcols = node.validate(sdfg, state)
        hname = list(state.in_edges_by_connector(node, '_h1'))[0].data.data
        htype = sdfg.arrays[hname].dtype
        sname = state.out_edges(node)[0].data.data
        stype = sdfg.arrays[sname].dtype

        @dace.program
        def ahht_pure(_a_row: dace.int32[nnz], _a_col: dace.int32[nnz],
                      _h1: htype[arows, hcols], _h2: htype[acols, hcols],
                      _h1_norm: htype[arows], _h2_norm: htype[acols],
                      _s_data: stype[nnz]):
            for i in dace.map[0:nnz]:
                _s_data[i] = 0
                for k in dace.map[0:hcols]:
                    _s_data[i] = _h1[_a_row[i], k] * _h2[_a_col[i], k] + _s_data[i]
                _s_data[i] = _s_data[i] / (_h1_norm[_a_row[i]] * _h2_norm[_a_col[i]])
        
        return ahht_pure.to_sdfg()


@dace.library.expansion
class ExpandAHHTNormCUDA(ExpandTransformation):
    from dace.libraries.standard.environments import CUDA
    environments = [CUDA]

    @staticmethod
    def expansion(node, state: SDFGState, sdfg: SDFG):
        nnz, arows, acols, hcols = node.validate(sdfg, state)
        sname = state.out_edges(node)[0].data.data
        stype = sdfg.arrays[sname].dtype
        node_id = state.node_id(node)
        state_id = sdfg.node_id(state)
        idstr = '{sdfg}_{state}_{node}'.format(sdfg=sdfg.name, state=state_id, node=node_id)

        from dace.codegen.prettycode import CodeIOStream
        cuda_globalcode = CodeIOStream()
        cuda_globalcode.write(
            """
#define FULL_MASK_{id} 0xFFFFFFFF
#define MAX_GRID_Y_{id} 65535

__device__ {T} sumWarp_{id}({T} a) {{
    a += __shfl_xor_sync(FULL_MASK_{id}, a, 16);
    a += __shfl_xor_sync(FULL_MASK_{id}, a, 8);
    a += __shfl_xor_sync(FULL_MASK_{id}, a, 4);
    a += __shfl_xor_sync(FULL_MASK_{id}, a, 2);
    a += __shfl_xor_sync(FULL_MASK_{id}, a, 1);
    return a;
}}

//__device__ {T} dotBlock_{id}({T} a, {T} b) {{
__device__ {T} dotBlock_{id}({T} a) {{
    int idx = threadIdx.x;
    __shared__ {T} warp_sums[32];

    //{T} warp_sum = sumWarp_{id}(a * b);
    {T} warp_sum = sumWarp_{id}(a);
    if (idx < 32) {{
        warp_sums[idx] = 0;
    }}
    __syncthreads();

    if ((idx & 31) == 0) {{
        warp_sums[idx >> 5] = warp_sum;
    }}
    __syncthreads();

    if (idx < 32) {{
        a = sumWarp_{id}(warp_sums[idx]);
    }}
    return a;
}}
            """.format(id=idstr, T=stype)
        )

        nsdfg = SDFG('nested_sdfg')
        nstate = nsdfg.add_state('nested_state')

        tasklet_code = """
for (int idx = i; idx < {nnz}; idx += gridDim.x) {{
    if (j == 0) {{
        __s_data[idx] = 0;
    }}
    __syncthreads();
    int r = __a_row[idx];
    int c = __a_col[idx];
    {T} sum = {T}(0);
    for (int k = j; k < {cols}; k += blockDim.x) {{
        {T} a = __h1[r * {cols} + k];
        {T} b = __h2[c * {cols} + k];
        sum += a * b;
    }}
    for (int offset = warpSize/2; offset > 0; offset /= 2) {{
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }}
    if (j % warpSize == 0) {{
        atomicAdd(__s_data + idx, sum);
    }}
    __syncthreads();
    if (j == 0) {{
        __s_data[idx] /= __h1_norm[r] * __h2_norm[c];
    }}
}}
    """.format(nnz=nnz, cols=hcols, id=idstr, T=stype)
    
        datadict = {}

        for e in state.all_edges(node):
            if e.src is node:
                cname = e.src_conn
            else:
                cname = e.dst_conn
            desc = sdfg.arrays[e.data.data]
            nname, ndesc = nsdfg.add_array(cname, desc.shape, desc.dtype)
            nnode = nstate.add_access(nname)
            datadict[cname] = nnode
            if desc.storage not in (dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned):
                nname, ndesc = nsdfg.add_array(nname, desc.shape, desc.dtype, storage=dtypes.StorageType.GPU_Global,
                                               find_new_name=True)
                gnode = nstate.add_access(nname)
                datadict[cname] = gnode
                if cname == '_s_data':
                    nstate.add_nedge(gnode, nnode)
                else:
                    nstate.add_nedge(nnode, gnode)
                # nnode = gnode
            # if cname == '_s_data':
            #     nstate.add_edge(tasklet, '_s_data', nnode, None, dace.Memlet.from_array(nname, ndesc))
            # else:
            #     nstate.add_edge(nnode, None, tasklet, cname, dace.Memlet.from_array(nname, ndesc))

        tasklet, me, mx = nstate.add_mapped_tasklet(
            name='callingKernel',
            map_ranges={'i': f'0:min({nnz}, 65536)', 'j': f'0:64'},
            inputs={
                '__a_row': dace.Memlet(f'{datadict["_a_row"].data}[0:{nnz}]'),
                '__a_col': dace.Memlet(f'{datadict["_a_col"].data}[0:{nnz}]'),
                '__h1': dace.Memlet(f'{datadict["_h1"].data}[0:{arows}, 0:{hcols}]'),
                '__h2': dace.Memlet(f'{datadict["_h2"].data}[0:{acols}, 0:{hcols}]'),
                '__h1_norm': dace.Memlet(f'{datadict["_h1_norm"].data}[0:{arows}]'),
                '__h2_norm': dace.Memlet(f'{datadict["_h2_norm"].data}[0:{acols}]')},
            outputs={'__s_data': dace.Memlet(f'{datadict["_s_data"].data}[0:{nnz}]')},
            code=tasklet_code,
            language=dace.dtypes.Language.CPP,
            external_edges=False
        )

        for k, v in datadict.items():
            if k == '_s_data':
                nstate.add_nedge(mx, v, dace.Memlet.from_array(v.data, nsdfg.arrays[v.data]))
            else:
                nstate.add_nedge(v, me, dace.Memlet.from_array(v.data, nsdfg.arrays[v.data]))
        nstate.fill_scope_connectors()

        me.map.schedule = dace.dtypes.ScheduleType.GPU_Device

        from dace.transformation.dataflow import MapExpansion
        from dace.sdfg import nodes
        nsdfg.apply_transformations_repeated(MapExpansion)
        for n in nstate.nodes():
            if isinstance(n, nodes.MapEntry) and "j" in n.map.params:
                n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock
        
        return nsdfg


@dace.library.node
class AHHTNorm(dace.sdfg.nodes.LibraryNode):
    """
    Executes A ⊙ (H x HT). A is a sparse adjacency matrix in COO format, while H is dense.
    """

    # Global properties
    implementations = {
        "pure": ExpandAHHTNormPure,
        "CUDA": ExpandAHHTNormCUDA
    }
    default_implementation = None

    def __init__(self, name, location=None):
        super().__init__(name,
                         location=location,
                         inputs=({"_a_row", "_a_col", "_h1", "_h2", "_h1_norm", "_h2_norm"}),
                         outputs={"_s_data"})

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 6:
            raise ValueError("Expected 6 inputs to AHHT")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a_row':
                subset = dc(memlet.subset)
                subset.squeeze()
                size0 = subset.size()
            if dst_conn == '_a_col':
                subset = dc(memlet.subset)
                subset.squeeze()
                size1 = subset.size()
            if dst_conn == '_h1':
                subset = dc(memlet.subset)
                subset.squeeze()
                size2 = subset.size()
            if dst_conn == '_h2':
                subset = dc(memlet.subset)
                subset.squeeze()
                size3 = subset.size()
            if dst_conn == '_h1_norm':
                subset = dc(memlet.subset)
                subset.squeeze()
                size4 = subset.size()
            if dst_conn == '_h2_norm':
                subset = dc(memlet.subset)
                subset.squeeze()
                size5 = subset.size()

        assert size0[0] == size1[0]
        assert size2[1] == size3[1]
        
        nnz = size0[0]
        arows = size2[0]
        acols = size3[0]
        hcols = size2[1]

        return nnz, arows, acols, hcols


@dace.library.expansion
class ExpandA_exp_relu_CPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, state: SDFGState, sdfg: SDFG):
        nnz, arows, acols = node.validate(sdfg, state)
        vname = list(state.in_edges_by_connector(node, '_vl'))[0].data.data
        vtype = sdfg.arrays[vname].dtype
        sname = state.out_edges(node)[0].data.data
        stype = sdfg.arrays[sname].dtype

        @dace.program
        def ahht_pure(_a_row: dace.int32[nnz], _a_col: dace.int32[nnz],
                      _vl: vtype[arows], _vr: vtype[acols], _s_data: stype[nnz]):
            for i in dace.map[0:nnz]:
                _s_data[i] = np.exp(np.max(_vl[_a_row[i]] + _vr[_a_col[i]]), 0)
        
        return ahht_pure.to_sdfg()


@dace.library.expansion
class ExpandA_exp_relu_CCUDA(ExpandTransformation):
    from dace.libraries.standard.environments import CUDA
    environments = [CUDA]

    @staticmethod
    def expansion(node, state: SDFGState, sdfg: SDFG):
        nnz, arows, acols = node.validate(sdfg, state)
        sname = state.out_edges(node)[0].data.data
        stype = sdfg.arrays[sname].dtype
        node_id = state.node_id(node)
        state_id = sdfg.node_id(state)

        nsdfg = SDFG('nested_sdfg')
        nstate = nsdfg.add_state('nested_state')

        tasklet_code = """
for (int idx = i; idx < {nnz}; idx += gridDim.x) {{
    int r = __a_row[idx];
    int c = __a_col[idx];
    __s_data[idx] = exp(max(__vl[r] + __vr[c], 0.0));
}}
    """.format(nnz=nnz, T=stype)
    
        datadict = {}

        for e in state.all_edges(node):
            if e.src is node:
                cname = e.src_conn
            else:
                cname = e.dst_conn
            desc = sdfg.arrays[e.data.data]
            nname, ndesc = nsdfg.add_array(cname, desc.shape, desc.dtype)
            nnode = nstate.add_access(nname)
            datadict[cname] = nnode
            if desc.storage not in (dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned):
                nname, ndesc = nsdfg.add_array(nname, desc.shape, desc.dtype, storage=dtypes.StorageType.GPU_Global,
                                               find_new_name=True)
                gnode = nstate.add_access(nname)
                datadict[cname] = gnode
                if cname == '_s_data':
                    nstate.add_nedge(gnode, nnode)
                else:
                    nstate.add_nedge(nnode, gnode)
                # nnode = gnode
            # if cname == '_s_data':
            #     nstate.add_edge(tasklet, '_s_data', nnode, None, dace.Memlet.from_array(nname, ndesc))
            # else:
            #     nstate.add_edge(nnode, None, tasklet, cname, dace.Memlet.from_array(nname, ndesc))

        tasklet, me, mx = nstate.add_mapped_tasklet(
            name='callingKernel',
            map_ranges={'i': f'0:int_ceil({nnz}, 512)', 'j': f'0:512'},
            inputs={
                '__a_row': dace.Memlet(f'{datadict["_a_row"].data}[0:{nnz}]'),
                '__a_col': dace.Memlet(f'{datadict["_a_col"].data}[0:{nnz}]'),
                '__vl': dace.Memlet(f'{datadict["_h1"].data}[0:{arows}]'),
                '__vr': dace.Memlet(f'{datadict["_h2"].data}[0:{acols}]')},
            outputs={'__s_data': dace.Memlet(f'{datadict["_s_data"].data}[0:{nnz}]')},
            code=tasklet_code,
            language=dace.dtypes.Language.CPP,
            external_edges=False
        )

        for k, v in datadict.items():
            if k == '_s_data':
                nstate.add_nedge(mx, v, dace.Memlet.from_array(v.data, nsdfg.arrays[v.data]))
            else:
                nstate.add_nedge(v, me, dace.Memlet.from_array(v.data, nsdfg.arrays[v.data]))
        nstate.fill_scope_connectors()

        me.map.schedule = dace.dtypes.ScheduleType.GPU_Device

        from dace.transformation.dataflow import MapExpansion
        from dace.sdfg import nodes
        nsdfg.apply_transformations_repeated(MapExpansion)
        for n in nstate.nodes():
            if isinstance(n, nodes.MapEntry) and "j" in n.map.params:
                n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock
        
        return nsdfg


@dace.library.node
class A_exp_relu_C(dace.sdfg.nodes.LibraryNode):
    """
    Executes A ⊙ (H x HT). A is a sparse adjacency matrix in COO format, while H is dense.
    """

    # Global properties
    implementations = {
        "pure": ExpandA_exp_relu_CPure,
        "CUDA": ExpandA_exp_relu_CCUDA
    }
    default_implementation = None

    def __init__(self, name, location=None):
        super().__init__(name,
                         location=location,
                         inputs=({"_a_row", "_a_col", "_vl", "_vr"}),
                         outputs={"_s_data"})

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 6:
            raise ValueError("Expected 4 inputs to AHHT")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a_row':
                subset = dc(memlet.subset)
                subset.squeeze()
                size0 = subset.size()
            if dst_conn == '_a_col':
                subset = dc(memlet.subset)
                subset.squeeze()
                size1 = subset.size()
            if dst_conn == '_vl':
                subset = dc(memlet.subset)
                subset.squeeze()
                size2 = subset.size()
            if dst_conn == '_vr':
                subset = dc(memlet.subset)
                subset.squeeze()
                size3 = subset.size()

        assert size0[0] == size1[0]
        
        nnz = size0[0]
        arows = size2[0]
        acols = size3[0]

        return nnz, arows, acols
