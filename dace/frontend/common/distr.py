# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from numbers import Number
from typing import Sequence, Union

import dace
from dace import dtypes
from dace.frontend.common import op_repository as oprepo
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState

import sympy as sp

from dace.frontend.python.replacements import _define_local_scalar


@oprepo.replaces('dace.comm.Bcast')
def _bcast(pv: 'ProgramVisitor',
           sdfg: SDFG,
           state: SDFGState,
           buffer: str,
           root: Union[str, sp.Expr, Number] = 0):

    from dace.libraries.mpi.nodes.bcast import Bcast

    libnode = Bcast('_Bcast_')
    desc = sdfg.arrays[buffer]
    in_buffer = state.add_read(buffer)
    out_buffer = state.add_write(buffer)
    if isinstance(root, str) and root in sdfg.arrays.keys():
        root_node = state.add_read(root)
    else:
        storage = desc.storage
        root_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        root_node = state.add_access(root_name)
        root_tasklet = state.add_tasklet('_set_root_', {}, {'__out'},
                                         '__out = {}'.format(root))
        state.add_edge(root_tasklet, '__out', root_node, None,
                       Memlet.simple(root_name, '0'))
    state.add_edge(in_buffer, None, libnode, '_inbuffer',
                   Memlet.from_array(buffer, desc))
    state.add_edge(root_node, None, libnode, '_root',
                   Memlet.simple(root_node.data, '0'))
    state.add_edge(libnode, '_outbuffer', out_buffer, None,
                   Memlet.from_array(buffer, desc))

    return None


@oprepo.replaces('dace.comm.Scatter')
def _scatter(pv: 'ProgramVisitor',
             sdfg: SDFG,
             state: SDFGState,
             in_buffer: str,
             out_buffer: str,
             root: Union[str, sp.Expr, Number] = 0):

    from dace.libraries.mpi.nodes.scatter import Scatter

    libnode = Scatter('_Scatter_')
    in_desc = sdfg.arrays[in_buffer]
    out_desc = sdfg.arrays[out_buffer]
    in_node = state.add_read(in_buffer)
    out_node = state.add_write(out_buffer)
    if isinstance(root, str) and root in sdfg.arrays.keys():
        root_node = state.add_read(root)
    else:
        storage = in_desc.storage
        root_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        root_node = state.add_access(root_name)
        root_tasklet = state.add_tasklet('_set_root_', {}, {'__out'},
                                         '__out = {}'.format(root))
        state.add_edge(root_tasklet, '__out', root_node, None,
                       Memlet.simple(root_name, '0'))
    state.add_edge(in_node, None, libnode, '_inbuffer',
                   Memlet.from_array(in_buffer, in_desc))
    state.add_edge(root_node, None, libnode, '_root',
                   Memlet.simple(root_node.data, '0'))
    state.add_edge(libnode, '_outbuffer', out_node, None,
                   Memlet.from_array(out_buffer, out_desc))

    return None


@oprepo.replaces('dace.comm.Gather')
def _gather(pv: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            in_buffer: str,
            out_buffer: str,
            root: Union[str, sp.Expr, Number] = 0):

    from dace.libraries.mpi.nodes.gather import Gather

    libnode = Gather('_Gather_')
    in_desc = sdfg.arrays[in_buffer]
    out_desc = sdfg.arrays[out_buffer]
    in_node = state.add_read(in_buffer)
    out_node = state.add_write(out_buffer)
    if isinstance(root, str) and root in sdfg.arrays.keys():
        root_node = state.add_read(root)
    else:
        storage = in_desc.storage
        root_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        root_node = state.add_access(root_name)
        root_tasklet = state.add_tasklet('_set_root_', {}, {'__out'},
                                         '__out = {}'.format(root))
        state.add_edge(root_tasklet, '__out', root_node, None,
                       Memlet.simple(root_name, '0'))
    state.add_edge(in_node, None, libnode, '_inbuffer',
                   Memlet.from_array(in_buffer, in_desc))
    state.add_edge(root_node, None, libnode, '_root',
                   Memlet.simple(root_node.data, '0'))
    state.add_edge(libnode, '_outbuffer', out_node, None,
                   Memlet.from_array(out_buffer, out_desc))

    return None


@oprepo.replaces('dace.comm.Send')
def _send(pv: 'ProgramVisitor',
          sdfg: SDFG,
          state: SDFGState,
          buffer: str,
          dst: Union[str, sp.Expr, Number],
          tag: Union[str, sp.Expr, Number] = 0):

    from dace.libraries.mpi.nodes.send import Send

    libnode = Send('_Send_')

    buf_range = None
    if isinstance(buffer, tuple):
        buf_name, buf_range = buffer
    else:
        buf_name = buffer

    desc = sdfg.arrays[buf_name]
    conn = libnode.in_connectors
    conn = {
        c: (dtypes.pointer(desc.dtype) if c == '_buffer' else t)
        for c, t in conn.items()
    }
    libnode.in_connectors = conn
    buf_node = state.add_write(buf_name)

    dst_range = None
    if isinstance(dst, tuple):
        dst_name, dst_range = dst
        dst_node = state.add_read(dst_name)
    elif isinstance(dst, str) and dst in sdfg.arrays.keys():
        dst_name = dst
        dst_node = state.add_read(dst_name)
    else:
        storage = desc.storage
        dst_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        dst_node = state.add_access(dst_name)
        dst_tasklet = state.add_tasklet('_set_dst_', {}, {'__out'},
                                        '__out = {}'.format(dst))
        state.add_edge(dst_tasklet, '__out', dst_node, None,
                       Memlet.simple(dst_name, '0'))

    tag_range = None
    if isinstance(tag, tuple):
        tag_name, tag_range = tag
        tag_node = state.add_read(tag_name)
    if isinstance(tag, str) and tag in sdfg.arrays.keys():
        tag_name = tag
        tag_node = state.add_read(tag)
    else:
        storage = desc.storage
        tag_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        tag_node = state.add_access(tag_name)
        tag_tasklet = state.add_tasklet('_set_tag_', {}, {'__out'},
                                        '__out = {}'.format(tag))
        state.add_edge(tag_tasklet, '__out', tag_node, None,
                       Memlet.simple(tag_name, '0'))

    if buf_range:
        buf_mem = Memlet.simple(buf_name, buf_range)
    else:
        buf_mem = Memlet.from_array(buf_name, desc)
    if dst_range:
        dst_mem = Memlet.simple(dst_name, dst_range)
    else:
        dst_mem = Memlet.simple(dst_name, '0')
    if tag_range:
        tag_mem = Memlet.simple(tag_name, tag_range)
    else:
        tag_mem = Memlet.simple(tag_name, '0')

    state.add_edge(buf_node, None, libnode, '_buffer', buf_mem)
    state.add_edge(dst_node, None, libnode, '_dest', dst_mem)
    state.add_edge(tag_node, None, libnode, '_tag', tag_mem)

    return None


@oprepo.replaces('dace.comm.Isend')
def _isend(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, buffer: str,
           dst: Union[str, sp.Expr, Number], tag: Union[str, sp.Expr,
                                                        Number], request: str):

    from dace.libraries.mpi.nodes.isend import Isend

    libnode = Isend('_Isend_')

    buf_range = None
    if isinstance(buffer, tuple):
        buf_name, buf_range = buffer
    else:
        buf_name = buffer
    desc = sdfg.arrays[buf_name]
    buf_node = state.add_read(buf_name)

    req_range = None
    if isinstance(request, tuple):
        req_name, req_range = request
    else:
        req_name = request
    req_desc = sdfg.arrays[req_name]
    req_node = state.add_write(req_name)

    iconn = libnode.in_connectors
    iconn = {
        c: (dtypes.pointer(desc.dtype) if c == '_buffer' else t)
        for c, t in iconn.items()
    }
    libnode.in_connectors = iconn
    oconn = libnode.out_connectors
    oconn = {
        c: (dtypes.pointer(req_desc.dtype) if c == '_request' else t)
        for c, t in oconn.items()
    }
    libnode.out_connectors = oconn

    dst_range = None
    if isinstance(dst, tuple):
        dst_name, dst_range = dst
        dst_node = state.add_read(dst_name)
    elif isinstance(dst, str) and dst in sdfg.arrays.keys():
        dst_name = dst
        dst_node = state.add_read(dst_name)
    else:
        storage = desc.storage
        dst_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        dst_node = state.add_access(dst_name)
        dst_tasklet = state.add_tasklet('_set_dst_', {}, {'__out'},
                                        '__out = {}'.format(dst))
        state.add_edge(dst_tasklet, '__out', dst_node, None,
                       Memlet.simple(dst_name, '0'))

    tag_range = None
    if isinstance(tag, tuple):
        tag_name, tag_range = tag
        tag_node = state.add_read(tag_name)
    if isinstance(tag, str) and tag in sdfg.arrays.keys():
        tag_name = tag
        tag_node = state.add_read(tag)
    else:
        storage = desc.storage
        tag_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        tag_node = state.add_access(tag_name)
        tag_tasklet = state.add_tasklet('_set_tag_', {}, {'__out'},
                                        '__out = {}'.format(tag))
        state.add_edge(tag_tasklet, '__out', tag_node, None,
                       Memlet.simple(tag_name, '0'))

    if buf_range:
        buf_mem = Memlet.simple(buf_name, buf_range)
    else:
        buf_mem = Memlet.from_array(buf_name, desc)
    if req_range:
        req_mem = Memlet.simple(req_name, req_range)
    else:
        req_mem = Memlet.from_array(req_name, req_desc)
    if dst_range:
        dst_mem = Memlet.simple(dst_name, dst_range)
    else:
        dst_mem = Memlet.simple(dst_name, '0')
    if tag_range:
        tag_mem = Memlet.simple(tag_name, tag_range)
    else:
        tag_mem = Memlet.simple(tag_name, '0')

    state.add_edge(buf_node, None, libnode, '_buffer', buf_mem)
    state.add_edge(dst_node, None, libnode, '_dest', dst_mem)
    state.add_edge(tag_node, None, libnode, '_tag', tag_mem)
    state.add_edge(libnode, '_request', req_node, None, req_mem)

    return None


@oprepo.replaces('dace.comm.Recv')
def _recv(pv: 'ProgramVisitor',
          sdfg: SDFG,
          state: SDFGState,
          buffer: str,
          src: Union[str, sp.Expr, Number],
          tag: Union[str, sp.Expr, Number] = 0):

    from dace.libraries.mpi.nodes.recv import Recv

    libnode = Recv('_Recv_')

    buf_range = None
    if isinstance(buffer, tuple):
        buf_name, buf_range = buffer
    else:
        buf_name = buffer

    desc = sdfg.arrays[buf_name]
    conn = libnode.out_connectors
    conn = {
        c: (dtypes.pointer(desc.dtype) if c == '_buffer' else t)
        for c, t in conn.items()
    }
    libnode.out_connectors = conn
    buf_node = state.add_write(buf_name)

    src_range = None
    if isinstance(src, tuple):
        src_name, src_range = src
        src_node = state.add_read(src_name)
    elif isinstance(src, str) and src in sdfg.arrays.keys():
        src_name = src
        src_node = state.add_read(src_name)
    else:
        storage = desc.storage
        src_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        src_node = state.add_access(src_name)
        src_tasklet = state.add_tasklet('_set_src_', {}, {'__out'},
                                        '__out = {}'.format(src))
        state.add_edge(src_tasklet, '__out', src_node, None,
                       Memlet.simple(src_name, '0'))

    tag_range = None
    if isinstance(tag, tuple):
        tag_name, tag_range = tag
        tag_node = state.add_read(tag_name)
    if isinstance(tag, str) and tag in sdfg.arrays.keys():
        tag_name = tag
        tag_node = state.add_read(tag)
    else:
        storage = desc.storage
        tag_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        tag_node = state.add_access(tag_name)
        tag_tasklet = state.add_tasklet('_set_tag_', {}, {'__out'},
                                        '__out = {}'.format(tag))
        state.add_edge(tag_tasklet, '__out', tag_node, None,
                       Memlet.simple(tag_name, '0'))

    if buf_range:
        buf_mem = Memlet.simple(buf_name, buf_range)
    else:
        buf_mem = Memlet.from_array(buf_name, desc)
    if src_range:
        src_mem = Memlet.simple(src_name, src_range)
    else:
        src_mem = Memlet.simple(src_name, '0')
    if tag_range:
        tag_mem = Memlet.simple(tag_name, tag_range)
    else:
        tag_mem = Memlet.simple(tag_name, '0')

    state.add_edge(libnode, '_buffer', buf_node, None, buf_mem)
    state.add_edge(src_node, None, libnode, '_src', src_mem)
    state.add_edge(tag_node, None, libnode, '_tag', tag_mem)

    return None


@oprepo.replaces('dace.comm.Irecv')
def _irecv(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, buffer: str,
           src: Union[str, sp.Expr, Number], tag: Union[str, sp.Expr,
                                                        Number], request: str):

    from dace.libraries.mpi.nodes.irecv import Irecv

    libnode = Irecv('_Irecv_')

    buf_range = None
    if isinstance(buffer, tuple):
        buf_name, buf_range = buffer
    else:
        buf_name = buffer
    desc = sdfg.arrays[buf_name]
    buf_node = state.add_read(buf_name)

    req_range = None
    if isinstance(request, tuple):
        req_name, req_range = request
    else:
        req_name = request
    req_desc = sdfg.arrays[req_name]
    req_node = state.add_write(req_name)

    conn = libnode.out_connectors
    conn = {
        c: (dtypes.pointer(desc.dtype) if c == '_buffer' else t)
        for c, t in conn.items()
    }
    conn = {
        c: (dtypes.pointer(req_desc.dtype) if c == '_request' else t)
        for c, t in conn.items()
    }
    libnode.out_connectors = conn

    src_range = None
    if isinstance(src, tuple):
        src_name, src_range = src
        src_node = state.add_read(src_name)
    elif isinstance(src, str) and src in sdfg.arrays.keys():
        src_name = src
        src_node = state.add_read(src_name)
    else:
        storage = desc.storage
        src_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        src_node = state.add_access(src_name)
        src_tasklet = state.add_tasklet('_set_src_', {}, {'__out'},
                                        '__out = {}'.format(src))
        state.add_edge(src_tasklet, '__out', src_node, None,
                       Memlet.simple(src_name, '0'))

    tag_range = None
    if isinstance(tag, tuple):
        tag_name, tag_range = tag
        tag_node = state.add_read(tag_name)
    if isinstance(tag, str) and tag in sdfg.arrays.keys():
        tag_name = tag
        tag_node = state.add_read(tag)
    else:
        storage = desc.storage
        tag_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        tag_node = state.add_access(tag_name)
        tag_tasklet = state.add_tasklet('_set_tag_', {}, {'__out'},
                                        '__out = {}'.format(tag))
        state.add_edge(tag_tasklet, '__out', tag_node, None,
                       Memlet.simple(tag_name, '0'))

    if buf_range:
        buf_mem = Memlet.simple(buf_name, buf_range)
    else:
        buf_mem = Memlet.from_array(buf_name, desc)
    if req_range:
        req_mem = Memlet.simple(req_name, req_range)
    else:
        req_mem = Memlet.from_array(req_name, req_desc)
    if src_range:
        src_mem = Memlet.simple(src_name, src_range)
    else:
        src_mem = Memlet.simple(src_name, '0')
    if tag_range:
        tag_mem = Memlet.simple(tag_name, tag_range)
    else:
        tag_mem = Memlet.simple(tag_name, '0')

    state.add_edge(libnode, '_buffer', buf_node, None, buf_mem)
    state.add_edge(src_node, None, libnode, '_src', src_mem)
    state.add_edge(tag_node, None, libnode, '_tag', tag_mem)
    state.add_edge(libnode, '_request', req_node, None, req_mem)

    return None


@oprepo.replaces('dace.comm.Wait')
def _wait(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, request: str):

    from dace.libraries.mpi.nodes.wait import Wait

    libnode = Wait('_Wait_')

    req_range = None
    if isinstance(request, tuple):
        req_name, req_range = request
    else:
        req_name = request

    desc = sdfg.arrays[req_name]
    req_node = state.add_access(req_name)

    src = sdfg.add_temp_transient([1], dtypes.int32)
    src_node = state.add_write(src[0])
    tag = sdfg.add_temp_transient([1], dtypes.int32)
    tag_node = state.add_write(tag[0])

    if req_range:
        req_mem = Memlet.simple(req_name, req_range)
    else:
        req_mem = Memlet.from_array(req_name, desc)

    state.add_edge(req_node, None, libnode, '_request', req_mem)
    state.add_edge(libnode, '_stat_source', src_node, None,
                   Memlet.from_array(*src))
    state.add_edge(libnode, '_stat_tag', tag_node, None,
                   Memlet.from_array(*tag))

    return None


@oprepo.replaces('dace.comm.Waitall')
def _wait(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, request: str):

    from dace.libraries.mpi.nodes.wait import Waitall

    libnode = Waitall('_Waitall_')

    req_range = None
    if isinstance(request, tuple):
        req_name, req_range = request
    else:
        req_name = request

    desc = sdfg.arrays[req_name]
    req_node = state.add_access(req_name)

    if req_range:
        req_mem = Memlet.simple(req_name, req_range)
    else:
        req_mem = Memlet.from_array(req_name, desc)

    state.add_edge(req_node, None, libnode, '_request', req_mem)

    return None


@oprepo.replaces('dace.comm.BCScatter')
def _bcscatter(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState,
               in_buffer: str, out_buffer: str,
               block_sizes: Union[str, Sequence[Union[sp.Expr, Number]]]):

    from dace.libraries.pblas.nodes.pgeadd import BlockCyclicScatter

    libnode = BlockCyclicScatter('_BCScatter_')

    inbuf_range = None
    if isinstance(in_buffer, tuple):
        inbuf_name, inbuf_range = in_buffer
    else:
        inbuf_name = in_buffer
    in_desc = sdfg.arrays[inbuf_name]
    inbuf_node = state.add_read(inbuf_name)

    bsizes_range = None
    if isinstance(block_sizes, (list, tuple)):
        if isinstance(block_sizes[0], str):
            bsizes_name, bsizes_range = block_sizes
            bsizes_desc = sdfg.arrays[bsizes_name]
            bsizes_node = state.add_read(bsizes_name)
        else:
            bsizes_name, bsizes_desc = sdfg.add_temp_transient(
                (len(block_sizes), ), dtype=dace.int32)
            bsizes_node = state.add_access(bsizes_name)
            bsizes_tasklet = state.add_tasklet(
                '_set_bsizes_', {}, {'__out'}, ";".join([
                    "__out[{}] = {}".format(i, sz)
                    for i, sz in enumerate(block_sizes)
                ]))
            state.add_edge(bsizes_tasklet, '__out', bsizes_node, None,
                           Memlet.from_array(bsizes_name, bsizes_desc))
    else:
        bsizes_name = block_sizes
        bsizes_desc = sdfg.arrays[bsizes_name]
        bsizes_node = state.add_read(bsizes_name)

    outbuf_range = None
    if isinstance(out_buffer, tuple):
        outbuf_name, outbuf_range = out_buffer
    else:
        outbuf_name = out_buffer
    out_desc = sdfg.arrays[outbuf_name]
    outbuf_node = state.add_write(outbuf_name)

    gdesc = sdfg.add_temp_transient((9, ), dtype=dace.int32)
    gdesc_node = state.add_write(gdesc[0])

    ldesc = sdfg.add_temp_transient((9, ), dtype=dace.int32)
    ldesc_node = state.add_write(ldesc[0])

    if inbuf_range:
        inbuf_mem = Memlet.simple(inbuf_name, inbuf_range)
    else:
        inbuf_mem = Memlet.from_array(inbuf_name, in_desc)
    if bsizes_range:
        bsizes_mem = Memlet.simple(bsizes_name, bsizes_range)
    else:
        bsizes_mem = Memlet.from_array(bsizes_name, bsizes_desc)
    if outbuf_range:
        outbuf_mem = Memlet.simple(outbuf_name, outbuf_range)
    else:
        outbuf_mem = Memlet.from_array(outbuf_name, out_desc)
    gdesc_mem = Memlet.from_array(*gdesc)
    ldesc_mem = Memlet.from_array(*ldesc)

    state.add_edge(inbuf_node, None, libnode, '_inbuffer', inbuf_mem)
    state.add_edge(bsizes_node, None, libnode, '_block_sizes', bsizes_mem)
    state.add_edge(libnode, '_outbuffer', outbuf_node, None, outbuf_mem)
    state.add_edge(libnode, '_gdescriptor', gdesc_node, None, gdesc_mem)
    state.add_edge(libnode, '_ldescriptor', ldesc_node, None, ldesc_mem)

    return [gdesc[0], ldesc[0]]


@oprepo.replaces('dace.comm.BCGather')
def _bcgather(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState,
              in_buffer: str, out_buffer: str,
              block_sizes: Union[str, Sequence[Union[sp.Expr, Number]]]):

    from dace.libraries.pblas.nodes.pgeadd import BlockCyclicGather

    libnode = BlockCyclicGather('_BCGather_')

    inbuf_range = None
    if isinstance(in_buffer, tuple):
        inbuf_name, inbuf_range = in_buffer
    else:
        inbuf_name = in_buffer
    in_desc = sdfg.arrays[inbuf_name]
    inbuf_node = state.add_read(inbuf_name)

    bsizes_range = None
    if isinstance(block_sizes, (list, tuple)):
        if isinstance(block_sizes[0], str):
            bsizes_name, bsizes_range = block_sizes
            bsizes_desc = sdfg.arrays[bsizes_name]
            bsizes_node = state.add_read(bsizes_name)
        else:
            bsizes_name, bsizes_desc = sdfg.add_temp_transient(
                (len(block_sizes), ), dtype=dace.int32)
            bsizes_node = state.add_access(bsizes_name)
            bsizes_tasklet = state.add_tasklet(
                '_set_bsizes_', {}, {'__out'}, ";".join([
                    "__out[{}] = {}".format(i, sz)
                    for i, sz in enumerate(block_sizes)
                ]))
            state.add_edge(bsizes_tasklet, '__out', bsizes_node, None,
                           Memlet.from_array(bsizes_name, bsizes_desc))
    else:
        bsizes_name = block_sizes
        bsizes_desc = sdfg.arrays[bsizes_name]
        bsizes_node = state.add_read(bsizes_name)

    outbuf_range = None
    if isinstance(out_buffer, tuple):
        outbuf_name, outbuf_range = out_buffer
    else:
        outbuf_name = out_buffer
    out_desc = sdfg.arrays[outbuf_name]
    outbuf_node = state.add_write(outbuf_name)

    if inbuf_range:
        inbuf_mem = Memlet.simple(inbuf_name, inbuf_range)
    else:
        inbuf_mem = Memlet.from_array(inbuf_name, in_desc)
    if bsizes_range:
        bsizes_mem = Memlet.simple(bsizes_name, bsizes_range)
    else:
        bsizes_mem = Memlet.from_array(bsizes_name, bsizes_desc)
    if outbuf_range:
        outbuf_mem = Memlet.simple(outbuf_name, outbuf_range)
    else:
        outbuf_mem = Memlet.from_array(outbuf_name, out_desc)

    state.add_edge(inbuf_node, None, libnode, '_inbuffer', inbuf_mem)
    state.add_edge(bsizes_node, None, libnode, '_block_sizes', bsizes_mem)
    state.add_edge(libnode, '_outbuffer', outbuf_node, None, outbuf_mem)

    return None


@oprepo.replaces('distr.MatMult')
def _distr_matmult(pv: 'ProgramVisitor',
                   sdfg: SDFG,
                   state: SDFGState,
                   opa: str,
                   opb: str,
                   shape: Sequence[Union[sp.Expr, Number]],
                   a_block_sizes: Union[str, Sequence[Union[sp.Expr,
                                                            Number]]] = None,
                   b_block_sizes: Union[str, Sequence[Union[sp.Expr,
                                                            Number]]] = None,
                   c_block_sizes: Union[str, Sequence[Union[sp.Expr,
                                                            Number]]] = None):

    arra = sdfg.arrays[opa]
    arrb = sdfg.arrays[opb]

    if len(shape) == 3:
        gm, gn, gk = shape
    else:
        gm, gn = shape

    a_block_sizes = a_block_sizes or arra.shape
    if len(a_block_sizes) < 2:
        a_block_sizes = (a_block_sizes[0], 1)
    b_block_sizes = b_block_sizes or arrb.shape
    if len(b_block_sizes) < 2:
        b_block_sizes = (b_block_sizes[0], 1)

    if len(arra.shape) == 1 and len(arrb.shape) == 2:
        a_block_sizes, b_block_sizes = b_block_sizes, a_block_sizes

    a_bsizes_range = None
    if isinstance(a_block_sizes, (list, tuple)):
        if isinstance(a_block_sizes[0], str):
            a_bsizes_name, a_bsizes_range = a_block_sizes
            a_bsizes_desc = sdfg.arrays[a_bsizes_name]
            a_bsizes_node = state.add_read(a_bsizes_name)
        else:
            a_bsizes_name, a_bsizes_desc = sdfg.add_temp_transient(
                (len(a_block_sizes), ), dtype=dace.int32)
            a_bsizes_node = state.add_access(a_bsizes_name)
            a_bsizes_tasklet = state.add_tasklet(
                '_set_a_bsizes_', {}, {'__out'}, ";".join([
                    "__out[{}] = {}".format(i, sz)
                    for i, sz in enumerate(a_block_sizes)
                ]))
            state.add_edge(a_bsizes_tasklet, '__out', a_bsizes_node, None,
                           Memlet.from_array(a_bsizes_name, a_bsizes_desc))
    else:
        a_bsizes_name = a_block_sizes
        a_bsizes_desc = sdfg.arrays[a_bsizes_name]
        a_bsizes_node = state.add_read(a_bsizes_name)

    b_bsizes_range = None
    if isinstance(a_block_sizes, (list, tuple)):
        if isinstance(a_block_sizes[0], str):
            b_bsizes_name, b_sizes_range = b_block_sizes
            b_bsizes_desc = sdfg.arrays[b_bsizes_name]
            b_bsizes_node = state.add_read(b_bsizes_name)
        else:
            b_bsizes_name, b_bsizes_desc = sdfg.add_temp_transient(
                (len(b_block_sizes), ), dtype=dace.int32)
            b_bsizes_node = state.add_access(b_bsizes_name)
            b_bsizes_tasklet = state.add_tasklet(
                '_set_b_sizes_', {}, {'__out'}, ";".join([
                    "__out[{}] = {}".format(i, sz)
                    for i, sz in enumerate(b_block_sizes)
                ]))
            state.add_edge(b_bsizes_tasklet, '__out', b_bsizes_node, None,
                           Memlet.from_array(b_bsizes_name, b_bsizes_desc))
    else:
        b_bsizes_name = b_block_sizes
        b_bsizes_desc = sdfg.arrays[b_bsizes_name]
        b_bsizes_node = state.add_read(b_bsizes_name)

    if len(arra.shape) == 2 and len(arrb.shape) == 2:
        # Gemm
        from dace.libraries.pblas.nodes.pgemm import Pgemm
        tasklet = Pgemm("__DistrMatMult__", gm, gn, gk)
        m = arra.shape[0]
        n = arrb.shape[-1]
        out = sdfg.add_temp_transient((m, n), dtype=arra.dtype)
    elif len(arra.shape) == 2 and len(arrb.shape) == 1:
        # Gemv
        from dace.libraries.pblas.nodes.pgemv import Pgemv
        tasklet = Pgemv("__DistrMatVecMult__", m=gm, n=gn)
        if c_block_sizes:
            m = c_block_sizes[0]
        else:
            m = arra.shape[0]
        out = sdfg.add_temp_transient((m, ), dtype=arra.dtype)
    elif len(arra.shape) == 1 and len(arrb.shape) == 2:
        # Gemv transposed
        # Swap a and b
        opa, opb = opb, opa
        arra, arrb = arrb, arra
        from dace.libraries.pblas.nodes.pgemv import Pgemv
        tasklet = Pgemv("__DistrMatVecMult__", transa='T', m=gm, n=gn)
        if c_block_sizes:
            n = c_block_sizes[0]
        else:
            n = arra.shape[1]
        out = sdfg.add_temp_transient((n, ), dtype=arra.dtype)

    anode = state.add_read(opa)
    bnode = state.add_read(opb)
    cnode = state.add_write(out[0])

    if a_bsizes_range:
        a_bsizes_mem = Memlet.simple(a_bsizes_name, a_bsizes_range)
    else:
        a_bsizes_mem = Memlet.from_array(a_bsizes_name, a_bsizes_desc)
    if b_bsizes_range:
        b_bsizes_mem = Memlet.simple(b_bsizes_name, b_bsizes_range)
    else:
        b_bsizes_mem = Memlet.from_array(b_bsizes_name, b_bsizes_desc)

    state.add_edge(anode, None, tasklet, '_a', Memlet.from_array(opa, arra))
    state.add_edge(bnode, None, tasklet, '_b', Memlet.from_array(opb, arrb))
    state.add_edge(a_bsizes_node, None, tasklet, '_a_block_sizes', a_bsizes_mem)
    state.add_edge(b_bsizes_node, None, tasklet, '_b_block_sizes', b_bsizes_mem)
    state.add_edge(tasklet, '_c', cnode, None, Memlet.from_array(*out))

    return out[0]
