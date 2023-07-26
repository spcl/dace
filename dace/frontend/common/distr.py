# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from numbers import Integral, Number
from typing import Sequence, Tuple, Union

import dace
from dace import dtypes, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState

import sympy as sp

from dace.frontend.python.replacements import _define_local_scalar

ShapeType = Sequence[Union[Integral, str, symbolic.symbol, symbolic.SymExpr, symbolic.sympy.Basic]]
RankType = Union[Integral, str, symbolic.symbol, symbolic.SymExpr, symbolic.sympy.Basic]
ProgramVisitor = 'dace.frontend.python.newast.ProgramVisitor'

##### MPI Communicators
# a helper function for getting an access node by argument name
# creates a scalar if it's a number
def _get_int_arg_node(pv: ProgramVisitor,
                     sdfg: SDFG,
                     state: SDFGState,
                     arg_name: Union[str, sp.Expr, Number]
                    ):
    if isinstance(arg_name, str) and arg_name in sdfg.arrays.keys():
        arg_name = arg_name
        arg_node = state.add_read(arg_name)
    else:
        # create a transient scalar and take its name
        arg_name = _define_local_scalar(pv, sdfg, state, dace.int32)
        arg_node = state.add_access(arg_name)
        # every tasklet is in different scope, no need to worry about name confilct
        color_tasklet = state.add_tasklet(f'_set_{arg_name}_', {}, {'__out'}, f'__out = {arg_name}')
        state.add_edge(color_tasklet, '__out', arg_node, None, Memlet.simple(arg_node, '0'))

    return arg_name, arg_node


@oprepo.replaces('mpi4py.MPI.COMM_WORLD.Split')
@oprepo.replaces('dace.comm.Split')
def _comm_split(pv: 'ProgramVisitor',
                sdfg: SDFG,
                state: SDFGState,
                color: Union[str, sp.Expr, Number] = 0,
                key: Union[str, sp.Expr, Number] = 0,
                grid: str = None):
    """ Splits communicator
    """

    from dace.libraries.mpi.nodes.comm_split import Comm_split

    # fine a new comm world name
    comm_name = sdfg.add_comm()

    comm_split_node = Comm_split(comm_name, grid)

    _, color_node = _get_int_arg_node(pv, sdfg, state, color)
    _, key_node = _get_int_arg_node(pv, sdfg, state, key)

    state.add_edge(color_node, None, comm_split_node, '_color', Memlet.simple(color_node, "0:1", num_accesses=1))
    state.add_edge(key_node, None, comm_split_node, '_key', Memlet.simple(key_node, "0:1", num_accesses=1))

    # Pseudo-writing for newast.py #3195 check and complete Processcomm creation
    _, scal = sdfg.add_scalar(comm_name, dace.int32, transient=True)
    wnode = state.add_write(comm_name)
    state.add_edge(comm_split_node, "_out", wnode, None, Memlet.from_array(comm_name, scal))

    # return value will be the name of this splited communicator
    return comm_name


@oprepo.replaces_method('Cartcomm', 'Split')
@oprepo.replaces_method('Intracomm', 'Split')
def _intracomm_comm_split(pv: 'ProgramVisitor',
                     sdfg: SDFG,
                     state: SDFGState,
                     comm: Tuple[str, 'Comm'],
                     color: Union[str, sp.Expr, Number] = 0,
                     key: Union[str, sp.Expr, Number] = 0):
    """ Equivalent to `dace.comm.Bcast(buffer, root)`. """
    from mpi4py import MPI
    comm_name, comm_obj = comm
    if comm_obj == MPI.COMM_WORLD:
        return _comm_split(pv, sdfg, state, color, key)
    raise ValueError('Only the mpi4py.MPI.COMM_WORLD Intracomm is supported in DaCe Python programs.')


@oprepo.replaces_method('ProcessComm', 'Split')
def _intracomm_comm_split(pv: 'ProgramVisitor',
                     sdfg: SDFG,
                     state: SDFGState,
                     comm: Tuple[str, 'Comm'],
                     color: Union[str, sp.Expr, Number] = 0,
                     key: Union[str, sp.Expr, Number] = 0):
    """ Equivalent to `dace.comm.Bcast(buffer, root)`. """
    return _comm_split(pv, sdfg, state, color, key, grid=comm)


##### MPI Cartesian Communicators


@oprepo.replaces('mpi4py.MPI.COMM_WORLD.Create_cart')
@oprepo.replaces('dace.comm.Cart_create')
def _cart_create(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, dims: ShapeType):
    """ Creates a process-grid and adds it to the DaCe program. The process-grid is implemented with [MPI_Cart_create](https://www.mpich.org/static/docs/latest/www3/MPI_Cart_create.html).
        :param dims: Shape of the process-grid (see `dims` parameter of `MPI_Cart_create`), e.g., [2, 3, 3].
        :return: Name of the new process-grid descriptor.
    """
    pgrid_name = sdfg.add_pgrid(dims)

    # Dummy tasklet adds MPI variables to the program's state.
    from dace.libraries.mpi import Dummy
    tasklet = Dummy(pgrid_name, [
        f'MPI_Comm {pgrid_name}_comm;',
        f'MPI_Group {pgrid_name}_group;',
        f'int {pgrid_name}_coords[{len(dims)}];',
        f'int {pgrid_name}_dims[{len(dims)}];',
        f'int {pgrid_name}_rank;',
        f'int {pgrid_name}_size;',
        f'bool {pgrid_name}_valid;',
    ])

    state.add_node(tasklet)

    # Pseudo-writing to a dummy variable to avoid removal of Dummy node by transformations.
    _, scal = sdfg.add_scalar(pgrid_name, dace.int32, transient=True)
    wnode = state.add_write(pgrid_name)
    state.add_edge(tasklet, '__out', wnode, None, Memlet.from_array(pgrid_name, scal))

    return pgrid_name


@oprepo.replaces_method('Intracomm', 'Create_cart')
def _intracomm_create(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, icomm: 'Intracomm', dims: ShapeType):
    """ Equivalent to `dace.comm.Cart_create(dims).
        :param dims: Shape of the process-grid (see `dims` parameter of `MPI_Cart_create`), e.g., [2, 3, 3].
        :return: Name of the new process-grid descriptor.
    """

    from mpi4py import MPI
    icomm_name, icomm_obj = icomm
    if icomm_obj != MPI.COMM_WORLD:
        raise ValueError('Only the mpi4py.MPI.COMM_WORLD Intracomm is supported in DaCe Python programs.')
    return _cart_create(pv, sdfg, state, dims)


@oprepo.replaces('dace.comm.Cart_sub')
def _cart_sub(pv: 'ProgramVisitor',
              sdfg: SDFG,
              state: SDFGState,
              parent_grid: str,
              color: Sequence[Union[Integral, bool]],
              exact_grid: RankType = None):
    """ Partitions the `parent_grid` to lower-dimensional sub-grids and adds them to the DaCe program.
        The sub-grids are implemented with [MPI_Cart_sub](https://www.mpich.org/static/docs/latest/www3/MPI_Cart_sub.html).
        :param parent_grid: Parent process-grid (similar to the `comm` parameter of `MPI_Cart_sub`).
        :param color: The i-th entry specifies whether the i-th dimension is kept in the sub-grid or is dropped (see `remain_dims` input of `MPI_Cart_sub`).
        :param exact_grid: [DEVELOPER] If set then, out of all the sub-grids created, only the one that contains the rank with id `exact_grid` will be utilized for collective communication.
        :return: Name of the new sub-grid descriptor.
    """
    pgrid_name = sdfg.add_pgrid(parent_grid=parent_grid, color=color, exact_grid=exact_grid)

    # Count sub-grid dimensions.
    pgrid_ndims = sum([bool(c) for c in color])

    # Dummy tasklet adds MPI variables to the program's state.
    from dace.libraries.mpi import Dummy
    tasklet = Dummy(pgrid_name, [
        f'MPI_Comm {pgrid_name}_comm;',
        f'MPI_Group {pgrid_name}_group;',
        f'int {pgrid_name}_coords[{pgrid_ndims}];',
        f'int {pgrid_name}_dims[{pgrid_ndims}];',
        f'int {pgrid_name}_rank;',
        f'int {pgrid_name}_size;',
        f'bool {pgrid_name}_valid;',
    ])

    state.add_node(tasklet)

    # Pseudo-writing to a dummy variable to avoid removal of Dummy node by transformations.
    _, scal = sdfg.add_scalar(pgrid_name, dace.int32, transient=True)
    wnode = state.add_write(pgrid_name)
    state.add_edge(tasklet, '__out', wnode, None, Memlet.from_array(pgrid_name, scal))

    return pgrid_name


@oprepo.replaces_method('ProcessGrid', 'Sub')
def _pgrid_sub(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, parent_grid: str, color: Sequence[Union[Integral,
                                                                                                           bool]]):
    """ Equivalent to `dace.comm.Cart_sub(parent_grid, color).
        :param parent_grid: Parent process-grid (similar to the `comm` parameter of `MPI_Cart_sub`).
        :param color: The i-th entry specifies whether the i-th dimension is kept in the sub-grid or is dropped (see `remain_dims` input of `MPI_Cart_sub`).
        :return: Name of the new sub-grid descriptor.
    """

    return _cart_sub(pv, sdfg, state, parent_grid, color)


@oprepo.replaces_operator('ProcessGrid', 'Eq', otherclass='Comm')
@oprepo.replaces_operator('ProcessGrid', 'Is', otherclass='Comm')
def _pgrid_eq_comm(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: 'Comm'):
    from mpi4py import MPI
    if op2 is MPI.COMM_WORLD or op2 is MPI.COMM_NULL:
        return False
    return True


@oprepo.replaces_operator('Comm', 'Eq', otherclass='ProcessGrid')
@oprepo.replaces_operator('Comm', 'Is', otherclass='ProcessGrid')
def _comm_eq_pgrid(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: 'Comm', op2: 'str'):
    from mpi4py import MPI
    if op1 is MPI.COMM_WORLD or op1 is MPI.COMM_NULL:
        return False
    return True


@oprepo.replaces_operator('ProcessGrid', 'NotEq', otherclass='Comm')
@oprepo.replaces_operator('ProcessGrid', 'IsNot', otherclass='Comm')
def _pgrid_neq_comm(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: 'Comm'):
    from mpi4py import MPI
    if op2 is MPI.COMM_WORLD or op2 is MPI.COMM_NULL:
        return True
    return False


@oprepo.replaces_operator('Comm', 'NotEq', otherclass='ProcessGrid')
@oprepo.replaces_operator('Comm', 'IsNot', otherclass='ProcessGrid')
def _comm_neq_pgrid(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: 'Comm', op2: 'str'):
    from mpi4py import MPI
    if op1 is MPI.COMM_WORLD or op1 is MPI.COMM_NULL:
        return True
    return False


##### MPI Collectives


# @oprepo.replaces('mpi4py.MPI.COMM_WORLD.Bcast')
@oprepo.replaces('dace.comm.Bcast')
def _bcast(pv: ProgramVisitor,
           sdfg: SDFG,
           state: SDFGState,
           buffer: str,
           root: Union[str, sp.Expr, Number] = 0,
           grid: str = None,
           fcomm: str = None):

    from dace.libraries.mpi.nodes.bcast import Bcast

    libnode = Bcast('_Bcast_', grid, fcomm)
    desc = sdfg.arrays[buffer]
    in_buffer = state.add_read(buffer)
    out_buffer = state.add_write(buffer)
    if grid:
        comm_node = state.add_read(grid)
        comm_desc = sdfg.arrays[grid]
        state.add_edge(comm_node, None, libnode, None, Memlet.from_array(grid, comm_desc))

    if isinstance(root, str) and root in sdfg.arrays.keys():
        root_node = state.add_read(root)
    else:
        storage = desc.storage
        root_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        root_node = state.add_access(root_name)
        root_tasklet = state.add_tasklet('_set_root_', {}, {'__out'}, '__out = {}'.format(root))
        state.add_edge(root_tasklet, '__out', root_node, None, Memlet.simple(root_name, '0'))
    state.add_edge(in_buffer, None, libnode, '_inbuffer', Memlet.from_array(buffer, desc))
    state.add_edge(root_node, None, libnode, '_root', Memlet.simple(root_node.data, '0'))
    state.add_edge(libnode, '_outbuffer', out_buffer, None, Memlet.from_array(buffer, desc))

    return None


@oprepo.replaces_method('Cartcomm', 'Bcast')
@oprepo.replaces_method('Intracomm', 'Bcast')
def _intracomm_bcast(pv: 'ProgramVisitor',
                     sdfg: SDFG,
                     state: SDFGState,
                     comm: Tuple[str, 'Comm'],
                     buffer: str,
                     root: Union[str, sp.Expr, Number] = 0):
    """ Equivalent to `dace.comm.Bcast(buffer, root)`. """

    from mpi4py import MPI
    comm_name, comm_obj = comm
    if comm_obj == MPI.COMM_WORLD:
        return _bcast(pv, sdfg, state, buffer, root)
    # NOTE: Highly experimental
    sdfg.add_scalar(comm_name, dace.int32)
    return _bcast(pv, sdfg, state, buffer, root, fcomm=comm_name)


@oprepo.replaces_method('ProcessComm', 'Bcast')
@oprepo.replaces_method('ProcessGrid', 'Bcast')
def _pgrid_bcast(pv: 'ProgramVisitor',
                 sdfg: SDFG,
                 state: SDFGState,
                 pgrid: str,
                 buffer: str,
                 root: Union[str, sp.Expr, Number] = 0):
    """ Equivalent to `dace.comm.Bcast(buffer, root, grid=pgrid)`. """

    return _bcast(pv, sdfg, state, buffer, root, grid=pgrid)


def _mpi4py_to_MPI(MPI, op):
    if op is MPI.SUM:
        return 'MPI_SUM'
    raise NotImplementedError


@oprepo.replaces('dace.comm.Reduce')
def _Reduce(pv: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            buffer: str,
            op: str,
            root: Union[str, sp.Expr, Number] = 0,
            grid: str = None):

    from dace.libraries.mpi.nodes.reduce import Reduce

    libnode = Reduce('_Reduce_', op, grid)
    desc = sdfg.arrays[buffer]
    in_buffer = state.add_read(buffer)
    out_buffer = state.add_write(buffer)
    if isinstance(root, str) and root in sdfg.arrays.keys():
        root_node = state.add_read(root)
    else:
        storage = desc.storage
        root_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        root_node = state.add_access(root_name)
        root_tasklet = state.add_tasklet('_set_root_', {}, {'__out'}, '__out = {}'.format(root))
        state.add_edge(root_tasklet, '__out', root_node, None, Memlet.simple(root_name, '0'))
    state.add_edge(in_buffer, None, libnode, '_inbuffer', Memlet.from_array(buffer, desc))
    state.add_edge(root_node, None, libnode, '_root', Memlet.simple(root_node.data, '0'))
    state.add_edge(libnode, '_outbuffer', out_buffer, None, Memlet.from_array(buffer, desc))

    return None


@oprepo.replaces('dace.comm.Alltoall')
def _alltoall(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, inbuffer: str, outbuffer: str, grid: str = None):

    from dace.libraries.mpi.nodes.alltoall import Alltoall

    libnode = Alltoall('_Alltoall_', grid)
    in_desc = sdfg.arrays[inbuffer]
    in_buffer = state.add_read(inbuffer)
    out_desc = sdfg.arrays[outbuffer]
    out_buffer = state.add_write(outbuffer)
    state.add_edge(in_buffer, None, libnode, '_inbuffer', Memlet.from_array(in_buffer, in_desc))
    state.add_edge(libnode, '_outbuffer', out_buffer, None, Memlet.from_array(out_buffer, out_desc))

    return None


@oprepo.replaces_method('Intracomm', 'Alltoall')
def _intracomm_alltoall(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, icomm: 'Intracomm', inp_buffer: str,
                        out_buffer: str):
    """ Equivalent to `dace.comm.Alltoall(inp_buffer, out_buffer)`. """

    from mpi4py import MPI
    icomm_name, icomm_obj = icomm
    if icomm_obj != MPI.COMM_WORLD:
        raise ValueError('Only the mpi4py.MPI.COMM_WORLD Intracomm is supported in DaCe Python programs.')
    return _alltoall(pv, sdfg, state, inp_buffer, out_buffer)


@oprepo.replaces_method('ProcessComm', 'Alltoall')
@oprepo.replaces_method('ProcessGrid', 'Alltoall')
def _pgrid_alltoall(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, pgrid: str, inp_buffer: str, out_buffer: str):
    """ Equivalent to `dace.comm.Alltoall(inp_buffer, out_buffer, grid=pgrid)`. """

    from mpi4py import MPI
    return _alltoall(pv, sdfg, state, inp_buffer, out_buffer, grid=pgrid)


@oprepo.replaces('dace.comm.Allreduce')
def _allreduce(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, buffer: str, op: str, grid: str = None):

    from dace.libraries.mpi.nodes.allreduce import Allreduce

    libnode = Allreduce('_Allreduce_', op, grid)
    desc = sdfg.arrays[buffer]
    in_buffer = state.add_read(buffer)
    out_buffer = state.add_write(buffer)
    state.add_edge(in_buffer, None, libnode, '_inbuffer', Memlet.from_array(buffer, desc))
    state.add_edge(libnode, '_outbuffer', out_buffer, None, Memlet.from_array(buffer, desc))

    return None


@oprepo.replaces_method('Intracomm', 'Allreduce')
def _intracomm_allreduce(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, icomm: 'Intracomm', inp_buffer: 'InPlace',
                         out_buffer: str, op: str):
    """ Equivalent to `dace.comm.Allreduce(out_buffer, op)`. """

    from mpi4py import MPI
    icomm_name, icomm_obj = icomm
    if icomm_obj != MPI.COMM_WORLD:
        raise ValueError('Only the mpi4py.MPI.COMM_WORLD Intracomm is supported in DaCe Python programs.')
    if inp_buffer != MPI.IN_PLACE:
        raise ValueError('DaCe currently supports in-place Allreduce only.')
    if isinstance(op, MPI.Op):
        op = _mpi4py_to_MPI(MPI, op)
    return _allreduce(pv, sdfg, state, out_buffer, op)


@oprepo.replaces_method('ProcessGrid', 'Allreduce')
def _pgrid_allreduce(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, pgrid: str, inp_buffer: 'InPlace',
                     out_buffer: str, op: str):
    """ Equivalent to `dace.comm.Allreduce(out_buffer, op, grid=pgrid)`. """

    from mpi4py import MPI
    if inp_buffer != MPI.IN_PLACE:
        raise ValueError('DaCe currently supports in-place Allreduce only.')
    if isinstance(op, MPI.Op):
        op = _mpi4py_to_MPI(MPI, op)
    return _allreduce(pv, sdfg, state, out_buffer, op, grid=pgrid)


@oprepo.replaces('dace.comm.Scatter')
def _scatter(pv: ProgramVisitor,
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
        root_tasklet = state.add_tasklet('_set_root_', {}, {'__out'}, '__out = {}'.format(root))
        state.add_edge(root_tasklet, '__out', root_node, None, Memlet.simple(root_name, '0'))
    state.add_edge(in_node, None, libnode, '_inbuffer', Memlet.from_array(in_buffer, in_desc))
    state.add_edge(root_node, None, libnode, '_root', Memlet.simple(root_node.data, '0'))
    state.add_edge(libnode, '_outbuffer', out_node, None, Memlet.from_array(out_buffer, out_desc))

    return None


@oprepo.replaces('dace.comm.Gather')
def _gather(pv: ProgramVisitor,
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
        root_tasklet = state.add_tasklet('_set_root_', {}, {'__out'}, '__out = {}'.format(root))
        state.add_edge(root_tasklet, '__out', root_node, None, Memlet.simple(root_name, '0'))
    state.add_edge(in_node, None, libnode, '_inbuffer', Memlet.from_array(in_buffer, in_desc))
    state.add_edge(root_node, None, libnode, '_root', Memlet.simple(root_node.data, '0'))
    state.add_edge(libnode, '_outbuffer', out_node, None, Memlet.from_array(out_buffer, out_desc))

    return None


##### Point-To-Point Communication


@oprepo.replaces('dace.comm.Send')
def _send(pv: ProgramVisitor,
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
    conn = {c: (dtypes.pointer(desc.dtype) if c == '_buffer' else t) for c, t in conn.items()}
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
        dst_tasklet = state.add_tasklet('_set_dst_', {}, {'__out'}, '__out = {}'.format(dst))
        state.add_edge(dst_tasklet, '__out', dst_node, None, Memlet.simple(dst_name, '0'))

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
        tag_tasklet = state.add_tasklet('_set_tag_', {}, {'__out'}, '__out = {}'.format(tag))
        state.add_edge(tag_tasklet, '__out', tag_node, None, Memlet.simple(tag_name, '0'))

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


@oprepo.replaces_method('Intracomm', 'Send')
def _intracomm_send(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, icomm: 'Intracomm', buffer: str,
                     dst: Union[str, sp.Expr, Number], tag: Union[str, sp.Expr, Number]):
    """ Equivalent to `dace.comm.end(buffer, dst, tag)`. """

    from mpi4py import MPI
    icomm_name, icomm_obj = icomm
    if icomm_obj != MPI.COMM_WORLD:
        raise ValueError('Only the mpi4py.MPI.COMM_WORLD Intracomm is supported in DaCe Python programs.')
    return _send(pv, sdfg, state, buffer, dst, tag)


@oprepo.replaces_method('ProcessGrid', 'Send')
def _pgrid_send(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, pgrid: str, buffer: str,
                 dst: Union[str, sp.Expr, Number], tag: Union[str, sp.Expr, Number]):
    """ Equivalent to `dace.comm.Send(buffer, dst, tag, grid=pgrid)`. """

    raise NotImplementedError('ProcessGrid.Send is not supported yet.')
    # return _send(pv, sdfg, state, buffer, dst, tag, grid=pgrid)


@oprepo.replaces('dace.comm.Isend')
def _isend(pv: ProgramVisitor,
           sdfg: SDFG,
           state: SDFGState,
           buffer: str,
           dst: Union[str, sp.Expr, Number],
           tag: Union[str, sp.Expr, Number],
           request: str = None,
           grid: str = None):

    from dace.libraries.mpi.nodes.isend import Isend

    ret_req = False
    if not request:
        ret_req = True
        request, _ = sdfg.add_array("isend_req", [1],
                                    dace.dtypes.opaque("MPI_Request"),
                                    transient=True,
                                    find_new_name=True)

    libnode = Isend('_Isend_', grid=grid)

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
    iconn = {c: (dtypes.pointer(desc.dtype) if c == '_buffer' else t) for c, t in iconn.items()}
    libnode.in_connectors = iconn
    oconn = libnode.out_connectors
    oconn = {c: (dtypes.pointer(req_desc.dtype) if c == '_request' else t) for c, t in oconn.items()}
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
        dst_tasklet = state.add_tasklet('_set_dst_', {}, {'__out'}, '__out = {}'.format(dst))
        state.add_edge(dst_tasklet, '__out', dst_node, None, Memlet.simple(dst_name, '0'))

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
        tag_tasklet = state.add_tasklet('_set_tag_', {}, {'__out'}, '__out = {}'.format(tag))
        state.add_edge(tag_tasklet, '__out', tag_node, None, Memlet.simple(tag_name, '0'))

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

    if ret_req:
        return request
    return None


@oprepo.replaces_method('Intracomm', 'Isend')
def _intracomm_isend(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, icomm: 'Intracomm', buffer: str,
                     dst: Union[str, sp.Expr, Number], tag: Union[str, sp.Expr, Number]):
    """ Equivalent to `dace.comm.Isend(buffer, dst, tag, req)`. """

    from mpi4py import MPI
    icomm_name, icomm_obj = icomm
    if icomm_obj != MPI.COMM_WORLD:
        raise ValueError('Only the mpi4py.MPI.COMM_WORLD Intracomm is supported in DaCe Python programs.')
    req, _ = sdfg.add_array("isend_req", [1], dace.dtypes.opaque("MPI_Request"), transient=True, find_new_name=True)
    _isend(pv, sdfg, state, buffer, dst, tag, req)
    return req


@oprepo.replaces_method('ProcessGrid', 'Isend')
def _pgrid_isend(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, pgrid: str, buffer: str,
                 dst: Union[str, sp.Expr, Number], tag: Union[str, sp.Expr, Number]):
    """ Equivalent to `dace.comm.Isend(buffer, dst, tag, req, grid=pgrid)`. """

    from mpi4py import MPI
    req, _ = sdfg.add_array("isend_req", [1], dace.dtypes.opaque("MPI_Request"), transient=True, find_new_name=True)
    _isend(pv, sdfg, state, buffer, dst, tag, req, grid=pgrid)
    return req


@oprepo.replaces('dace.comm.Recv')
def _recv(pv: ProgramVisitor,
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
    conn = {c: (dtypes.pointer(desc.dtype) if c == '_buffer' else t) for c, t in conn.items()}
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
        src_tasklet = state.add_tasklet('_set_src_', {}, {'__out'}, '__out = {}'.format(src))
        state.add_edge(src_tasklet, '__out', src_node, None, Memlet.simple(src_name, '0'))

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
        tag_tasklet = state.add_tasklet('_set_tag_', {}, {'__out'}, '__out = {}'.format(tag))
        state.add_edge(tag_tasklet, '__out', tag_node, None, Memlet.simple(tag_name, '0'))

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


@oprepo.replaces_method('Intracomm', 'Recv')
def _intracomm_Recv(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, icomm: 'Intracomm', buffer: str,
                     src: Union[str, sp.Expr, Number], tag: Union[str, sp.Expr, Number]):
    """ Equivalent to `dace.comm.Recv(buffer, src, tagq)`. """

    from mpi4py import MPI
    icomm_name, icomm_obj = icomm
    if icomm_obj != MPI.COMM_WORLD:
        raise ValueError('Only the mpi4py.MPI.COMM_WORLD Intracomm is supported in DaCe Python programs.')
    return _recv(pv, sdfg, state, buffer, src, tag)


@oprepo.replaces_method('ProcessGrid', 'Recv')
def _pgrid_irecv(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, pgrid: str, buffer: str,
                 src: Union[str, sp.Expr, Number], tag: Union[str, sp.Expr, Number]):
    """ Equivalent to `dace.comm.Recv(buffer, dst, tag, grid=pgrid)`. """

    raise NotImplementedError('ProcessGrid.Recv is not supported yet.')
    # return _recv(pv, sdfg, state, buffer, src, tag, req, grid=pgrid)


@oprepo.replaces('dace.comm.Irecv')
def _irecv(pv: ProgramVisitor,
           sdfg: SDFG,
           state: SDFGState,
           buffer: str,
           src: Union[str, sp.Expr, Number],
           tag: Union[str, sp.Expr, Number],
           request: str = None,
           grid: str = None):

    from dace.libraries.mpi.nodes.irecv import Irecv

    ret_req = False
    if not request:
        ret_req = True
        request, _ = sdfg.add_array("irecv_req", [1],
                                    dace.dtypes.opaque("MPI_Request"),
                                    transient=True,
                                    find_new_name=True)

    libnode = Irecv('_Irecv_', grid=grid)

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
    conn = {c: (dtypes.pointer(desc.dtype) if c == '_buffer' else t) for c, t in conn.items()}
    conn = {c: (dtypes.pointer(req_desc.dtype) if c == '_request' else t) for c, t in conn.items()}
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
        src_tasklet = state.add_tasklet('_set_src_', {}, {'__out'}, '__out = {}'.format(src))
        state.add_edge(src_tasklet, '__out', src_node, None, Memlet.simple(src_name, '0'))

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
        tag_tasklet = state.add_tasklet('_set_tag_', {}, {'__out'}, '__out = {}'.format(tag))
        state.add_edge(tag_tasklet, '__out', tag_node, None, Memlet.simple(tag_name, '0'))

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

    if ret_req:
        return request
    return None


@oprepo.replaces_method('Intracomm', 'Irecv')
def _intracomm_irecv(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, icomm: 'Intracomm', buffer: str,
                     src: Union[str, sp.Expr, Number], tag: Union[str, sp.Expr, Number]):
    """ Equivalent to `dace.comm.Irecv(buffer, src, tag, req)`. """

    from mpi4py import MPI
    icomm_name, icomm_obj = icomm
    if icomm_obj != MPI.COMM_WORLD:
        raise ValueError('Only the mpi4py.MPI.COMM_WORLD Intracomm is supported in DaCe Python programs.')
    req, _ = sdfg.add_array("irecv_req", [1], dace.dtypes.opaque("MPI_Request"), transient=True, find_new_name=True)
    _irecv(pv, sdfg, state, buffer, src, tag, req)
    return req


@oprepo.replaces_method('ProcessGrid', 'Irecv')
def _pgrid_irecv(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, pgrid: str, buffer: str,
                 src: Union[str, sp.Expr, Number], tag: Union[str, sp.Expr, Number]):
    """ Equivalent to `dace.comm.Isend(buffer, dst, tag, req, grid=pgrid)`. """

    from mpi4py import MPI
    req, _ = sdfg.add_array("irecv_req", [1], dace.dtypes.opaque("MPI_Request"), transient=True, find_new_name=True)
    _irecv(pv, sdfg, state, buffer, src, tag, req, grid=pgrid)
    return req


@oprepo.replaces('dace.comm.Wait')
def _wait(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, request: str):

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
    state.add_edge(libnode, '_stat_source', src_node, None, Memlet.from_array(*src))
    state.add_edge(libnode, '_stat_tag', tag_node, None, Memlet.from_array(*tag))

    return None


@oprepo.replaces('mpi4py.MPI.Request.Waitall')
@oprepo.replaces('dace.comm.Waitall')
def _wait(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, request: str):

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


@oprepo.replaces('dace.comm.Cart_create')
def _cart_create(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, dims: ShapeType):
    """ Creates a process-grid and adds it to the DaCe program. The process-grid is implemented with [MPI_Cart_create](https://www.mpich.org/static/docs/latest/www3/MPI_Cart_create.html).

        :param dims: Shape of the process-grid (see `dims` parameter of `MPI_Cart_create`), e.g., [2, 3, 3].
        :return: Name of the new process-grid descriptor.
    """
    pgrid_name = sdfg.add_pgrid(dims)

    # Dummy tasklet adds MPI variables to the program's state.
    from dace.libraries.mpi import Dummy
    tasklet = Dummy(pgrid_name, [
        f'MPI_Comm {pgrid_name}_comm;',
        f'MPI_Group {pgrid_name}_group;',
        f'int {pgrid_name}_coords[{len(dims)}];',
        f'int {pgrid_name}_dims[{len(dims)}];',
        f'int {pgrid_name}_rank;',
        f'int {pgrid_name}_size;',
        f'bool {pgrid_name}_valid;',
    ])

    state.add_node(tasklet)

    # Pseudo-writing to a dummy variable to avoid removal of Dummy node by transformations.
    _, scal = sdfg.add_scalar(pgrid_name, dace.int32, transient=True)
    wnode = state.add_write(pgrid_name)
    state.add_edge(tasklet, '__out', wnode, None, Memlet.from_array(pgrid_name, scal))

    return pgrid_name


@oprepo.replaces('dace.comm.Cart_sub')
def _cart_sub(pv: ProgramVisitor,
              sdfg: SDFG,
              state: SDFGState,
              parent_grid: str,
              color: Sequence[Union[Integral, bool]],
              exact_grid: RankType = None):
    """ Partitions the `parent_grid` to lower-dimensional sub-grids and adds them to the DaCe program.
        The sub-grids are implemented with [MPI_Cart_sub](https://www.mpich.org/static/docs/latest/www3/MPI_Cart_sub.html).

        :param parent_grid: Parent process-grid (similar to the `comm` parameter of `MPI_Cart_sub`).
        :param color: The i-th entry specifies whether the i-th dimension is kept in the sub-grid or is dropped (see `remain_dims` input of `MPI_Cart_sub`).
        :param exact_grid: [DEVELOPER] If set then, out of all the sub-grids created, only the one that contains the rank with id `exact_grid` will be utilized for collective communication.
        :return: Name of the new sub-grid descriptor.
    """
    pgrid_name = sdfg.add_pgrid(parent_grid=parent_grid, color=color, exact_grid=exact_grid)

    # Count sub-grid dimensions.
    pgrid_ndims = sum([bool(c) for c in color])

    # Dummy tasklet adds MPI variables to the program's state.
    from dace.libraries.mpi import Dummy
    tasklet = Dummy(pgrid_name, [
        f'MPI_Comm {pgrid_name}_comm;',
        f'MPI_Group {pgrid_name}_group;',
        f'int {pgrid_name}_coords[{pgrid_ndims}];',
        f'int {pgrid_name}_dims[{pgrid_ndims}];',
        f'int {pgrid_name}_rank;',
        f'int {pgrid_name}_size;',
        f'bool {pgrid_name}_valid;',
    ])

    state.add_node(tasklet)

    # Pseudo-writing to a dummy variable to avoid removal of Dummy node by transformations.
    _, scal = sdfg.add_scalar(pgrid_name, dace.int32, transient=True)
    wnode = state.add_write(pgrid_name)
    state.add_edge(tasklet, '__out', wnode, None, Memlet.from_array(pgrid_name, scal))

    return pgrid_name


@oprepo.replaces('dace.comm.Subarray')
def _subarray(pv: ProgramVisitor,
              sdfg: SDFG,
              state: SDFGState,
              array: Union[str, ShapeType],
              subarray: Union[str, ShapeType],
              dtype: dtypes.typeclass = None,
              process_grid: str = None,
              correspondence: Sequence[Integral] = None):
    """ Adds a sub-array descriptor to the DaCe Program.
        Sub-arrays are implemented (when `process_grid` is set) with [MPI_Type_create_subarray](https://www.mpich.org/static/docs/v3.2/www3/MPI_Type_create_subarray.html).

        :param array: Either the name of an Array descriptor or the shape of the array (similar to the `array_of_sizes` parameter of `MPI_Type_create_subarray`).
        :param subarray: Either the name of an Array descriptor or the sub-shape of the (sub-)array (similar to the `array_of_subsizes` parameter of `MPI_Type_create_subarray`).
        :param dtype: Datatype of the array/sub-array (similar to the `oldtype` parameter of `MPI_Type_create_subarray`).
        :process_grid: Name of the process-grid for collective scatter/gather operations.
        :param correspondence: Matching of the array/sub-array's dimensions to the process-grid's dimensions.
        :return: Name of the new sub-array descriptor.
    """
    # Get dtype, shape, and subshape
    if isinstance(array, str):
        shape = sdfg.arrays[array].shape
        arr_dtype = sdfg.arrays[array].dtype
    else:
        shape = array
        arr_dtype = None
    if isinstance(subarray, str):
        subshape = sdfg.arrays[subarray].shape
        sub_dtype = sdfg.arrays[subarray].dtype
    else:
        subshape = subarray
        sub_dtype = None
    dtype = dtype or arr_dtype or sub_dtype

    subarray_name = sdfg.add_subarray(dtype, shape, subshape, process_grid, correspondence)

    # Generate subgraph only if process-grid is set, i.e., the sub-array will be used for collective scatter/gather ops.
    if process_grid:
        # Dummy tasklet adds MPI variables to the program's state.
        from dace.libraries.mpi import Dummy
        tasklet = Dummy(
            subarray_name,
            [f'MPI_Datatype {subarray_name};', f'int* {subarray_name}_counts;', f'int* {subarray_name}_displs;'])

        state.add_node(tasklet)

        # Pseudo-writing to a dummy variable to avoid removal of Dummy node by transformations.
        _, scal = sdfg.add_scalar(subarray_name, dace.int32, transient=True)
        wnode = state.add_write(subarray_name)
        state.add_edge(tasklet, '__out', wnode, None, Memlet.from_array(subarray_name, scal))

    return subarray_name


@oprepo.replaces('dace.comm.BlockScatter')
def _block_scatter(pv: ProgramVisitor,
                   sdfg: SDFG,
                   state: SDFGState,
                   in_buffer: str,
                   out_buffer: str,
                   scatter_grid: str,
                   bcast_grid: str = None,
                   correspondence: Sequence[Integral] = None):
    """ Block-scatters an Array using process-grids, sub-arrays, and the BlockScatter library node.
        This method currently does not support Array slices and imperfect tiling.

        :param in_buffer: Name of the (global) Array descriptor.
        :param out_buffer: Name of the (local) Array descriptor.
        :param scatter_grid: Name of the sub-grid used for scattering the Array (replication group leaders).
        :param bcast_grid: Name of the sub-grid used for broadcasting the Array (replication groups). 
        :param correspondence: Matching of the array/sub-array's dimensions to the process-grid's dimensions.
        :return: Name of the new sub-array descriptor.
    """
    in_desc = sdfg.arrays[in_buffer]
    out_desc = sdfg.arrays[out_buffer]

    if in_desc.dtype != out_desc.dtype:
        raise ValueError("Input/output buffer datatypes must match!")

    subarray_name = _subarray(pv,
                              sdfg,
                              state,
                              in_buffer,
                              out_buffer,
                              process_grid=scatter_grid,
                              correspondence=correspondence)

    from dace.libraries.mpi import BlockScatter
    libnode = BlockScatter('_BlockScatter_', subarray_name, scatter_grid, bcast_grid)

    inbuf_name = in_buffer
    in_desc = sdfg.arrays[inbuf_name]
    inbuf_node = state.add_read(inbuf_name)
    inbuf_mem = Memlet.from_array(inbuf_name, in_desc)

    outbuf_name = out_buffer
    out_desc = sdfg.arrays[outbuf_name]
    outbuf_node = state.add_write(outbuf_name)
    outbuf_mem = Memlet.from_array(outbuf_name, out_desc)

    state.add_edge(inbuf_node, None, libnode, '_inp_buffer', inbuf_mem)
    state.add_edge(libnode, '_out_buffer', outbuf_node, None, outbuf_mem)

    return subarray_name


@oprepo.replaces('dace.comm.BlockGather')
def _block_gather(pv: ProgramVisitor,
                  sdfg: SDFG,
                  state: SDFGState,
                  in_buffer: str,
                  out_buffer: str,
                  gather_grid: str,
                  reduce_grid: str = None,
                  correspondence: Sequence[Integral] = None):
    """ Block-gathers an Array using process-grids, sub-arrays, and the BlockGather library node.
        This method currently does not support Array slices and imperfect tiling.

        :param in_buffer: Name of the (local) Array descriptor.
        :param out_buffer: Name of the (global) Array descriptor.
        :param gather_grid: Name of the sub-grid used for gathering the Array (reduction group leaders).
        :param reduce_grid: Name of the sub-grid used for broadcasting the Array (reduction groups). 
        :param correspondence: Matching of the array/sub-array's dimensions to the process-grid's dimensions.
        :return: Name of the new sub-array descriptor.
    """
    in_desc = sdfg.arrays[in_buffer]
    out_desc = sdfg.arrays[out_buffer]

    if in_desc.dtype != out_desc.dtype:
        raise ValueError("Input/output buffer datatypes must match!")

    subarray_name = _subarray(pv,
                              sdfg,
                              state,
                              out_buffer,
                              in_buffer,
                              process_grid=gather_grid,
                              correspondence=correspondence)

    from dace.libraries.mpi import BlockGather
    libnode = BlockGather('_BlockGather_', subarray_name, gather_grid, reduce_grid)

    inbuf_name = in_buffer
    in_desc = sdfg.arrays[inbuf_name]
    inbuf_node = state.add_read(inbuf_name)
    inbuf_mem = Memlet.from_array(inbuf_name, in_desc)

    outbuf_name = out_buffer
    out_desc = sdfg.arrays[outbuf_name]
    outbuf_node = state.add_write(outbuf_name)
    outbuf_mem = Memlet.from_array(outbuf_name, out_desc)

    state.add_edge(inbuf_node, None, libnode, '_inp_buffer', inbuf_mem)
    state.add_edge(libnode, '_out_buffer', outbuf_node, None, outbuf_mem)

    return subarray_name


@oprepo.replaces('dace.comm.Redistribute')
def _redistribute(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, in_buffer: str, in_subarray: str, out_buffer: str,
                  out_subarray: str):
    """ Redistributes an Array using process-grids, sub-arrays, and the Redistribute library node.
    
        :param in_buffer: Name of the (local) input Array descriptor.
        :param in_subarray: Input sub-array descriptor.
        :param out_buffer: Name of the (local) output Array descriptor.
        :param out_subarray: Output sub-array descriptor.
        :return: Name of the new redistribution descriptor.
    """
    in_desc = sdfg.arrays[in_buffer]
    out_desc = sdfg.arrays[out_buffer]

    rdistrarray_name = sdfg.add_rdistrarray(in_subarray, out_subarray)

    from dace.libraries.mpi import Dummy, Redistribute
    tasklet = Dummy(rdistrarray_name, [
        f'MPI_Datatype {rdistrarray_name};', f'int {rdistrarray_name}_sends;',
        f'MPI_Datatype* {rdistrarray_name}_send_types;', f'int* {rdistrarray_name}_dst_ranks;',
        f'int {rdistrarray_name}_recvs;', f'MPI_Datatype* {rdistrarray_name}_recv_types;',
        f'int* {rdistrarray_name}_src_ranks;', f'int {rdistrarray_name}_self_copies;',
        f'int* {rdistrarray_name}_self_src;', f'int* {rdistrarray_name}_self_dst;',
        f'int* {rdistrarray_name}_self_size;'
    ])
    state.add_node(tasklet)
    _, scal = sdfg.add_scalar(rdistrarray_name, dace.int32, transient=True)
    wnode = state.add_write(rdistrarray_name)
    state.add_edge(tasklet, '__out', wnode, None, Memlet.from_array(rdistrarray_name, scal))

    libnode = Redistribute('_Redistribute_', rdistrarray_name)

    inbuf_range = None
    if isinstance(in_buffer, tuple):
        inbuf_name, inbuf_range = in_buffer
    else:
        inbuf_name = in_buffer
    in_desc = sdfg.arrays[inbuf_name]
    inbuf_node = state.add_read(inbuf_name)

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
    if outbuf_range:
        outbuf_mem = Memlet.simple(outbuf_name, outbuf_range)
    else:
        outbuf_mem = Memlet.from_array(outbuf_name, out_desc)

    state.add_edge(inbuf_node, None, libnode, '_inp_buffer', inbuf_mem)
    state.add_edge(libnode, '_out_buffer', outbuf_node, None, outbuf_mem)

    return rdistrarray_name


@oprepo.replaces('dace.comm.BCScatter')
def _bcscatter(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, in_buffer: str, out_buffer: str,
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
            bsizes_name, bsizes_desc = sdfg.add_temp_transient((len(block_sizes), ), dtype=dace.int32)
            bsizes_node = state.add_access(bsizes_name)
            bsizes_tasklet = state.add_tasklet(
                '_set_bsizes_', {}, {'__out'},
                ";".join(["__out[{}] = {}".format(i, sz) for i, sz in enumerate(block_sizes)]))
            state.add_edge(bsizes_tasklet, '__out', bsizes_node, None, Memlet.from_array(bsizes_name, bsizes_desc))
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
def _bcgather(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, in_buffer: str, out_buffer: str,
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
            bsizes_name, bsizes_desc = sdfg.add_temp_transient((len(block_sizes), ), dtype=dace.int32)
            bsizes_node = state.add_access(bsizes_name)
            bsizes_tasklet = state.add_tasklet(
                '_set_bsizes_', {}, {'__out'},
                ";".join(["__out[{}] = {}".format(i, sz) for i, sz in enumerate(block_sizes)]))
            state.add_edge(bsizes_tasklet, '__out', bsizes_node, None, Memlet.from_array(bsizes_name, bsizes_desc))
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


@oprepo.replaces('dace.distr.MatMult')
@oprepo.replaces('distr.MatMult')
def _distr_matmult(pv: ProgramVisitor,
                   sdfg: SDFG,
                   state: SDFGState,
                   opa: str,
                   opb: str,
                   shape: Sequence[Union[sp.Expr, Number]],
                   a_block_sizes: Union[str, Sequence[Union[sp.Expr, Number]]] = None,
                   b_block_sizes: Union[str, Sequence[Union[sp.Expr, Number]]] = None,
                   c_block_sizes: Union[str, Sequence[Union[sp.Expr, Number]]] = None):

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
            a_bsizes_name, a_bsizes_desc = sdfg.add_temp_transient((len(a_block_sizes), ), dtype=dace.int32)
            a_bsizes_node = state.add_access(a_bsizes_name)
            a_bsizes_tasklet = state.add_tasklet(
                '_set_a_bsizes_', {}, {'__out'},
                ";".join(["__out[{}] = {}".format(i, sz) for i, sz in enumerate(a_block_sizes)]))
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
            b_bsizes_name, b_bsizes_desc = sdfg.add_temp_transient((len(b_block_sizes), ), dtype=dace.int32)
            b_bsizes_node = state.add_access(b_bsizes_name)
            b_bsizes_tasklet = state.add_tasklet(
                '_set_b_sizes_', {}, {'__out'},
                ";".join(["__out[{}] = {}".format(i, sz) for i, sz in enumerate(b_block_sizes)]))
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
