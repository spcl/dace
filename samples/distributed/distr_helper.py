import dace
import math
import numpy

from dace.sdfg import SDFG
from dace.transformation.transformation import Transformation
from dace.codegen.compiler import CompiledSDFG, ReloadableDLL

from inspect import getargspec
from copy import deepcopy
from functools import reduce
from itertools import product
from mpi4py import MPI
from typing import Any, Callable, Dict, List, Iterable, Tuple, Union


DSize = Union[int, dace.symbol]
DList = Iterable[DSize]
Distr = Iterable[DList]


def validate_process_grid(rank: int,  # MPI_COMM_WORLD rank
                          size: int,  # MPI_COMM_WORLD size
                          grid: List[DSize],  # process grid
                          label: str,  # label for debugging/error messages
                          default_grid: List[int] = None  # default grid
                         ) -> DList:
    # Compute the size of the requested process grid. 
    int_grid = [p.get() if isinstance(p, dace.symbol) else p for p in grid]
    grid_size = reduce(lambda a, b: a * b, int_grid, 1)

    # If the size of the requested process grid is too large,
    # rever to the default grid.
    if grid_size > size or numpy.any(numpy.array(int_grid) <= 0):
        # Validate default grid.
        defgrid_err = False
        if default_grid:
            int_defgrid = [p.get() if isinstance(p, dace.symbol) else p
                           for p in default_grid]
            defgrid_size = reduce(lambda a, b: a * b, int_defgrid, 1)
            if defgrid_size > size:
                default_grid = None
                defgrid_err = True
        if not default_grid:
            default_grid = default_grid or [1] * (len(grid) - 1) + [size]
        return_grid = []
        for op, np in zip(grid, default_grid):
            if isinstance(op, dace.symbol):
                op.set(np)
                return_grid.append(op)
            else:
                return_grid.append(np)
        # Print error messages
        if rank == 0:
            print("Not enough rank for the requested {l} decomposition "
                  "(comm size = {cs}, req size/grid = {rs}/{rg}.). "
                  " Using the default grid {df}.".format(
                      l=label, cs=size, rs=grid_size, rg=int_grid,
                      df=default_grid))
            if defgrid_err:
                print("The default grid set by the user is also too large "
                      "(def size/grid = {ds}/{dg}".format(
                          ds=defgrid_size, dg=int_defgrid))
        return return_grid
    
    return grid


def validate_block_sizes(rank: int,  # MPI_COMM_WORLD rank
                         bsizes: List[DSize],  # block sizes
                         label: str,  # label for debugging/error messages
                        ) -> DList:
    int_bsizes = [b.get() if isinstance(b, dace.symbol) else b for b in bsizes]
    if numpy.any(numpy.array(int_bsizes) <= 0):
        default_sizes = [b if b > 0 else 1 for b in int_bsizes]
        return_sizes = []
        for ob, nb in zip(bsizes, default_sizes):
            if isinstance(ob, dace.symbol):
                ob.set(nb)
                return_sizes.append(ob)
            else:
                return_sizes.append(nb)
        if rank == 0:
            print("Block sizes for {l} decomposition may not be less than 1 "
                  "(req bsizes = {bs}). Using {ds} instead.".format(
                      l=label, bs=int_bsizes, ds=default_sizes))
        return return_sizes
    return bsizes


def validate_decomposition(rank: int,  # MPI_COMM_WORLD rank
                           size: int,  # MPI_COMM_WORLD size
                           pgrid: List[DSize],  # process grid
                           bsizes: List[DSize],  # block sizes
                           label: str,  # label for debugging/error messages
                           default_pgrid: List[int] = None  # default pgrid
                          ) -> Tuple[DList]:
    valid_pgrid = validate_process_grid(rank, size, pgrid, label, default_pgrid)
    valid_bsizes = validate_block_sizes(rank, bsizes, label)
    return (valid_pgrid, valid_bsizes)


def distribute_sdfg(sdfg: Union[SDFG, dace.program],  # SDFG or DaceProgram
                    data_distr: Dict[str, Distr] = None,  # Data distributions
                    itsp_distr: Distr = None,  # Iteration space distribution
                    other_trans: Dict[Transformation, Dict[str, Any]] = None  # transformations
                   ) -> SDFG:
    data_distr = data_distr or dict()
    other_trans = other_trans or dict()

    from dace.transformation.dataflow import (BlockCyclicData, BlockCyclicMap)

    # Generate initial SDFG
    if isinstance(sdfg, SDFG):
        distr_sdfg = deepcopy(sdfg)
    else:
        distr_sdfg = sdfg.to_sdfg()
    
    # Apply strict and other transformations
    distr_sdfg.apply_strict_transformations()
    for trans, options in other_trans.items():
        distr_sdfg.apply_transformations([trans], options=[options])

    # Add process grids
    for dataname, (pgrid, bsizes) in data_distr.items():
        distr_sdfg.add_process_grid(dataname, pgrid)
    if itsp_distr:
        distr_sdfg.add_process_grid("itspace", itsp_distr[0])
    
    # Apply block-cyclic transformations
    for dataname, (pgrid, bsizes) in data_distr.items():
        distr_sdfg.apply_transformations([BlockCyclicData],
                                         options=[{'dataname':dataname,
                                                   'gridname': dataname,
                                                   'block': bsizes}],
                                         validate=False)
    if itsp_distr:
        distr_sdfg.apply_transformations([BlockCyclicMap],
                                         options=[{'gridname': "itspace",
                                                   'block': itsp_distr[1]}])
    
    return distr_sdfg


def get_coords(rank: int,  # MPI_COMM_WORLD rank
               grid: DList  # process grid
              ) -> Tuple[bool, List[int]]:
    # Check if rank belongs to comm
    int_grid = [p.get() if isinstance(p, dace.symbol) else p for p in grid]
    grid_size = reduce(lambda a, b: a * b, int_grid, 1)
    if rank >= grid_size:
        return False, []
    # Compute strides
    n = len(int_grid)
    size = 1
    strides = [None] * n
    for i in range(n - 1, -1, -1):
        strides[i] = size
        size *= int_grid[i]
    # Compute coords
    rem = rank
    coords = [None] * n
    for i in range(n):
        coords[i] = int(rem / strides[i])
        rem %= strides[i]
    return True, coords


def extract_local(global_data: numpy.ndarray,  # global data
                  rank: int,  # MPI_COMM_WORLD rank
                  pgrid: DList,  # process grid
                  bsizes: DList  # block sizes
                 ):
    fits, coords = get_coords(rank, pgrid)
    if not fits:
        return numpy.empty([0], dtype=global_data.dtype)
    
    int_pgrid = [p.get() if isinstance(p, dace.symbol) else p for p in pgrid]
    int_bsizes = [b.get() if isinstance(b, dace.symbol) else b for b in bsizes]

    local_shape = [math.ceil(n / (b * p))
                   for n, p, b in zip(global_data.shape, int_pgrid, int_bsizes)]
    local_shape.extend([b.get() for b in bsizes])
    local_data = numpy.zeros(local_shape, dtype=global_data.dtype)

    n = len(global_data.shape)
    for l in product(*[range(ls) for ls in local_shape[:n]]):
        gstart = [(li * p + c) * b
                  for li, p, c, b in zip(l, int_pgrid, coords, int_bsizes)]
        gfinish = [min(n, s + b)
                   for n, s, b in zip(global_data.shape, gstart, int_bsizes)]
        gindex = tuple(slice(s, f, 1) for s, f in zip(gstart, gfinish))
        # Validate range
        rng = [f - s for s, f in zip(gstart, gfinish)]
        if numpy.any(numpy.less(rng, 0)):
            continue
        block_slice = tuple(slice(0, r, 1) for r in rng)
        lindex = l + block_slice
        try:
            local_data[lindex] = global_data[gindex]
        except Exception as e:
            print(rank, coords, pgrid, int_bsizes)
            print(l, lindex, gstart, gfinish, gindex)
            raise e
    
    return local_data


def distr_exec(sdfg: Union[SDFG, dace.program],  # SDFG or DaceProgram
               args: Dict[str, Union[numpy.ndarray, dace.symbol]],  # arguments
               output: Iterable[str] = None,  # output data names
               ref_func: Callable = None, # reference output function
               data_distr: Dict[str, Distr] = None,  # Data distributions
               itsp_distr: Distr = None,  # Iteration space distribution
               other_trans: Dict[Transformation, Dict[str, Any]] = None  # transformations
              ):
    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Validate distributions
    data_distr = data_distr or dict()
    valid_data_distr = dict()
    for dataname, (pgrid, bsizes) in data_distr.items():
        pgrid = validate_process_grid(rank, size, pgrid, dataname)
        bsizes = validate_block_sizes(rank, bsizes, dataname)
        valid_data_distr[dataname] = (pgrid, bsizes)
    valid_itsp_distr = None
    if itsp_distr:
        pgrid = validate_process_grid(rank, size, itsp_distr[0],
                                      "iteration space")
        bsizes = validate_block_sizes(rank, itsp_distr[1], "iteration space")
        valid_itsp_distr = (pgrid, bsizes)
    
    # Distribute sdfg/program and load function
    if rank == 0:
        distr_sdfg = distribute_sdfg(
            sdfg, valid_data_distr, valid_itsp_distr, other_trans)
        distr_sdfg.save("rma_{}.sdfg".format(sdfg.name))
        distr_func = distr_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        distr_sdfg = SDFG.from_file("rma_{}.sdfg".format(sdfg.name))
        distr_func = CompiledSDFG(distr_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    
    # Distribute data (bcast + local extraction)
    valid_args = dict()
    for arg, argdesc in args.items():
        if arg in distr_sdfg.arrays:
            if isinstance(argdesc, numpy.ndarray):
                comm.Bcast(argdesc, root=0)
            else:  # Assume scalar
                argndarray = numpy.ndarray([1], dtype=type(argdesc))
                argndarray[0] = argdesc
                comm.Bcast(argndarray, root=0)
                argdesc = argndarray[0]
            arrstorage = distr_sdfg.arrays[arg].storage
            if arrstorage == dace.dtypes.StorageType.Distributed:
                pgrid, bsizes = valid_data_distr[arg]
                localdesc = extract_local(argdesc, rank, pgrid, bsizes)
                valid_args[arg] = localdesc
            else:
                if output and arg in output:
                    valid_args[arg] = deepcopy(argdesc)
                else:
                    valid_args[arg] = argdesc
        else:
            if output and arg in output:
                valid_args[arg] = deepcopy(arg)
            else:
                valid_args[arg] = argdesc
    
    # Add distribution symbols to arguments
    for _, (pgrid, bsizes) in valid_data_distr.items():
        for p in pgrid:
            if isinstance(p, dace.symbol):
                valid_args[p.name] = p
        for b in bsizes:
            if isinstance(b, dace.symbol):
                valid_args[b.name] = b
    if valid_itsp_distr:
        pgrid, bsizes = valid_itsp_distr
        for p in pgrid:
            if isinstance(p, dace.symbol):
                valid_args[p.name] = p
        for b in bsizes:
            if isinstance(b, dace.symbol):
                valid_args[b.name] = b
    
    # Execute distributed function
    distr_func(**valid_args)

    # Validate
    if output and ref_func:
        # Compute reference output and bcast
        if rank == 0:
            ref_func_args = {arg: args[arg]
                             for arg in getargspec(ref_func).args}
            ref_output = ref_func(**ref_func_args)
            for ref in ref_output:
                comm.Bcast(ref, root=0)
        else:
            ref_output = []
            for outname in output:
                desc = args[outname]
                tmp = deepcopy(desc)
                comm.Bcast(tmp, root=0)
                ref_output.append(tmp)
        # Validation per rank holding part of the output
        for outname, global_ref in zip(output, ref_output):
            outstorage = distr_sdfg.arrays[outname].storage
            if outstorage == dace.dtypes.StorageType.Distributed:
                pgrid, bsizes = valid_data_distr[outname]
                ref = extract_local(global_ref, rank, pgrid, bsizes)
            else:
                ref = global_ref
            val = valid_args[outname]
            if val.size > 0:
                norm_diff = numpy.linalg.norm(val - ref)
                norm_ref = numpy.linalg.norm(ref)
                relerror = norm_diff / norm_ref
                print("Rank {r} relative error: {e}".format(r=rank, e=relerror))
