# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" A module that contains type definitions for distributed SDFGs. """
from numbers import Integral
from typing import Sequence, Union

import dace.dtypes as dtypes
from dace import symbolic, serialize
from dace.properties import (Property, make_properties, ShapeProperty, SymbolicProperty, TypeClassProperty,
                             ListProperty)

ShapeType = Sequence[Union[Integral, str, symbolic.symbol, symbolic.SymExpr, symbolic.sympy.Basic]]
RankType = Union[Integral, str, symbolic.symbol, symbolic.SymExpr, symbolic.sympy.Basic]


@make_properties
class ProcessGrid(object):
    """
    Process-grids implement cartesian topologies similarly to cartesian communicators created with [MPI_Cart_create](https://www.mpich.org/static/docs/latest/www3/MPI_Cart_create.html)
    and [MPI_Cart_sub](https://www.mpich.org/static/docs/v3.2/www3/MPI_Cart_sub.html).

    The boolean property `is_subgrid` provides a switch between "parent" process-grids (equivalent to communicators
    create with `MPI_Cart_create`) and sub-grids (equivalent to communicators created with `MPI_Cart_sub`).
    
    If `is_subgrid` is false, a "parent" process-grid is created. The `shape` property is equivalent to the `dims`
    parameter of `MPI_Cart_create`. The other properties are ignored. All "parent" process-grids spawn out of
    `MPI_COMM_WORLD`, while their `periods` and `reorder` parameters are set to False.

    If `is_subgrid` is true, then the `parent_grid` is partitioned to lower-dimensional cartesian sub-grids (for more
    details, see the documentation of `MPI_Cart_sub`). The `parent_grid` property is equivalent to the `comm` parameter
    of `MPI_Cart_sub`. The `color` property corresponds to the `remain_dims` parameter of `MPI_Cart_sub`, i.e., the i-th
    entry specifies whether the i-th dimension is kep in the sub-grid or is dropped.
    
    The following properties store information used in the redistribution of data:

    The `exact_grid` property is either None or the rank of an MPI process in the `parent_grid`. If set then, out of all
    the sub-grids created, only the one that contains this rank is used for collective communication. The `root`
    property is used to select the root rank for purposed of collective communication (by default 0).
    """

    name = Property(dtype=str, desc="The process-grid's name.")
    is_subgrid = Property(dtype=bool, default=False, desc="If true, spanws sub-grids out of the parent process-grid.")
    shape = ShapeProperty(default=[], desc="The process-grid's shape.")
    parent_grid = Property(dtype=str,
                           allow_none=True,
                           default=None,
                           desc="Name of the parent process-grid "
                           "(mandatory if `is_subgrid` is true, otherwise ignored).")
    color = ListProperty(int,
                         allow_none=True,
                         default=None,
                         desc="The i-th entry specifies whether the i-th dimension is kept in the sub-grid or is "
                         "dropped (mandatory if `is_subgrid` is true, otherwise ignored).")
    exact_grid = SymbolicProperty(allow_none=True,
                                  default=None,
                                  desc="If set then, out of all the sub-grids created, only the one that contains the "
                                  "rank with id `exact_grid` will be utilized for collective communication "
                                  "(optional if `is_subgrid` is true, otherwise ignored).")
    root = SymbolicProperty(default=0, desc="The root rank for collective communication.")

    def __init__(self,
                 name: str,
                 is_subgrid: bool,
                 shape: ShapeType = None,
                 parent_grid: str = None,
                 color: Sequence[Union[Integral, bool]] = None,
                 exact_grid: RankType = None,
                 root: RankType = 0):
        self.name = name
        self.is_subgrid = is_subgrid
        if is_subgrid:
            self.parent_grid = parent_grid.name
            self.color = color
            self.exact_grid = exact_grid
            self.shape = [parent_grid.shape[i] for i, remain in enumerate(color) if remain]
        else:
            self.shape = shape
        self.root = root
        self._validate()

    def validate(self):
        """ Validate the correctness of this object.
            Raises an exception on error. """
        self._validate()

    # Validation of this class is in a separate function, so that this
    # class can call `_validate()` without calling the subclasses'
    # `validate` function.
    def _validate(self):
        if self.is_subgrid:
            if not self.parent_grid or len(self.parent_grid) == 0:
                raise ValueError('Sub-grid misses its corresponding parent process-grid')
        if any(not isinstance(s, (Integral, symbolic.SymExpr, symbolic.symbol, symbolic.sympy.Basic))
               for s in self.shape):
            raise TypeError('Shape must be a list or tuple of integer values or symbols')
        if self.color and any(c < 0 or c > 1 for c in self.color):
            raise ValueError('Color must have only logical true (1) or false (0) values.')
        return True

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)
        retdict = {"type": type(self).__name__, "attributes": attrs}
        return retdict

    @classmethod
    def from_json(cls, json_obj, context=None):
        # Create dummy object
        ret = cls('tmp', False, [])
        serialize.set_properties_from_json(ret, json_obj, context=context)
        # Check validity now
        ret.validate()
        return ret

    def init_code(self):
        """ Outputs MPI allocation/initialization code for the process-grid.
            It is assumed that the following variables exist in the SDFG program's state:
            - MPI_Comm {self.name}_comm
            - MPI_Group {self.name}_group
            - int {self.name}_rank
            - int {self.name}_size
            - int* {self.name}_dims
            - int* {self.name}_remain
            - int* {self.name}_coords
            - bool {self.name})_valid

            These variables are typically added to the program's state through a Tasklet, e.g., the Dummy MPI node (for
            more details, check the DaCe MPI library in `dace/libraries/mpi`).

        """
        if self.is_subgrid:
            tmp = ""
            for i, s in enumerate(self.shape):
                tmp += f"__state->{self.name}_dims[{i}] = {s};\n"
            tmp += f"""
                __state->{self.name}_valid = false;
                if (__state->{self.parent_grid}_valid) {{
                    int {self.name}_remain[{len(self.color)}] = {{{', '.join(['1' if c else '0' for c in self.color])}}};
                    MPI_Cart_sub(__state->{self.parent_grid}_comm, {self.name}_remain, &__state->{self.name}_comm);
                    MPI_Comm_group(__state->{self.name}_comm, &__state->{self.name}_group);
                    MPI_Comm_rank(__state->{self.name}_comm, &__state->{self.name}_rank);
                    MPI_Comm_size(__state->{self.name}_comm, &__state->{self.name}_size);
                    MPI_Cart_coords(__state->{self.name}_comm, __state->{self.name}_rank, {len(self.shape)}, __state->{self.name}_coords);
            """
            if self.exact_grid is not None:
                tmp += f"""
                    int ranks1[1] = {{{self.exact_grid}}};
                    int ranks2[1];
                    MPI_Group_translate_ranks(__state->{self.parent_grid}_group, 1, ranks1, __state->{self.name}_group, ranks2);
                    __state->{self.name}_valid = (ranks2[0] != MPI_PROC_NULL && ranks2[0] != MPI_UNDEFINED);
                }}
                """
            else:
                tmp += f"""
                    __state->{self.name}_valid = true;
                }}
                """
            return tmp
        else:
            tmp = ""
            for i, s in enumerate(self.shape):
                tmp += f"__state->{self.name}_dims[{i}] = {s};\n"
            tmp += f"""
                int {self.name}_periods[{len(self.shape)}] = {{0}};
                MPI_Cart_create(MPI_COMM_WORLD, {len(self.shape)}, __state->{self.name}_dims, {self.name}_periods, 0, &__state->{self.name}_comm);
                if (__state->{self.name}_comm != MPI_COMM_NULL) {{
                    MPI_Comm_group(__state->{self.name}_comm, &__state->{self.name}_group);
                    MPI_Comm_rank(__state->{self.name}_comm, &__state->{self.name}_rank);
                    MPI_Comm_size(__state->{self.name}_comm, &__state->{self.name}_size);
                    MPI_Cart_coords(__state->{self.name}_comm, __state->{self.name}_rank, {len(self.shape)}, __state->{self.name}_coords);
                    __state->{self.name}_valid = true;
                }} else {{
                    __state->{self.name}_group = MPI_GROUP_NULL;
                    __state->{self.name}_rank = MPI_PROC_NULL;
                    __state->{self.name}_size = 0;
                    __state->{self.name}_valid = false;
                }}
            """
            return tmp

    def exit_code(self):
        """ Outputs MPI deallocation code for the process-grid. """
        return f"""
            if (__state->{self.name}_valid) {{
                MPI_Group_free(&__state->{self.name}_group);
                MPI_Comm_free(&__state->{self.name}_comm);
            }}
        """


@make_properties
class SubArray(object):
    """
    Sub-arrays describe subsets of Arrays (see `dace::data::Array`) for purposes of distributed communication. They are
    implemented with [MPI_Type_create_subarray](https://www.mpich.org/static/docs/v3.2/www3/MPI_Type_create_subarray.html).
    Sub-arrays can be also used for collective scatter/gather communication in a process-grid.

    The `shape`, `subshape`, and `dtype` properties correspond to the `array_of_sizes`, `array_of_subsizes`, and
    `oldtype` parameters of `MPI_Type_create_subarray`.

    The following properties are used for collective scatter/gather communication in a process-grid:

    The `pgrid` property is the name of the process-grid where the data will be distributed. The `correspondence`
    property matches the arrays dimensions to the process-grid's dimensions. For example, if one wants to distribute
    a matrix to a 2D process-grid, but tile the matrix rows over the grid's columns, then `correspondence = [1, 0]`.
    """

    name = Property(dtype=str, desc="The type's name.")
    dtype = TypeClassProperty(default=dtypes.int32, choices=dtypes.Typeclasses)
    shape = ShapeProperty(default=[], desc="The array's shape.")
    subshape = ShapeProperty(default=[], desc="The sub-array's shape.")
    pgrid = Property(dtype=str,
                     allow_none=True,
                     default=None,
                     desc="Name of the process-grid where the data are distributed.")
    correspondence = ListProperty(int,
                                  allow_none=True,
                                  default=None,
                                  desc="Correspondence of the array's indices to the process grid's "
                                  "indices.")

    def __init__(self,
                 name: str,
                 dtype: dtypes.typeclass,
                 shape: ShapeType,
                 subshape: ShapeType,
                 pgrid: str = None,
                 correspondence: Sequence[Integral] = None):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.subshape = subshape
        self.pgrid = pgrid
        self.correspondence = correspondence or list(range(len(shape)))
        self._validate()

    def validate(self):
        """ Validate the correctness of this object.
            Raises an exception on error. """
        self._validate()

    # Validation of this class is in a separate function, so that this
    # class can call `_validate()` without calling the subclasses'
    # `validate` function.
    def _validate(self):
        if any(not isinstance(s, (Integral, symbolic.SymExpr, symbolic.symbol, symbolic.sympy.Basic))
               for s in self.shape):
            raise TypeError('Shape must be a list or tuple of integer values or symbols')
        if any(not isinstance(s, (Integral, symbolic.SymExpr, symbolic.symbol, symbolic.sympy.Basic))
               for s in self.subshape):
            raise TypeError('Sub-shape must be a list or tuple of integer values or symbols')
        if any(not isinstance(i, Integral) for i in self.correspondence):
            raise TypeError('Correspondence must be a list or tuple of integer values')
        if len(self.shape) != len(self.subshape):
            raise ValueError('The dimensionality of the shape and sub-shape must match')
        if len(self.correspondence) != len(self.shape):
            raise ValueError('The dimensionality of the shape and correspondence list must match')
        return True

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)
        retdict = {"type": type(self).__name__, "attributes": attrs}
        return retdict

    @classmethod
    def from_json(cls, json_obj, context=None):
        # Create dummy object
        ret = cls('tmp', dtypes.int8, [], [], 'tmp', [])
        serialize.set_properties_from_json(ret, json_obj, context=context)
        # Check validity now
        ret.validate()
        return ret

    def init_code(self):
        """ Outputs MPI allocation/initialization code for the sub-array MPI datatype ONLY if the process-grid is set.
            It is assumed that the following variables exist in the SDFG program's state:
            - MPI_Datatype {self.name}
            - int* {self.name}_counts
            - int* {self.name}_displs

            These variables are typically added to the program's state through a Tasklet, e.g., the Dummy MPI node (for
            more details, check the DaCe MPI library in `dace/libraries/mpi`).
        """
        from dace.libraries.mpi import utils
        if self.pgrid:
            return f"""
                if (__state->{self.pgrid}_valid) {{
                    int sizes[{len(self.shape)}] = {{{', '.join([str(s) for s in self.shape])}}};
                    int subsizes[{len(self.shape)}] = {{{', '.join([str(s) for s in self.subshape])}}};
                    int corr[{len(self.shape)}] = {{{', '.join([str(i) for i in self.correspondence])}}};

                    int basic_stride = subsizes[{len(self.shape)} - 1];

                    int process_strides[{len(self.shape)}];
                    int block_strides[{len(self.shape)}];
                    int data_strides[{len(self.shape)}];

                    process_strides[{len(self.shape)} - 1] = 1;
                    block_strides[{len(self.shape)} - 1] = subsizes[{len(self.shape)} - 1];
                    data_strides[{len(self.shape)} - 1] = 1;

                    for (auto i = {len(self.shape)} - 2; i >= 0; --i) {{
                        block_strides[i] = block_strides[i+1] * subsizes[i];
                        process_strides[i] = process_strides[i+1] * __state->{self.pgrid}_dims[corr[i+1]];
                        data_strides[i] = block_strides[i] * process_strides[i] / basic_stride;
                    }}

                    MPI_Datatype type;
                    int origin[{len(self.shape)}] = {{{','.join(['0'] * len(self.shape))}}};
                    MPI_Type_create_subarray({len(self.shape)}, sizes, subsizes, origin, MPI_ORDER_C, {utils.MPI_DDT(self.dtype.base_type)}, &type);
                    MPI_Type_create_resized(type, 0, basic_stride*sizeof({self.dtype.ctype}), &__state->{self.name});
                    MPI_Type_commit(&__state->{self.name});

                    __state->{self.name}_counts = new int[__state->{self.pgrid}_size];
                    __state->{self.name}_displs = new int[__state->{self.pgrid}_size];
                    int block_id[{len(self.shape)}] = {{0}};
                    int displ = 0;
                    for (auto i = 0; i < __state->{self.pgrid}_size; ++i) {{
                        __state->{self.name}_counts[i] = 1;
                        __state->{self.name}_displs[i] = displ;
                        int idx = {len(self.shape)} - 1;
                        while (idx >= 0 && block_id[idx] + 1 >= __state->{self.pgrid}_dims[corr[idx]]) {{
                            block_id[idx] = 0;
                            displ -= data_strides[idx] * (__state->{self.pgrid}_dims[corr[idx]] - 1);
                            idx--;
                        }}
                        if (idx >= 0) {{ 
                            block_id[idx] += 1;
                            displ += data_strides[idx];
                        }} else {{
                            assert(i == __state->{self.pgrid}_size - 1);
                        }}
                    }}
                }}
            """
        else:
            return ""

    def exit_code(self):
        """ Outputs MPI deallocation code for the sub-array MPI datatype ONLY if the process-grid is set. """
        if self.pgrid:
            return f"""
                if (__state->{self.pgrid}_valid) {{
                    delete[] __state->{self.name}_counts;
                    delete[] __state->{self.name}_displs;
                    MPI_Type_free(&__state->{self.name});
                }}
            """
        else:
            return ""


@make_properties
class RedistrArray(object):
    """
    Describes the redistribution of an Array from one process-grid and sub-array descriptor (`array_a`) to another
    (`array_b`). The redistribution is implemented with MPI datatypes (see [MPI_Type_create_subarray](https://www.mpich.org/static/docs/v3.2/www3/MPI_Type_create_subarray.html)
    and point-to-point communication through the `MPI_COMM_WORLD` communicator.
    TODO: Add reference to publication describing the redistribution scheme.
    """

    name = Property(dtype=str, desc="The redistribution's name.")
    array_a = Property(dtype=str, allow_none=True, default=None, desc="Sub-array that will be redistributed.")
    array_b = Property(dtype=str, allow_none=True, default=None, desc="Output sub-array.")

    def __init__(self, name: str, array_a: str, array_b: str):
        self.name = name
        self.array_a = array_a
        self.array_b = array_b
        self._validate()

    def validate(self):
        """ Validate the correctness of this object.
            Raises an exception on error. """
        self._validate()

    # Validation of this class is in a separate function, so that this
    # class can call `_validate()` without calling the subclasses'
    # `validate` function.
    def _validate(self):
        return True

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)
        retdict = {"type": type(self).__name__, "attributes": attrs}
        return retdict

    @classmethod
    def from_json(cls, json_obj, context=None):
        # Create dummy object
        ret = cls('tmp', 'tmp', 'tmp')
        serialize.set_properties_from_json(ret, json_obj, context=context)
        # Check validity now
        ret.validate()
        return ret

    def init_code(self, sdfg):
        """ Outputs MPI allocation/initialization code for the redistribution.
            It is assumed that the following variables exist in the SDFG program's state:
            - MPI_Datatype {self.name}
            - int {self.name}_sends
            - MPI_Datatype* {self.name}_send_types
            - int* {self.name}_dst_ranks
            - int {self.name}_recvs
            - MPI_Datatype* {self.name}_recv_types
            - int* {self.name}_src_ranks
            - int {self.name}_self_copies
            - int* {self.name}_self_src
            - int* {self.name}_self_dst
            - int* {self.name}_self_size

            These variables are typically added to the program's state through a Tasklet, e.g., the Dummy MPI node (for
            more details, check the DaCe MPI library in `dace/libraries/mpi`).
        """
        array_a = sdfg.subarrays[self.array_a]
        array_b = sdfg.subarrays[self.array_b]
        from dace.libraries.mpi import utils
        tmp = f"""{{
            __state->{self.name}_sends = 0;
            __state->{self.name}_recvs = 0;
            __state->{self.name}_self_copies = 0;
            int max_sends = 1;
            int max_recvs = 1;

            int kappa[{len(array_b.shape)}];
            int lambda[{len(array_b.shape)}];
            int xi[{len(array_b.shape)}];
            int pcoords[{len(array_b.shape)}];

            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        """
        for i, (sa, sb) in enumerate(zip(array_a.subshape, array_b.subshape)):
            sa = f"int({sa})"
            sb = f"int({sb})"
            tmp += f"""
                max_sends *= int_ceil({sa} + {sb} - 1, {sb});
                max_recvs *= int_ceil({sb} - 1, {sa}) + 1;
            """
        tmp += f"""
            __state->{self.name}_send_types = new MPI_Datatype[max_sends];
            __state->{self.name}_dst_ranks = new int[max_sends];
            __state->{self.name}_recv_types = new MPI_Datatype[max_recvs];
            __state->{self.name}_src_ranks = new int[max_recvs];
            __state->{self.name}_self_src = new int[max_sends * {len(array_a.shape)}];
            __state->{self.name}_self_dst = new int[max_sends * {len(array_b.shape)}];
            __state->{self.name}_self_size = new int[max_sends * {len(array_a.shape)}];
        """
        tmp += f"""
            if (__state->{array_b.pgrid}_valid) {{
        """
        grid_a = sdfg.process_grids[array_a.pgrid]
        if grid_a.is_subgrid:
            pgrid_a = sdfg.process_grids[grid_a.parent_grid]
            tmp += f"""
                int pgrid_exact_coords[{len(pgrid_a.shape)}];
                dace::comm::cart_coords({grid_a.exact_grid}, {len(pgrid_a.shape)}, __state->{pgrid_a.name}_dims, pgrid_exact_coords);
                int pgrid_coords[{len(pgrid_a.shape)}];
            """
        tmp += f"""
            int sizes[{len(array_b.subshape)}] = {{{', '.join([str(s) for s in array_b.subshape])}}};
            int subsizes[{len(array_b.subshape)}];
            int origin[{len(array_b.subshape)}];
        """
        for i, (sa, sb, cb) in enumerate(zip(array_a.subshape, array_b.subshape, array_b.correspondence)):
            sa = f"int({sa})"
            sb = f"int({sb})"
            pcoord = f"__state->{array_b.pgrid}_coords[{cb}]"
            tmp += f"""
                xi[{i}] = ({pcoord} * {sb}) / {sa};
                lambda[{i}] = {pcoord} * {sb} % {sa};
                kappa[{i}] = int_ceil({sb} + lambda[{i}], {sa});
            """
        for i in range(len(array_b.shape)):
            tmp += f"""
                int rem{i} = {array_b.subshape[i]};
                for (auto idx{i} = 0; idx{i} < kappa[{i}]; ++idx{i}) {{
                    int actual_idx{i} = {array_a.correspondence[i]};
                    pcoords[actual_idx{i}] = xi[{i}] + idx{i};
                    int lo{i} = (idx{i} == 0 ? lambda[{i}] : 0);
                    int uo{i} = std::min(int({array_a.subshape[i]}), lo{i} + rem{i});
                    subsizes[{i}] = uo{i} - lo{i};
                    origin[{i}] = {array_b.subshape[i]} - rem{i};
                    rem{i} -= uo{i} - lo{i};
            """
        if grid_a.is_subgrid:
            j = 0
            for i, c in enumerate(grid_a.color):
                if c:
                    tmp += f"pgrid_coords[{i}] = pcoords[{j}];\n"
                    j += 1
                else:
                    tmp += f"pgrid_coords[{i}] = pgrid_exact_coords[{i}];\n"
            tmp += f"int cart_rank = dace::comm::cart_rank({len(pgrid_a.shape)}, __state->{pgrid_a.name}_dims, pgrid_coords);\n"
        else:
            tmp += f"int cart_rank = dace::comm::cart_rank({len(grid_a.shape)}, __state->{grid_a.name}_dims, pcoords);\n"
        tmp += f"if (myrank == cart_rank) {{ // self-copy"
        for i in range(len(array_b.shape)):
            tmp += f"""
                __state->{self.name}_self_src[__state->{self.name}_self_copies * {len(array_a.shape)} + {i}] = lo{i};
                __state->{self.name}_self_dst[__state->{self.name}_self_copies * {len(array_b.shape)} + {i}] = origin[{i}];
                __state->{self.name}_self_size[__state->{self.name}_self_copies * {len(array_a.shape)} + {i}] = subsizes[{i}];
            """
        tmp += f"""
                __state->{self.name}_self_copies++;
                // printf("({self.array_a} -> {self.array_b}) I am rank %d and I self-copy {{I receive from %d%d (%d - %d) in (%d, %d) size (%d, %d)}} \\n", myrank, pcoords[0], pcoords[1], cart_rank, cart_rank, origin[0], origin[1], subsizes[0], subsizes[1]);
            }} else {{
                MPI_Type_create_subarray({len(array_b.shape)},  sizes, subsizes, origin, MPI_ORDER_C, {utils.MPI_DDT(array_b.dtype.base_type)}, &__state->{self.name}_recv_types[__state->{self.name}_recvs]);
                MPI_Type_commit(&__state->{self.name}_recv_types[__state->{self.name}_recvs]);
                __state->{self.name}_src_ranks[__state->{self.name}_recvs] = cart_rank;
                // printf("({self.array_a} -> {self.array_b}) I am rank %d and I receive from %d%d (%d - %d) in (%d, %d) size (%d, %d) \\n", myrank, pcoords[0], pcoords[1], cart_rank, __state->{self.name}_src_ranks[__state->{self.name}_recvs], origin[0], origin[1], subsizes[0], subsizes[1]);
                __state->{self.name}_recvs++;
            }}
        """
        for i in range(len(array_b.shape)):
            tmp += f"}}"
        tmp += "}"
        tmp += f"""
            if (__state->{array_a.pgrid}_valid) {{
        """
        grid_b = sdfg.process_grids[array_b.pgrid]
        if grid_b.is_subgrid:
            pgrid_b = sdfg.process_grids[grid_b.parent_grid]
            tmp += f"""
                int pgrid_exact_coords[{len(pgrid_b.shape)}];
                dace::comm::cart_coords({grid_b.exact_grid}, {len(pgrid_b.shape)}, __state->{pgrid_b.name}_dims, pgrid_exact_coords);
                int pgrid_coords[{len(pgrid_b.shape)}];
            """
        tmp += f"""
            int sizes[{len(array_a.subshape)}] = {{{', '.join([str(s) for s in array_a.subshape])}}};
            int subsizes[{len(array_a.subshape)}];
            int origin[{len(array_a.subshape)}];
        """
        for i in range(len(array_b.shape)):
            pcoord = f"__state->{array_a.pgrid}_coords[{array_a.correspondence[i]}]"
            sa = f"int({array_a.subshape[i]})"
            sb = f"int({array_b.subshape[i]})"
            tmp += f"""
                // int_ceil(x, y) := (x + y - 1) / y
                // int_ceil(pcoord * sa - sb + 1, sb) = (pcoord * sa) / sb
                int lp{i} = std::max(0, ({pcoord} * {sa}) / {sb}); // int_ceil(x, y) := (x + y - 1) / y
                int up{i} = std::min(__state->{array_b.pgrid}_dims[{array_b.correspondence[i]}], int_ceil(({pcoord} + 1) * {sa}, {sb}));
                // printf("I am rank %d and I have {i}-th bounds [%d, %d)\\n", myrank, lp{i}, up{i});
                for (auto idx{i} = lp{i}; idx{i} < up{i}; ++idx{i}) {{
                    int actual_idx{i} = {array_b.correspondence[i]};

                    xi[{i}] = (idx{i} * {sb}) / {sa};
                    lambda[{i}] = idx{i} * {sb} % {sa};
                    kappa[{i}] = int_ceil({sb} + lambda[{i}], {sa});
                    int idx{i}_dst = {pcoord} - xi[{i}];

                    if (idx{i}_dst < 0 || idx{i}_dst >= kappa[{i}]) continue;
                    int lo{i} = (idx{i}_dst == 0 ? lambda[{i}] : 0);
                    int uo{i} = (idx{i}_dst == kappa[{i}] - 1 ? {sb} + lambda[{i}] - idx{i}_dst * {sa} : {sa});
                    subsizes[{i}] = uo{i} - lo{i};
                    origin[{i}] = lo{i};
                    pcoords[actual_idx{i}] = idx{i};

            """
        if grid_b.is_subgrid:
            j = 0
            for i, c in enumerate(grid_b.color):
                if c:
                    tmp += f"pgrid_coords[{i}] = pcoords[{j}];\n"
                    j += 1
                else:
                    tmp += f"pgrid_coords[{i}] = pgrid_exact_coords[{i}];\n"
            tmp += f"int cart_rank = dace::comm::cart_rank({len(pgrid_b.shape)}, __state->{pgrid_b.name}_dims, pgrid_coords);\n"
        else:
            tmp += f"int cart_rank = dace::comm::cart_rank({len(grid_b.shape)}, __state->{grid_b.name}_dims, pcoords);\n"
        tmp += f"""
            if (myrank != cart_rank) {{ // not self-copy
                MPI_Type_create_subarray({len(array_a.shape)},  sizes, subsizes, origin, MPI_ORDER_C, {utils.MPI_DDT(array_a.dtype.base_type)}, &__state->{self.name}_send_types[__state->{self.name}_sends]);
                MPI_Type_commit(&__state->{self.name}_send_types[__state->{self.name}_sends]);
                __state->{self.name}_dst_ranks[__state->{self.name}_sends] = cart_rank;
                // printf("({self.array_a} -> {self.array_b}) I am rank %d and I send to %d%d (%d - %d) from (%d, %d) size (%d, %d)\\n", myrank, pcoords[0], pcoords[1], cart_rank, __state->{self.name}_dst_ranks[__state->{self.name}_sends], origin[0], origin[1], subsizes[0], subsizes[1]);
                __state->{self.name}_sends++;
            }}
        """
        for i in range(len(array_b.shape)):
            tmp += f"}}"
        tmp += "}"
        tmp += "}"
        return tmp

    def exit_code(self, sdfg):
        """ Outputs MPI deallocation code for the redistribution. """
        array_a = sdfg.subarrays[self.array_a]
        return f"""
            if (__state->{array_a.pgrid}_valid) {{
                for (auto __idx = 0; __idx < __state->{self.name}_sends; ++__idx) {{
                    MPI_Type_free(&__state->{self.name}_send_types[__idx]);
                }}
            }}
            delete[] __state->{self.name}_send_types;
            delete[] __state->{self.name}_dst_ranks;
            delete[] __state->{self.name}_recv_types;
            delete[] __state->{self.name}_src_ranks;
            delete[] __state->{self.name}_self_src;
            delete[] __state->{self.name}_self_dst;
            delete[] __state->{self.name}_self_size;
        """

@make_properties
class RMA_window(object):
    """
    RMA_window is the descriptor class for MPI Remote Memory Access window
    Real window creation is implemented in mpi.nodes.win_create.Win_create
    """

    name = Property(dtype=str, desc="The name of new window.")
    def __init__(self,
                 name: str):
        self.name = name
        self._validate()

    def validate(self):
        """ Validate the correctness of this object.
            Raises an exception on error. """
        self._validate()

    # Validation of this class is in a separate function, so that this
    # class can call `_validate()` without calling the subclasses'
    # `validate` function.
    def _validate(self):
        return True
