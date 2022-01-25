# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" A module that contains type definitions for distributed SDFGs. """
import functools
import re
import copy as cp
import sympy as sp
import numpy
from numbers import Number, Integral
from typing import Set, Sequence, Tuple

import dace.dtypes as dtypes
from dace import symbolic, serialize
from dace.properties import (EnumProperty, Property, make_properties,
                             DictProperty, SubArrayProperty, ShapeProperty,
                             SubsetProperty, SymbolicProperty,
                             TypeClassProperty, DebugInfoProperty, CodeProperty,
                             ListProperty)


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


@make_properties
class ProcessGrid(object):
    """ Process grids are used to create cartesian communicators.
    """

    name = Property(dtype=str, desc="The grid's name.")
    is_subgrid = Property(
        dtype=bool,
        default=False,
        desc="True if the grid is a subset of another grid.")
    shape = ShapeProperty(default=[], desc="The grid's shape.")
    parent_grid = Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Name of the parent grid (mandatory only if is_subgrid == True).")
    color = ListProperty(
        int,
        allow_none=True,
        default=None,
        desc="The color that can be used to split the parent-grid.")
    exact_grid = SymbolicProperty(
        allow_none=True,
        default=None,
        desc="An MPI process's rank in the parent grid. Defines the exact "
             "sub-grid that includes this rank "
             "(optional only if is_subgrid == True).")
    root = SymbolicProperty(default=0, desc="The root rank for collectives.")

    def __init__(self, name, is_subgrid, shape, parent_grid=None, color=None, exact_grid=None,
                 root=0):
        self.name = name
        self.is_subgrid = is_subgrid
        if is_subgrid:
            self.parent_grid = parent_grid.name
            # self.correspondence = correspondence
            self.color = color
            self.exact_grid = exact_grid

            self.shape = [parent_grid.shape[i]
                          for i, remain in enumerate(color) if remain]
            # self.color = [1 if i in correspondence else 0
            #               for i in range(len(parent_grid.shape))]
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
                raise ValueError('Sub-grid misses its corresponding main-grid')
        if any(not isinstance(
                s, (Integral, symbolic.SymExpr, symbolic.symbol,
                    symbolic.sympy.Basic)) for s in self.shape):
                raise TypeError('Shape must be a list or tuple of integer '
                                'values or symbols')
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
        if self.is_subgrid:
            tmp = f"""
                if (__state->{self.parent_grid}_valid) {{
                    int {self.name}_remain[{len(self.color)}] = {{{', '.join(['1' if c else '0' for c in self.color])}}};
                    MPI_Cart_sub(__state->{self.parent_grid}_comm, {self.name}_remain, &__state->{self.name}_comm);
                    MPI_Comm_group(__state->{self.name}_comm, &__state->{self.name}_group);
                    MPI_Comm_rank(__state->{self.name}_comm, &__state->{self.name}_rank);
                    MPI_Comm_size(__state->{self.name}_comm, &__state->{self.name}_size);
                    MPI_Cart_coords(__state->{self.name}_comm, __state->{self.name}_rank, {len(self.shape)}, __state->{self.name}_coords);

                    int periods[{len(self.color)}];
                    int coords[{len(self.color)}];
                    MPI_Cart_get(__state->{self.name}_comm, {len(self.color)}, __state->{self.name}_dims, periods, coords);
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
                // TODO: Do we need this check?
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
        return f"""
            MPI_Group_free(&__state->{self.name}_group);
            MPI_Comm_free(&__state->{self.name}_comm);
        """


@make_properties
class SubArray(object):
    """ MPI sub-array data type.
    """

    name = Property(dtype=str, desc="The type's name.")
    dtype = TypeClassProperty(default=dtypes.int32, choices=dtypes.Typeclasses)
    shape = ShapeProperty(default=[], desc="The array's shape.")
    subshape = ShapeProperty(default=[], desc="The sub-array's shape.")
    pgrid = Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Name of the process grid where the data are distributed.")
    correspondence = ListProperty(
        int,
        allow_none=True,
        default=None,
        desc="Correspondence of the array's indices to the process grid's "
             "indices.")
    
    def __init__(self, name, dtype, shape, subshape, pgrid, correspondence):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.subshape = subshape
        self.pgrid = pgrid
        self.correspondence = correspondence
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
        ret = cls('tmp', dtypes.int8, [], [], 'tmp', [])
        serialize.set_properties_from_json(ret, json_obj, context=context)
        # Check validity now
        ret.validate()
        return ret
    
    def init_code(self):
        from dace.libraries.mpi import utils
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
                    while (block_id[idx] + 1 >= __state->{self.pgrid}_dims[corr[idx]]) {{
                        block_id[idx] = 0;
                        displ -= data_strides[idx] * (__state->{self.pgrid}_dims[corr[idx]] - 1);
                        idx--;
                    }}
                    block_id[idx] += 1;
                    displ += data_strides[idx];
                }}
            }}
        """
    
    def exit_code(self):
        return f"""
            if (__state->{self.pgrid}_valid) {{
                delete[] __state->{self.name}_counts;
                delete[] __state->{self.name}_displs;
                MPI_Type_free(&__state->{self.name});
            }}
        """


@make_properties
class RedistrArray(object):
    """ Array redistribution.
    """

    name = Property(dtype=str, desc="The redistribution's name.")
    array_a = Property(
        dtype=str,
        allow_none=True,
        default=None,desc="Sub-array that will be redistributed.")
    array_b = Property(
        dtype=str,
        allow_none=True,
        default=None,desc="Output sub-array.")
    
    def __init__(self, name, array_a, array_b):
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
        array_a = sdfg.subarrays[self.array_a]
        array_b = sdfg.subarrays[self.array_b]
        from dace.libraries.mpi import utils
        tmp = f"""
            __state->{self.name}_sends = 0;
            __state->{self.name}_recvs = 0;
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
            tmp += f"""
                max_sends *= std::ceil(({sa} - 1) / (double){sb});
                max_recvs *= std::ceil(({sb} - 1) / (double){sa}) + 1;
            """
        tmp += f"""
            __state->{self.name}_send_types = new MPI_Datatype[max_sends];
            __state->{self.name}_dst_ranks = new int[max_sends];
            __state->{self.name}_recv_types = new MPI_Datatype[max_recvs];
            __state->{self.name}_src_ranks = new int[max_recvs];
        """
        tmp += f"""
            if (__state->{array_b.pgrid}_valid) {{
        """
        grid_a = sdfg.process_grids[array_a.pgrid]
        if grid_a.is_subgrid:
            pgrid_a = sdfg.process_grids[grid_a.parent_grid]
            tmp += f"""
                int pgrid_exact_coords[{len(pgrid_a.shape)}];
                MPI_Cart_coords(__state->{pgrid_a.name}_comm, {grid_a.exact_grid}, {len(pgrid_a.shape)}, pgrid_exact_coords);
                int pgrid_coords[{len(pgrid_a.shape)}];
            """
        tmp += f"""
            int sizes[{len(array_b.subshape)}] = {{{', '.join([str(s) for s in array_b.subshape])}}};
            //int subsizes[{len(array_a.subshape)}] = {{{', '.join([str(s) for s in array_a.subshape])}}};
            int subsizes[{len(array_b.subshape)}];
            int origin[{len(array_b.subshape)}];
        """
        for i, (sa, sb, cb) in enumerate(zip(array_a.subshape,
                                             array_b.subshape,
                                             array_b.correspondence)):
            pcoord = f"__state->{array_b.pgrid}_coords[{cb}]"
            tmp += f"""
                xi[{i}] = std::floor({pcoord} * {sb} / (double){sa});
                lambda[{i}] = {pcoord} * {sb} % {sa};
                kappa[{i}] = std::ceil(({sb} + lambda[{i}]) / (double){sa});
            """
        for i in range(len(array_b.shape)):
            tmp += f"""
                int rem{i} = {array_b.subshape[i]};
                for (auto idx{i} = 0; idx{i} < kappa[{i}]; ++idx{i}) {{
                    pcoords[{i}] = xi[{i}] + idx{i};
                    int lo{i} = (idx{i} == 0 ? lambda[{i}] : 0);
                    int uo{i} = std::min({array_a.subshape[i]}, lo{i} + rem{i});
                    subsizes[{i}] = uo{i} - lo{i};
                    origin[{i}] = {array_b.subshape[i]} - rem{i};
                    rem{i} -= uo{i} - lo{i};
            """
        tmp += f"""
            MPI_Type_create_subarray({len(array_b.shape)},  sizes, subsizes, origin, MPI_ORDER_C, {utils.MPI_DDT(array_b.dtype.base_type)}, &__state->{self.name}_recv_types[__state->{self.name}_recvs]);
            MPI_Type_commit(&__state->{self.name}_recv_types[__state->{self.name}_recvs]);
            int cart_rank;
            MPI_Cart_rank(__state->{array_a.pgrid}_comm, pcoords, &cart_rank);
        """
        if grid_a.is_subgrid:
            tmp += f"""
                int ranks1[1] = {{cart_rank}};
                int ranks2[1];
                MPI_Group_translate_ranks(__state->{array_a.pgrid}_group, 1, ranks1, __state->{pgrid_a.name}_group, ranks2);
                MPI_Cart_coords(__state->{pgrid_a.name}_comm, ranks2[0], {len(pgrid_a.shape)}, pgrid_coords);
            """
            for i, c in enumerate(grid_a.color):
                tmp += f"""
                    //pgrid_coords[{i}] = ({0 if c else 1} && pgrid_exact_coords[{i}]) || ({1 if c else 0} && pgrid_coords[{i}]);
                    pgrid_coords[{i}] = {f"pgrid_coords[{i}]" if c else f"pgrid_exact_coords[{i}]"};                
                """
            tmp += f"""
                MPI_Cart_rank(__state->{pgrid_a.name}_comm, pgrid_coords, &cart_rank);
                MPI_Group world_group;
                MPI_Comm_group(MPI_COMM_WORLD, &world_group);
                ranks1[0] = {{cart_rank}};
                MPI_Group_translate_ranks(__state->{pgrid_a.name}_group, 1, ranks1, world_group, ranks2);
                __state->{self.name}_src_ranks[__state->{self.name}_recvs] = ranks2[0];
                printf("(init) I am rank %d and I receive from %d%d (%d - %d) in (%d, %d)\\n", myrank, pcoords[0], pcoords[1], cart_rank, __state->{self.name}_src_ranks[__state->{self.name}_recvs], origin[0], origin[1]);
                __state->{self.name}_recvs++;
                MPI_Group_free(&world_group);
            """
        else:
            tmp += f"""
                MPI_Group world_group;
                MPI_Comm_group(MPI_COMM_WORLD, &world_group);
                int ranks1[1] = {{cart_rank}};
                int ranks2[1];
                MPI_Group_translate_ranks(__state->{array_a.pgrid}_group, 1, ranks1, world_group, ranks2);
                __state->{self.name}_src_ranks[__state->{self.name}_recvs] = ranks2[0];
                printf("(init) I am rank %d and I receive from %d%d (%d - %d) in (%d, %d)\\n", myrank, pcoords[0], pcoords[1], cart_rank, __state->{self.name}_src_ranks[__state->{self.name}_recvs], origin[0], origin[1]);
                __state->{self.name}_recvs++;
                MPI_Group_free(&world_group);
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
                MPI_Cart_coords(__state->{pgrid_b.name}_comm, {grid_b.exact_grid}, {len(pgrid_b.shape)}, pgrid_exact_coords);
                int pgrid_coords[{len(pgrid_b.shape)}];
            """
        tmp += f"""
            int sizes[{len(array_a.subshape)}] = {{{', '.join([str(s) for s in array_a.subshape])}}};
            //int subsizes[{len(array_b.subshape)}] = {{{', '.join([str(s) for s in array_b.subshape])}}};
            int subsizes[{len(array_a.subshape)}];
            int origin[{len(array_a.subshape)}];
        """
        for i in range(len(array_b.shape)):
            pcoord = f"__state->{array_a.pgrid}_coords[{array_a.correspondence[i]}]"
            sa = array_a.subshape[i]
            sb = array_b.subshape[i]
            tmp += f"""
                int lp{i} = std::max(0, (int)std::ceil(({pcoord} * {sa} - {sb} + 1) / (double){sb}));
                int up{i} = std::min(__state->{array_b.pgrid}_dims[{array_b.correspondence[i]}], (int)std::ceil(({pcoord} + 1) * {sa} / (double){sb}));
                //printf("I am rank %d and I have {i}-th bounds [%d, %d)\\n", myrank, lp{i}, up{i});
                for (auto idx{i} = lp{i}; idx{i} < up{i}; ++idx{i}) {{
                    xi[{i}] = std::floor(idx{i} * {sb} / (double){sa});
                    lambda[{i}] = idx{i} * {sb} % {sa};
                    kappa[{i}] = std::ceil(({sb} + lambda[{i}]) / (double){sa});
                    int idx{i}_dst = {pcoord} - xi[{i}];

                    //if (myrank == 2) {{
                    //    printf("dims[{i}] = %d, coord[{i}] = %d\\n", __state->{array_a.pgrid}_dims[{i}], {pcoord});
                    //    printf("xi[{i}] = %d, lambda[{i}] = %d, kappa[{i}] = %d, idx{i}_dst = %d\\n", xi[{i}], lambda[{i}], kappa[{i}], idx{i}_dst);
                    //}}

                    if (idx{i}_dst < 0 || idx{i}_dst >= kappa[{i}]) continue;
                    int lo{i} = (idx{i}_dst == 0 ? lambda[{i}] : 0);
                    int uo{i} = (idx{i}_dst == kappa[{i}] ? {sb} + lambda[{i}] - idx{i}_dst * {sa} : {sa});
                    subsizes[{i}] = uo{i} - lo{i};
                    origin[{i}] = lo{i};
                    pcoords[{i}] = idx{i};

            """
        tmp += f"""
            MPI_Type_create_subarray({len(array_a.shape)},  sizes, subsizes, origin, MPI_ORDER_C, {utils.MPI_DDT(array_a.dtype.base_type)}, &__state->{self.name}_send_types[__state->{self.name}_sends]);
            MPI_Type_commit(&__state->{self.name}_send_types[__state->{self.name}_sends]);
            int cart_rank;
            MPI_Cart_rank(__state->{array_b.pgrid}_comm, pcoords, &cart_rank);
            if (myrank == 2) {{
                printf("pcoords = %d, %d, cart_rank = %d\\n", pcoords[0], pcoords[1], cart_rank);
            }}
        """
        if grid_b.is_subgrid:
            tmp += f"""
                int ranks1[1] = {{cart_rank}};
                int ranks2[1];
                MPI_Group_translate_ranks(__state->{array_b.pgrid}_group, 1, ranks1, __state->{pgrid_b.name}_group, ranks2);
                MPI_Cart_coords(__state->{pgrid_b.name}_comm, ranks2[0], {len(pgrid_b.shape)}, pgrid_coords);
                if (myrank == 2) {{
                    printf("subgrid rank %d -> pgrid rank %d\\n", ranks1[0], ranks2[0]);
                }}
            """
            for i, c in enumerate(grid_b.color):
                tmp += f"""
                    //pgrid_coords[{i}] = ({0 if c else 1} && pgrid_exact_coords[{i}]) || ({1 if c else 0} && pgrid_coords[{i}]);
                    pgrid_coords[{i}] = {f"pgrid_coords[{i}]" if c else f"pgrid_exact_coords[{i}]"};
                """
            tmp += f"""
                MPI_Cart_rank(__state->{pgrid_b.name}_comm, pgrid_coords, &cart_rank);
                if (myrank == 2) {{
                    printf("pgrid_coords = %d, %d, %d, cart_rank = %d\\n", pgrid_coords[0], pgrid_coords[1], pgrid_coords[2], cart_rank);
                }}
                MPI_Group world_group;
                MPI_Comm_group(MPI_COMM_WORLD, &world_group);
                ranks1[0] = {{cart_rank}};
                MPI_Group_translate_ranks(__state->{pgrid_b.name}_group, 1, ranks1, world_group, ranks2);
                __state->{self.name}_dst_ranks[__state->{self.name}_sends] = ranks2[0];
                printf("(init) I am rank %d and I send to %d%d (%d - %d) from (%d, %d)\\n", myrank, pcoords[0], pcoords[1], cart_rank, __state->{self.name}_dst_ranks[__state->{self.name}_sends], origin[0], origin[1]);
                __state->{self.name}_sends++;
                MPI_Group_free(&world_group);
            """
        else:
            tmp += f"""
                MPI_Group world_group;
                MPI_Comm_group(MPI_COMM_WORLD, &world_group);
                int ranks1[1] = {{cart_rank}};
                int ranks2[1];
                MPI_Group_translate_ranks(__state->{array_b.pgrid}_group, 1, ranks1, world_group, ranks2);
                __state->{self.name}_dst_ranks[__state->{self.name}_sends] = ranks2[0];
                printf("(init) I am rank %d and I send to %d%d (%d - %d) from (%d, %d)\\n", myrank, pcoords[0], pcoords[1], cart_rank, __state->{self.name}_dst_ranks[__state->{self.name}_sends], origin[0], origin[1]);
                __state->{self.name}_sends++;
                MPI_Group_free(&world_group);
            """
        for i in range(len(array_b.shape)):
            tmp += f"}}"
        tmp += "}"
        return tmp
    
    def exit_code(self, sdfg):
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
        """