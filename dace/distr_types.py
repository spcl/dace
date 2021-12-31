# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" A module that contains type definitions for distributed SDFGs. """
import functools
import re
import copy as cp
import sympy as sp
import numpy
from numbers import Number
from typing import Set, Sequence, Tuple

import dace.dtypes as dtypes
from dace import symbolic, serialize
from dace.properties import (EnumProperty, Property, make_properties,
                             DictProperty, ReferenceProperty, ShapeProperty,
                             SubsetProperty, SymbolicProperty,
                             TypeClassProperty, DebugInfoProperty, CodeProperty,
                             ListProperty)


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
    # correspondence = ListProperty(
    #     int,
    #     allow_none=True,
    #     default=None,
    #     desc="Correspondence of the sub-grid's indices to parent-grid's "
    #          "indices (mandatory only if is_subgrid == True).")
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

    def __init__(self, name, is_subgrid, shape, parent_grid, color, exact_grid,
                 root):
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
                s, (int, symbolic.SymExpr, symbolic.symbol,
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
        ret = cls(dtypes.int8, ())
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
        ret = cls(dtypes.int8, ())
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
