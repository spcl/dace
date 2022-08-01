# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Helper functions for C++ code generation.
NOTE: The C++ code generator is currently located in cpu.py.
"""
import ast
import copy
import functools
import itertools
import math
import warnings

import sympy as sp
from six import StringIO
from typing import IO, List, Optional, Tuple, Union

import dace
from dace import data, subsets, symbolic, dtypes, memlet as mmlt, nodes
from dace.codegen import cppunparse
from dace.codegen.targets.common import (sym2cpp, find_incoming_edges, codeblock_to_cpp)
from dace.codegen.dispatcher import DefinedType
from dace.config import Config
from dace.frontend import operations
from dace.frontend.python import astutils
from dace.frontend.python.astutils import ExtNodeTransformer, rname, unparse
from dace.sdfg import nodes, graph as gr, utils
from dace.properties import LambdaProperty
from dace.sdfg import SDFG, is_devicelevel_gpu, SDFGState
from dace.codegen.targets import fpga


def copy_expr(
    dispatcher,
    sdfg,
    data_name,
    memlet,
    is_write=None,  # Otherwise it's a read
    offset=None,
    relative_offset=True,
    packed_types=False,
):
    data_desc = sdfg.arrays[data_name]
    ptrname = ptr(data_name, data_desc, sdfg, dispatcher.frame)
    if relative_offset:
        s = memlet.subset
        o = offset
    else:
        if offset is None:
            s = None
        elif not isinstance(offset, subsets.Subset):
            s = subsets.Indices(offset)
        else:
            s = offset
        o = None
    if s is not None:
        offset_cppstr = cpp_offset_expr(data_desc, s, o)
    else:
        offset_cppstr = "0"
    dt = ""

    is_global = data_desc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent)
    defined_types = None
    # Non-free symbol dependent Arrays due to their shape
    dependent_shape = (isinstance(data_desc, data.Array) and not isinstance(data_desc, data.View) and any(
        str(s) not in dispatcher.frame.symbols_and_constants(sdfg) for s in dispatcher.frame.free_symbols(data_desc)))
    try:
        # NOTE: It is hard to get access to the view-edge here, so always check
        # the declared-arrays dictionary for Views.
        if dependent_shape or isinstance(data_desc, data.View):
            defined_types = dispatcher.declared_arrays.get(ptrname, is_global=is_global)
    except KeyError:
        pass
    if not defined_types:
        defined_types = dispatcher.defined_vars.get(ptrname, is_global=is_global)
    def_type, _ = defined_types
    if fpga.is_fpga_array(data_desc):
        # get conf flag
        decouple_array_interfaces = Config.get_bool("compiler", "xilinx", "decouple_array_interfaces")

        expr = fpga.fpga_ptr(
            data_name,
            data_desc,
            sdfg,
            s,
            is_write,
            dispatcher,
            0,
            def_type == DefinedType.ArrayInterface
            # If this is a view, it has already been renamed
            and not isinstance(data_desc, data.View),
            decouple_array_interfaces=decouple_array_interfaces)
    else:
        expr = ptr(data_name, data_desc, sdfg, dispatcher.frame)

    add_offset = offset_cppstr != "0"

    if def_type in [DefinedType.Pointer, DefinedType.ArrayInterface]:
        return "{}{}{}".format(dt, expr, " + {}".format(offset_cppstr) if add_offset else "")

    elif def_type == DefinedType.StreamArray:
        return "{}[{}]".format(expr, offset_cppstr)

    elif def_type == DefinedType.FPGA_ShiftRegister:
        return expr

    elif def_type in [DefinedType.Scalar, DefinedType.Stream]:

        if add_offset:
            raise TypeError("Tried to offset address of scalar {}: {}".format(data_name, offset_cppstr))

        if def_type == DefinedType.Scalar:
            return "{}&{}".format(dt, expr)
        else:
            return data_name
    else:
        raise NotImplementedError("copy_expr not implemented " "for connector type: {}".format(def_type))


def memlet_copy_to_absolute_strides(dispatcher, sdfg, memlet, src_node, dst_node, packed_types=False):
    # TODO: Take both source and destination subset into account for computing
    # copy shape.
    copy_shape = memlet.subset.size_exact()
    src_nodedesc = src_node.desc(sdfg)
    dst_nodedesc = dst_node.desc(sdfg)
    src_expr, dst_expr = None, None

    if memlet.data == src_node.data:
        if dispatcher is not None:
            src_expr = copy_expr(dispatcher, sdfg, src_node.data, memlet, is_write=False, packed_types=packed_types)
        if memlet.other_subset is not None:
            if dispatcher is not None:
                dst_expr = copy_expr(dispatcher,
                                     sdfg,
                                     dst_node.data,
                                     memlet,
                                     is_write=True,
                                     offset=memlet.other_subset,
                                     relative_offset=False,
                                     packed_types=packed_types)
            dst_subset = memlet.other_subset
        else:
            if dispatcher is not None:
                dst_expr = copy_expr(dispatcher,
                                     sdfg,
                                     dst_node.data,
                                     memlet,
                                     is_write=True,
                                     offset=None,
                                     relative_offset=False,
                                     packed_types=packed_types)
            dst_subset = subsets.Range.from_array(dst_nodedesc)
        src_subset = memlet.subset

    else:
        if dispatcher is not None:
            dst_expr = copy_expr(dispatcher, sdfg, dst_node.data, memlet, is_write=True, packed_types=packed_types)
        if memlet.other_subset is not None:
            if dispatcher is not None:
                src_expr = copy_expr(dispatcher,
                                     sdfg,
                                     src_node.data,
                                     memlet,
                                     is_write=False,
                                     offset=memlet.other_subset,
                                     relative_offset=False,
                                     packed_types=packed_types)
            src_subset = memlet.other_subset
        else:
            if dispatcher is not None:
                src_expr = copy_expr(dispatcher,
                                     sdfg,
                                     src_node.data,
                                     memlet,
                                     is_write=False,
                                     offset=None,
                                     relative_offset=False,
                                     packed_types=packed_types)
            src_subset = subsets.Range.from_array(src_nodedesc)
        dst_subset = memlet.subset

    src_strides = src_subset.absolute_strides(src_nodedesc.strides)
    dst_strides = dst_subset.absolute_strides(dst_nodedesc.strides)

    # Try to turn into degenerate/strided ND copies
    result = ndcopy_to_strided_copy(
        copy_shape,
        src_nodedesc.shape,
        src_strides,
        dst_nodedesc.shape,
        dst_strides,
        memlet.subset,
        src_subset,
        dst_subset,
    )
    if result is not None:
        copy_shape, src_strides, dst_strides = result
    else:
        # If other_subset is defined, reduce its dimensionality by
        # removing the "empty" dimensions (size = 1) and filter the
        # corresponding strides out
        src_strides = ([stride
                        for stride, s in zip(src_strides, src_subset.size()) if s != 1] + src_strides[len(src_subset):]
                       )  # Include tiles
        if not src_strides:
            src_strides = [1]
        dst_strides = ([stride
                        for stride, s in zip(dst_strides, dst_subset.size()) if s != 1] + dst_strides[len(dst_subset):]
                       )  # Include tiles
        if not dst_strides:
            dst_strides = [1]
        copy_shape = [s for s in copy_shape if s != 1]
        if not copy_shape:
            copy_shape = [1]

    # Extend copy shape to the largest among the data dimensions,
    # and extend other array with the appropriate strides
    if len(dst_strides) != len(copy_shape) or len(src_strides) != len(copy_shape):
        if memlet.data == src_node.data:
            copy_shape, dst_strides = reshape_strides(src_subset, src_strides, dst_strides, copy_shape)
        elif memlet.data == dst_node.data:
            copy_shape, src_strides = reshape_strides(dst_subset, dst_strides, src_strides, copy_shape)

    return copy_shape, src_strides, dst_strides, src_expr, dst_expr


def ptr(name: str, desc: data.Data, sdfg: SDFG = None, framecode=None) -> str:
    """
    Returns a string that points to the data based on its name and descriptor.
    :param name: Data name.
    :param desc: Data descriptor.
    :return: C-compatible name that can be used to access the data.
    """
    from dace.codegen.targets.framecode import DaCeCodeGenerator  # Avoid import loop
    framecode: DaCeCodeGenerator = framecode

    # Special case: If memory is persistent and defined in this SDFG, add state
    # struct to name
    if desc.storage != dtypes.StorageType.CPU_ThreadLocal:
        if (desc.transient and desc.lifetime is dtypes.AllocationLifetime.Persistent):
            from dace.codegen.targets.cuda import CUDACodeGen  # Avoid import loop
            if not CUDACodeGen._in_device_code:  # GPU kernels cannot access state
                return f'__state->__{sdfg.sdfg_id}_{name}'
            elif (sdfg, name) in framecode.where_allocated and framecode.where_allocated[(sdfg, name)] is not sdfg:
                return f'__{sdfg.sdfg_id}_{name}'
        elif (desc.transient and sdfg is not None and framecode is not None
              and (sdfg, name) in framecode.where_allocated and framecode.where_allocated[(sdfg, name)] is not sdfg):
            # Array allocated for another SDFG, use unambiguous name
            return f'__{sdfg.sdfg_id}_{name}'

    return name


def emit_memlet_reference(dispatcher,
                          sdfg: SDFG,
                          memlet: mmlt.Memlet,
                          pointer_name: str,
                          conntype: dtypes.typeclass,
                          ancestor: int = 1,
                          is_write: bool = None,
                          device_code: bool = False,
                          decouple_array_interfaces: bool = False) -> Tuple[str, str, str]:
    """
    Returns a tuple of three strings with a definition of a reference to an
    existing memlet. Used in nested SDFG arguments.
    :param device_code: boolean flag indicating whether we are in the process of generating FPGA device code
    :param decouple_array_interfaces: boolean flag, used for Xilinx FPGA code generation. It indicates whether or not
        we are generating code by decoupling reads/write from memory.
    :return: A tuple of the form (type, name, value).
    """
    desc = sdfg.arrays[memlet.data]
    typedef = conntype.ctype
    offset = cpp_offset_expr(desc, memlet.subset)
    offset_expr = '[' + offset + ']'
    is_scalar = not isinstance(conntype, dtypes.pointer)
    ptrname = ptr(memlet.data, desc, sdfg, dispatcher.frame)
    ref = ''

    # Get defined type (pointer, stream etc.) and change the type definition
    # accordingly.
    defined_types = None
    try:
        if (isinstance(desc, data.Array) and not isinstance(desc, data.View) and any(
                str(s) not in dispatcher.frame.symbols_and_constants(sdfg)
                for s in dispatcher.frame.free_symbols(desc))):
            defined_types = dispatcher.declared_arrays.get(ptrname, ancestor)
    except KeyError:
        pass
    if not defined_types:
        defined_types = dispatcher.defined_vars.get(ptrname, ancestor)
    defined_type, defined_ctype = defined_types

    if fpga.is_fpga_array(desc):

        datadef = fpga.fpga_ptr(memlet.data,
                                desc,
                                sdfg,
                                memlet.subset,
                                is_write,
                                dispatcher,
                                ancestor,
                                defined_type == DefinedType.ArrayInterface,
                                decouple_array_interfaces=decouple_array_interfaces)

    else:
        datadef = ptr(memlet.data, desc, sdfg, dispatcher.frame)

    if (defined_type == DefinedType.Pointer
            or (defined_type == DefinedType.ArrayInterface and isinstance(desc, data.View))):
        if not is_scalar and desc.dtype == conntype.base_type:
            # Cast potential consts
            typedef = defined_ctype

        make_const = False
        if is_scalar:
            defined_type = DefinedType.Scalar
            if is_write is False:
                make_const = True
            ref = '&'
        else:
            # constexpr arrays
            if memlet.data in dispatcher.frame.symbols_and_constants(sdfg):
                make_const = True
                ref = '*'

        # check whether const has already been added before
        if make_const and not typedef.startswith("const "):
            typedef = f'const {typedef}'

    elif defined_type == DefinedType.ArrayInterface:
        base_ctype = conntype.base_type.ctype
        typedef = f"{base_ctype}*" if is_write else f"const {base_ctype}*"
        is_scalar = False
    elif defined_type == DefinedType.Scalar:
        typedef = defined_ctype if is_scalar else (defined_ctype + '*')
        ref = '&' if is_scalar else ''
        defined_type = DefinedType.Scalar if is_scalar else DefinedType.Pointer
        offset_expr = ''
    elif defined_type == DefinedType.Stream:
        typedef = defined_ctype
        ref = '&'
        offset_expr = ''
        if not is_scalar:
            conntype = conntype.base_type
            is_scalar = True
    elif defined_type == DefinedType.StreamArray:
        # Stream array to stream (reference)
        if memlet.subset.num_elements() == 1:
            ref = '&'
            typedef = defined_ctype
            is_scalar = True  # Avoid "&" in expression below
            conntype = conntype.base_type  # Avoid vector-esque casts
            defined_type = DefinedType.Stream
        else:
            # Stream array to stream array (pointer)
            ref = ''
            typedef = defined_ctype
            defined_type = DefinedType.StreamArray
    elif defined_type == DefinedType.FPGA_ShiftRegister:
        ref = '&' if is_scalar else ''
        defined_type = DefinedType.Pointer
    else:
        raise TypeError('Unsupported memlet type "%s"' % defined_type.name)

    if (not device_code and defined_type != DefinedType.ArrayInterface
            and desc.storage == dace.StorageType.FPGA_Global):
        # This is a device buffer accessed on the host.
        # Can not be accessed with offset different than zero. Check this if we can:
        if (isinstance(offset, int) and int(offset) != 0) or (isinstance(offset, str) and offset.isnumeric()
                                                              and int(offset) != 0):
            raise TypeError("Can not offset device buffers from host code ({}, offset {})".format(datadef, offset))
        # Device buffers are passed by reference
        expr = datadef
        ref = '&'
    else:
        # Cast as necessary
        expr = make_ptr_vector_cast(datadef + offset_expr, desc.dtype, conntype, is_scalar, defined_type)

    # Register defined variable
    dispatcher.defined_vars.add(pointer_name, defined_type, typedef, allow_shadowing=True)

    return (typedef + ref, pointer_name, expr)


def reshape_strides(subset, strides, original_strides, copy_shape):
    """ Helper function that reshapes a shape to the given strides. """
    # TODO(later): Address original strides in the computation of the
    #              result strides.
    original_copy_shape = subset.size()
    dims = len(copy_shape)

    reduced_tile_sizes = [ts for ts, s in zip(subset.tile_sizes, original_copy_shape) if s != 1]

    reshaped_copy = copy_shape + [ts for ts in subset.tile_sizes if ts != 1]
    reshaped_copy[:len(copy_shape)] = [s / ts for s, ts in zip(copy_shape, reduced_tile_sizes)]

    new_strides = [0] * len(reshaped_copy)
    elements_remaining = functools.reduce(sp.Mul, copy_shape, 1)
    tiledim = 0
    for i in range(len(copy_shape)):
        new_strides[i] = elements_remaining / reshaped_copy[i]
        elements_remaining = new_strides[i]
        if reduced_tile_sizes[i] != 1:
            new_strides[dims + tiledim] = (elements_remaining / reshaped_copy[dims + tiledim])
            elements_remaining = new_strides[dims + tiledim]
            tiledim += 1

    return reshaped_copy, new_strides


def _is_c_contiguous(shape, strides):
    """
    Returns True if the strides represent a non-padded, C-contiguous (last
    dimension contiguous) array.
    """
    computed_strides = tuple(data._prod(shape[i + 1:]) for i in range(len(shape)))
    return tuple(strides) == computed_strides


def ndcopy_to_strided_copy(
    copy_shape,
    src_shape,
    src_strides,
    dst_shape,
    dst_strides,
    subset,
    src_subset,
    dst_subset,
):
    """ Detects situations where an N-dimensional copy can be degenerated into
        a (faster) 1D copy or 2D strided copy. Returns new copy
        dimensions and offsets to emulate the requested copy.

        :return: a 3-tuple: copy_shape, src_strides, dst_strides
    """

    # Cannot degenerate tiled copies
    if any(ts != 1 for ts in subset.tile_sizes):
        return None

    # If the copy is contiguous, the difference between the first and last
    # pointers should be the shape of the copy
    first_src_index = src_subset.at([0] * src_subset.dims(), src_strides)
    first_dst_index = dst_subset.at([0] * dst_subset.dims(), dst_strides)
    last_src_index = src_subset.at([d - 1 for d in src_subset.size()], src_strides)
    last_dst_index = dst_subset.at([d - 1 for d in dst_subset.size()], dst_strides)
    copy_length = functools.reduce(lambda x, y: x * y, copy_shape)
    src_copylen = last_src_index - first_src_index + 1
    dst_copylen = last_dst_index - first_dst_index + 1

    # Make expressions symbolic and simplify
    copy_length = symbolic.pystr_to_symbolic(copy_length).simplify()
    src_copylen = symbolic.pystr_to_symbolic(src_copylen).simplify()
    dst_copylen = symbolic.pystr_to_symbolic(dst_copylen).simplify()

    # Detect 1D copies. The first condition is the general one, whereas the
    # second one applies when the arrays are completely equivalent in strides
    # and shapes to the copy. The second condition is there because sometimes
    # the symbolic math engine fails to produce the same expressions for both
    # arrays.
    if (tuple(src_strides) == tuple(dst_strides)
            and ((src_copylen == copy_length and dst_copylen == copy_length) or
                 (tuple(src_shape) == tuple(copy_shape) and tuple(dst_shape) == tuple(copy_shape)))):
        # Emit 1D copy of the whole array
        return [src_copylen], [1], [1]
    # Another case of non-strided 1D copy: all indices match and copy length
    # matches pointer difference, as well as match in contiguity and padding
    elif (first_src_index == first_dst_index and last_src_index == last_dst_index and copy_length == src_copylen
          and _is_c_contiguous(src_shape, src_strides) and _is_c_contiguous(dst_shape, dst_strides)):
        # Emit 1D copy of the whole array
        return [src_copylen], [1], [1]
    # 1D strided copy
    elif (sum([0 if c == 1 else 1 for c in copy_shape]) == 1 and len(src_subset) == len(dst_subset)):
        # Find the copied dimension:
        # In copy shape
        copydim = next(i for i, c in enumerate(copy_shape) if c != 1)

        # In source strides
        src_copy_shape = src_subset.size_exact()
        if copy_shape == src_copy_shape:
            srcdim = copydim
        else:
            try:
                srcdim = next(i for i, c in enumerate(src_copy_shape) if c != 1)
            except StopIteration:
                # NOTE: This is the old stride computation code for FPGA
                # compatibility
                if len(copy_shape) == len(src_shape):
                    srcdim = copydim
                else:
                    srcdim = next(i for i, c in enumerate(src_shape) if c != 1)

        # In destination strides
        dst_copy_shape = dst_subset.size_exact()
        if copy_shape == dst_copy_shape:
            dstdim = copydim
        else:
            try:
                dstdim = next(i for i, c in enumerate(dst_copy_shape) if c != 1)
            except StopIteration:
                # NOTE: This is the old stride computation code for FPGA
                # compatibility
                if len(copy_shape) == len(dst_shape):
                    dstdim = copydim
                else:
                    dstdim = next(i for i, c in enumerate(dst_shape) if c != 1)

        # Return new copy
        return [copy_shape[copydim]], [src_strides[srcdim]], [dst_strides[dstdim]]
    else:
        return None


def cpp_offset_expr(d: data.Data, subset_in: subsets.Subset, offset=None, packed_veclen=1, indices=None):
    """ Creates a C++ expression that can be added to a pointer in order
        to offset it to the beginning of the given subset and offset.
        :param d: The data structure to use for sizes/strides.
        :param subset_in: The subset to offset by.
        :param offset: An additional list of offsets or a Subset object
        :param packed_veclen: If packed types are targeted, specifies the
                              vector length that the final offset should be
                              divided by.
        :param indices: A tuple of indices to use for expression.
        :return: A string in C++ syntax with the correct offset
    """
    if fpga.is_multibank_array_with_distributed_index(d):
        subset_in = fpga.modify_distributed_subset(subset_in, 0)

    # Offset according to parameters, then offset according to array
    if offset is not None:
        subset = subset_in.offset_new(offset, False)
        subset.offset(d.offset, False)
    else:
        subset = subset_in.offset_new(d.offset, False)

    # Obtain start range from offsetted subset
    indices = indices or ([0] * len(d.strides))

    index = subset.at(indices, d.strides)
    if packed_veclen > 1:
        index /= packed_veclen

    return sym2cpp(index)


def cpp_array_expr(sdfg,
                   memlet,
                   with_brackets=True,
                   offset=None,
                   relative_offset=True,
                   packed_veclen=1,
                   use_other_subset=False,
                   indices=None,
                   referenced_array=None,
                   codegen=None):
    """ Converts an Indices/Range object to a C++ array access string. """
    subset = memlet.subset if not use_other_subset else memlet.other_subset
    s = subset if relative_offset else subsets.Indices(offset)
    o = offset if relative_offset else None
    desc = (sdfg.arrays[memlet.data] if referenced_array is None else referenced_array)
    offset_cppstr = cpp_offset_expr(desc, s, o, packed_veclen, indices=indices)

    if with_brackets:
        if fpga.is_fpga_array(desc):
            # get conf flag
            decouple_array_interfaces = Config.get_bool("compiler", "xilinx", "decouple_array_interfaces")
            ptrname = fpga.fpga_ptr(memlet.data,
                                    desc,
                                    sdfg,
                                    subset,
                                    decouple_array_interfaces=decouple_array_interfaces)
        else:
            ptrname = ptr(memlet.data, desc, sdfg, codegen)
        return "%s[%s]" % (ptrname, offset_cppstr)
    else:
        return offset_cppstr


def make_ptr_vector_cast(dst_expr, dst_dtype, src_dtype, is_scalar, defined_type):
    """
    If there is a type mismatch, cast pointer type. Used mostly in vector types.
    """
    if src_dtype != dst_dtype:
        if is_scalar:
            dst_expr = '*(%s *)(&%s)' % (src_dtype.ctype, dst_expr)
        elif src_dtype.base_type != dst_dtype:
            dst_expr = '(%s)(&%s)' % (src_dtype.ctype, dst_expr)
        elif defined_type in [DefinedType.Pointer, DefinedType.ArrayInterface]:
            dst_expr = '&' + dst_expr
    elif not is_scalar:
        dst_expr = '&' + dst_expr
    return dst_expr


def cpp_ptr_expr(sdfg,
                 memlet,
                 defined_type,
                 offset=None,
                 relative_offset=True,
                 use_other_subset=False,
                 indices=None,
                 is_write=None,
                 codegen=None,
                 decouple_array_interface=False):
    """ Converts a memlet to a C++ pointer expression. """
    subset = memlet.subset if not use_other_subset else memlet.other_subset
    s = subset if relative_offset else subsets.Indices(offset)
    o = offset if relative_offset else None
    desc = sdfg.arrays[memlet.data]
    if isinstance(indices, str):
        offset_cppstr = indices
    else:
        offset_cppstr = cpp_offset_expr(desc, s, o, indices=indices)
    if fpga.is_fpga_array(desc):
        dname = fpga.fpga_ptr(memlet.data,
                              desc,
                              sdfg,
                              s,
                              is_write,
                              None,
                              None,
                              defined_type == DefinedType.ArrayInterface,
                              decouple_array_interfaces=decouple_array_interface)
    else:
        dname = ptr(memlet.data, desc, sdfg, codegen)

    if defined_type == DefinedType.Scalar:
        dname = '&' + dname

    if offset_cppstr == '0':
        return dname
    else:
        return '%s + %s' % (dname, offset_cppstr)


def _check_range_conflicts(subset, a, itersym, b, step):
    found = False
    if isinstance(step, symbolic.SymExpr):
        step = step.approx
    for rb, re, _ in subset.ndrange():
        m = rb.match(a * itersym + b)
        if m is None:
            continue
        if (m[a] >= 1) != True:
            continue
        if re != rb:
            if isinstance(rb, symbolic.SymExpr):
                rb = rb.approx
            if isinstance(re, symbolic.SymExpr):
                re = re.approx

            # If False or indeterminate, the range may
            # overlap across iterations
            if ((re - rb) > m[a] * step) != False:
                continue

            m = re.match(a * itersym + b)
            if m is None:
                continue
            if (m[a] >= 1) != True:
                continue
        found = True
        break
    return found


def _check_map_conflicts(map, edge):
    for itervar, (_, _, mapskip) in zip(map.params, map.range):
        itersym = symbolic.pystr_to_symbolic(itervar)
        a = sp.Wild('a', exclude=[itersym])
        b = sp.Wild('b', exclude=[itersym])
        if not _check_range_conflicts(edge.data.subset, a, itersym, b, mapskip):
            return False
    # If matches all map params, good to go
    return True


def write_conflicted_map_params(map, edge):
    result = []
    for itervar, (_, _, mapskip) in zip(map.params, map.range):
        itersym = symbolic.pystr_to_symbolic(itervar)
        a = sp.Wild('a', exclude=[itersym])
        b = sp.Wild('b', exclude=[itersym])
        if not _check_range_conflicts(edge.data.subset, a, itersym, b, mapskip):
            result.append(itervar)

    return result


def is_write_conflicted(dfg, edge, datanode=None, sdfg_schedule=None):
    """
    Detects whether a write-conflict-resolving edge can be emitted without
    using atomics or critical sections.
    """
    return (is_write_conflicted_with_reason(dfg, edge, datanode, sdfg_schedule) is not None)


def is_write_conflicted_with_reason(dfg, edge, datanode=None, sdfg_schedule=None):
    """
    Detects whether a write-conflict-resolving edge can be emitted without
    using atomics or critical sections, returning the node or SDFG that caused
    the decision.
    :return: None if the conflict is nonatomic, otherwise returns the scope entry
             node or SDFG that caused the decision to be made.
    """

    if edge.data.wcr_nonatomic or edge.data.wcr is None:
        return None

    # If it's an entire SDFG, it's probably write-conflicted
    if isinstance(dfg, SDFG):
        if datanode is None:
            return dfg
        in_edges = find_incoming_edges(datanode, dfg)
        if len(in_edges) != 1:
            return dfg
        if (isinstance(in_edges[0].src, nodes.ExitNode) and
                (in_edges[0].src.map.schedule == dtypes.ScheduleType.Sequential or
                 in_edges[0].src.map.schedule == dtypes.ScheduleType.Snitch)):
            return None
        return dfg
    elif isinstance(dfg, gr.SubgraphView):
        dfg = dfg.graph

    # Traverse memlet path to determine conflicts.
    # If no conflicts will occur, write without atomics
    # (e.g., if the array has been defined in a non-parallel schedule context)
    while edge is not None:
        path = dfg.memlet_path(edge)
        for e in path:
            if (isinstance(e.dst, nodes.ExitNode)
                    and (e.dst.map.schedule != dtypes.ScheduleType.Sequential and
                 e.dst.map.schedule != dtypes.ScheduleType.Snitch)):
                if _check_map_conflicts(e.dst.map, e):
                    # This map is parallel w.r.t. WCR
                    # print('PAR: Continuing from map')
                    continue
                # print('SEQ: Map is conflicted')
                return dfg.entry_node(e.dst)
            # Should never happen (no such thing as write-conflicting reads)
            if (isinstance(e.src, nodes.EntryNode) and e.src.map.schedule != dtypes.ScheduleType.Sequential):
                warnings.warn('Unexpected WCR path to have write-conflicting reads')
                return e.src

        sdfg = dfg.parent
        dst = path[-1].dst
        # Unexpected case
        if not isinstance(dst, nodes.AccessNode):
            warnings.warn('Unexpected WCR path to not end in access node')
            return dst

        if dfg.in_degree(dst) > 0:
            for x, y in itertools.combinations(dfg.in_edges(dst), 2):
                x, y = x.data.subset, y.data.subset
                if subsets.intersects(x, y):
                    return dst

        # If this is a nested SDFG and the access leads outside
        if not sdfg.arrays[dst.data].transient:
            if sdfg.parent_nsdfg_node is not None:
                dfg = sdfg.parent
                nsdfg = sdfg.parent_nsdfg_node
                edge = next(iter(dfg.out_edges_by_connector(nsdfg, dst.data)))
            else:
                break
        else:
            # Memlet path ends here, transient. We can thus safely write here
            edge = None
            # print('PAR: Reached transient')
            return None

    return None


class LambdaToFunction(ast.NodeTransformer):
    def visit_Lambda(self, node: ast.Lambda):
        newbody = [ast.Return(value=node.body)]
        newnode = ast.FunctionDef(name="_anonymous", args=node.args, body=newbody, decorator_list=[])
        newnode = ast.copy_location(newnode, node)
        return ast.fix_missing_locations(newnode)


def unparse_cr_split(sdfg, wcr_ast):
    """ Parses various types of WCR functions, returning a 2-tuple of body (in
        C++), and a list of arguments. """
    if isinstance(wcr_ast, ast.Lambda):
        # Convert the lambda expression into a function that we can parse
        funcdef = LambdaToFunction().visit(wcr_ast)
        return unparse_cr_split(sdfg, funcdef)
    elif isinstance(wcr_ast, ast.FunctionDef):
        # Process data structure initializers
        sinit = StructInitializer(sdfg)
        body = [sinit.visit(stmt) for stmt in wcr_ast.body]

        # Construct a C++ lambda function out of a function
        args = [n.arg for n in wcr_ast.args.args]
        return cppunparse.cppunparse(body, expr_semicolon=False), args
    elif isinstance(wcr_ast, ast.Module):
        return unparse_cr_split(sdfg, wcr_ast.body[0].value)
    elif isinstance(wcr_ast, str):
        return unparse_cr_split(sdfg, LambdaProperty.from_string(wcr_ast))
    else:
        raise NotImplementedError("INVALID TYPE OF WCR: " + type(wcr_ast).__name__)


def unparse_cr(sdfg, wcr_ast, dtype):
    """ Outputs a C++ version of a conflict resolution lambda. """
    body_cpp, args = unparse_cr_split(sdfg, wcr_ast)

    ctype = 'auto' if dtype is None else dtype.ctype

    # Construct a C++ lambda function out of a function
    return '[] (%s) { %s }' % (', '.join('const %s& %s' % (ctype, a) for a in args), body_cpp)


def connected_to_gpu_memory(node: nodes.Node, state: SDFGState, sdfg: SDFG):
    for e in state.all_edges(node):
        path = state.memlet_path(e)
        if ((isinstance(path[0].src, nodes.AccessNode)
             and path[0].src.desc(sdfg).storage is dtypes.StorageType.GPU_Global)):
            return True
    return False


def unparse_tasklet(sdfg, state_id, dfg, node, function_stream, callsite_stream, locals, ldepth, toplevel_schedule,
                    codegen):

    if node.label is None or node.label == "":
        return ""

    state_dfg = sdfg.nodes()[state_id]

    # Not [], "" or None
    if not node.code:
        return ""

    # Not [], "" or None
    if node.code_global and node.code_global.code:
        function_stream.write(
            codeblock_to_cpp(node.code_global),
            sdfg,
            state_id,
            node,
        )
        function_stream.write("\n", sdfg, state_id, node)

    # add node state_fields to the statestruct
    codegen._frame.statestruct.extend(node.state_fields)

    # If raw C++ code, return the code directly
    if node.language != dtypes.Language.Python:
        # If this code runs on the host and is associated with a GPU stream,
        # set the stream to a local variable.
        max_streams = int(Config.get("compiler", "cuda", "max_concurrent_streams"))
        if not is_devicelevel_gpu(sdfg, state_dfg, node) and (hasattr(node, "_cuda_stream")
                                                              or connected_to_gpu_memory(node, state_dfg, sdfg)):
            if max_streams >= 0:
                callsite_stream.write(
                    'int __dace_current_stream_id = %d;\n%sStream_t __dace_current_stream = __state->gpu_context->streams[__dace_current_stream_id];'
                    % (node._cuda_stream, Config.get('compiler', 'cuda', 'backend')),
                    sdfg,
                    state_id,
                    node,
                )
            else:
                callsite_stream.write(
                    '%sStream_t __dace_current_stream = nullptr;' % Config.get('compiler', 'cuda', 'backend'),
                    sdfg,
                    state_id,
                    node,
                )

        if node.language != dtypes.Language.CPP and node.language != dtypes.Language.MLIR:
            raise ValueError("Only Python, C++ or MLIR code supported in CPU codegen, got: {}".format(node.language))

        if node.language == dtypes.Language.MLIR:
            # Inline import because mlir.utils depends on pyMLIR which may not be installed
            # Doesn't cause crashes due to missing pyMLIR if a MLIR tasklet is not present
            from dace.codegen.targets.mlir import utils

            mlir_func_uid = "_" + str(sdfg.sdfg_id) + "_" + str(state_id) + "_" + str(dfg.node_id(node))

            mlir_ast = utils.get_ast(node.code.code)
            mlir_is_generic = utils.is_generic(mlir_ast)
            mlir_entry_func = utils.get_entry_func(mlir_ast, mlir_is_generic, mlir_func_uid)

            # Arguments of the MLIR must match the input connector names of the tasklet (the "%" excluded)
            mlir_in_typed = ""
            mlir_in_untyped = ""

            for mlir_arg in utils.get_entry_args(mlir_entry_func, mlir_is_generic):
                mlir_arg_name = mlir_arg[0]
                mlir_arg_type = node.in_connectors[mlir_arg_name].ctype
                mlir_in_typed = mlir_in_typed + mlir_arg_type + " " + mlir_arg_name + ", "
                mlir_in_untyped = mlir_in_untyped + mlir_arg_name + ", "

            mlir_in_typed = mlir_in_typed[:-2]
            mlir_in_untyped = mlir_in_untyped[:-2]

            mlir_out = next(iter(node.out_connectors.items()))
            mlir_out_type = mlir_out[1].ctype
            mlir_out_name = mlir_out[0]

            # MLIR tools such as mlir-opt and mlir-translate as well as the LLVM compiler "lc" will be required to compile the MLIR tasklet
            function_stream.write('extern "C" ' + mlir_out_type + ' mlir_entry' + mlir_func_uid + '(' + mlir_in_typed +
                                  ');\n\n')
            callsite_stream.write(mlir_out_name + " = mlir_entry" + mlir_func_uid + "(" + mlir_in_untyped + ");")

        if node.language == dtypes.Language.CPP:
            callsite_stream.write(type(node).__properties__["code"].to_string(node.code), sdfg, state_id, node)

        if hasattr(node, "_cuda_stream") and not is_devicelevel_gpu(sdfg, state_dfg, node):
            synchronize_streams(sdfg, state_dfg, state_id, node, node, callsite_stream)
        return

    body = node.code.code

    # Map local names to memlets (for WCR detection)
    memlets = {}
    for edge in state_dfg.all_edges(node):
        u, uconn, v, vconn, memlet = edge
        if u == node:
            memlet_nc = not is_write_conflicted(dfg, edge, sdfg_schedule=toplevel_schedule)
            memlet_wcr = memlet.wcr
            if uconn in u.out_connectors:
                conntype = u.out_connectors[uconn]
            else:
                conntype = None

            memlets[uconn] = (memlet, memlet_nc, memlet_wcr, conntype)
        elif v == node:
            if vconn in v.in_connectors:
                conntype = v.in_connectors[vconn]
            else:
                conntype = None

            memlets[vconn] = (memlet, False, None, conntype)

    # To prevent variables-redefinition, build dictionary with all the previously defined symbols
    defined_symbols = state_dfg.symbols_defined_at(node)

    defined_symbols.update(
        {k: v.dtype if hasattr(v, 'dtype') else dtypes.typeclass(type(v))
         for k, v in sdfg.constants.items()})

    for connector, (memlet, _, _, conntype) in memlets.items():
        if connector is not None:
            defined_symbols.update({connector: conntype})

    callsite_stream.write("// Tasklet code (%s)\n" % node.label, sdfg, state_id, node)
    for stmt in body:
        stmt = copy.deepcopy(stmt)
        rk = StructInitializer(sdfg).visit(stmt)
        if isinstance(stmt, ast.Expr):
            rk = DaCeKeywordRemover(sdfg, memlets, sdfg.constants, codegen).visit_TopLevelExpr(stmt)
        else:
            rk = DaCeKeywordRemover(sdfg, memlets, sdfg.constants, codegen).visit(stmt)

        if rk is not None:
            # Unparse to C++ and add 'auto' declarations if locals not declared
            result = StringIO()
            cppunparse.CPPUnparser(rk, ldepth + 1, locals, result, defined_symbols=defined_symbols)
            callsite_stream.write(result.getvalue(), sdfg, state_id, node)


def shape_to_strides(shape):
    """ Constructs strides from shape (for objects with no special strides). """
    strides = []
    curstride = 1
    for s in reversed(shape):
        strides.append(curstride)
        curstride *= s
    return list(reversed(strides))


class InterstateEdgeUnparser(cppunparse.CPPUnparser):
    """
    An extension of the Python->C++ unparser that allows including
    multidimensional array expressions from an existing SDFGs. Used in
    inter-state edge code generation.
    """
    def __init__(self, sdfg: SDFG, tree: ast.AST, file: IO[str], defined_symbols=None, codegen=None):
        self.sdfg = sdfg
        self.codegen = codegen
        super().__init__(tree, 0, cppunparse.CPPLocals(), file, expr_semicolon=False, defined_symbols=defined_symbols)

    def _Name(self, t: ast.Name):
        if t.id not in self.sdfg.arrays:
            return super()._Name(t)

        # Replace values with their code-generated names (for example,
        # persistent arrays)
        desc = self.sdfg.arrays[t.id]
        self.write(ptr(t.id, desc, self.sdfg, self.codegen))

    def _Subscript(self, t: ast.Subscript):
        from dace.frontend.python.astutils import subscript_to_slice
        target, rng = subscript_to_slice(t, self.sdfg.arrays)
        rng = subsets.Range(rng)
        if rng.num_elements() != 1:
            raise SyntaxError('Range subscripts disallowed in interstate edges')

        memlet = mmlt.Memlet(data=target, subset=rng)

        if target not in self.sdfg.arrays:
            # This could be an FPGA array whose name has been mangled
            unqualified = fpga.unqualify_fpga_array_name(self.sdfg, target)
            desc = self.sdfg.arrays[unqualified]
            self.write(cpp_array_expr(self.sdfg, memlet, referenced_array=desc, codegen=self.codegen))
        else:
            self.write(cpp_array_expr(self.sdfg, memlet, codegen=self.codegen))


def unparse_interstate_edge(code_ast: Union[ast.AST, str], sdfg: SDFG, symbols=None, codegen=None) -> str:
    # Convert from code to AST as necessary
    if isinstance(code_ast, str):
        code_ast = ast.parse(code_ast).body[0]

    strio = StringIO()
    InterstateEdgeUnparser(sdfg, code_ast, strio, symbols, codegen)
    return strio.getvalue().strip()


class DaCeKeywordRemover(ExtNodeTransformer):
    """ Removes memlets and other DaCe keywords from a Python AST, and
        converts array accesses to C++ methods that can be generated.

        Used for unparsing Python tasklets into C++ that uses the DaCe
        runtime.

        :note: Assumes that the DaCe syntax is correct (as verified by the
               Python frontend).
    """
    def __init__(self, sdfg, memlets, constants, codegen):
        self.sdfg = sdfg
        self.memlets = memlets
        self.constants = constants
        self.codegen = codegen
        self.allow_casts = True
        self._decouple_array_interfaces = Config.get_bool("compiler", "xilinx", "decouple_array_interfaces")

    def visit_TopLevelExpr(self, node):
        # This is a DaCe shift, omit it
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.op, ast.LShift) or isinstance(node.value.op, ast.RShift):
                return None
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        if not isinstance(node.target, ast.Subscript):
            return self.generic_visit(node)

        target = rname(node.target)
        if target not in self.memlets:
            return self.generic_visit(node)

        raise SyntaxError("Augmented assignments (e.g. +=) not allowed on " + "array memlets")

    def _replace_assignment(self, newnode: ast.AST, node: ast.Assign) -> ast.AST:
        locfix = ast.copy_location(newnode, node.value)
        if len(node.targets) == 1:
            return locfix
        # More than one target, i.e., x = y = z
        return ast.copy_location(ast.Assign(targets=node.targets[:-1], value=locfix), node)

    def _subscript_expr(self, slicenode: ast.AST, target: str) -> symbolic.SymbolicType:
        visited_slice = self.visit(slicenode)

        if isinstance(visited_slice, ast.Index):
            visited_slice = visited_slice.value

        # Collect strides for index expressions
        if target in self.constants:
            strides = shape_to_strides(self.constants[target].shape)
        else:
            memlet = self.memlets[target][0]
            dtype = self.memlets[target][3]
            dname = memlet.data
            strides = self.sdfg.arrays[dname].strides
            # Get memlet absolute strides, including tile sizes
            strides = memlet.subset.absolute_strides(strides)
            # Filter ("squeeze") strides w.r.t. scalar dimensions
            dimlen = dtype.veclen if isinstance(dtype, dtypes.vector) else 1
            subset_size = memlet.subset.size()
            indexdims = [i for i, s in enumerate(subset_size) if s == 1]
            # Pointer to a single element can use all strides
            is_scalar = not isinstance(dtype, dtypes.pointer)
            if is_scalar or data._prod(subset_size) != 1:
                strides = [
                    s for i, s in enumerate(strides) if i not in indexdims and not (s == 1 and subset_size[i] == dimlen)
                ]

        if isinstance(visited_slice, ast.Tuple):
            if len(strides) != len(visited_slice.elts):
                raise SyntaxError('Invalid number of dimensions in expression (expected %d, '
                                  'got %d)' % (len(strides), len(visited_slice.elts)))

            return sum(symbolic.pystr_to_symbolic(unparse(elt)) * s for elt, s in zip(visited_slice.elts, strides))

        if len(strides) != 1:
            raise SyntaxError('Missing dimensions in expression (expected %d, got one)' % len(strides))

        try:
            return symbolic.pystr_to_symbolic(unparse(visited_slice)) * strides[0]
        except (TypeError, sp.SympifyError):
            # Fallback in case of .pop() or other C++ mannerisms
            return f'({unparse(visited_slice)}) * {strides[0]}'

    def visit_Assign(self, node):
        target = rname(node.targets[-1])
        if target not in self.memlets:
            return self.generic_visit(node)

        memlet, nc, wcr, dtype = self.memlets[target]
        value = self.visit(astutils.copy_tree(node.value))

        if not isinstance(node.targets[-1], ast.Subscript):
            # Dynamic accesses or streams -> every access counts
            try:
                desc = (self.sdfg.arrays[memlet.data] if memlet and memlet.data else None)
                if memlet and memlet.data and (memlet.dynamic or isinstance(desc, data.Stream)):
                    ptrname = ptr(memlet.data, desc, self.sdfg, self.codegen._frame)
                    if wcr is not None:
                        newnode = ast.Name(
                            id=self.codegen.write_and_resolve_expr(self.sdfg,
                                                                   memlet,
                                                                   nc,
                                                                   target,
                                                                   cppunparse.cppunparse(value, expr_semicolon=False),
                                                                   dtype=dtype))
                        node.value = ast.copy_location(newnode, node.value)
                        return node
                    elif isinstance(desc, data.Stream):
                        if desc.is_stream_array():
                            index = cpp_offset_expr(desc, memlet.subset)
                            target = f"{ptrname}[{index}]"
                        else:
                            target = ptrname
                        newnode = ast.Name(id="%s.push(%s);" % (
                            target,
                            cppunparse.cppunparse(value, expr_semicolon=False),
                        ))
                    else:
                        var_type, ctypedef = self.codegen._dispatcher.defined_vars.get(ptrname)
                        if var_type == DefinedType.Scalar:
                            newnode = ast.Name(id="%s = %s;" % (
                                ptrname,
                                cppunparse.cppunparse(value, expr_semicolon=False),
                            ))
                        elif (var_type != DefinedType.ArrayInterface or isinstance(desc, data.View)):
                            newnode = ast.Name(id="%s = %s;" % (
                                cpp_array_expr(self.sdfg, memlet, codegen=self.codegen._frame),
                                cppunparse.cppunparse(value, expr_semicolon=False),
                            ))
                        else:
                            array_interface_name = fpga.fpga_ptr(
                                ptrname,
                                desc,
                                self.sdfg,
                                memlet.dst_subset,
                                True,
                                None,
                                None,
                                True,
                                decouple_array_interfaces=self._decouple_array_interfaces)
                            newnode = ast.Name(
                                id=f"{array_interface_name}"
                                f"[{cpp_array_expr(self.sdfg, memlet, with_brackets=False, codegen=self.codegen._frame)}]"
                                f" = {cppunparse.cppunparse(value, expr_semicolon=False)};")

                    return self._replace_assignment(newnode, node)
            except TypeError:  # cannot determine truth value of Relational
                pass

            return self.generic_visit(node)

        subscript = self._subscript_expr(node.targets[-1].slice, target)

        if wcr is not None:
            newnode = ast.Name(
                id=self.codegen.write_and_resolve_expr(self.sdfg,
                                                       memlet,
                                                       nc,
                                                       target,
                                                       cppunparse.cppunparse(value, expr_semicolon=False),
                                                       indices=sym2cpp(subscript),
                                                       dtype=dtype) + ';')
        else:
            newnode = ast.Name(id="%s[%s] = %s;" %
                               (target, sym2cpp(subscript), cppunparse.cppunparse(value, expr_semicolon=False)))

        return self._replace_assignment(newnode, node)

    def visit_Subscript(self, node):
        target = rname(node)
        if target not in self.memlets and target not in self.constants:
            return self.generic_visit(node)

        subscript = self._subscript_expr(node.slice, target)

        # New subscript is created as a name AST object (rather than a
        # subscript), as otherwise the visitor will recursively descend into
        # the new expression and modify it erroneously.
        defined = set(self.memlets.keys()) | set(self.constants.keys())
        newnode = ast.Name(id="%s[%s]" % (target, sym2cpp(subscript, defined)))

        return ast.copy_location(newnode, node)

    def visit_Call(self, node: ast.Call):
        funcname = rname(node.func)
        if (funcname in self.sdfg.symbols and isinstance(self.sdfg.symbols[funcname], dtypes.callback)):
            # Visit arguments without changing their types
            self.allow_casts = False
            result = self.generic_visit(node)
            self.allow_casts = True
            return result
        else:
            return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        name = rname(node)
        if name not in self.memlets:
            return self.generic_visit(node)
        memlet, nc, wcr, dtype = self.memlets[name]
        if node.id in self.sdfg.arrays:
            ptrname = ptr(node.id, self.sdfg.arrays[node.id], self.sdfg, self.codegen._frame)
        else:
            ptrname = node.id
        try:
            defined_type, _ = self.codegen._dispatcher.defined_vars.get(ptrname)
        except KeyError:
            defined_type = None
        if (self.allow_casts and isinstance(dtype, dtypes.pointer) and memlet.subset.num_elements() == 1):
            return ast.parse(f"{name}[0]").body[0].value
        elif (self.allow_casts and (defined_type == DefinedType.Stream or defined_type == DefinedType.StreamArray)
              and memlet.dynamic):
            return ast.parse(f"{name}.pop()").body[0].value
        else:
            return self.generic_visit(node)

    def visit_Expr(self, node):
        # Check for DaCe function calls
        if isinstance(node.value, ast.Call):
            # Some calls should not be parsed
            if rname(node.value.func) == "define_local":
                return None
            elif rname(node.value.func) == "define_local_scalar":
                return None
            elif rname(node.value.func) == "define_stream":
                return None
            elif rname(node.value.func) == "define_streamarray":
                return None

        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Do not parse internal functions
        return None

    def visit_BinOp(self, node: ast.BinOp):
        # Special case for integer powers
        if isinstance(node.op, ast.Pow):
            from dace.frontend.python import astutils
            try:
                unparsed = symbolic.pystr_to_symbolic(
                    astutils.evalnode(node.right, {
                        **self.constants, 'dace': dace,
                        'math': math
                    }))
                evaluated = symbolic.symstr(symbolic.evaluate(unparsed, self.constants))
                node.right = ast.parse(evaluated).body[0].value
            except (TypeError, AttributeError, NameError, KeyError, ValueError, SyntaxError):
                return self.generic_visit(node)

        return self.generic_visit(node)

    # Replace default modules (e.g., math) with dace::math::
    def visit_Attribute(self, node):
        attrname = rname(node)
        module_name = attrname[:attrname.rfind(".")]
        func_name = attrname[attrname.rfind(".") + 1:]
        if module_name in dtypes._ALLOWED_MODULES:
            cppmodname = dtypes._ALLOWED_MODULES[module_name]
            return ast.copy_location(ast.Name(id=(cppmodname + func_name), ctx=ast.Load), node)
        return self.generic_visit(node)


class StructInitializer(ExtNodeTransformer):
    """ Replace struct creation calls with compound literal struct
        initializers in tasklets. """
    def __init__(self, sdfg: SDFG):
        self._structs = {}
        if sdfg is None:
            return

        # Find all struct types in SDFG
        for array in sdfg.arrays.values():
            if array is None or not hasattr(array, "dtype"):
                continue
            if isinstance(array.dtype, dace.dtypes.struct):
                self._structs[array.dtype.name] = array.dtype

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and (node.func.id.startswith('__DACESTRUCT_')
                                                or node.func.id in self._structs):
            fields = ', '.join([
                '.%s = %s' % (rname(arg.arg), cppunparse.pyexpr2cpp(arg.value))
                for arg in sorted(node.keywords, key=lambda x: x.arg)
            ])

            tname = node.func.id
            if node.func.id.startswith('__DACESTRUCT_'):
                tname = node.func.id[len('__DACESTRUCT_'):]

            return ast.copy_location(ast.Name(id="%s { %s }" % (tname, fields), ctx=ast.Load), node)

        return self.generic_visit(node)


# TODO: This should be in the CUDA code generator. Add appropriate conditions to node dispatch predicate
def presynchronize_streams(sdfg, dfg, state_id, node, callsite_stream):
    state_dfg = sdfg.nodes()[state_id]
    if hasattr(node, "_cuda_stream") or is_devicelevel_gpu(sdfg, state_dfg, node):
        return
    backend = Config.get('compiler', 'cuda', 'backend')
    for e in state_dfg.in_edges(node):
        if hasattr(e.src, "_cuda_stream"):
            cudastream = "__state->gpu_context->streams[%d]" % e.src._cuda_stream
            callsite_stream.write(
                "%sStreamSynchronize(%s);" % (backend, cudastream),
                sdfg,
                state_id,
                [e.src, e.dst],
            )


# TODO: This should be in the CUDA code generator. Add appropriate conditions to node dispatch predicate
def synchronize_streams(sdfg, dfg, state_id, node, scope_exit, callsite_stream):
    # Post-kernel stream synchronization (with host or other streams)
    max_streams = int(Config.get("compiler", "cuda", "max_concurrent_streams"))
    backend = Config.get('compiler', 'cuda', 'backend')
    if max_streams >= 0:
        cudastream = "__state->gpu_context->streams[%d]" % node._cuda_stream
        for edge in dfg.out_edges(scope_exit):
            # Synchronize end of kernel with output data (multiple kernels
            # lead to same data node)
            if (isinstance(edge.dst, nodes.AccessNode) and hasattr(edge.dst, '_cuda_stream')
                    and edge.dst._cuda_stream != node._cuda_stream):
                callsite_stream.write(
                    """{backend}EventRecord(__state->gpu_context->events[{ev}], {src_stream});
{backend}StreamWaitEvent(__state->gpu_context->streams[{dst_stream}], __state->gpu_context->events[{ev}], 0);""".format(
                        ev=edge._cuda_event if hasattr(edge, "_cuda_event") else 0,
                        src_stream=cudastream,
                        dst_stream=edge.dst._cuda_stream,
                        backend=backend,
                    ),
                    sdfg,
                    state_id,
                    [edge.src, edge.dst],
                )
                continue

            # If a view, get the relevant access node
            dstnode = edge.dst
            while isinstance(sdfg.arrays[dstnode.data], data.View):
                dstnode = dfg.out_edges(dstnode)[0].dst

            # We need the streams leading out of the output data
            for e in dfg.out_edges(dstnode):
                if isinstance(e.dst, nodes.AccessNode):
                    continue
                # If no stream at destination: synchronize stream with host.
                if not hasattr(e.dst, "_cuda_stream"):
                    pass
                    # Done at destination

                # If different stream at destination: record event and wait
                # for it in target stream.
                elif e.dst._cuda_stream != node._cuda_stream:
                    callsite_stream.write(
                        """{backend}EventRecord(__state->gpu_context->events[{ev}], {src_stream});
    {backend}StreamWaitEvent(__state->gpu_context->streams[{dst_stream}], __state->gpu_context->events[{ev}], 0);""".
                        format(
                            ev=e._cuda_event if hasattr(e, "_cuda_event") else 0,
                            src_stream=cudastream,
                            dst_stream=e.dst._cuda_stream,
                            backend=backend,
                        ),
                        sdfg,
                        state_id,
                        [e.src, e.dst],
                    )
                # Otherwise, no synchronization necessary
