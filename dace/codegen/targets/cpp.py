import ast
import copy
import functools

import sympy as sp
from six import StringIO

import dace
from dace import data, subsets, symbolic, dtypes, memlet as mmlt
from dace.codegen import cppunparse
from dace.codegen.targets.common import (sym2cpp, find_incoming_edges,
                                         codeblock_to_cpp)
from dace.codegen.targets.target import DefinedType
from dace.config import Config
from dace.frontend import operations
from dace.frontend.python.astutils import ExtNodeTransformer, rname, unparse
from dace.sdfg import nodes
from dace.properties import LambdaProperty
from dace.sdfg import SDFG, is_devicelevel_gpu


def copy_expr(
    dispatcher,
    sdfg,
    dataname,
    memlet,
    offset=None,
    relative_offset=True,
    packed_types=False,
):
    datadesc = sdfg.arrays[dataname]
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
        offset_cppstr = cpp_offset_expr(datadesc, s, o,
                                        memlet.veclen if packed_types else 1)
    else:
        offset_cppstr = "0"
    dt = ""

    if memlet.veclen != 1 and not packed_types:
        offset_cppstr = "(%s) / %s" % (offset_cppstr, sym2cpp(memlet.veclen))
        dt = "(dace::vec<%s, %s> *)" % (
            datadesc.dtype.ctype,
            sym2cpp(memlet.veclen),
        )

    expr = dataname

    def_type = dispatcher.defined_vars.get(dataname)

    add_offset = offset_cppstr != "0"

    if def_type == DefinedType.Pointer:
        return "{}{}{}".format(
            dt, expr, " + {}".format(offset_cppstr) if add_offset else "")

    elif def_type == DefinedType.ArrayView:
        return "{}{}.ptr(){}".format(
            dt, expr, " + {}".format(offset_cppstr) if add_offset else "")

    elif def_type == DefinedType.StreamArray:
        return "{}[{}]".format(expr, offset_cppstr)

    elif def_type == DefinedType.FPGA_ShiftRegister:
        return expr

    elif def_type in [
            DefinedType.Scalar, DefinedType.Stream, DefinedType.StreamView
    ]:

        if add_offset:
            raise TypeError("Tried to offset address of scalar {}: {}".format(
                dataname, offset_cppstr))

        if def_type == DefinedType.Scalar:
            return "{}&{}".format(dt, expr)
        else:
            return dataname

    else:
        raise NotImplementedError("copy_expr not implemented "
                                  "for connector type: {}".format(def_type))


def memlet_copy_to_absolute_strides(dispatcher,
                                    sdfg,
                                    memlet,
                                    src_node,
                                    dst_node,
                                    packed_types=False):
    # TODO: Take both source and destination subset into account for computing
    # copy shape.
    copy_shape = memlet.subset.size_exact()
    src_nodedesc = src_node.desc(sdfg)
    dst_nodedesc = dst_node.desc(sdfg)

    if memlet.data == src_node.data:
        src_expr = copy_expr(dispatcher,
                             sdfg,
                             src_node.data,
                             memlet,
                             packed_types=packed_types)
        dst_expr = copy_expr(dispatcher,
                             sdfg,
                             dst_node.data,
                             memlet,
                             None,
                             False,
                             packed_types=packed_types)
        if memlet.other_subset is not None:
            dst_expr = copy_expr(
                dispatcher,
                sdfg,
                dst_node.data,
                memlet,
                memlet.other_subset,
                False,
                packed_types=packed_types,
            )
            dst_subset = memlet.other_subset
        else:
            dst_subset = subsets.Range.from_array(dst_nodedesc)
        src_subset = memlet.subset

    else:
        src_expr = copy_expr(dispatcher,
                             sdfg,
                             src_node.data,
                             memlet,
                             None,
                             False,
                             packed_types=packed_types)
        dst_expr = copy_expr(dispatcher,
                             sdfg,
                             dst_node.data,
                             memlet,
                             packed_types=packed_types)
        if memlet.other_subset is not None:
            src_expr = copy_expr(
                dispatcher,
                sdfg,
                src_node.data,
                memlet,
                memlet.other_subset,
                False,
                packed_types=packed_types,
            )
            src_subset = memlet.other_subset
        else:
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
        src_strides = ([
            stride
            for stride, s in zip(src_strides, src_subset.size()) if s != 1
        ] + src_strides[len(src_subset):])  # Include tiles
        if not src_strides:
            src_strides = [1]
        dst_strides = ([
            stride
            for stride, s in zip(dst_strides, dst_subset.size()) if s != 1
        ] + dst_strides[len(dst_subset):])  # Include tiles
        if not dst_strides:
            dst_strides = [1]
        copy_shape = [s for s in copy_shape if s != 1]
        if not copy_shape:
            copy_shape = [1]

    # Extend copy shape to the largest among the data dimensions,
    # and extend other array with the appropriate strides
    if len(dst_strides) != len(copy_shape) or len(src_strides) != len(
            copy_shape):
        if memlet.data == src_node.data:
            copy_shape, dst_strides = reshape_strides(src_subset, src_strides,
                                                      dst_strides, copy_shape)
        elif memlet.data == dst_node.data:
            copy_shape, src_strides = reshape_strides(dst_subset, dst_strides,
                                                      src_strides, copy_shape)

    if memlet.veclen != 1:
        int_floor = sp.Function("int_floor")
        src_strides[:-1] = [
            int_floor(s, memlet.veclen) for s in src_strides[:-1]
        ]
        dst_strides[:-1] = [
            int_floor(s, memlet.veclen) for s in dst_strides[:-1]
        ]
        if not packed_types:
            copy_shape[-1] = int_floor(copy_shape[-1], memlet.veclen)

    return copy_shape, src_strides, dst_strides, src_expr, dst_expr


def emit_memlet_reference(dispatcher, sdfg: SDFG, memlet: mmlt.Memlet,
                          pointer_name: str):
    """ Returns a string with a definition of a reference to an existing
        memlet. Used in nested SDFG arguments. """
    desc = sdfg.arrays[memlet.data]
    typedef = 'auto'  # TODO(later): Use non-auto types (vec, Stream<>...)
    datadef = memlet.data
    offset_expr = '[' + cpp_offset_expr(desc, memlet.subset) + ']'

    # Get defined type (pointer, stream etc.) and change the type definition
    # accordingly.
    defined_type = dispatcher.defined_vars.get(memlet.data)
    if defined_type == DefinedType.Pointer:
        typedef += '*'
        datadef = '&' + datadef
    elif defined_type == DefinedType.Scalar:
        typedef += '&'
        offset_expr = ''
    elif defined_type == DefinedType.Stream:
        typedef += '&'
        offset_expr = ''
    else:
        typedef += '&'
        defined_type = DefinedType.Pointer

    # Register defined variable
    dispatcher.defined_vars.add(pointer_name,
                                defined_type,
                                allow_shadowing=True)

    return '%s %s = %s%s;' % (typedef, pointer_name, datadef, offset_expr)


def reshape_strides(subset, strides, original_strides, copy_shape):
    """ Helper function that reshapes a shape to the given strides. """
    # TODO(later): Address original strides in the computation of the
    #              result strides.
    original_copy_shape = subset.size()
    dims = len(copy_shape)

    reduced_tile_sizes = [
        ts for ts, s in zip(subset.tile_sizes, original_copy_shape) if s != 1
    ]

    reshaped_copy = copy_shape + [ts for ts in subset.tile_sizes if ts != 1]
    reshaped_copy[:len(copy_shape)] = [
        s / ts for s, ts in zip(copy_shape, reduced_tile_sizes)
    ]

    new_strides = [0] * len(reshaped_copy)
    elements_remaining = functools.reduce(sp.mul.Mul, copy_shape, 1)
    tiledim = 0
    for i in range(len(copy_shape)):
        new_strides[i] = elements_remaining / reshaped_copy[i]
        elements_remaining = new_strides[i]
        if reduced_tile_sizes[i] != 1:
            new_strides[dims + tiledim] = (elements_remaining /
                                           reshaped_copy[dims + tiledim])
            elements_remaining = new_strides[dims + tiledim]
            tiledim += 1

    return reshaped_copy, new_strides


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
    last_src_index = src_subset.at([d - 1 for d in src_subset.size()],
                                   src_strides)
    last_dst_index = dst_subset.at([d - 1 for d in dst_subset.size()],
                                   dst_strides)
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
    if ((src_copylen == copy_length and dst_copylen == copy_length)
            or (tuple(src_shape) == tuple(copy_shape)
                and tuple(dst_shape) == tuple(copy_shape)
                and tuple(src_strides) == tuple(dst_strides))):
        # Emit 1D copy of the whole array
        copy_shape = [functools.reduce(lambda x, y: x * y, copy_shape)]
        return copy_shape, [1], [1]
    # 1D strided copy
    elif sum([0 if c == 1 else 1 for c in copy_shape]) == 1:
        # Find the copied dimension:
        # In copy shape
        copydim = next(i for i, c in enumerate(copy_shape) if c != 1)

        # In source strides
        if len(copy_shape) == len(src_shape):
            srcdim = copydim
        else:
            srcdim = next(i for i, c in enumerate(src_shape) if c != 1)

        # In destination strides
        if len(copy_shape) == len(dst_shape):
            dstdim = copydim
        else:
            dstdim = next(i for i, c in enumerate(dst_shape) if c != 1)

        # Return new copy
        return [copy_shape[copydim]], [src_strides[srcdim]
                                       ], [dst_strides[dstdim]]
    else:
        return None


def cpp_offset_expr(d: data.Data,
                    subset_in: subsets.Subset,
                    offset=None,
                    packed_veclen=1,
                    indices=None):
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
    subset = copy.deepcopy(subset_in)

    # Offset according to parameters
    if offset is not None:
        if isinstance(offset, subsets.Subset):
            subset.offset(offset, False)
        else:
            subset.offset(subsets.Indices(offset), False)

    # Then, offset according to array
    subset.offset(subsets.Indices(d.offset), False)

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
                   indices=None):
    """ Converts an Indices/Range object to a C++ array access string. """
    subset = memlet.subset if not use_other_subset else memlet.other_subset
    s = subset if relative_offset else subsets.Indices(offset)
    o = offset if relative_offset else None
    offset_cppstr = cpp_offset_expr(sdfg.arrays[memlet.data],
                                    s,
                                    o,
                                    packed_veclen,
                                    indices=indices)

    if with_brackets:
        return "%s[%s]" % (memlet.data, offset_cppstr)
    else:
        return offset_cppstr


def write_and_resolve_expr(sdfg, memlet, nc, outname, inname, indices=None):
    """ Helper function that emits a write_and_resolve call from a memlet. """

    redtype = operations.detect_reduction_type(memlet.wcr)

    nc = "_nc" if nc else ""
    indstr = (", " + indices) if indices is not None else ""

    reduction_tmpl = ""
    custom_reduction = ""

    # Special call for detected reduction types
    if redtype != dtypes.ReductionType.Custom:
        credtype = "dace::ReductionType::" + str(
            redtype)[str(redtype).find(".") + 1:]
        reduction_tmpl = "<%s>" % credtype
    else:
        custom_reduction = ', %s' % unparse_cr(sdfg, memlet.wcr)

    return "{oname}.write_and_resolve{nc}{tmpl}({iname}{wcr}{ind});".format(
        oname=outname,
        nc=nc,
        tmpl=reduction_tmpl,
        iname=inname,
        wcr=custom_reduction,
        ind=indstr,
    )


def is_write_conflicted(dfg, edge, datanode=None, sdfg_schedule=None):
    """ Detects whether a write-conflict-resolving edge can be emitted without
        using atomics or critical sections. """

    if edge.data.wcr_nonatomic:
        return False

    # If it's an entire SDFG, it's probably write-conflicted
    if isinstance(dfg, SDFG):
        if datanode is None:
            return True
        in_edges = find_incoming_edges(datanode, dfg)
        if len(in_edges) != 1:
            return True
        if (isinstance(in_edges[0].src, nodes.ExitNode) and
                in_edges[0].src.map.schedule == dtypes.ScheduleType.Sequential):
            return False
        return True

    # Traverse memlet path to determine conflicts.
    # If no conflicts will occur, write without atomics
    # (e.g., if the array has been defined in a non-parallel schedule context)
    # TODO: This is not perfect (need to take indices into consideration)
    path = dfg.memlet_path(edge)
    for e in path:
        if (isinstance(e.dst, nodes.ExitNode)
                and e.dst.map.schedule != dtypes.ScheduleType.Sequential):
            return True
        # Should never happen (no such thing as write-conflicting reads)
        if (isinstance(e.src, nodes.EntryNode)
                and e.src.map.schedule != dtypes.ScheduleType.Sequential):
            return True

    # If SDFG schedule is not None (top-level) or not sequential
    if (sdfg_schedule is not None
            and sdfg_schedule != dtypes.ScheduleType.Sequential):
        return True

    return False


class LambdaToFunction(ast.NodeTransformer):
    def visit_Lambda(self, node: ast.Lambda):
        newbody = [ast.Return(value=node.body)]
        newnode = ast.FunctionDef(name="_anonymous",
                                  args=node.args,
                                  body=newbody,
                                  decorator_list=[])
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
        raise NotImplementedError("INVALID TYPE OF WCR: " +
                                  type(wcr_ast).__name__)


def unparse_cr(sdfg, wcr_ast):
    """ Outputs a C++ version of a conflict resolution lambda. """
    body_cpp, args = unparse_cr_split(sdfg, wcr_ast)

    # Construct a C++ lambda function out of a function
    return '[] (%s) { %s }' % (', '.join('const auto& %s' % a
                                         for a in args), body_cpp)


def unparse_tasklet(sdfg, state_id, dfg, node, function_stream, callsite_stream,
                    locals, ldepth, toplevel_schedule):

    if node.label is None or node.label == "":
        return ""

    state_dfg = sdfg.nodes()[state_id]

    # Not [], "" or None
    if not node.code:
        return ""

    # If raw C++ code, return the code directly
    if node.language != dtypes.Language.Python:
        # If this code runs on the host and is associated with a CUDA stream,
        # set the stream to a local variable.
        max_streams = int(
            Config.get("compiler", "cuda", "max_concurrent_streams"))
        if (max_streams >= 0 and not is_devicelevel_gpu(sdfg, state_dfg, node)
                and hasattr(node, "_cuda_stream")):
            callsite_stream.write(
                'int __dace_current_stream_id = %d;\ncudaStream_t __dace_current_stream = dace::cuda::__streams[__dace_current_stream_id];'
                % node._cuda_stream,
                sdfg,
                state_id,
                node,
            )

        if node.language != dtypes.Language.CPP:
            raise ValueError(
                "Only Python or C++ code supported in CPU codegen, got: {}".
                format(node.language))
        callsite_stream.write(
            type(node).__properties__["code"].to_string(node.code), sdfg,
            state_id, node)

        if hasattr(node, "_cuda_stream") and not is_devicelevel_gpu(
                sdfg, state_dfg, node):
            synchronize_streams(sdfg, state_dfg, state_id, node, node,
                                callsite_stream)
        return

    body = node.code.code

    # Map local names to memlets (for WCR detection)
    memlets = {}
    for edge in state_dfg.all_edges(node):
        u, uconn, v, vconn, memlet = edge
        if u == node:
            memlet_nc = not is_write_conflicted(
                dfg, edge, sdfg_schedule=toplevel_schedule)
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

    callsite_stream.write("// Tasklet code (%s)\n" % node.label, sdfg, state_id,
                          node)
    for stmt in body:
        stmt = copy.deepcopy(stmt)
        rk = StructInitializer(sdfg).visit(stmt)
        if isinstance(stmt, ast.Expr):
            rk = DaCeKeywordRemover(sdfg, memlets,
                                    sdfg.constants).visit_TopLevelExpr(stmt)
        else:
            rk = DaCeKeywordRemover(sdfg, memlets, sdfg.constants).visit(stmt)

        if rk is not None:
            # Unparse to C++ and add 'auto' declarations if locals not declared
            result = StringIO()
            cppunparse.CPPUnparser(rk, ldepth + 1, locals, result)
            callsite_stream.write(result.getvalue(), sdfg, state_id, node)


class DaCeKeywordRemover(ExtNodeTransformer):
    """ Removes memlets and other DaCe keywords from a Python AST, and
        converts array accesses to C++ methods that can be generated.

        Used for unparsing Python tasklets into C++ that uses the DaCe
        runtime.

        @note: Assumes that the DaCe syntax is correct (as verified by the
               Python frontend).
    """
    def __init__(self, sdfg, memlets, constants):
        self.sdfg = sdfg
        self.memlets = memlets
        self.constants = constants

    def visit_TopLevelExpr(self, node):
        # This is a DaCe shift, omit it
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.op, ast.LShift) or isinstance(
                    node.value.op, ast.RShift):
                return None
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        if not isinstance(node.target, ast.Subscript):
            return self.generic_visit(node)

        target = rname(node.target)
        if target not in self.memlets:
            return self.generic_visit(node)

        raise SyntaxError("Augmented assignments (e.g. +=) not allowed on " +
                          "array memlets")

    def visit_Assign(self, node):
        target = rname(node.targets[0])
        if target not in self.memlets:
            return self.generic_visit(node)

        memlet, nc, wcr, dtype = self.memlets[target]
        value = self.visit(node.value)

        if not isinstance(node.targets[0], ast.Subscript):
            # Dynamic accesses -> every access counts
            try:
                if memlet is not None and memlet.dynamic:
                    if wcr is not None:
                        newnode = ast.Name(id=write_and_resolve_expr(
                            self.sdfg, memlet, nc, '__' + target,
                            cppunparse.cppunparse(value, expr_semicolon=False)))
                    else:
                        newnode = ast.Name(id="__%s.write(%s);" % (
                            target,
                            cppunparse.cppunparse(value, expr_semicolon=False),
                        ))

                    return ast.copy_location(newnode, node)
            except TypeError:  # cannot determine truth value of Relational
                pass

            return self.generic_visit(node)

        slice = self.visit(node.targets[0].slice)
        if not isinstance(slice, ast.Index):
            raise NotImplementedError("Range subscripting not implemented")

        if isinstance(slice.value, ast.Tuple):
            subscript = unparse(slice)[1:-1]
        else:
            subscript = unparse(slice)

        if wcr is not None:
            newnode = ast.Name(id=write_and_resolve_expr(
                self.sdfg,
                memlet,
                nc,
                "__" + target,
                cppunparse.cppunparse(value, expr_semicolon=False),
                indices=subscript,
            ))
        else:
            newnode = ast.Name(id="__%s.write(%s, %s);" % (
                target,
                cppunparse.cppunparse(value, expr_semicolon=False),
                subscript,
            ))

        return ast.copy_location(newnode, node)

    # TODO: Remove!
    @staticmethod
    def ndslice_cpp(slice, dims, rowmajor=True):
        result = StringIO()

        if len(slice) == 0:  # Scalar
            return "0"

        for i, d in enumerate(slice):
            if isinstance(d, tuple):
                raise SyntaxError(
                    "CPU backend does not yet support ranges as inputs/outputs")

            result.write(sym2cpp(d))

            # If not last
            if i < len(slice) - 1:
                # We use the shape as-is since this function is intended for
                # constant arrays only
                strdims = [str(dim) for dim in dims[i + 1:]]
                result.write(
                    "*%s + " %
                    "*".join(strdims))  # Multiply by leading dimensions

        return result.getvalue()

    def visit_Subscript(self, node):
        target = rname(node)
        if target not in self.memlets and target not in self.constants:
            return self.generic_visit(node)

        slice = self.visit(node.slice)
        if not isinstance(slice, ast.Index):
            raise NotImplementedError("Range subscripting not implemented")

        if isinstance(slice.value, ast.Tuple):
            subscript = unparse(slice)[1:-1]
        else:
            subscript = unparse(slice)

        if target in self.constants:
            slice_str = DaCeKeywordRemover.ndslice_cpp(
                subscript.split(", "), self.constants[target].shape)
            newnode = ast.parse("%s[%s]" % (target, slice_str)).body[0].value
        else:
            newnode = ast.parse("__%s(%s)" % (target, subscript)).body[0].value
        return ast.copy_location(newnode, node)

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

    # Replace default modules (e.g., math) with dace::math::
    def visit_Attribute(self, node):
        attrname = rname(node)
        module_name = attrname[:attrname.rfind(".")]
        func_name = attrname[attrname.rfind(".") + 1:]
        if module_name in dtypes._ALLOWED_MODULES:
            cppmodname = dtypes._ALLOWED_MODULES[module_name]
            return ast.copy_location(
                ast.Name(id=(cppmodname + func_name), ctx=ast.Load), node)
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
        if isinstance(node.func,
                      ast.Name) and (node.func.id.startswith('__DACESTRUCT_')
                                     or node.func.id in self._structs):
            fields = ', '.join([
                '.%s = %s' % (rname(arg.arg), cppunparse.pyexpr2cpp(arg.value))
                for arg in sorted(node.keywords, key=lambda x: x.arg)
            ])

            tname = node.func.id
            if node.func.id.startswith('__DACESTRUCT_'):
                tname = node.func.id[len('__DACESTRUCT_'):]

            return ast.copy_location(
                ast.Name(id="%s { %s }" % (tname, fields), ctx=ast.Load), node)

        return self.generic_visit(node)


# TODO: This should be in the CUDA code generator. Add appropriate conditions to node dispatch predicate
def presynchronize_streams(sdfg, dfg, state_id, node, callsite_stream):
    state_dfg = sdfg.nodes()[state_id]
    if hasattr(node, "_cuda_stream") or is_devicelevel_gpu(
            sdfg, state_dfg, node):
        return
    for e in state_dfg.in_edges(node):
        if hasattr(e.src, "_cuda_stream"):
            cudastream = "dace::cuda::__streams[%d]" % e.src._cuda_stream
            callsite_stream.write(
                "cudaStreamSynchronize(%s);" % cudastream,
                sdfg,
                state_id,
                [e.src, e.dst],
            )


# TODO: This should be in the CUDA code generator. Add appropriate conditions to node dispatch predicate
def synchronize_streams(sdfg, dfg, state_id, node, scope_exit, callsite_stream):
    # Post-kernel stream synchronization (with host or other streams)
    max_streams = int(Config.get("compiler", "cuda", "max_concurrent_streams"))
    if max_streams >= 0:
        cudastream = "dace::cuda::__streams[%d]" % node._cuda_stream
        for edge in dfg.out_edges(scope_exit):
            # Synchronize end of kernel with output data (multiple kernels
            # lead to same data node)
            if (isinstance(edge.dst, nodes.AccessNode)
                    and edge.dst._cuda_stream != node._cuda_stream):
                callsite_stream.write(
                    """cudaEventRecord(dace::cuda::__events[{ev}], {src_stream});
cudaStreamWaitEvent(dace::cuda::__streams[{dst_stream}], dace::cuda::__events[{ev}], 0);"""
                    .format(
                        ev=edge._cuda_event
                        if hasattr(edge, "_cuda_event") else 0,
                        src_stream=cudastream,
                        dst_stream=edge.dst._cuda_stream,
                    ),
                    sdfg,
                    state_id,
                    [edge.src, edge.dst],
                )
                continue

            # We need the streams leading out of the output data
            for e in dfg.out_edges(edge.dst):
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
                        """cudaEventRecord(dace::cuda::__events[{ev}], {src_stream});
    cudaStreamWaitEvent(dace::cuda::__streams[{dst_stream}], dace::cuda::__events[{ev}], 0);"""
                        .format(
                            ev=e._cuda_event
                            if hasattr(e, "_cuda_event") else 0,
                            src_stream=cudastream,
                            dst_stream=e.dst._cuda_stream,
                        ),
                        sdfg,
                        state_id,
                        [e.src, e.dst],
                    )
                # Otherwise, no synchronization necessary
