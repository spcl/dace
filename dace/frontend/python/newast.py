# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from collections import OrderedDict
import copy
import itertools
import re
import sys
from typing import Any, Dict, List, Tuple, Union, Callable, Optional

import dace
from dace import data, dtypes, subsets, symbolic, sdfg as sd
from dace.config import Config
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import astutils
from dace.frontend.python.common import DaceSyntaxError, inverse_dict_lookup
from dace.frontend.python.astutils import ExtNodeVisitor, ExtNodeTransformer
from dace.frontend.python.astutils import rname
from dace.frontend.python import nested_call
from dace.frontend.python.memlet_parser import (DaceSyntaxError, parse_memlet,
                                                pyexpr_to_symbolic, ParseMemlet,
                                                inner_eval_ast, MemletExpr)
from dace.sdfg import nodes
from dace.sdfg.propagation import propagate_memlet
from dace.memlet import Memlet
from dace.properties import LambdaProperty, CodeBlock
from dace.sdfg import SDFG, SDFGState
from dace.symbolic import pystr_to_symbolic

import sympy

# register replacements in oprepo
import dace.frontend.python.replacements

# Type hints
Size = Union[int, dace.symbolic.symbol]
ShapeTuple = Tuple[Size]
ShapeList = List[Size]
Shape = Union[ShapeTuple, ShapeList]


def until(val, substr):
    """ Helper function that returns the substring of a string until a certain pattern. """
    if substr not in val:
        return val
    return val[:val.find(substr)]


augassign_ops = {
    'Add': '+',
    'Sub': '-',
    'Mult': '*',
    'Div': '/',
    'FloorDiv': '//',
    'Mod': '%',
    'Pow': '**',
    'LShift': '<<',
    'RShift': '>>',
    'BitOr': '|',
    'BitXor': '^',
    'BitAnd': '&'
}


class AddTransientMethods(object):
    """ A management singleton for methods that add transient data to SDFGs. """

    _methods = {}

    @staticmethod
    def get(datatype):
        """ Returns a method. """
        if datatype not in AddTransientMethods._methods:
            return None
        return AddTransientMethods._methods[datatype]


@dtypes.paramdec
def specifies_datatype(func: Callable[[Any, data.Data], Tuple[str, data.Data]],
                       datatype=None):
    AddTransientMethods._methods[datatype] = func
    return func


@specifies_datatype(datatype=data.Scalar)
def _method(sdfg: SDFG, sample_data: data.Scalar):
    name = sdfg.temp_data_name()
    new_data = sdfg.add_scalar(name, sample_data.dtype, transient=True)
    return name, new_data


@specifies_datatype(datatype=data.Array)
def _method(sdfg: SDFG, sample_data: data.Array):
    name, new_data = sdfg.add_temp_transient(sample_data.shape,
                                             sample_data.dtype)
    return name, new_data


@specifies_datatype(datatype=data.Stream)
def _method(sdfg: SDFG, sample_data: data.Stream):
    name = sdfg.temp_data_name()
    new_data = sdfg.add_stream(name,
                               sample_data.dtype,
                               buffer_size=sample_data.buffer_size,
                               shape=sample_data.shape,
                               transient=True)
    return name, new_data


def _add_transient_data(sdfg: SDFG, sample_data: data.Data):
    """ Adds to the sdfg transient data of the same dtype, shape and other
        parameters as sample_data. """
    try:
        func = AddTransientMethods._methods[type(sample_data)]
        return func(sdfg, sample_data)
    except KeyError:
        raise NotImplementedError


def parse_dace_program(f, argtypes, global_vars, modules, other_sdfgs,
                       constants):
    """ Parses a `@dace.program` function into a _ProgramNode object.
        :param f: A Python function to parse.
        :param argtypes: An dictionary of (name, type) for the given
                         function's arguments, which may pertain to data
                         nodes or symbols (scalars).
        :param global_vars: A dictionary of global variables in the closure
                            of `f`.
        :param modules: A dictionary from an imported module name to the
                        module itself.
        :param other_sdfgs: Other SDFG and DaceProgram objects in the context
                            of this function.
        :param constants: A dictionary from a name to a constant value.
        :return: Hierarchical tree of `astnodes._Node` objects, where the top
                 level node is an `astnodes._ProgramNode`.
        @rtype: SDFG
    """
    src_ast, src_file, src_line, src = astutils.function_to_ast(f)

    # Resolve symbols to their names
    symrepl = {
        k: v.name
        for k, v in global_vars.items() if isinstance(v, symbolic.symbol)
    }
    src_ast = astutils.ASTFindReplace(symrepl).visit(src_ast)

    # Resolve data structures
    src_ast = StructTransformer(global_vars).visit(src_ast)

    src_ast = ModuleResolver(modules).visit(src_ast)
    # Convert modules after resolution
    for mod, modval in modules.items():
        if mod == 'builtins':
            continue
        newmod = global_vars[mod]
        del global_vars[mod]
        global_vars[modval] = newmod

    # Resolve constants to their values (if they are not already defined in this scope)
    src_ast = GlobalResolver({
        k: v
        for k, v in global_vars.items()
        if dtypes.isconstant(v) and not k in argtypes and k != '_'
    }).visit(src_ast)

    pv = ProgramVisitor(name=f.__name__,
                        filename=src_file,
                        line_offset=src_line,
                        col_offset=0,
                        global_vars=global_vars,
                        constants=constants,
                        scope_arrays=argtypes,
                        scope_vars={},
                        other_sdfgs=other_sdfgs)

    sdfg, _, _ = pv.parse_program(src_ast.body[0])
    sdfg.set_sourcecode(src, 'python')

    return sdfg


class StructTransformer(ast.NodeTransformer):
    """ A Python AST transformer that replaces `Call`s to create structs with
        the custom StructInitializer AST node. """
    def __init__(self, gvars):
        super().__init__()
        self._structs = {
            k: v
            for k, v in gvars.items() if isinstance(v, dtypes.struct)
        }

    def visit_Call(self, node: ast.Call):
        # Struct initializer
        name = rname(node.func)
        if name not in self._structs:
            return self.generic_visit(node)

        # Parse name and fields
        struct = self._structs[name]
        name = struct.name
        fields = {rname(arg.arg): arg.value for arg in node.keywords}
        if tuple(sorted(fields.keys())) != tuple(sorted(struct.fields.keys())):
            raise SyntaxError('Mismatch in fields in struct definition')

        # Create custom node
        #new_node = astutils.StructInitializer(name, fields)
        #return ast.copy_location(new_node, node)

        node.func = ast.copy_location(
            ast.Name(id='__DACESTRUCT_' + name, ctx=ast.Load()), node.func)

        return node


# Replaces instances of modules Y imported with "import X as Y" by X
class ModuleResolver(ast.NodeTransformer):
    def __init__(self, modules: Dict[str, str]):
        self.modules = modules

    def visit_Attribute(self, node):
        # Traverse AST until reaching the top-level value (could be a name
        # or a function)
        cnode = node
        while isinstance(cnode.value, ast.Attribute):
            cnode = cnode.value

        if (isinstance(cnode.value, ast.Name)
                and cnode.value.id in self.modules):
            cnode.value.id = self.modules[cnode.value.id]

        return self.generic_visit(node)


def _targets(node: ast.Assign):
    for target in node.targets:
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                yield elt
        else:
            yield target


# AST node types that are disallowed in DaCe programs
_DISALLOWED_STMTS = [
    'Global', 'Delete', 'Import', 'ImportFrom', 'Assert', 'Pass', 'Exec',
    'Print', 'Nonlocal', 'Yield', 'YieldFrom', 'Raise', 'Try', 'TryExcept',
    'TryFinally', 'ExceptHandler', 'Starred', 'Ellipsis', 'ClassDef',
    'AsyncFor', 'Await', 'Bytes', 'Set', 'Dict', 'ListComp', 'GeneratorExp',
    'SetComp', 'DictComp', 'comprehension'
]

TaskletType = Union[ast.FunctionDef, ast.With, ast.For]


def _disallow_stmt(visitor, node):
    raise DaceSyntaxError(visitor, node,
                          'Keyword "%s" disallowed' % (type(node).__name__))


###############################################################
# Parsing functions
###############################################################


def _subset_has_indirection(subset):
    for dim in subset:
        if not isinstance(dim, tuple):
            dim = [dim]
        for r in dim:
            if symbolic.contains_sympy_functions(r):
                return True
    return False


def add_indirection_subgraph(sdfg: SDFG,
                             graph: SDFGState,
                             src: nodes.Node,
                             dst: nodes.Node,
                             memlet: Memlet,
                             local_name: str,
                             PVisitor: 'ProgramVisitor',
                             output: bool = False):
    """ Replaces the specified edge in the specified graph with a subgraph that
        implements indirection without nested memlet subsets. """

    array = sdfg.arrays[memlet.data]
    indirect_inputs = set()
    indirect_outputs = set()

    # Scheme for multi-array indirection:
    # 1. look for all arrays and accesses, create set of arrays+indices
    #    from which the index memlets will be constructed from
    # 2. each separate array creates a memlet, of which num_accesses = len(set)
    # 3. one indirection tasklet receives them all + original array and
    #    produces the right output index/range memlet
    #########################
    # Step 1
    accesses = OrderedDict()
    newsubset = copy.deepcopy(memlet.subset)
    for dimidx, dim in enumerate(memlet.subset):
        # Range/Index disambiguation
        direct_assignment = False
        if not isinstance(dim, tuple):
            dim = [dim]
            direct_assignment = True
        elif dim[0] == dim[1]:
            dim = [dim[0]]
            direct_assignment = True

        for i, r in enumerate(dim):
            for expr in symbolic.swalk(r, enter_functions=True):
                if symbolic.is_sympy_userfunction(expr):
                    fname = expr.func.__name__
                    if fname not in accesses:
                        accesses[fname] = []

                    # Replace function with symbol (memlet local name to-be)
                    if expr.args in accesses[fname]:
                        aindex = accesses[fname].index(expr.args)
                        toreplace = 'index_' + fname + '_' + str(aindex)
                    else:
                        accesses[fname].append(expr.args)
                        toreplace = 'index_' + fname + '_' + str(
                            len(accesses[fname]) - 1)

                    if direct_assignment:
                        # newsubset[dimidx] = newsubset[dimidx].subs(expr, toreplace)
                        newsubset[dimidx] = r.subs(expr, toreplace)
                    else:
                        rng = list(newsubset[dimidx])
                        rng[i] = rng[i].subs(expr, toreplace)
                        newsubset[dimidx] = tuple(rng)
                        # newsubset[dimidx][i] = r.subs(expr, toreplace)
    #########################
    # Step 2
    if output:
        ind_inputs = {'lookup': None}
        ind_outputs = {('__ind_' + local_name): None}
    else:
        ind_inputs = {('__ind_' + local_name): None}
        ind_outputs = {'lookup': None}
    # Add accesses to inputs
    for arrname, arr_accesses in accesses.items():
        for i in range(len(arr_accesses)):
            ind_inputs['index_%s_%d' % (arrname, i)] = None

    tasklet = nodes.Tasklet("Indirection", ind_inputs, ind_outputs)

    # Create map if indirected subset is a range
    ind_entry = None
    ind_exit = None
    inp_base_path = [tasklet]
    out_base_path = [tasklet]
    if (isinstance(memlet.subset, subsets.Range)
            and memlet.subset.num_elements() != 1):
        rng = copy.deepcopy(memlet.subset)
        nonsqz_dims = rng.squeeze()
        ind_entry, ind_exit = graph.add_map('indirection', {
            '__i%d' % i: '%s:%s+1:%s' % (s, e, t)
            for i, (s, e, t) in enumerate(rng)
        },
                                            debuginfo=PVisitor.current_lineinfo)
        inp_base_path.insert(0, ind_entry)
        out_base_path.append(ind_exit)

    input_index_memlets = []
    for arrname, arr_accesses in accesses.items():
        arr_name = arrname
        for i, access in enumerate(arr_accesses):
            if isinstance(access, (list, tuple)):
                access = access[0]
            if isinstance(access, sympy.Tuple):
                access = list(access)
            if not isinstance(access, (list, tuple)):
                access = [access]
            conn = None
            if PVisitor.nested:
                arr_rng = dace.subsets.Range([(a, a, 1) for a in access])
                if output:
                    arrname = PVisitor._add_write_access(arr_name,
                                                         arr_rng,
                                                         target=None)
                else:
                    arrname = PVisitor._add_read_access(arr_name,
                                                        arr_rng,
                                                        target=None)
                access = [0] * len(access)
                conn = 'index_%s_%d' % (arr_name, i)
            arr = sdfg.arrays[arrname]
            # Memlet to load the indirection index
            indexMemlet = Memlet.simple(arrname, subsets.Indices(access))
            input_index_memlets.append(indexMemlet)
            read_node = graph.add_read(arrname,
                                       debuginfo=PVisitor.current_lineinfo)
            if PVisitor.nested or not isinstance(src, nodes.EntryNode):
                path = [read_node] + inp_base_path
            else:
                if output:
                    # TODO: This only works for Maps. Perhaps it should be
                    # generalized for other pairs of entry/exit nodes.
                    entry = None
                    if isinstance(dst, nodes.MapExit):
                        for node in graph.nodes():
                            if (isinstance(node, nodes.MapEntry)
                                    and node.map is dst.map):
                                entry = node
                                break
                    else:
                        raise NotImplementedError
                else:
                    entry = src
                path = [read_node, entry] + inp_base_path
            graph.add_memlet_path(*path,
                                  dst_conn="index_%s_%d" % (arr_name, i),
                                  memlet=indexMemlet)

    #########################
    # Step 3
    # Create new tasklet that will perform the indirection
    if output:
        code = "{arr}[{index}] = lookup"
    else:
        code = "lookup = {arr}[{index}]"

    newsubset = [r[0] if isinstance(r, tuple) else r for r in newsubset]
    if ind_entry:  # Amend newsubset when a range is indirected
        for i, idx in enumerate(nonsqz_dims):
            newsubset[idx] = '__i%d' % i

    tasklet.code = CodeBlock(
        code.format(arr='__ind_' + local_name,
                    index=', '.join([symbolic.symstr(s) for s in newsubset])))

    # Create transient variable to trigger the indirect load
    tmp_name = '__' + local_name + '_value'
    start_src = None
    end_dst = None
    if memlet.num_accesses == 1 and dst is not None:
        _, storage = sdfg.add_scalar(tmp_name, array.dtype, transient=True)
    else:
        rng = copy.deepcopy(memlet.subset)
        if isinstance(rng, subsets.Range):
            rng.squeeze()
        _, storage = sdfg.add_array(tmp_name,
                                    rng.bounding_box_size(),
                                    array.dtype,
                                    storage=dtypes.StorageType.Default,
                                    transient=True)
        # Force creation of transients for range indirection
        if output:
            if src:
                start_src = src
                src = None
        else:
            if dst:
                end_dst = dst
                dst = None

    # Create transients when implementing indirection
    # through slicing or when indirecting a range.
    if src is None:
        if start_src:
            src = graph.add_access(tmp_name,
                                   debuginfo=PVisitor.current_lineinfo)
        else:
            src = graph.add_read(tmp_name, debuginfo=PVisitor.current_lineinfo)
    elif dst is None:
        if end_dst:
            dst = graph.add_access(tmp_name,
                                   debuginfo=PVisitor.current_lineinfo)
        else:
            dst = graph.add_write(tmp_name, debuginfo=PVisitor.current_lineinfo)

    tmp_shape = storage.shape
    indirectRange = subsets.Range([(0, s - 1, 1) for s in tmp_shape])
    if ind_entry:  # Amend indirected range
        indirectRange = ','.join([ind for ind in ind_entry.map.params])

    # Create memlet that depends on the full array that we look up in
    fullRange = subsets.Range([(0, s - 1, 1) for s in array.shape])
    fullMemlet = Memlet.simple(memlet.data,
                               fullRange,
                               num_accesses=memlet.num_accesses)
    fullMemlet.dynamic = memlet.dynamic

    if output:
        if isinstance(dst, nodes.ExitNode):
            full_write_node = graph.add_write(
                memlet.data, debuginfo=PVisitor.current_lineinfo)
            path = out_base_path + [dst, full_write_node]
        elif isinstance(dst, nodes.AccessNode):
            path = out_base_path + [dst]
        else:
            raise Exception("Src node type for indirection is invalid.")
        graph.add_memlet_path(*path,
                              src_conn='__ind_' + local_name,
                              memlet=fullMemlet)
    else:
        if isinstance(src, nodes.EntryNode):
            full_read_node = graph.add_read(memlet.data,
                                            debuginfo=PVisitor.current_lineinfo)
            path = [full_read_node, src] + inp_base_path
        elif isinstance(src, nodes.AccessNode):
            path = [src] + inp_base_path
        else:
            raise Exception("Src node type for indirection is invalid.")
        graph.add_memlet_path(*path,
                              dst_conn='__ind_' + local_name,
                              memlet=fullMemlet)

    # Memlet to store the final value into the transient, and to load it into
    # the tasklet that needs it
    # indirectMemlet = Memlet.simple('__' + local_name + '_value',
    #                         indirectRange, num_accesses=memlet.num_accesses)
    # graph.add_edge(tasklet, 'lookup', dataNode, None, indirectMemlet)

    valueMemlet = Memlet.simple(tmp_name, indirectRange, num_accesses=1)
    if output:
        path = [src] + inp_base_path
        if isinstance(src, nodes.AccessNode):
            src_conn = None
        else:
            src_conn = local_name
        graph.add_memlet_path(*path,
                              src_conn=src_conn,
                              dst_conn='lookup',
                              memlet=valueMemlet)
        # Connect original source to the indirected-range-transient
        if start_src:
            if isinstance(start_src, nodes.AccessNode):
                src_conn = None
            else:
                src_conn = local_name
            graph.add_edge(start_src, src_conn, src, None,
                           Memlet.from_array(tmp_name, storage))
    else:
        path = out_base_path + [dst]
        if isinstance(dst, nodes.AccessNode):
            dst_conn = None
        else:
            dst_conn = local_name
        graph.add_memlet_path(*path,
                              src_conn='lookup',
                              dst_conn=dst_conn,
                              memlet=valueMemlet)
        # Connect original destination to the indirected-range-transient
        if end_dst:
            if isinstance(end_dst, nodes.AccessNode):
                dst_conn = None
            else:
                dst_conn = local_name
            graph.add_edge(dst, None, end_dst, dst_conn,
                           Memlet.from_array(tmp_name, storage))

    return tmp_name


class GlobalResolver(ast.NodeTransformer):
    """ Resolves global constants and lambda expressions if not
        already defined in the given scope. """
    def __init__(self, globals: Dict[str, Any]):
        self.globals = globals
        self.current_scope = set()

    def generic_visit(self, node: ast.AST):
        if hasattr(node, 'body') or hasattr(node, 'orelse'):
            oldscope = self.current_scope
            self.current_scope = set()
            self.current_scope.update(oldscope)
            result = super().generic_visit(node)
            self.current_scope = oldscope
            return result
        else:
            return super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, (ast.Store, ast.AugStore)):
            self.current_scope.add(node.id)
        else:
            if node.id in self.current_scope:
                return node
            if node.id in self.globals:
                # Compatibility check since Python changed their AST nodes
                if sys.version_info >= (3, 8):
                    newnode = ast.Constant(value=self.globals[node.id], kind='')
                else:
                    newnode = ast.Num(n=self.globals[node.id])
                return ast.copy_location(newnode, node)
        return node


class TaskletTransformer(ExtNodeTransformer):
    """ A visitor that traverses a data-centric tasklet, removes memlet
        annotations and returns input and output memlets.
    """
    def __init__(self,
                 defined,
                 sdfg: SDFG,
                 state: SDFGState,
                 filename: str,
                 lang=dtypes.Language.Python,
                 location: dict = {},
                 nested: bool = False,
                 scope_arrays: Dict[str, data.Data] = dict(),
                 scope_vars: Dict[str, str] = dict(),
                 variables: Dict[str, str] = dict(),
                 accesses: Dict[Tuple[str, dace.subsets.Subset, str],
                                str] = dict()):
        """ Creates an AST parser for tasklets.
            :param sdfg: The SDFG to add the tasklet in (used for defined arrays and symbols).
            :param state: The SDFG state to add the tasklet to.
        """
        self.sdfg = sdfg
        self.state = state
        self.defined = defined

        # For syntax errors
        self.filename = filename

        # Connectors generated from memlets
        self.inputs = {}  # type: Dict[str, Memlet]
        self.outputs = {}  # type: Dict[str, Memlet]

        self.extcode = None
        self.lang = lang
        self.location = location

        self.nested = nested
        self.scope_arrays = scope_arrays
        self.scope_vars = scope_vars
        self.variables = variables
        self.accesses = accesses

        self.sdfg_inputs = {}
        self.sdfg_outputs = {}

        # Disallow keywords
        for stmt in _DISALLOWED_STMTS:
            setattr(self, 'visit_' + stmt, lambda n: _disallow_stmt(self, n))

    def parse_tasklet(self,
                      tasklet_ast: TaskletType,
                      name: Optional[str] = None):
        """ Parses the AST of a tasklet and returns the tasklet node, as well as input and output memlets.
            :param tasklet_ast: The Tasklet's Python AST to parse.
            :param name: Optional name to use as prefix for tasklet.
            :return: 3-tuple of (Tasklet node, input memlets, output memlets).
            @rtype: Tuple[Tasklet, Dict[str, Memlet], Dict[str, Memlet]]
        """
        # Should return a tasklet object (with connectors)
        self.visit(tasklet_ast)

        # Location identifier
        locinfo = dtypes.DebugInfo(tasklet_ast.lineno, tasklet_ast.col_offset,
                                   tasklet_ast.body[-1].lineno,
                                   tasklet_ast.body[-1].col_offset,
                                   self.filename)

        # Determine tasklet name (either declared as a function or use line #)
        if name is not None:
            name += '_' + str(tasklet_ast.lineno)
        else:
            name = getattr(tasklet_ast, 'name',
                           'tasklet_%d' % tasklet_ast.lineno)

        t = self.state.add_tasklet(name,
                                   set(self.inputs.keys()),
                                   set(self.outputs.keys()),
                                   self.extcode or tasklet_ast.body,
                                   language=self.lang,
                                   location=self.location,
                                   debuginfo=locinfo)

        return t, self.inputs, self.outputs, self.accesses

    def _add_access(
            self,
            name: str,
            rng: subsets.Range,
            access_type: str,  # 'r' or 'w'
            target: Union[ast.Name, ast.Subscript],
            new_name: str = None,
            arr_type: data.Data = None) -> str:
        if access_type not in ('r', 'w'):
            raise ValueError("Access type {} is invalid".format(access_type))
        if new_name:
            var_name = new_name
        elif target:
            var_name = "__tmp_{l}_{c}".format(l=target.lineno,
                                              c=target.col_offset)
        else:
            var_name = self.sdfg.temp_data_name()

        parent_name = self.scope_vars[name]
        parent_array = self.scope_arrays[parent_name]
        if _subset_has_indirection(rng):
            squeezed_rng = list(range(len(rng)))
            shape = parent_array.shape
            strides = [parent_array.strides[d] for d in squeezed_rng]
        else:
            squeezed_rng = copy.deepcopy(rng)
            non_squeezed = squeezed_rng.squeeze()
            shape = squeezed_rng.size()
            if non_squeezed:
                strides = [parent_array.strides[d] for d in non_squeezed]
            else:
                strides = [1]
        dtype = parent_array.dtype

        if arr_type is None:
            arr_type = type(parent_array)
        if arr_type == data.Scalar:
            self.sdfg.add_scalar(var_name, dtype)
        elif arr_type == data.Array:
            self.sdfg.add_array(var_name, shape, dtype, strides=strides)
        elif arr_type == data.Stream:
            self.sdfg.add_stream(var_name, dtype)
        else:
            raise NotImplementedError(
                "Data type {} is not implemented".format(arr_type))

        self.accesses[(name, rng, access_type)] = (var_name, squeezed_rng)

        inner_indices = set()
        for n, r in reversed(list(enumerate(squeezed_rng))):
            if r == rng[n]:
                inner_indices.add(n)

        if access_type == 'r':
            if _subset_has_indirection(rng):
                self.sdfg_inputs[var_name] = (dace.Memlet.from_array(
                    parent_name, parent_array), inner_indices)
            else:
                self.sdfg_inputs[var_name] = (dace.Memlet.simple(
                    parent_name, rng), inner_indices)
        else:
            if _subset_has_indirection(rng):
                self.sdfg_outputs[var_name] = (dace.Memlet.from_array(
                    parent_name, parent_array), inner_indices)
            else:
                self.sdfg_outputs[var_name] = (dace.Memlet.simple(
                    parent_name, rng), inner_indices)

        return (var_name, squeezed_rng)

    def _add_read_access(self,
                         name: str,
                         rng: subsets.Range,
                         target: Union[ast.Name, ast.Subscript],
                         new_name: str = None,
                         arr_type: data.Data = None):

        if (name, rng, 'w') in self.accesses:
            return self.accesses[(name, rng, 'w')]
        elif (name, rng, 'r') in self.accesses:
            return self.accesses[(name, rng, 'r')]
        elif name in self.variables:
            return (self.variables[name], None)
        elif name in self.scope_vars:
            return self._add_access(name, rng, 'r', target, new_name, arr_type)
        else:
            raise NotImplementedError

    def _add_write_access(self,
                          name: str,
                          rng: subsets.Range,
                          target: Union[ast.Name, ast.Subscript],
                          new_name: str = None,
                          arr_type: data.Data = None):

        if (name, rng, 'w') in self.accesses:
            return self.accesses[(name, rng, 'w')]
        elif name in self.variables:
            return (self.variables[name], None)
        elif (name, rng, 'r') in self.accesses or name in self.scope_vars:
            return self._add_access(name, rng, 'w', target, new_name, arr_type)
        else:
            raise NotImplementedError

    def _get_range(self, node: Union[ast.Name, ast.Subscript, ast.Call],
                   name: str):
        if isinstance(node, ast.Name):
            actual_node = copy.deepcopy(node)
            actual_node.id = name
            rng = dace.subsets.Range(
                astutils.subscript_to_slice(actual_node, {
                    **self.sdfg.arrays,
                    **self.scope_arrays
                })[1])
        elif isinstance(node, ast.Subscript):
            actual_node = copy.deepcopy(node)
            if isinstance(actual_node.value, ast.Call):
                actual_node.value.func.id = name
            else:
                actual_node.value.id = name
            rng = dace.subsets.Range(
                astutils.subscript_to_slice(actual_node, {
                    **self.sdfg.arrays,
                    **self.scope_arrays
                })[1])
        elif isinstance(node, ast.Call):
            rng = dace.subsets.Range.from_array({
                **self.sdfg.arrays,
                **self.scope_arrays
            }[name])
        else:
            raise NotImplementedError

        return rng

    def _update_names(self,
                      node: Union[ast.Name, ast.Subscript, ast.Call],
                      name: str,
                      name_subscript: bool = False):
        if isinstance(node, ast.Name):
            node.id = name
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Call):
                node = node.value
                node.func.id = name
            elif name_subscript:
                node = node.value
                node.id = name
            else:
                node.value.id = name
        elif isinstance(node, ast.Call):
            node.func.id = name
        else:
            raise NotImplementedError

        return node

    def visit_TopLevelExpr(self, node):
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.op, (ast.LShift, ast.RShift)):
                variables = {**self.variables, **self.scope_vars}
                target = node.value.right
                name = rname(target)
                name_sub = False
                if isinstance(node.value.op, ast.LShift):
                    if self.nested:
                        real_name = variables[name]
                        rng = self._get_range(target, real_name)
                        name, squeezed_rng = self._add_read_access(
                            name, rng, target)
                        if squeezed_rng is not None:
                            name_sub = True
                    else:
                        if name in variables:
                            name = variables[name]
                    node.value.right = self._update_names(
                        node.value.right, name, name_subscript=name_sub)
                    connector, memlet = parse_memlet(self, node.value.right,
                                                     node.value.left,
                                                     self.sdfg.arrays)
                    if self.nested and _subset_has_indirection(rng):
                        memlet = dace.Memlet.simple(memlet.data, rng)
                    if connector in self.inputs or connector in self.outputs:
                        raise DaceSyntaxError(
                            self, node,
                            'Local variable is already a tasklet input or output'
                        )
                    self.inputs[connector] = memlet
                    return None  # Remove from final tasklet code
                elif isinstance(node.value.op, ast.RShift):
                    if self.nested:
                        real_name = variables[name]
                        rng = self._get_range(target, real_name)
                        name, squeezed_rng = self._add_write_access(
                            name, rng, target)
                        if squeezed_rng is not None:
                            name_sub = True
                    else:
                        if name in variables:
                            name = variables[name]
                    node.value.right = self._update_names(
                        node.value.right, name, name_subscript=name_sub)
                    connector, memlet = parse_memlet(self, node.value.left,
                                                     node.value.right,
                                                     self.sdfg.arrays)
                    if self.nested and _subset_has_indirection(rng):
                        memlet = dace.Memlet.simple(memlet.data, rng)
                    if self.nested and name in self.sdfg_outputs:
                        out_memlet = self.sdfg_outputs[name][0]
                        out_memlet.volume = memlet.volume
                        out_memlet.dynamic = memlet.dynamic
                        out_memlet.wcr = memlet.wcr
                        out_memlet.wcr_nonatomic = memlet.wcr_nonatomic
                    if connector in self.inputs or connector in self.outputs:
                        raise DaceSyntaxError(
                            self, node,
                            'Local variable is already a tasklet input or output'
                        )
                    self.outputs[connector] = memlet
                    return None  # Remove from final tasklet code
        elif isinstance(node.value, ast.Str):
            return self.visit_TopLevelStr(node.value)

        return self.generic_visit(node)

    # Detect external tasklet code
    def visit_TopLevelStr(self, node: ast.Str):
        if self.extcode != None:
            raise DaceSyntaxError(
                self, node,
                'Cannot provide more than one intrinsic implementation ' +
                'for tasklet')
        self.extcode = node.s
        # TODO(later): Syntax for other languages?
        self.lang = dtypes.Language.CPP

        return node

    def visit_Name(self, node: ast.Name):
        # If accessing a symbol, add it to the SDFG symbol list
        if (isinstance(node.ctx, ast.Load) and node.id in self.defined
                and isinstance(self.defined[node.id], symbolic.symbol)):
            if node.id not in self.sdfg.symbols:
                self.sdfg.add_symbol(node.id, self.defined[node.id].dtype)
        return self.generic_visit(node)


class ProgramVisitor(ExtNodeVisitor):
    """ A visitor that traverses a data-centric Python program AST and
        constructs an SDFG.
    """
    def __init__(
        self,
        name: str,
        filename: str,
        line_offset: int,
        col_offset: int,
        global_vars: Dict[str, Any],
        constants: Dict[str, Any],
        scope_arrays: Dict[str, data.Data],
        scope_vars: Dict[str, str],
        other_sdfgs: Dict[str, SDFG],  # Dict[str, Union[SDFG, DaceProgram]]
        nested: bool = False,
        tmp_idx: int = 0):
        """ ProgramVisitor init method

        Arguments:
            name {str} -- Name of DaCe program
            filename {str} -- Name of file containing DaCe program
            line_offset {int} -- Line where DaCe program is called
            col_offset {int} -- Column where DaCe program is called
            global_vars {Dict[str, Any]} -- Global variables
            constants {Dict[str, Any]} -- Constant values
            scope_arrays {Dict[str, data.Data]} -- Scope arrays
            scope_vars {Dict[str, str]} -- Scope variables
            other_sdfgs {Dict[str, Union[SDFG, DaceProgram]]} -- Other SDFGs

        Keyword Arguments:
            nested {bool} -- True, if SDFG is nested (default: {False})
            tmp_idx {int} -- First idx for tmp transient names (default: {0})
        """

        self.filename = filename
        if nested:
            self.name = "{n}_{l}_{c}".format(n=name,
                                             l=line_offset,
                                             c=col_offset)
        else:
            self.name = name

        self.globals = global_vars
        self.other_sdfgs = other_sdfgs
        self.nested = nested

        # Keeps track of scope arrays, numbers, variables and accesses
        self.scope_arrays = OrderedDict()
        self.scope_arrays.update(scope_arrays)
        self.scope_vars = {k: k for k in scope_arrays.keys()}
        self.scope_vars.update(scope_vars)
        self.numbers = dict()  # Dict[str, str]
        self.variables = dict()  # Dict[str, str]
        self.accesses = dict()

        # Entry point to the program
        # self.program = None
        self.sdfg = SDFG(self.name)
        if not self.nested:
            self.sdfg.arrays.update(scope_arrays)
            for arr in self.sdfg.arrays.values():
                for sym in arr.free_symbols:
                    if sym.name not in self.sdfg.symbols:
                        self.sdfg.add_symbol(sym.name, sym.dtype)
        self.sdfg._temp_transients = tmp_idx
        self.last_state = self.sdfg.add_state('init', is_start_state=True)
        self.inputs = {}
        self.outputs = {}
        self.current_lineinfo = dtypes.DebugInfo(line_offset, col_offset,
                                                 line_offset, col_offset,
                                                 filename)

        # Add constants
        for cstname, cstval in constants.items():
            self.sdfg.add_constant(cstname, cstval)

        # Add symbols
        for arr in scope_arrays.values():
            self.scope_vars.update(
                {str(k): self.globals[str(k)]
                 for k in arr.free_symbols})

        # Disallow keywords
        for stmt in _DISALLOWED_STMTS:
            setattr(self, 'visit_' + stmt, lambda n: _disallow_stmt(self, n))

    def visit(self, node: ast.AST):
        """Visit a node."""
        self.current_lineinfo = dtypes.DebugInfo(node.lineno, node.col_offset,
                                                 node.lineno, node.col_offset,
                                                 self.filename)
        return super().visit(node)

    def parse_program(self, program: ast.FunctionDef, is_tasklet: bool = False):
        """ Parses a DaCe program or tasklet

        Arguments:
            program {ast.FunctionDef} -- DaCe program or tasklet

        Keyword Arguments:
            is_tasklet {bool} -- True, if program is tasklet (default: {False})

        Returns:
            Tuple[SDFG, Dict, Dict] -- Parsed SDFG, its inputs and outputs
        """

        if is_tasklet:
            program.decorator_list = []
            self.visit_FunctionDef(program)
        else:
            for stmt in program.body:
                self.visit_TopLevel(stmt)
        if len(self.sdfg.nodes()) == 0:
            self.sdfg.add_state("EmptyState")

        # Handle return values
        # Assignments to return values become __return* arrays
        for vname, arrname in self.variables.items():
            if vname.startswith('__return'):
                self.sdfg.replace(arrname, vname)

        # Return values become non-transient (accessible by the outside)
        for arrname, arr in self.sdfg.arrays.items():
            if arrname.startswith('__return'):
                arr.transient = False
                self.outputs[arrname] = Memlet.from_array(arrname, arr)
        ####

        # Try to replace transients with their python-assigned names
        for pyname, arrname in self.variables.items():
            if arrname in self.sdfg.arrays:
                if self.sdfg.arrays[arrname].transient:
                    if (pyname and dtypes.validate_name(pyname)
                            and pyname not in self.sdfg.arrays):
                        self.sdfg.replace(arrname, pyname)

        return self.sdfg, self.inputs, self.outputs

    @property
    def defined(self):
        # Check parent SDFG arrays first
        # result = {
        #     k: self.parent_arrays[v]
        #     for k, v in self.variables.items() if v in self.parent_arrays
        # }
        result = {}
        result.update({
            k: v
            for k, v in self.globals.items() if isinstance(v, symbolic.symbol)
        })
        result.update({
            k: self.sdfg.arrays[v]
            for k, v in self.scope_vars.items() if v in self.sdfg.arrays
        })
        result.update({
            k: self.scope_arrays[v]
            for k, v in self.scope_vars.items() if v in self.scope_arrays
        })
        result.update({
            k: self.sdfg.arrays[v]
            for k, v in self.variables.items() if v in self.sdfg.arrays
        })
        result.update({
            v: self.sdfg.arrays[v]
            for _, v in self.variables.items() if v in self.sdfg.arrays
        })
        # TODO: Is there a case of a variable-symbol?
        result.update({
            k: self.sdfg.symbols[v]
            for k, v in self.variables.items() if v in self.sdfg.symbols
        })

        return result

    def _add_state(self, label=None):
        state = self.sdfg.add_state(label)
        if self.last_state is not None:
            self.sdfg.add_edge(self.last_state, state, dace.InterstateEdge())
        self.last_state = state
        return state

    def _parse_arg(self, arg: Any, as_list=True):
        """ Parse possible values to slices or objects that can be used in
            the SDFG API. """
        if isinstance(arg, ast.Subscript) and rname(arg) == '_':
            # result = [
            #     ':'.join([str(d) for d in dim]) for dim in
            #     astutils.subscript_to_slice(arg, self.sdfg.arrays)[1]
            # ]
            rng = dace.subsets.Range(
                astutils.subscript_to_slice(arg, self.sdfg.arrays)[1])
            result = rng.string_list()
            if as_list is False and len(result) == 1:
                return result[0]
            return result

        return arg

    def _decorator_or_annotation_params(
            self, node: ast.FunctionDef) -> List[Tuple[str, Any]]:
        """ Returns a list of parameters, either from the function parameters
            and decorator arguments or parameters and their annotations (type
            hints).
            :param node: The given function definition node.
            :return: A list of 2-tuples (name, value).
        """
        # If the arguments are defined in the decorator
        dec = node.decorator_list[0]
        if 'args' in dir(dec) and len(dec.args) > 0:
            # If it's one argument of the form of ND range, e.g., "_[0:M, 0:N]"
            parg0 = self._parse_arg(dec.args[0])
            if isinstance(parg0, list):
                args = parg0
            else:
                args = [self._parse_arg(arg) for arg in dec.args]
        else:  # Otherwise, use annotations
            args = [
                self._parse_arg(arg.annotation, as_list=False)
                for arg in node.args.args
            ]

        result = [(rname(arg), argval)
                  for arg, argval in zip(node.args.args, args)]

        # Ensure all arguments are annotated
        if len(result) != len(node.args.args):
            raise DaceSyntaxError(
                self, node,
                'All arguments in primitive %s must be annotated' % node.name)
        return result

    def _parse_subprogram(self,
                          name,
                          node,
                          is_tasklet=False,
                          extra_symbols=None):
        extra_symbols = extra_symbols or {}
        local_vars = {}
        local_vars.update(self.globals)
        local_vars.update(extra_symbols)
        pv = ProgramVisitor(name=name,
                            filename=self.filename,
                            line_offset=node.lineno,
                            col_offset=node.col_offset,
                            global_vars=local_vars,
                            constants=self.sdfg.constants,
                            scope_arrays={
                                **self.scope_arrays,
                                **self.sdfg.arrays
                            },
                            scope_vars={
                                **self.scope_vars,
                                **self.variables,
                            },
                            other_sdfgs=self.other_sdfgs,
                            nested=True,
                            tmp_idx=self.sdfg._temp_transients + 1)

        return pv.parse_program(node, is_tasklet)

    def _symbols_from_params(
            self, params: List[Tuple[str, Union[str, dtypes.typeclass]]],
            memlet_inputs: Dict[str, Memlet]) -> Dict[str, symbolic.symbol]:
        """
        Returns a mapping between symbol names to their type, as a symbol 
        object to maintain compatibility with global symbols. Used to maintain 
        typed symbols in SDFG scopes (e.g., map, consume).
        """
        from dace.codegen.tools.type_inference import infer_expr_type
        result = {}

        # Add map inputs first
        dyn_inputs = {}
        for name, val in memlet_inputs.items():
            dyn_inputs[name] = symbolic.symbol(
                name, self.scope_arrays[val.data].dtype)
        result.update(dyn_inputs)

        for name, val in params:
            if isinstance(val, dtypes.typeclass):
                result[name] = symbolic.symbol(name, dtype=val)
            else:
                values = str(val).split(':')
                if len(values) == 2:
                    result[name] = symbolic.symbol(
                        name,
                        dtypes.result_type_of(
                            infer_expr_type(values[0], {
                                **self.globals,
                                **dyn_inputs
                            }),
                            infer_expr_type(values[1], {
                                **self.globals,
                                **dyn_inputs
                            })))
                else:
                    result[name] = symbolic.symbol(
                        name,
                        infer_expr_type(values[0], {
                            **self.globals,
                            **dyn_inputs
                        }))

        return result

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Supported decorated function types: map, mapscope, consume,
        # consumescope, tasklet, program

        if len(node.decorator_list) > 1:
            raise DaceSyntaxError(
                self, node,
                'Exactly one DaCe decorator is allowed on a function')
        if len(node.decorator_list) == 0:
            dec = 'dace.tasklet'
        else:
            dec = rname(node.decorator_list[0])

        # Create a new state for the statement
        state = self._add_state("s{l}_{c}".format(l=node.lineno,
                                                  c=node.col_offset))

        # Define internal node for reconnection
        internal_node = None

        # Select primitive according to function type
        if dec == 'dace.tasklet':  # Tasklet
            internal_node, inputs, outputs, sdfg_inp, sdfg_out = self._parse_tasklet(
                state, node)

            # Add memlets
            self._add_dependencies(state, internal_node, None, None, inputs,
                                   outputs)
            self.inputs.update(sdfg_inp)
            self.outputs.update(sdfg_out)

        elif dec.startswith('dace.map') or dec.startswith(
                'dace.consume'):  # Scope or scope+tasklet
            if 'map' in dec:
                params = self._decorator_or_annotation_params(node)
                params, map_inputs = self._parse_map_inputs(
                    node.name, params, node)
                entry, exit = state.add_map(node.name,
                                            ndrange=params,
                                            debuginfo=self.current_lineinfo)
            elif 'consume' in dec:
                (stream_name, stream_elem, PE_tuple, condition,
                 chunksize) = self._parse_consume_inputs(node)
                params = [
                    PE_tuple, (stream_elem, self.sdfg.arrays[stream_name].dtype)
                ]
                map_inputs = {}
                entry, exit = state.add_consume(node.name,
                                                PE_tuple,
                                                condition,
                                                chunksize=chunksize,
                                                debuginfo=self.current_lineinfo)

            if dec.endswith('scope'):  # @dace.mapscope or @dace.consumescope
                sdfg, inputs, outputs = self._parse_subprogram(
                    node.name,
                    node,
                    extra_symbols=self._symbols_from_params(params, map_inputs))
            else:  # Scope + tasklet (e.g., @dace.map)
                name = "{}_body".format(entry.label)
                sdfg, inputs, outputs = self._parse_subprogram(
                    name,
                    node,
                    True,
                    extra_symbols=self._symbols_from_params(params, map_inputs))

            internal_node = state.add_nested_sdfg(
                sdfg,
                self.sdfg,
                set(inputs.keys()),
                set(outputs.keys()),
                debuginfo=self.current_lineinfo)
            self._add_nested_symbols(internal_node)

            # If consume scope, inject stream inputs to the internal SDFG
            if 'consume' in dec:
                free_symbols_before = copy.copy(sdfg.free_symbols)
                self._inject_consume_memlets(dec, entry, inputs, internal_node,
                                             sdfg, state, stream_elem,
                                             stream_name)
                # Remove symbols defined after injection
                syms_to_remove = free_symbols_before - sdfg.free_symbols
                syms_to_remove.add(stream_elem)
                for sym in syms_to_remove:
                    del internal_node.symbol_mapping[sym]
                    del sdfg.symbols[sym]

            # Connect internal node with scope/access nodes
            self._add_dependencies(state, internal_node, entry, exit, inputs,
                                   outputs, map_inputs)

        elif dec == 'dace.program':  # Nested SDFG
            raise DaceSyntaxError(
                self, node, 'Nested programs must be '
                'defined outside existing programs')
        else:
            raise DaceSyntaxError(self, node, 'Unsupported function decorator')

    def _inject_consume_memlets(self, dec, entry, inputs, internal_node, sdfg,
                                state, stream_elem, stream_name):
        """ Inject stream inputs to subgraph when creating a consume scope. """

        # Inject element to internal SDFG arrays
        ntrans = sdfg.temp_data_name()
        sdfg.add_array(ntrans, [1], self.sdfg.arrays[stream_name].dtype)
        internal_memlet = dace.Memlet.simple(ntrans, subsets.Indices([0]))
        external_memlet = dace.Memlet.simple(stream_name,
                                             subsets.Indices([0]),
                                             num_accesses=-1)

        # Inject to internal tasklet
        if not dec.endswith('scope'):
            injected_node_count = 0
            for s in sdfg.nodes():
                for n in s.nodes():
                    if isinstance(n, nodes.Tasklet):
                        n.add_in_connector(stream_elem)
                        rnode = s.add_read(ntrans,
                                           debuginfo=self.current_lineinfo)
                        s.add_edge(rnode, None, n, stream_elem, internal_memlet)
                        injected_node_count += 1
            assert injected_node_count == 1

        # Inject to nested SDFG node
        internal_node.add_in_connector(ntrans)
        stream_node = state.add_read(stream_name,
                                     debuginfo=self.current_lineinfo)
        state.add_edge_pair(entry,
                            internal_node,
                            stream_node,
                            external_memlet,
                            scope_connector='stream',
                            internal_connector=ntrans)

        # Mark as input so that no extra edges are added
        inputs[ntrans] = None

    def _parse_for_indices(self, node: ast.Expr):
        """Parses the indices of a for-loop statement

        Arguments:
            node {ast.Expr} -- Target of ast.For node

        Raises:
            DaceSyntaxError: If target is not ast.Tuple
            DaceSyntaxError: If index is not ast.Name
            DaceSyntaxError: If index ID is duplicate

        Returns:
            List[str] -- List of index IDs
        """

        if not isinstance(node, (ast.Name, ast.Tuple)):
            raise DaceSyntaxError(
                self, node, "Target of ast.For must be a name or a tuple")

        if isinstance(node, ast.Name):
            elts = (node, )
        else:
            elts = node.elts

        indices = []
        for n in elts:
            if not isinstance(n, ast.Name):
                raise DaceSyntaxError(self, n,
                                      "For-loop iterator must be ast.Name")
            idx_id = n.id
            if idx_id in indices:
                raise DaceSyntaxError(
                    self, n,
                    "Duplicate index id ({}) in for-loop".format(idx_id))
            indices.append(idx_id)

        return indices

    def _parse_value(self, node: Union[ast.Name, ast.Num, ast.Constant]):
        """Parses a value

        Arguments:
            node {Union[ast.Name, ast.Num, ast.Constant]} -- Value node

        Raises:
            DaceSyntaxError: If node is not ast.Name or ast.Num/Constant

        Returns:
            str -- Value id or number as string
        """

        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Num):
            return str(node.n)
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(pyexpr_to_symbolic(self.defined, node))

    def _parse_slice(self, node: ast.Slice):
        """Parses a range

        Arguments:
            node {ast.Slice} -- Slice node

        Returns:
            Tuple[str] -- Range in (from, to, step) format
        """

        return (self._parse_value(node.lower), self._parse_value(node.upper),
                self._parse_value(node.step) if node.step is not None else "1")

    def _parse_index_as_range(self, node: ast.Index):
        """Parses an index as range

        Arguments:
            node {ast.Index} -- Index node

        Returns:
            Tuple[str] -- Range in (from, to, step) format
        """

        val = self._parse_value(node.value)
        return (val, val, "1")

    def _parse_for_iterator(self, node: ast.Expr):
        """Parses the iterator of a for-loop statement

        Arguments:
            node {ast.Expr} -- Iterator (iter) of ast.For node

        Raises:
            DaceSyntaxError: If iterator is not ast.Subscript
            DaceSyntaxError: If iterator type is not supported
            NotImplementedError: If iterator type is not implemented

        Returns:
            Tuple[str, List[str]] -- Iterator type and iteration ranges
        """

        if not isinstance(node, (ast.Call, ast.Subscript)):
            raise DaceSyntaxError(
                self, node,
                "Iterator of ast.For must be a function or a subscript")

        iterator = rname(node)
        if iterator not in {'range', 'parrange', 'dace.map'}:
            raise DaceSyntaxError(self, node,
                                  "Iterator {} is unsupported".format(iterator))
        elif iterator in ['range', 'parrange']:
            if len(node.args) == 1:  # (par)range(stop)
                ranges = [('0', self._parse_value(node.args[0]), '1')]
            elif len(node.args) == 2:  # (par)range(start, stop)
                ranges = [(self._parse_value(node.args[0]),
                           self._parse_value(node.args[1]), '1')]
            elif len(node.args) == 3:  # (par)range(start, stop, step)
                ranges = [(self._parse_value(node.args[0]),
                           self._parse_value(node.args[1]),
                           self._parse_value(node.args[2]))]
            else:
                raise DaceSyntaxError(
                    self, node,
                    'Invalid number of arguments for "%s"' % iterator)
            if iterator == 'parrange':
                iterator = 'dace.map'
        else:
            ranges = []
            if isinstance(node.slice, ast.ExtSlice):
                for s in node.slice.dims:
                    ranges.append(self._parse_slice(s))
            elif isinstance(node.slice, ast.Slice):
                ranges.append(self._parse_slice(node.slice))
            else:  # isinstance(node.slice, ast.Index) is True
                ranges.append(self._parse_index_as_range(node.slice))

        return (iterator, ranges)

    def _parse_map_inputs(
            self, name: str, params: List[Tuple[str, str]],
            node: ast.AST) -> Tuple[Dict[str, str], Dict[str, Memlet]]:
        """ Parse map parameters for data-dependent inputs, modifying the
            parameter dictionary and returning relevant memlets.
            :return: A 2-tuple of (parameter dictionary, mapping from connector
                     name to memlet).
        """
        new_params = []
        map_inputs = {}
        for k, v in params:
            vsp = list(v.split(':'))
            for i, (val, vid) in enumerate(zip(vsp, 'best')):
                # Walk through expression, find functions and replace with
                # variables
                ctr = 0
                repldict = {}
                symval = pystr_to_symbolic(val)

                for atom in symval.free_symbols:
                    if symbolic.issymbolic(atom, self.sdfg.constants):
                        # Check for undefined variables
                        if str(atom) not in self.defined:
                            raise DaceSyntaxError(
                                self, node, 'Undefined variable "%s"' % atom)
                        # Add to global SDFG symbols
                        if str(atom) not in self.sdfg.symbols:
                            self.sdfg.add_symbol(str(atom), atom.dtype)

                for expr in symbolic.swalk(symval):
                    if symbolic.is_sympy_userfunction(expr):
                        # If function contains a function
                        if any(
                                symbolic.contains_sympy_functions(a)
                                for a in expr.args):
                            raise DaceSyntaxError(
                                self, node,
                                'Indirect accesses not supported in map ranges')
                        arr = expr.func.__name__
                        newvar = '__%s_%s%d' % (name, vid, ctr)
                        repldict[arr] = newvar
                        # Create memlet
                        args = ','.join([str(a) for a in expr.args])
                        if arr in self.variables:
                            arr = self.variables[arr]
                        if arr not in self.sdfg.arrays:
                            rng = subsets.Range.from_string(args)
                            args = str(rng)
                        map_inputs[newvar] = Memlet.simple(arr, args)
                        # ','.join([str(a) for a in expr.args]))
                        ctr += 1
                # Replace functions with new variables
                for find, replace in repldict.items():
                    val = re.sub(r"%s\(.*?\)" % find, replace, val)
                vsp[i] = val

            new_params.append((k, ':'.join(vsp)))

        return new_params, map_inputs

    def _parse_consume_inputs(
        self, node: ast.FunctionDef
    ) -> Tuple[str, str, Tuple[str, str], str, str]:
        """ Parse consume parameters from AST.
            :return: A 5-tuple of Stream name, internal stream name,
                     (PE index, number of PEs), condition, chunk size.
        """

        # Consume scopes in Python are defined as functions with the following
        # syntax:
        # @dace.consume(<stream name>, <number of PEs>[, <quiescence condition>,
        #               <chunk size>)
        # def func(<internal stream element name>, <internal PE index name>):

        # Parse decorator
        dec = node.decorator_list[0]
        if hasattr(dec, 'args') and len(dec.args) >= 2:
            stream_name = self.visit(dec.args[0])
            num_PEs = pystr_to_symbolic(self.visit(dec.args[1]))
            if len(dec.args) >= 3:
                # TODO: Does not work if the condition uses arrays
                condition = astutils.unparse(dec.args[2])
            else:
                condition = None  # Run until stream is empty
            if len(dec.args) >= 4:
                chunksize = pystr_to_symbolic(self.visit(dec.args[3]))
            else:
                chunksize = 1
        else:
            raise DaceSyntaxError(
                self, node, 'Consume scope decorator must '
                'contain at least two arguments')

        # Parse function
        if len(node.args.args) != 2:
            raise DaceSyntaxError(
                self, node, 'Consume scope function must '
                'contain two arguments')

        stream_elem, PE_index = tuple(a.arg for a in node.args.args)

        return (stream_name, stream_elem, (PE_index, num_PEs), condition,
                chunksize)

    def _find_access(self, name: str, rng: subsets.Range, mode: str):
        for n, r, m in self.accesses:
            if n == name and m == mode:
                if r == rng:
                    return True
                elif r.covers(rng):
                    print("WARNING: New access {n}[{rng}] already covered by"
                          " {n}[{r}]".format(n=name, rng=rng, r=r))
                elif rng.covers(r):
                    print("WARNING: New access {n}[{rng}] covers previous"
                          " access {n}[{r}]".format(n=name, rng=rng, r=r))
                return False

    def _add_dependencies(self,
                          state: SDFGState,
                          internal_node: nodes.CodeNode,
                          entry_node: nodes.EntryNode,
                          exit_node: nodes.ExitNode,
                          inputs: Dict[str, Memlet],
                          outputs: Dict[str, Memlet],
                          map_inputs: Dict[str, Memlet] = None):

        # Parse map inputs (for memory-based ranges)
        if map_inputs is not None:
            for conn, memlet in map_inputs.items():
                if self.nested:
                    new_name = self._add_read_access(memlet.data, memlet.subset,
                                                     None)
                    memlet = Memlet.from_array(new_name,
                                               self.sdfg.arrays[new_name])
                else:
                    new_name = memlet.data

                read_node = state.add_read(new_name,
                                           debuginfo=self.current_lineinfo)
                entry_node.add_in_connector(conn)
                state.add_edge(read_node, None, entry_node, conn, memlet)

        # Parse internal node inputs and indirect memory accesses
        if inputs:
            for conn, v in inputs.items():
                if v is None:  # Input already handled outside
                    continue
                if isinstance(v, nodes.Tasklet):
                    # Create a code->code node
                    new_scalar = self.sdfg.temp_data_name()
                    if isinstance(internal_node, nodes.NestedSDFG):
                        dtype = internal_node.sdfg.arrays[conn].dtype
                    else:
                        raise SyntaxError('Cannot determine connector type for '
                                          'tasklet input dependency')
                    self.sdfg.add_scalar(new_scalar, dtype, transient=True)
                    state.add_edge(v, conn, internal_node, conn,
                                   dace.Memlet.simple(new_scalar, '0'))
                    if entry_node is not None:
                        state.add_edge(entry_node, None, v, None, dace.Memlet())
                    continue

                if isinstance(v, tuple):
                    memlet, inner_indices = v
                else:
                    memlet, inner_indices = v, set()
                if _subset_has_indirection(memlet.subset):
                    read_node = entry_node
                    if entry_node is None:
                        read_node = state.add_read(
                            memlet.data, debuginfo=self.current_lineinfo)
                    add_indirection_subgraph(self.sdfg, state, read_node,
                                             internal_node, memlet, conn, self)
                    continue
                if memlet.data not in self.sdfg.arrays:
                    arr = self.scope_arrays[memlet.data]
                    if entry_node:
                        scope_memlet = propagate_memlet(state, memlet,
                                                        entry_node, True, arr)
                    else:
                        scope_memlet = copy.deepcopy(memlet)
                    irng = memlet.subset
                    orng = copy.deepcopy(scope_memlet.subset)
                    outer_indices = []
                    for n, (i, o) in enumerate(zip(irng, orng)):
                        if i == o and n not in inner_indices:
                            outer_indices.append(n)
                        elif n not in inner_indices:
                            inner_indices.add(n)
                    irng.pop(outer_indices)
                    orng.pop(outer_indices)
                    irng.offset(orng, True)
                    if (memlet.data, scope_memlet.subset, 'w') in self.accesses:
                        vname = self.accesses[(memlet.data, scope_memlet.subset,
                                               'w')][0]
                        memlet = Memlet.simple(vname, str(irng))
                    elif (memlet.data, scope_memlet.subset,
                          'r') in self.accesses:
                        vname = self.accesses[(memlet.data, scope_memlet.subset,
                                               'r')][0]
                        memlet = Memlet.simple(vname, str(irng))
                    else:
                        name = memlet.data
                        vname = "{c}_in_from_{s}{n}".format(
                            c=conn,
                            s=self.sdfg.nodes().index(state),
                            n=('_%s' %
                               state.node_id(entry_node) if entry_node else ''))
                        self.accesses[(name, scope_memlet.subset,
                                       'r')] = (vname, orng)
                        orig_shape = orng.size()
                        shape = [d for d in orig_shape if d != 1]
                        strides = [
                            i for j, i in enumerate(arr.strides)
                            if j not in outer_indices
                        ]
                        strides = [
                            s for d, s in zip(orig_shape, strides) if d != 1
                        ]
                        if not shape:
                            shape = [1]
                            strides = [1]
                        # TODO: Formulate this better
                        if not strides:
                            strides = [arr.strides[-1]]
                        dtype = arr.dtype
                        if isinstance(memlet.data, data.Stream):
                            self.sdfg.add_stream(vname, dtype)
                        else:
                            self.sdfg.add_array(vname,
                                                shape,
                                                dtype,
                                                strides=strides)
                        self.inputs[vname] = (scope_memlet, inner_indices)
                        # self.inputs[vname] = (memlet.data, scope_memlet.subset, inner_indices)
                        memlet.data = vname
                        # memlet.subset.offset(memlet.subset, True, outer_indices)
                else:
                    vname = memlet.data
            # for conn, memlet in inputs.items():
            #     if _subset_has_indirection(memlet.subset):
            #         read_node = entry_node
            #         if entry_node is None:
            #             read_node = state.add_read(memlet.data)
            #         add_indirection_subgraph(self.sdfg, state, read_node,
            #                                  internal_node, memlet, conn)
            #         continue

                read_node = state.add_read(vname,
                                           debuginfo=self.current_lineinfo)

                if entry_node is not None:
                    state.add_memlet_path(read_node,
                                          entry_node,
                                          internal_node,
                                          memlet=memlet,
                                          src_conn=None,
                                          dst_conn=conn)
                else:
                    state.add_edge(read_node, None, internal_node, conn, memlet)
        else:
            if entry_node is not None:
                state.add_nedge(entry_node, internal_node, dace.Memlet())

        # Parse internal node outputs
        if outputs:
            for conn, v in outputs.items():
                if v is None:  # Output already handled outside
                    continue
                if isinstance(v, tuple):
                    memlet, inner_indices = v
                else:
                    memlet, inner_indices = v, set()
                if _subset_has_indirection(memlet.subset):
                    write_node = exit_node
                    if exit_node is None:
                        write_node = state.add_write(
                            memlet.data, debuginfo=self.current_lineinfo)
                    add_indirection_subgraph(self.sdfg, state, internal_node,
                                             write_node, memlet, conn, self,
                                             True)
                    continue
                inner_memlet = memlet
                if memlet.data not in self.sdfg.arrays:
                    arr = self.scope_arrays[memlet.data]
                    if entry_node:
                        scope_memlet = propagate_memlet(state, memlet,
                                                        entry_node, True, arr)
                    else:
                        scope_memlet = copy.deepcopy(memlet)
                    irng = memlet.subset
                    orng = copy.deepcopy(scope_memlet.subset)
                    outer_indices = []
                    for n, (i, o) in enumerate(zip(irng, orng)):
                        if i == o and n not in inner_indices:
                            outer_indices.append(n)
                        elif n not in inner_indices:
                            inner_indices.add(n)
                    irng.pop(outer_indices)
                    orng.pop(outer_indices)
                    irng.offset(orng, True)
                    if self._find_access(memlet.data, scope_memlet.subset, 'w'):
                        vname = self.accesses[(memlet.data, scope_memlet.subset,
                                               'w')][0]
                        inner_memlet = Memlet.simple(vname, str(irng))
                        inner_memlet.num_accesses = memlet.num_accesses
                        inner_memlet.dynamic = memlet.dynamic
                    else:
                        name = memlet.data
                        vname = "{c}_out_of_{s}{n}".format(
                            c=conn,
                            s=self.sdfg.nodes().index(state),
                            n=('_%s' %
                               state.node_id(exit_node) if exit_node else ''))
                        self.accesses[(name, scope_memlet.subset,
                                       'w')] = (vname, orng)
                        orig_shape = orng.size()
                        shape = [d for d in orig_shape if d != 1]
                        strides = [
                            i for j, i in enumerate(arr.strides)
                            if j not in outer_indices
                        ]
                        strides = [
                            s for d, s in zip(orig_shape, strides) if d != 1
                        ]
                        if not shape:
                            shape = [1]
                            strides = [1]
                        # TODO: Formulate this better
                        if not strides:
                            strides = [arr.strides[-1]]
                        dtype = arr.dtype
                        if isinstance(memlet.data, data.Stream):
                            self.sdfg.add_stream(vname, dtype)
                        else:
                            self.sdfg.add_array(vname,
                                                shape,
                                                dtype,
                                                strides=strides)
                        self.outputs[vname] = (scope_memlet, inner_indices)
                        # self.outputs[vname] = (memlet.data, scope_memlet.subset, inner_indices)
                        inner_memlet.data = vname
                        # memlet.subset.offset(memlet.subset, True, outer_indices)
                else:
                    vname = memlet.data
                write_node = state.add_write(vname,
                                             debuginfo=self.current_lineinfo)
                if exit_node is not None:
                    state.add_memlet_path(internal_node,
                                          exit_node,
                                          write_node,
                                          memlet=inner_memlet,
                                          src_conn=conn,
                                          dst_conn=None)
                else:
                    state.add_edge(internal_node, conn, write_node, None,
                                   inner_memlet)
        else:
            if exit_node is not None:
                state.add_nedge(internal_node, exit_node, dace.Memlet())

    def _add_nested_symbols(self, nsdfg_node: nodes.NestedSDFG):
        """ 
        Adds symbols from nested SDFG mapping values (if appear as globals)
        to current SDFG.
        """
        for mv in nsdfg_node.symbol_mapping.values():
            for sym in mv.free_symbols:
                if (sym.name not in self.sdfg.symbols
                        and sym.name in self.globals):
                    self.sdfg.add_symbol(sym.name, self.globals[sym.name].dtype)

    def _recursive_visit(self,
                         body: List[ast.AST],
                         name: str,
                         lineno: int,
                         last_state=True,
                         extra_symbols=None):
        """ Visits a subtree of the AST, creating special states before and after the visit.
            Returns the previous state, and the first and last internal states of the
            recursive visit. """
        before_state = self.last_state
        self.last_state = None
        first_internal_state = self._add_state('%s_%d' % (name, lineno))

        # Add iteration variables to recursive visit
        if extra_symbols:
            old_globals = self.globals
            self.globals = copy.copy(self.globals)
            self.globals.update(extra_symbols)

        # Recursive loop processing
        for stmt in body:
            self.visit_TopLevel(stmt)

        # Create the next state
        last_internal_state = self.last_state
        if last_state:
            self.last_state = None
            self._add_state('end%s_%d' % (name, lineno))

        # Revert new symbols
        if extra_symbols:
            self.globals = old_globals

        return before_state, first_internal_state, last_internal_state

    def visit_For(self, node: ast.For):
        # We allow three types of for loops:
        # 1. `for i in range(...)`: Creates a looping state
        # 2. `for i in parrange(...)`: Creates a 1D map
        # 3. `for i,j,k in dace.map[0:M, 0:N, 0:K]`: Creates an ND map
        # print(ast.dump(node))
        indices = self._parse_for_indices(node.target)
        iterator, ranges = self._parse_for_iterator(node.iter)

        if len(indices) != len(ranges):
            raise DaceSyntaxError(
                self, node,
                "Number of indices and ranges of for-loop do not match")

        if iterator == 'dace.map':
            state = self._add_state('MapState')
            params = [(k, ':'.join(v)) for k, v in zip(indices, ranges)]
            params, map_inputs = self._parse_map_inputs('map_%d' % node.lineno,
                                                        params, node)
            me, mx = state.add_map(name='%s_%d' % (self.name, node.lineno),
                                   ndrange=params,
                                   debuginfo=self.current_lineinfo)
            # body = SDFG('MapBody')
            body, inputs, outputs = self._parse_subprogram(
                self.name,
                node,
                extra_symbols=self._symbols_from_params(params, map_inputs))
            tasklet = state.add_nested_sdfg(body,
                                            self.sdfg,
                                            inputs.keys(),
                                            outputs.keys(),
                                            debuginfo=self.current_lineinfo)
            self._add_nested_symbols(tasklet)
            self._add_dependencies(state, tasklet, me, mx, inputs, outputs,
                                   map_inputs)
        elif iterator == 'range':
            # Create an extra typed symbol for the loop iterate
            from dace.codegen.tools.type_inference import infer_expr_type
            extra_syms = {
                indices[0]:
                symbolic.symbol(
                    indices[0],
                    dtypes.result_type_of(
                        infer_expr_type(ranges[0][0], self.sdfg.symbols),
                        infer_expr_type(ranges[0][1], self.sdfg.symbols),
                        infer_expr_type(ranges[0][2], self.sdfg.symbols)))
            }

            # Add range symbols as necessary
            for rng in ranges[0]:
                rng = pystr_to_symbolic(rng)
                for atom in rng.free_symbols:
                    if symbolic.issymbolic(atom, self.sdfg.constants):
                        # Check for undefined variables
                        if str(atom) not in self.defined:
                            raise DaceSyntaxError(
                                self, node, 'Undefined variable "%s"' % atom)
                        # Add to global SDFG symbols
                        if str(atom) not in self.sdfg.symbols:
                            self.sdfg.add_symbol(str(atom), atom.dtype)

            # Add an initial loop state with a None last_state (so as to not
            # create an interstate edge)
            laststate, first_loop_state, last_loop_state = self._recursive_visit(
                node.body, 'for', node.lineno, extra_symbols=extra_syms)
            end_loop_state = self.last_state

            # Add loop to SDFG
            loop_cond = '>' if ((
                pystr_to_symbolic(ranges[0][2]) < 0) == True) else '<'
            self.sdfg.add_loop(
                laststate, first_loop_state, end_loop_state, indices[0],
                ranges[0][0],
                '%s %s %s' % (indices[0], loop_cond, ranges[0][1]),
                '%s + %s' % (indices[0], ranges[0][2]), last_loop_state)
        else:
            raise DaceSyntaxError(
                self, node, 'Unsupported for-loop iterator "%s"' % iterator)

    def visit_While(self, node: ast.While):
        # Add an initial loop state with a None last_state (so as to not
        # create an interstate edge)
        laststate, first_loop_state, last_loop_state = \
            self._recursive_visit(node.body, 'while', node.lineno)
        end_loop_state = self.last_state

        # Add loop to SDFG
        loop_cond = astutils.unparse(node.test)
        self.sdfg.add_loop(laststate, first_loop_state, end_loop_state, None,
                           None, loop_cond, None, last_loop_state)

    def visit_If(self, node: ast.If):
        # Add a guard state
        self._add_state('if_guard')

        # Visit recursively
        laststate, first_if_state, last_if_state = \
            self._recursive_visit(node.body, 'if', node.lineno)
        end_if_state = self.last_state

        # Connect the states
        cond = astutils.unparse(node.test)
        cond_else = astutils.unparse(astutils.negate_expr(node.test))
        self.sdfg.add_edge(laststate, first_if_state, dace.InterstateEdge(cond))
        self.sdfg.add_edge(last_if_state, end_if_state, dace.InterstateEdge())

        # Process 'else'/'elif' statements
        if len(node.orelse) > 0:
            # Visit recursively
            _, first_else_state, last_else_state = \
                self._recursive_visit(node.orelse, 'else', node.lineno, False)

            # Connect the states
            self.sdfg.add_edge(laststate, first_else_state,
                               dace.InterstateEdge(cond_else))
            self.sdfg.add_edge(last_else_state, end_if_state,
                               dace.InterstateEdge())
            self.last_state = end_if_state
        else:
            self.sdfg.add_edge(laststate, end_if_state,
                               dace.InterstateEdge(cond_else))

    def _parse_index(self, node: ast.Index):

        indices = []
        for idx in node.value.elts:
            indices.append(self._parse_value(idx))

        return indices

    def _parse_tasklet(self, state: SDFGState, node: TaskletType, name=None):
        ttrans = TaskletTransformer(self.defined,
                                    self.sdfg,
                                    state,
                                    self.filename,
                                    nested=self.nested,
                                    scope_arrays=self.scope_arrays,
                                    scope_vars=self.scope_vars,
                                    variables=self.variables,
                                    accesses=self.accesses)
        node, inputs, outputs, self.accesses = ttrans.parse_tasklet(node, name)

        # Convert memlets to their actual data nodes
        for i in inputs.values():
            if not isinstance(i, tuple) and i.data in self.scope_vars.keys():
                i.data = self.scope_vars[i.data]
        for o in outputs.values():
            if not isinstance(o, tuple) and o.data in self.scope_vars.keys():
                o.data = self.scope_vars[o.data]
        return node, inputs, outputs, ttrans.sdfg_inputs, ttrans.sdfg_outputs

    def _add_assignment(self,
                        node: Union[ast.Assign, ast.AugAssign],
                        target: Union[str, Tuple[str, subsets.Range]],
                        operand: Union[str, Tuple[str, subsets.Range]],
                        op: str = None):
        op_is_scalar = False
        if isinstance(target, tuple):
            target_name, target_subset = target
        else:
            target_name = target
            target_array = self.sdfg.arrays[target_name]
            target_subset = subsets.Range.from_array(target_array)
        if isinstance(operand, tuple):
            op_name, op_subset = operand
        elif isinstance(operand, (int, float, complex)):
            op_is_scalar = True
        elif symbolic.issymbolic(operand):
            op_is_scalar = True
            for sym in operand.free_symbols:
                if str(sym) not in self.sdfg.symbols:
                    self.sdfg.add_symbol(str(sym), self.globals[str(sym)].dtype)
            operand = symbolic.symstr(operand)
        else:
            op_name = operand
            op_array = self.sdfg.arrays[op_name]
            op_subset = subsets.Range.from_array(op_array)

        state = self._add_state("assign_{l}_{c}".format(l=node.lineno,
                                                        c=node.col_offset))

        if target_subset.num_elements() != 1:
            if not op_is_scalar and op_subset.num_elements() != 1:
                op1 = state.add_read(op_name, debuginfo=self.current_lineinfo)
                op2 = state.add_write(target_name,
                                      debuginfo=self.current_lineinfo)
                memlet = Memlet.simple(target_name, target_subset)
                memlet.other_subset = op_subset
                state.add_nedge(op1, op2, memlet)
            else:
                memlet = Memlet.simple(
                    target_name,
                    ','.join(['__i%d' % i for i in range(len(target_subset))]))
                if op:
                    memlet.wcr = LambdaProperty.from_string(
                        'lambda x, y: x {} y'.format(op))
                if op_is_scalar:
                    state.add_mapped_tasklet(state.label, {
                        '__i%d' % i: '%s:%s+1:%s' % (start, end, step)
                        for i, (start, end, step) in enumerate(target_subset)
                    }, {},
                                             '__out = %s' % operand,
                                             {'__out': memlet},
                                             external_edges=True,
                                             debuginfo=self.current_lineinfo)
                else:
                    state.add_mapped_tasklet(state.label, {
                        '__i%d' % i: '%s:%s+1:%s' % (start, end, step)
                        for i, (start, end, step) in enumerate(target_subset)
                    }, {
                        '__inp':
                        Memlet.simple(op_name, '%s' % op_subset[0][0])
                    },
                                             '__out = __inp', {'__out': memlet},
                                             external_edges=True,
                                             debuginfo=self.current_lineinfo)
        else:
            if not op_is_scalar and op_subset.num_elements() != 1:
                raise DaceSyntaxError(
                    self, node, "Incompatible subsets %s and %s" %
                    (target_subset, op_subset))
            if op_is_scalar:
                tasklet = state.add_tasklet(name=state.label,
                                            inputs={},
                                            outputs={'__out'},
                                            code='__out = %s' % operand,
                                            debuginfo=self.current_lineinfo)
            else:
                op1 = state.add_read(op_name, debuginfo=self.current_lineinfo)
                tasklet = state.add_tasklet(name=state.label,
                                            inputs={'__inp'},
                                            outputs={'__out'},
                                            code='__out = __inp',
                                            debuginfo=self.current_lineinfo)
                inp_memlet = Memlet.simple(op_name, '%s' % op_subset[0][0])
                state.add_edge(op1, None, tasklet, '__inp', inp_memlet)

            op2 = state.add_write(target_name, debuginfo=self.current_lineinfo)
            out_memlet = Memlet.simple(target_name, '%s' % target_subset[0][0])
            state.add_edge(tasklet, '__out', op2, None, out_memlet)

    def _add_aug_assignment(self, node: Union[ast.Assign, ast.AugAssign],
                            rtarget: Union[str, Tuple[str, subsets.Range]],
                            wtarget: Union[str, Tuple[str, subsets.Range]],
                            operand: Union[str, Tuple[str,
                                                      subsets.Range]], op: str):

        if isinstance(rtarget, tuple):
            rtarget_name, rtarget_subset = rtarget
        else:
            rtarget_name = rtarget
            rtarget_array = self.sdfg.arrays[rtarget_name]
            rtarget_subset = subsets.Range.from_array(rtarget_array)
        if isinstance(wtarget, tuple):
            wtarget_name, wtarget_subset = wtarget
        else:
            wtarget_name = wtarget
            wtarget_array = self.sdfg.arrays[wtarget_name]
            wtarget_subset = subsets.Range.from_array(wtarget_array)
        if isinstance(operand, tuple):
            op_name, op_subset = operand
        else:
            op_name = operand
            op_array = self.sdfg.arrays[op_name]
            op_subset = subsets.Range.from_array(op_array)

        state = self._add_state("augassign_{l}_{c}".format(l=node.lineno,
                                                           c=node.col_offset))

        if wtarget_subset.num_elements() != 1:
            if op_subset.num_elements() != 1:
                if wtarget_subset.size() == op_subset.size():
                    in1_subset = copy.deepcopy(rtarget_subset)
                    in1_subset.offset(wtarget_subset, True)
                    in1_memlet = Memlet.simple(
                        rtarget_name, ','.join([
                            '__i%d + %d' % (i, s)
                            for i, (s, _, _) in enumerate(in1_subset)
                        ]))
                    in2_subset = copy.deepcopy(op_subset)
                    in2_subset.offset(wtarget_subset, True)
                    in2_memlet = Memlet.simple(
                        op_name, ','.join([
                            '__i%d + %d' % (i, s)
                            for i, (s, _, _) in enumerate(in2_subset)
                        ]))
                    out_memlet = Memlet.simple(
                        wtarget_name, ','.join(
                            ['__i%d' % i for i in range(len(wtarget_subset))]))
                    state.add_mapped_tasklet(state.label, {
                        '__i%d' % i: '%s:%s+1:%s' % (start, end, step)
                        for i, (start, end, step) in enumerate(wtarget_subset)
                    }, {
                        '__in1': in1_memlet,
                        '__in2': in2_memlet
                    },
                                             '__out = __in1 {op} __in2'.format(
                                                 op=op), {'__out': out_memlet},
                                             external_edges=True,
                                             debuginfo=self.current_lineinfo)
                else:
                    op1 = state.add_read(op_name,
                                         debuginfo=self.current_lineinfo)
                    op2 = state.add_write(wtarget_name,
                                          debuginfo=self.current_lineinfo)
                    memlet = Memlet.simple(wtarget_name, wtarget_subset)
                    memlet.other_subset = op_subset
                    if op is not None:
                        memlet.wcr = LambdaProperty.from_string(
                            'lambda x, y: x {} y'.format(op))
                    state.add_nedge(op1, op2, memlet)
            else:
                in1_subset = copy.deepcopy(rtarget_subset)
                in1_subset.offset(wtarget_subset, True)
                in1_memlet = Memlet.simple(
                    rtarget_name, ','.join([
                        '__i%d + %d' % (i, s)
                        for i, (s, _, _) in enumerate(in1_subset)
                    ]))
                in2_memlet = Memlet.simple(op_name, '%s' % op_subset[0][0])
                out_memlet = Memlet.simple(
                    wtarget_name,
                    ','.join(['__i%d' % i for i in range(len(wtarget_subset))]))
                state.add_mapped_tasklet(
                    state.label, {
                        '__i%d' % i: '%s:%s+1:%s' % (start, end, step)
                        for i, (start, end, step) in enumerate(wtarget_subset)
                    }, {
                        '__in1': in1_memlet,
                        '__in2': in2_memlet
                    },
                    '__out = __in1 {op} __in2'.format(op=op),
                    {'__out': out_memlet},
                    external_edges=True,
                    debuginfo=self.current_lineinfo)
        else:
            if op_subset.num_elements() != 1:
                raise DaceSyntaxError(
                    self, node, "Incompatible subsets %s, %s and %s" %
                    (rtarget_subset, op_subset, wtarget_subset))
            else:
                op1 = state.add_read(rtarget_name,
                                     debuginfo=self.current_lineinfo)
                op2 = state.add_read(op_name, debuginfo=self.current_lineinfo)
                op3 = state.add_write(wtarget_name,
                                      debuginfo=self.current_lineinfo)
                tasklet = state.add_tasklet(
                    name=state.label,
                    inputs={'__in1', '__in2'},
                    outputs={'__out'},
                    code='__out = __in1 {op} __in2'.format(op=op),
                    debuginfo=self.current_lineinfo)
                in1_memlet = Memlet.simple(rtarget_name,
                                           '%s' % rtarget_subset[0][0])
                in2_memlet = Memlet.simple(op_name, '%s' % op_subset[0][0])
                out_memlet = Memlet.simple(wtarget_name,
                                           '%s' % wtarget_subset[0][0])
                state.add_edge(op1, None, tasklet, '__in1', in1_memlet)
                state.add_edge(op2, None, tasklet, '__in2', in2_memlet)
                state.add_edge(tasklet, '__out', op3, None, out_memlet)

    def _get_variable_name(self, node, name):
        if name in self.variables:
            return self.variables[name]
        elif name in self.scope_vars:
            return self.scope_vars[name]
        else:
            raise DaceSyntaxError(self, node,
                                  'Array "%s" used before definition' % name)

    def _add_access(
            self,
            name: str,
            rng: subsets.Range,
            access_type: str,  # 'r' or 'w'
            target: Union[ast.Name, ast.Subscript],
            new_name: str = None,
            arr_type: data.Data = None) -> str:
        if access_type not in ('r', 'w'):
            raise ValueError("Access type {} is invalid".format(access_type))
        if new_name:
            var_name = new_name
        elif target:
            var_name = "__tmp_{l}_{c}_{a}".format(l=target.lineno,
                                                  c=target.col_offset,
                                                  a=access_type)
        else:
            var_name = self.sdfg.temp_data_name()

        parent_name = self.scope_vars[name]
        parent_array = self.scope_arrays[parent_name]
        squeezed_rng = copy.deepcopy(rng)
        non_squeezed = squeezed_rng.squeeze()
        shape = squeezed_rng.size()
        dtype = parent_array.dtype

        if arr_type is None:
            arr_type = type(parent_array)
        if arr_type == data.Scalar:
            self.sdfg.add_scalar(var_name, dtype)
        elif arr_type == data.Array:
            if non_squeezed:
                strides = [parent_array.strides[d] for d in non_squeezed]
            else:
                strides = [1]
            self.sdfg.add_array(var_name, shape, dtype, strides=strides)
        elif arr_type == data.Stream:
            self.sdfg.add_stream(var_name, dtype)
        else:
            raise NotImplementedError(
                "Data type {} is not implemented".format(arr_type))

        self.accesses[(name, rng, access_type)] = (var_name, squeezed_rng)

        inner_indices = set(non_squeezed)

        if access_type == 'r':
            self.inputs[var_name] = (dace.Memlet.simple(parent_name,
                                                        rng), inner_indices)
        else:
            self.outputs[var_name] = (dace.Memlet.simple(parent_name,
                                                         rng), inner_indices)

        return var_name

    def _add_read_access(self,
                         name: str,
                         rng: subsets.Range,
                         target: Union[ast.Name, ast.Subscript],
                         new_name: str = None,
                         arr_type: data.Data = None):

        if (name, rng, 'w') in self.accesses:
            return self.accesses[(name, rng, 'w')][0]
        elif (name, rng, 'r') in self.accesses:
            return self.accesses[(name, rng, 'r')][0]
        elif name in self.variables:
            return self.variables[name]
        elif name in self.scope_vars:
            return self._add_access(name, rng, 'r', target, new_name, arr_type)
        else:
            raise NotImplementedError

    def _add_write_access(self,
                          name: str,
                          rng: subsets.Range,
                          target: Union[ast.Name, ast.Subscript],
                          new_name: str = None,
                          arr_type: data.Data = None):

        if (name, rng, 'w') in self.accesses:
            return self.accesses[(name, rng, 'w')][0]
        elif name in self.variables:
            return self.variables[name]
        elif (name, rng, 'r') in self.accesses or name in self.scope_vars:
            return self._add_access(name, rng, 'w', target, new_name, arr_type)
        else:
            raise NotImplementedError

    def visit_Assign(self, node: ast.Assign):

        self._visit_assign(node, node.targets[0], None)

    def _visit_assign(self, node, node_target, op):
        # Get targets (elts) and results
        elts = None
        results = None
        if isinstance(node_target, (ast.Tuple, ast.List)):
            elts = node_target.elts
        else:
            elts = [node_target]

        results = []
        if isinstance(node.value, (ast.Tuple, ast.List)):
            for n in node.value.elts:
                if isinstance(n, (ast.Num, ast.Constant)):
                    results.append(self._convert_num_to_array(n))
                else:
                    results.extend(self._gettype(n))
        else:
            if isinstance(node.value, (ast.Constant, ast.Num)):
                results.append(self._convert_num_to_array(node.value))
            else:
                results.extend(self._gettype(node.value))

        if len(results) != len(elts):
            raise DaceSyntaxError(
                self, node, 'Function returns %d values but %d provided' %
                (len(results), len(elts)))

        defined_vars = {**self.variables, **self.scope_vars}
        defined_arrays = {**self.sdfg.arrays, **self.scope_arrays}

        for target, (result, _) in zip(elts, results):

            name = rname(target)
            true_name = None
            if name in defined_vars:
                true_name = defined_vars[name]
                true_array = defined_arrays[true_name]

            if (isinstance(target, ast.Name) and true_name and not op
                    and not isinstance(true_array, data.Scalar)
                    and not (true_array.shape == (1, ))):
                raise DaceSyntaxError(
                    self, target,
                    'Cannot reassign value to variable "{}"'.format(name))

            if not true_name and op:
                raise DaceSyntaxError(
                    self, target,
                    'Variable "{}" used before definition'.format(name))

            new_data = None
            if not true_name:
                if (result in self.sdfg.arrays
                        and not self.sdfg.arrays[result].transient):
                    result_data = self.sdfg.arrays[result]
                    true_name, new_data = _add_transient_data(
                        self.sdfg, result_data)
                    self.variables[name] = true_name
                    defined_vars[name] = true_name
                else:
                    self.variables[name] = result
                    defined_vars[name] = result
                    continue

            if new_data:
                rng = dace.subsets.Range.from_array(new_data)
            else:
                true_target = copy.deepcopy(target)
                if isinstance(target, ast.Name):
                    true_target.id = true_name
                elif isinstance(target, ast.Subscript):
                    true_target.value.id = true_name
                rng = dace.subsets.Range(
                    astutils.subscript_to_slice(true_target, defined_arrays)[1])

            if self.nested and not new_data:  # Nested SDFG
                if op:
                    rtarget = self._add_read_access(name, rng, target)
                    wtarget = self._add_write_access(name, rng, target)
                    self._add_aug_assignment(node, rtarget, wtarget, result, op)
                else:
                    wtarget = self._add_write_access(name, rng, target)
                    self._add_assignment(node, wtarget, result)
            else:  # Top-level SDFG
                output_indirection = None
                if _subset_has_indirection(rng):
                    output_indirection = self.sdfg.add_state(
                        'slice_%s_%d' % (true_name, node.lineno))
                    wnode = output_indirection.add_write(
                        true_name, debuginfo=self.current_lineinfo)
                    memlet = Memlet.simple(true_name, str(rng))
                    tmp = self.sdfg.temp_data_name()
                    wtarget = add_indirection_subgraph(self.sdfg,
                                                       output_indirection, None,
                                                       wnode, memlet, tmp, self,
                                                       True)
                else:
                    wtarget = (true_name, rng)
                if op:
                    if _subset_has_indirection(rng):
                        self._add_state('slice_%s_%d' %
                                        (true_name, node.lineno))
                        rnode = self.last_state.add_read(
                            true_name, debuginfo=self.current_lineinfo)
                        memlet = Memlet.simple(true_name, str(rng))
                        tmp = self.sdfg.temp_data_name()
                        rtarget = add_indirection_subgraph(
                            self.sdfg, self.last_state, rnode, None, memlet,
                            tmp, self)
                    else:
                        rtarget = (true_name, rng)
                    self._add_aug_assignment(node, rtarget, wtarget, result, op)
                else:
                    self._add_assignment(node, wtarget, result)

                if output_indirection:
                    self.sdfg.add_edge(self.last_state, output_indirection,
                                       dace.sdfg.InterstateEdge())
                    self.last_state = output_indirection

    def visit_AugAssign(self, node: ast.AugAssign):

        self._visit_assign(node, node.target,
                           augassign_ops[type(node.op).__name__])

    def _get_keyword_value(self, keywords: List[ast.keyword], arg: str):
        """Finds a keyword in list and returns its value

        Arguments:
            keywords {List[ast.keyword]} -- Keyword list
            arg {str} -- Keyword ID

        Raises:
            DaceSyntaxError: If keyword is not found

        Returns:
            Any -- Keyword value
        """

        for kword in keywords:
            if kword.arg == arg:
                return kword.value

        raise DaceSyntaxError(self, keywords,
                              "Keyword {} not found".format(arg))

    def _parse_shape(self, node: Union[ast.List, ast.Tuple, ast.Attribute]):
        """Parses the shape of an array

        Arguments:
            node {Union[ast.List, ast.Tuple, ast.Attribute]} -- Shape node

        Raises:
            DaceSyntaxError: If shape node is ast.Attribute, but the attribute is not a shape
            DaceSyntaxError: If shape node is neither a list/tuple nor an attribute

        Returns:
            List[Union[str, int, dace.symbol]] -- Shape
        """

        if isinstance(node, (ast.List, ast.Tuple)):
            shape = []
            for length in node.elts:
                shape.append(self._parse_value(length))
        elif isinstance(node, ast.Attribute):
            if node.attr != "shape":
                raise DaceSyntaxError(
                    self, node, "Attribute {} is not shape".format(rname(node)))
            shape = self.scope_arrays[node.value.id].shape
        else:
            raise DaceSyntaxError(
                self, node,
                "Array shape must either be a list of dimension lengths or "
                " the shape attribute of another array.")

        return shape

    def _parse_dtype(self, node: ast.Attribute):
        """Parses the dtype of an array

        Arguments:
            node {ast.Attribute} -- Dtype node

        Raises:
            DaceSyntaxError: If dtype node is an ast.Attribute, but the attribute is not a dtype
            DaceSyntaxError: If dtype node is not ast.Attribute

        Returns:
            Any -- Dtype
        """

        if isinstance(node, ast.Attribute):
            if node.value.id in {"dace", "numpy"}:
                dtype = getattr(self.globals[node.value.id], node.attr)
            elif node.attr != "dtype":
                raise DaceSyntaxError(
                    self, node, "Attribute {} is not dtype".format(rname(node)))
            else:
                dtype = self.scope_arrays[node.value.id].dtype
        else:
            raise DaceSyntaxError(
                self, node, "Array dtype must either be a dace/numpy type or "
                " the dtype attribute of another array.")

        return dtype

    def _parse_ndarray(self, node: ast.Call):
        """Parses a call to numpy.ndarray

        Arguments:
            node {ast.Call} -- Call node

        Returns:
            Tuple[shape, dtype] -- Shape and dtype of the array
        """

        num_args = len(node.args)
        # num_kwargs = len(node.keywords)

        if num_args == 0:
            shape_node = self._get_keyword_value(node.keywords, "shape")
            shape = self._parse_shape(shape_node)
            print(shape)
            dtype_node = self._get_keyword_value(node.keywords, "dtype")
            dtype = self._parse_dtype(dtype_node)
            print(dtype)
        elif num_args == 1:
            shape_node = node.args[0]
            shape = self._parse_shape(shape_node)
            print(shape)
            dtype_node = self._get_keyword_value(node.keywords, "dtype")
            dtype = self._parse_dtype(dtype_node)
            print(dtype)
        elif num_args >= 2:
            shape_node = node.args[0]
            shape = self._parse_shape(shape_node)
            print(shape)
            dtype_node = node.args[1]
            dtype = self._parse_dtype(dtype_node)
            print(dtype)

        return (shape, dtype)

    def _parse_function_arg(self, arg: ast.AST):
        # Obtain a string representation
        return self.visit(arg)

    def _is_inputnode(self, sdfg: SDFG, name: str):
        visited_data = set()
        for state in sdfg.nodes():
            visited_state_data = set()
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data == name:
                    visited_state_data.add(node.data)
                    if (node.data not in visited_data
                            and state.in_degree(node) == 0):
                        return True
            visited_data = visited_data.union(visited_state_data)

    def _is_outputnode(self, sdfg: SDFG, name: str):
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data == name:
                    if state.in_degree(node) > 0:
                        return True

    def visit_Call(self, node: ast.Call):
        from dace.frontend.python.parser import DaceProgram  # Avoiding import loop

        funcname = rname(node)
        func = None

        # Check if the function exists as an SDFG in a different module
        modname = until(funcname, '.')
        if ('.' in funcname and len(modname) > 0 and modname in self.globals
                and dtypes.ismodule(self.globals[modname])):
            func = getattr(self.globals[modname], funcname[len(modname) + 1:])

            # Not an SDFG, ignore (might be a recognized function, see below)
            if not isinstance(func, (SDFG, DaceProgram)):
                func = None
            else:
                # An SDFG, replace dots in name with underscores
                funcname = funcname.replace('.', '_')

        # If the function exists as a global SDFG or @dace.program, use it
        if func or funcname in self.other_sdfgs:
            if func is None:
                func = self.other_sdfgs[funcname]
            if isinstance(func, SDFG):
                sdfg = copy.deepcopy(func)
                args = [(arg.arg, self._parse_function_arg(arg.value))
                        for arg in node.keywords]
                required_args = list(sdfg.arglist().keys())
                # Add keyword arguments to variables
                for (k, v) in args:
                    self.variables[k] = v
            elif isinstance(func, DaceProgram):
                args = [(aname, self._parse_function_arg(arg))
                        for aname, arg in zip(func.argnames, node.args)]
                args += [(arg.arg, self._parse_function_arg(arg.value))
                         for arg in node.keywords]
                required_args = func.argnames

                sdfg = copy.deepcopy(
                    func.to_sdfg(*({
                        **self.defined,
                        **self.sdfg.arrays,
                        **self.sdfg.symbols
                    }[arg] if isinstance(arg, str) else arg
                                   for aname, arg in args)))

            else:
                raise DaceSyntaxError(
                    self, node, 'Unrecognized SDFG type "%s" in call to "%s"' %
                    (type(func).__name__, funcname))

            # Avoid import loops
            from dace.frontend.python.parser import infer_symbols_from_shapes

            # Map internal SDFG symbols by adding keyword arguments
            symbols = set(sdfg.symbols.keys())
            try:
                mapping = infer_symbols_from_shapes(
                    sdfg, {
                        k: self.sdfg.arrays[v]
                        for k, v in args if v in self.sdfg.arrays
                    },
                    set(sym.arg for sym in node.keywords if sym.arg in symbols))
            except ValueError as ex:
                raise DaceSyntaxError(self, node, str(ex))
            if len(mapping) == 0:  # Default to same-symbol mapping
                mapping = None

            # Add undefined symbols to required arguments
            if mapping:
                required_args.extend(
                    [sym for sym in symbols if sym not in mapping])
            else:
                required_args.extend(symbols)

            # Argument checks
            for arg in node.keywords:
                if arg.arg not in required_args:
                    raise DaceSyntaxError(
                        self, node, 'Invalid keyword argument "%s" in call to '
                        '"%s"' % (arg.arg, funcname))
            if len(args) != len(required_args):
                raise DaceSyntaxError(
                    self, node, 'Argument number mismatch in'
                    ' call to "%s" (expected %d,'
                    ' got %d)' % (funcname, len(required_args), len(args)))

            # Remove newly-defined symbols from arguments
            if mapping is not None:
                symbols -= set(mapping.keys())
            if len(symbols) > 0:
                mapping = mapping or {}
            args_to_remove = []
            for i, (aname, arg) in enumerate(args):
                if aname in symbols:
                    args_to_remove.append(args[i])
                    mapping[aname] = arg
            for arg in args_to_remove:
                args.remove(arg)

            # Change transient names
            arrays_before = list(sdfg.arrays.items())
            for arrname, array in arrays_before:
                if array.transient and arrname[:5] == '__tmp':
                    if int(arrname[5:]) < self.sdfg._temp_transients:
                        if self.sdfg._temp_transients > sdfg._temp_transients:
                            new_name = self.sdfg.temp_data_name()
                        else:
                            new_name = sdfg.temp_data_name()
                        sdfg.replace(arrname, new_name)
            self.sdfg._temp_transients = max(self.sdfg._temp_transients,
                                             sdfg._temp_transients)
            sdfg._temp_transients = self.sdfg._temp_transients

            # TODO: This workaround needs to be formalized (pass-by-assignment)
            slice_state = None
            output_slices = set()
            for arg in itertools.chain(node.args,
                                       [kw.value for kw in node.keywords]):
                if isinstance(arg, ast.Subscript):
                    slice_state = self.last_state
                    break

            # Make sure that any scope vars in the arguments are substituted
            # by an access.
            for i, (aname, arg) in enumerate(args):
                if arg not in self.sdfg.arrays:
                    if isinstance(arg, str) and arg in self.scope_arrays:
                        newarg = self._add_read_access(
                            arg,
                            subsets.Range.from_array(self.scope_arrays[arg]),
                            node)
                    else:
                        newarg = arg
                    args[i] = (aname, newarg)

            state = self._add_state('call_%s_%d' % (funcname, node.lineno))
            argdict = {
                conn: Memlet.from_array(arg, self.sdfg.arrays[arg])
                for conn, arg in args if arg in self.sdfg.arrays
            }
            # Handle scalar inputs to nested SDFG calls
            for conn, arg in args:
                if (arg not in self.sdfg.arrays
                        and conn not in mapping.keys() | symbols):
                    argdict[conn] = state.add_tasklet(
                        'scalar', {}, {conn},
                        '%s = %s' % (conn, arg),
                        debuginfo=self.current_lineinfo)

            inputs = {
                k: v
                for k, v in argdict.items() if self._is_inputnode(sdfg, k)
            }
            outputs = {
                k: v
                for k, v in argdict.items() if self._is_outputnode(sdfg, k)
            }
            # Unset parent inputs/read accesses that
            # turn out to be outputs/write accesses.
            # TODO: Is there a case where some data is both input and output?
            # TODO: If yes, is it a problem?
            for memlet in outputs.values():
                aname = memlet.data
                rng = memlet.subset
                access_value = (aname, rng)
                access_key = inverse_dict_lookup(self.accesses, access_value)
                if access_key:
                    # Delete read access and create write access and output
                    vname = aname[:-1] + 'w'
                    name, rng, atype = access_key
                    if atype == 'r':
                        del self.accesses[access_key]
                        access_value = self._add_write_access(name,
                                                              rng,
                                                              node,
                                                              new_name=vname)
                        memlet.data = vname
                    # Delete the old read descriptor
                    conn_used = False
                    for s in self.sdfg.nodes():
                        for n in s.data_nodes():
                            if n.data == aname:
                                conn_used = True
                                break
                        if conn_used:
                            break
                    if not conn_used:
                        del self.sdfg.arrays[aname]
                if aname in self.inputs.keys():
                    # Delete input
                    del self.inputs[aname]
                # Delete potential input slicing
                if slice_state:
                    for n in slice_state.nodes():
                        if isinstance(n, nodes.AccessNode) and n.data == aname:
                            for e in slice_state.in_edges(n):
                                sub = None
                                for s in itertools.chain(
                                        node.args,
                                    [kw.value for kw in node.keywords]):
                                    if isinstance(s, ast.Subscript):
                                        if s.value.id == e.src.data:
                                            sub = s
                                            break
                                if not sub:
                                    raise KeyError("Did not find output "
                                                   "subscript")
                                output_slices.add((sub, ast.Name(id=aname)))
                                slice_state.remove_edge(e)
                                slice_state.remove_node(e.src)
                            slice_state.remove_node(n)
                            break

            # Add return values as additional outputs
            rets = []
            for arrname, arr in sdfg.arrays.items():
                if arrname.startswith('__return'):
                    # Add a transient to the current SDFG
                    new_arrname = '%s_ret_%d' % (sdfg.name, len(rets))
                    newarr = copy.deepcopy(arr)
                    newarr.transient = True

                    # Substitute symbol mapping to get actual shape/strides
                    if mapping is not None:
                        # Two-step replacement (N -> __dacesym_N --> mapping[N])
                        # to avoid clashes
                        for sym, symvalue in mapping.items():
                            if str(sym) != str(symvalue):
                                sd.replace_properties(newarr, sym,
                                                      '__dacesym_' + sym)
                        for sym, symvalue in mapping.items():
                            if str(sym) != str(symvalue):
                                sd.replace_properties(newarr,
                                                      '__dacesym_' + sym,
                                                      symvalue)

                    new_arrname = self.sdfg.add_datadesc(new_arrname,
                                                         newarr,
                                                         find_new_name=True)

                    # Create an output entry for the connectors
                    outputs[arrname] = dace.Memlet.from_array(
                        new_arrname, newarr)
                    rets.append(new_arrname)

            nsdfg = state.add_nested_sdfg(sdfg,
                                          self.sdfg,
                                          inputs.keys(),
                                          outputs.keys(),
                                          mapping,
                                          debuginfo=self.current_lineinfo)
            self._add_nested_symbols(nsdfg)
            self._add_dependencies(state, nsdfg, None, None, inputs, outputs)

            if output_slices:
                if len(rets) > 0:
                    raise DaceSyntaxError(
                        self, node, 'Both return values and output slices '
                        'unsupported')

                assign_node = ast.Assign()
                targets = []
                value = []
                for t, v in output_slices:
                    targets.append(t)
                    value.append(v)
                assign_node = ast.Assign(targets=ast.Tuple(elts=targets),
                                         value=ast.Tuple(elts=value),
                                         lineno=node.lineno,
                                         col_offset=node.col_offset)
                assign_node = ast.fix_missing_locations(assign_node)
                return self._visit_assign(assign_node, assign_node.targets,
                                          None)

            # Return SDFG return values, if exist
            if len(rets) == 1:
                return rets[0]
            return rets

        # TODO: If the function is a callback, implement it as a tasklet

        # Otherwise, try to find a default implementation for the SDFG
        func = oprepo.Replacements.get(funcname)
        if func is None:
            # Check for SDFG as fallback
            func = oprepo.Replacements.get(funcname)
            if func is None:
                raise DaceSyntaxError(
                    self, node,
                    'Function "%s" is not registered with an SDFG implementation'
                    % funcname)
            print(
                'WARNING: Function "%s" is not registered with an %s implementation, falling back to SDFG'
                % funcname)

        args = [self._parse_function_arg(arg) for arg in node.args]
        keywords = {
            arg.arg: self._parse_function_arg(arg.value)
            for arg in node.keywords
        }

        self._add_state('call_%d' % node.lineno)
        self.last_state.set_default_lineinfo(self.current_lineinfo)

        result = func(self.sdfg, self.last_state, *args, **keywords)

        self.last_state.set_default_lineinfo(None)

        if isinstance(result,
                      tuple) and type(result[0]) is nested_call.NestedCall:
            self.last_state = result[0].last_state
            result = result[1]

        if not isinstance(result, (tuple, list)):
            return [result]
        return result

    # Used for memlet expressions outside of tasklets, otherwise ignored
    def visit_TopLevelExpr(self, node: ast.Expr):
        if isinstance(node.value, ast.BinOp):
            # Add two access nodes and a memlet (the arrays must already exist)
            if isinstance(node.value.op, ast.LShift):
                src = node.value.right
                dst = node.value.left
            elif isinstance(node.value.op, ast.RShift):
                src = node.value.left
                dst = node.value.right
            else:
                # Top-level binary operator that is not a memlet, does nothing
                self.generic_visit(node)
                return

            # Create an edge between the two data descriptors
            state = self._add_state('globalmemlet_%d' % node.lineno)
            src_expr = ParseMemlet(self, self.defined, src)
            dst_expr = ParseMemlet(self, self.defined, dst)
            src_name = src_expr.name
            if src_name not in self.sdfg.arrays:
                src_name = self._add_read_access(src_name, src_expr.subset,
                                                 None)
            dst_name = dst_expr.name
            if dst_name not in self.sdfg.arrays:
                dst_name = self._add_write_access(dst_name, dst_expr.subset,
                                                  None)

            rnode = state.add_read(src_name, debuginfo=self.current_lineinfo)
            wnode = state.add_write(dst_name, debuginfo=self.current_lineinfo)
            state.add_nedge(
                rnode, wnode,
                Memlet.simple(src_name,
                              subsets.Range.from_array(
                                  self.sdfg.arrays[src_name]),
                              num_accesses=src_expr.accesses,
                              wcr_str=dst_expr.wcr))
            return

        # Calling reduction or other SDFGs / functions
        elif isinstance(node.value, ast.Call):
            # Handles reduction and calling other SDFGs / DaCe programs
            # self._add_state('call_%d' % node.lineno)
            self.visit_Call(node.value)
            return

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        # Modify node value to become an expression
        new_node = ast.copy_location(ast.Expr(value=node.value), node)

        # Return values can either be tuples or a single object
        if isinstance(node.value, (ast.Tuple, ast.List)):
            ast_tuple = ast.copy_location(
                ast.parse('(%s,)' % ','.join(
                    '__return_%d' % i
                    for i in range(len(node.value.elts)))).body[0].value, node)
            self._visit_assign(new_node, ast_tuple, None)
        else:
            ast_name = ast.copy_location(ast.Name(id='__return'), node)
            self._visit_assign(new_node, ast_name, None)

    def visit_With(self, node, is_async=False):
        # "with dace.tasklet" syntax
        if len(node.items) == 1:
            dec = node.items[0].context_expr
            funcname = rname(dec)
            if funcname == 'dace.tasklet':
                # Parse as tasklet
                state = self._add_state('with_%d' % node.lineno)

                # Parse tasklet name
                namelist = self.name.split('_')
                if len(namelist) > 2:  # Remove trailing line and column number
                    name = '_'.join(namelist[:-2])
                else:
                    name = self.name

                tasklet, inputs, outputs, sdfg_inp, sdfg_out = \
                    self._parse_tasklet(state, node, name)

                # Add memlets
                self._add_dependencies(state, tasklet, None, None, inputs,
                                       outputs)
                self.inputs.update(sdfg_inp)
                self.outputs.update(sdfg_out)
                return

        raise DaceSyntaxError(
            self, node, 'General "with" statements disallowed in DaCe programs')

    def visit_AsyncWith(self, node):
        return self.visit_With(node, is_async=True)

    def _visitname(self, name: str, node: ast.AST):
        # First, if it is defined in the parser, use the definition
        if name in self.variables:
            return self.variables[name]

        # If an allowed global, use directly
        if name in self.globals:
            return inner_eval_ast(self.globals, node)

        if name in self.sdfg.arrays:
            return name

        if name in self.sdfg.symbols:
            return name

        if name not in self.scope_vars:
            raise DaceSyntaxError(self, node,
                                  'Use of undefined variable "%s"' % name)
        return self.scope_vars[name]

    #### Visitors that return arrays
    def visit_Str(self, node: ast.Str):
        # A string constant returns itself
        return node.s

    def visit_Num(self, node: ast.Num):
        return node.n

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_Name(self, node: ast.Name):
        # If visiting a name, check if it is a defined variable or a global
        return self._visitname(node.id, node)

    def visit_Attribute(self, node: ast.Attribute):
        # If visiting an attribute, return attribute value if it's of an array or global
        name = until(astutils.unparse(node), '.')
        result = self._visitname(name, node)
        if result in self.sdfg.arrays:
            arr = self.sdfg.arrays[result]
        elif result in self.scope_arrays:
            arr = self.scope_arrays[result]
        else:
            return result
        try:
            return getattr(arr, node.attr)
        except KeyError:
            return result

    def visit_List(self, node: ast.List):
        # Recursively loop over elements
        return [self.visit(a) for a in node.elts]

    def visit_Tuple(self, node: ast.Tuple):
        # Recursively loop over elements
        return tuple(self.visit(a) for a in node.elts)

    def visit_Lambda(self, node: ast.Lambda):
        # Return a string representation of the function
        return astutils.unparse(node)

    ############################################################

    def _gettype(self, opnode: ast.AST) -> List[Tuple[str, str]]:
        """ Returns an operand and its type as a 2-tuple of strings. """
        operands = self.visit(opnode)
        if isinstance(operands, (list, tuple)):
            if len(operands) == 0:
                raise DaceSyntaxError(self, opnode,
                                      'Operand has no return value')
        else:
            operands = [operands]

        result = []
        for operand in operands:
            if isinstance(operand, str) and operand in self.sdfg.arrays:
                result.append(
                    (operand, type(self.sdfg.arrays[operand]).__name__))
            elif isinstance(operand, str) and operand in self.scope_arrays:
                result.append(
                    (operand, type(self.scope_arrays[operand]).__name__))
            else:
                result.append((operand, type(operand).__name__))

        return result

    def _convert_num_to_array(self, node: Union[ast.Num, ast.Constant]):
        name = None
        if node.n not in self.numbers:
            dtype = None
            if isinstance(node.n, int):
                dtype = dace.int64
            elif isinstance(node.n, float):
                dtype = dace.float64
            elif isinstance(node.n, complex):
                dtype = dace.complex128
            else:
                raise NotImplementedError
            name, _ = self.sdfg.add_temp_transient(
                [1], dtype, lifetime=dtypes.AllocationLifetime.SDFG)
            self.numbers[node.n] = name
            init_state = None
            if not self.sdfg.nodes():
                init_state = self.sdfg.add_state('init')
                self.last_state = init_state
            else:
                init_state = self.sdfg.nodes()[0]
            tasklet = init_state.add_tasklet('init_{}'.format(name), {},
                                             {'out'},
                                             'out = {}'.format(node.n),
                                             debuginfo=self.current_lineinfo)
            access = init_state.add_write(name, debuginfo=self.current_lineinfo)
            init_state.add_edge(tasklet, 'out', access, None,
                                dace.Memlet.simple(name, '0'))
        else:
            name = self.numbers[node.n]
        return name, 'Array'

    def _visit_op(self, node: Union[ast.UnaryOp, ast.BinOp, ast.BoolOp],
                  op1: ast.AST, op2: ast.AST):
        opname = None
        try:
            opname = type(node.op).__name__
        except:
            pass

        # Parse operands
        op1_parsed = self._gettype(op1)
        if len(op1_parsed) > 1:
            raise DaceSyntaxError(self, op1, 'Operand cannot be a tuple')
        operand1, op1type = op1_parsed[0]
        if op2 is not None:
            op2_parsed = self._gettype(op2)
            if len(op2_parsed) > 1:
                raise DaceSyntaxError(self, op2, 'Operand cannot be a tuple')
            operand2, op2type = op2_parsed[0]
        else:
            operand2, op2type = None, None

        if isinstance(node, ast.BinOp):
            if op1type == 'Array' and isinstance(op2, (ast.Constant, ast.Num)):
                operand2, op2type = self._convert_num_to_array(op2)
            elif op2type == 'Array' and isinstance(op1,
                                                   (ast.Constant, ast.Num)):
                operand1, op1type = self._convert_num_to_array(op1)

        func = oprepo.Replacements.getop(op1type, opname, otherclass=op2type)
        if func is None:
            # Check for SDFG as fallback
            func = oprepo.Replacements.getop(op1type,
                                             opname,
                                             otherclass=op2type)
            if func is None:
                raise DaceSyntaxError(
                    self, node,
                    'Operator "%s" is not defined for types %s and %s' %
                    (opname, op1type, op2type))
            print(
                'WARNING: Operator "%s" is not registered with an implementation for'
                'types %s and %s, falling back to SDFG' %
                (opname, op1type, op2type))

        self._add_state('%s_%d' % (type(node).__name__, node.lineno))
        self.last_state.set_default_lineinfo(self.current_lineinfo)
        try:
            result = func(self, self.sdfg, self.last_state, operand1, operand2)
        except SyntaxError as ex:
            raise DaceSyntaxError(self, node, str(ex))

        self.last_state.set_default_lineinfo(None)

        return result

    def visit_UnaryOp(self, node: ast.UnaryOp):
        return self._visit_op(node, node.operand, None)

    def visit_BinOp(self, node: ast.BinOp):
        return self._visit_op(node, node.left, node.right)

    def visit_BoolOp(self, node: ast.BoolOp):
        last = node.values[0]
        # Syntax of BoolOp is a list of values, we parse left to right
        for i in range(1, len(node.values)):
            last = self._visit_op(node, last, node.values[i])
        return last

    def visit_Compare(self, node: ast.Compare):
        if len(node.ops) > 1 or len(node.comparators) > 1:
            raise NotImplementedError
        binop_node = ast.BinOp(node.left,
                               node.ops[0],
                               node.comparators[0],
                               lineno=node.lineno,
                               col_offset=node.col_offset)
        return self.visit_BinOp(binop_node)

    ### Subscript (slicing) handling
    def visit_Subscript(self, node: ast.Subscript):

        if self.nested:

            defined_vars = {**self.variables, **self.scope_vars}
            defined_arrays = {**self.sdfg.arrays, **self.scope_arrays}

            name = rname(node)
            true_name = defined_vars[name]

            # If this subscript originates from an external array, create the
            # subset in the edge going to the connector, as well as a local
            # reference to the subset
            if (true_name not in self.sdfg.arrays
                    and isinstance(node.value, ast.Name)):
                true_node = copy.deepcopy(node)
                true_node.value.id = true_name
                rng = dace.subsets.Range(
                    astutils.subscript_to_slice(true_node, defined_arrays)[1])

                return self._add_read_access(name, rng, node)

        # Obtain array
        node_parsed = self._gettype(node.value)
        if len(node_parsed) > 1:
            raise DaceSyntaxError(self, node.value, 'Subscripted object cannot '
                                  'be a tuple')
        array, arrtype = node_parsed[0]
        if arrtype == 'str' or arrtype in dtypes._CTYPES:
            raise DaceSyntaxError(self, node,
                                  'Type "%s" cannot be sliced' % arrtype)

        # Try to construct memlet from subscript
        # expr: MemletExpr = ParseMemlet(self, self.defined, node)
        # TODO: This needs to be formalized better
        node.value = ast.Name(id=array)
        expr: MemletExpr = ParseMemlet(self, self.sdfg.arrays, node)
        arrobj = self.sdfg.arrays[array]

        # TODO: Check dimensionality of access and extend as necessary

        # Add slicing state
        self._add_state('slice_%s_%d' % (array, node.lineno))
        rnode = self.last_state.add_read(array, debuginfo=self.current_lineinfo)
        if _subset_has_indirection(expr.subset):
            memlet = Memlet.simple(array,
                                   expr.subset,
                                   num_accesses=expr.accesses,
                                   wcr_str=expr.wcr)
            tmp = self.sdfg.temp_data_name()
            return add_indirection_subgraph(self.sdfg, self.last_state, rnode,
                                            None, memlet, tmp, self)
        else:
            other_subset = copy.deepcopy(expr.subset)
            other_subset.squeeze()
            tmp, tmparr = self.sdfg.add_temp_transient(other_subset.size(),
                                                       arrobj.dtype,
                                                       arrobj.storage)
            wnode = self.last_state.add_write(tmp,
                                              debuginfo=self.current_lineinfo)
            self.last_state.add_nedge(
                rnode, wnode,
                Memlet.simple(array,
                              expr.subset,
                              num_accesses=expr.accesses,
                              wcr_str=expr.wcr,
                              other_subset_str=other_subset))
            return tmp

    ##################################
