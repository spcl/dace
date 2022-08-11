# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from collections import OrderedDict
import copy
import itertools
import inspect
import networkx as nx
import re
import sys
import time
from os import path
import warnings
from numbers import Number
from typing import Any, Dict, List, Set, Tuple, Union, Callable, Optional

import dace
from dace import data, dtypes, subsets, symbolic, sdfg as sd
from dace import sourcemap
from dace.config import Config
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import astutils
from dace.frontend.python.common import (DaceSyntaxError, SDFGClosure, SDFGConvertible, inverse_dict_lookup)
from dace.frontend.python.astutils import ExtNodeVisitor, ExtNodeTransformer
from dace.frontend.python.astutils import rname
from dace.frontend.python import nested_call, replacements, preprocessing
from dace.frontend.python.memlet_parser import (DaceSyntaxError, parse_memlet, pyexpr_to_symbolic, ParseMemlet,
                                                inner_eval_ast, MemletExpr)
from dace.sdfg import nodes
from dace.sdfg.propagation import propagate_memlet, propagate_subset, propagate_states
from dace.memlet import Memlet
from dace.properties import LambdaProperty, CodeBlock
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.replace import replace_datadesc_names
from dace.symbolic import pystr_to_symbolic, inequal_symbols

import numpy
import sympy

# register replacements in oprepo
import dace.frontend.python.replacements
from dace.frontend.python.replacements import _sym_type, _broadcast_to

# Type hints
Size = Union[int, dace.symbolic.symbol]
ShapeTuple = Tuple[Size]
ShapeList = List[Size]
Shape = Union[ShapeTuple, ShapeList]
DependencyType = Dict[str, Tuple[SDFGState, Union[Memlet, nodes.Tasklet], Tuple[int]]]


class SkipCall(Exception):
    """ Exception used to skip calls to functions that cannot be parsed. """
    pass


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
def specifies_datatype(func: Callable[[Any, data.Data, Any], Tuple[str, data.Data]], datatype=None):
    AddTransientMethods._methods[datatype] = func
    return func


@specifies_datatype(datatype=data.Scalar)
def _method(sdfg: SDFG, sample_data: data.Scalar, dtype: dtypes.typeclass):
    name = sdfg.temp_data_name()
    _, new_data = sdfg.add_scalar(name, dtype, transient=True)
    return name, new_data


@specifies_datatype(datatype=data.Array)
def _method(sdfg: SDFG, sample_data: data.Array, dtype):
    name, new_data = sdfg.add_temp_transient_like(sample_data, dtype=dtype)
    return name, new_data


@specifies_datatype(datatype=data.View)
def _method(sdfg: SDFG, sample_data: data.View, dtype):
    name, new_data = sdfg.add_temp_transient(sample_data.shape, dtype)
    return name, new_data


@specifies_datatype(datatype=data.Stream)
def _method(sdfg: SDFG, sample_data: data.Stream, dtype):
    name = sdfg.temp_data_name()
    new_data = sdfg.add_stream(name,
                               dtype,
                               buffer_size=sample_data.buffer_size,
                               shape=sample_data.shape,
                               transient=True)
    return name, new_data


def _add_transient_data(sdfg: SDFG, sample_data: data.Data, dtype: dtypes.typeclass = None):
    """ Adds to the sdfg transient data of the same dtype, shape and other
        parameters as sample_data. """
    func = AddTransientMethods.get(type(sample_data))
    if func is None:
        raise NotImplementedError
    if dtype is None:
        return func(sdfg, sample_data, sample_data.dtype)
    else:
        return func(sdfg, sample_data, dtype)


def _is_equivalent(first: data.Data, second: data.Data):
    if not first.is_equivalent(second):
        if any(not isinstance(d, data.Scalar) and not (isinstance(d, data.Array) and d.shape == (1, ))
               for d in (first, second)):
            return False
    return True


def parse_dace_program(name: str,
                       preprocessed_ast: ast.AST,
                       argtypes: Dict[str, data.Data],
                       constants: Dict[str, Any],
                       closure: SDFGClosure,
                       simplify: Optional[bool] = None,
                       save: bool = True,
                       progress: Optional[bool] = None) -> SDFG:
    """ Parses a `@dace.program` function into an SDFG.
        :param src_ast: The AST of the Python program to parse.
        :param visitor: A ProgramVisitor object returned from 
                        ``preprocess_dace_program``.
        :param closure: An object that contains the @dace.program closure.
        :param simplify: If True, simplification pass will be performed.
        :param save: If True, saves source mapping data for this SDFG.
        :param progress: If True, prints a progress bar of the parsing process. 
                         If None (default), prints after 5 seconds of parsing. 
                         If False, never prints progress.
        :return: A 2-tuple of SDFG and its reduced (used) closure.
    """
    # Progress bar handling (pre-parse)
    teardown_progress = False
    if progress is None and not Config.get_bool('progress'):
        progress = False
    if progress is None or progress is True:
        try:
            from tqdm import tqdm
        except (ImportError, ModuleNotFoundError):
            progress = False

        if progress is not False and ProgramVisitor.progress_bar is None:
            ctl = closure.call_tree_length()
            teardown_progress = True  # First parser should teardown progress bar
            ProgramVisitor.start_time = time.time()
            if progress is True:
                ProgramVisitor.progress_bar = tqdm(total=ctl, desc='Parsing Python program')
            else:
                ProgramVisitor.progress_bar = (0, ctl)  # Make a counter instead (tqdm cannot be enabled mid-progress)
    if (progress is None and isinstance(ProgramVisitor.progress_bar, tuple)
            and (time.time() - ProgramVisitor.start_time) >= 5):
        initial, total = ProgramVisitor.progress_bar
        ProgramVisitor.progress_bar = tqdm(total=total, initial=initial, desc='Parsing Python program')
    # End of progress bar

    visitor = ProgramVisitor(name=name,
                             filename=preprocessed_ast.filename,
                             line_offset=preprocessed_ast.src_line,
                             col_offset=0,
                             global_vars=preprocessed_ast.program_globals,
                             constants=constants,
                             scope_arrays=argtypes,
                             scope_vars={},
                             closure=closure,
                             simplify=simplify)

    try:
        sdfg, _, _, _ = visitor.parse_program(preprocessed_ast.preprocessed_ast.body[0])
        sdfg.set_sourcecode(preprocessed_ast.src, 'python')

        # Combine nested closures with the current one
        nested_closure_replacements: Dict[str, str] = {}
        for name, (arr, _) in visitor.nested_closure_arrays.items():
            # Check if the same array is already passed as part of a nested closure
            if id(arr) in closure.array_mapping:
                existing_name = closure.array_mapping[id(arr)]
                if name != existing_name:
                    if existing_name not in closure.callbacks:
                        nested_closure_replacements[name] = existing_name
                    else:  # Callbacks should be mapped
                        sdfg.callback_mapping[name] = existing_name

        # Make safe replacements
        def repl_callback(repldict):
            for state in sdfg.nodes():
                for name, new_name in repldict.items():
                    state.replace(name, new_name)
            for name, new_name in repldict.items():
                sdfg.arrays[new_name] = sdfg.arrays[name]
                del sdfg.arrays[name]
                if name in sdfg.constants_prop:
                    sdfg.constants_prop[new_name] = sdfg.constants_prop[name]
                    del sdfg.constants_prop[name]

        symbolic.safe_replace(nested_closure_replacements, repl_callback, value_as_string=True)

        sdfg.debuginfo = dtypes.DebugInfo(
            visitor.src_line + 1,
            end_line=visitor.src_line + len(preprocessed_ast.src.split("\n")) - 1,
            filename=path.abspath(preprocessed_ast.filename),
        )

        # Progress bar handling (post-parse)
        if (progress is None and isinstance(ProgramVisitor.progress_bar, tuple)
                and (time.time() - ProgramVisitor.start_time) >= 5):
            initial, total = ProgramVisitor.progress_bar
            ProgramVisitor.progress_bar = tqdm(total=total, initial=initial, desc='Parsing Python program')
        ProgramVisitor.increment_progress()
    finally:
        if teardown_progress:
            if not isinstance(ProgramVisitor.progress_bar, tuple):
                ProgramVisitor.progress_bar.close()
                print('Parsing complete.')
            ProgramVisitor.progress_bar = None
            ProgramVisitor.start_time = 0

    return sdfg


# AST node types that are disallowed in DaCe programs
DISALLOWED_STMTS = [
    'Delete', 'Import', 'ImportFrom', 'Exec', 'Yield', 'YieldFrom', 'ClassDef', 'Await', 'Try', 'TryExcept',
    'TryFinally', 'ExceptHandler'
]
# Extra AST node types that are disallowed after preprocessing
_DISALLOWED_STMTS = DISALLOWED_STMTS + [
    'Global', 'Assert', 'Print', 'Nonlocal', 'Raise', 'Starred', 'AsyncFor', 'Bytes', 'ListComp', 'GeneratorExp',
    'SetComp', 'DictComp', 'comprehension'
]

TaskletType = Union[ast.FunctionDef, ast.With, ast.For]


def _disallow_stmt(visitor, node):
    raise DaceSyntaxError(visitor, node, 'Keyword "%s" disallowed' % (type(node).__name__))


###############################################################
# Parsing functions
###############################################################


def _subset_has_indirection(subset, pvisitor: 'ProgramVisitor' = None):
    for dim in subset:
        if not isinstance(dim, tuple):
            dim = [dim]
        for r in dim:
            if not symbolic.issymbolic(r):
                continue
            if symbolic.contains_sympy_functions(r):
                return True
            if pvisitor:
                for s in r.free_symbols:
                    try:
                        name = pvisitor._visitname(str(s), None)
                        if isinstance(name, str) and name in pvisitor.sdfg.arrays:
                            return True
                    except DaceSyntaxError:
                        continue
    return False


def _subset_is_local_symbol_dependent(subset: subsets.Subset, pvisitor: 'ProgramVisitor') -> bool:
    if any(s not in pvisitor.map_symbols and s not in pvisitor.globals for s in subset.free_symbols):
        return True
    return False


def add_indirection_subgraph(sdfg: SDFG,
                             graph: SDFGState,
                             src: nodes.Node,
                             dst: nodes.Node,
                             memlet: Memlet,
                             local_name: str,
                             pvisitor: 'ProgramVisitor',
                             output: bool = False,
                             with_wcr: bool = False):
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
                fname = None
                if symbolic.is_sympy_userfunction(expr):
                    fname = expr.func.__name__
                else:
                    try:
                        rname = pvisitor._visitname(str(expr), None)
                    except DaceSyntaxError:
                        continue
                    if isinstance(rname, str) and rname in pvisitor.sdfg.arrays:
                        fname = rname
                if fname:
                    if fname not in accesses:
                        accesses[fname] = []

                    # Replace function with symbol (memlet local name to-be)
                    if expr.args in accesses[fname]:
                        aindex = accesses[fname].index(expr.args)
                        toreplace = 'index_' + fname + '_' + str(aindex)
                    else:
                        if expr.args:
                            accesses[fname].append(expr.args)
                        else:
                            # Scalar access
                            accesses[fname].append(0)
                        toreplace = 'index_' + fname + '_' + str(len(accesses[fname]) - 1)

                    if direct_assignment:
                        # newsubset[dimidx] = newsubset[dimidx].subs(expr, toreplace)
                        newsubset[dimidx] = r.subs(expr, toreplace)
                        r = newsubset[dimidx]
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
    if (isinstance(memlet.subset, subsets.Range) and memlet.subset.num_elements() != 1):
        rng = copy.deepcopy(memlet.subset)
        nonsqz_dims = rng.squeeze()
        mapped_rng = []
        for i, r in enumerate(memlet.subset):
            if i in nonsqz_dims:
                mapped_rng.append(r)
        ind_entry, ind_exit = graph.add_map(
            'indirection', {'__i%d' % i: '%s:%s+1:%s' % (s, e, t)
                            for i, (s, e, t) in enumerate(mapped_rng)},
            debuginfo=pvisitor.current_lineinfo)
        inp_base_path.insert(0, ind_entry)
        out_base_path.append(ind_exit)

    input_index_memlets = []
    for arrname, arr_accesses in accesses.items():
        arr_name = arrname
        for i, access in enumerate(arr_accesses):
            if isinstance(access, sympy.Tuple):
                access = list(access)
            if not isinstance(access, (list, tuple)):
                access = [access]
            conn = None
            if pvisitor.nested:
                # TODO: Make this work for nested for-loops
                arr_rng = dace.subsets.Range([(a, a, 1) for a in access])
                if output:
                    arrname, rng = pvisitor._add_write_access(arr_name, arr_rng, target=None)
                else:
                    arrname, rng = pvisitor._add_read_access(arr_name, arr_rng, target=None)
                conn = 'index_%s_%d' % (arr_name, i)
                arr = sdfg.arrays[arrname]
                subset = subsets.Range.from_array(arr)
            else:
                subset = subsets.Indices(access)
            # Memlet to load the indirection index
            indexMemlet = Memlet.simple(arrname, subset)
            input_index_memlets.append(indexMemlet)
            read_node = graph.add_read(arrname, debuginfo=pvisitor.current_lineinfo)
            if pvisitor.nested or not isinstance(src, nodes.EntryNode):
                path = [read_node] + inp_base_path
            else:
                if output:
                    # TODO: This only works for Maps. Perhaps it should be
                    # generalized for other pairs of entry/exit nodes.
                    entry = None
                    if isinstance(dst, nodes.MapExit):
                        for node in graph.nodes():
                            if (isinstance(node, nodes.MapEntry) and node.map is dst.map):
                                entry = node
                                break
                    else:
                        raise NotImplementedError
                else:
                    entry = src
                path = [read_node, entry] + inp_base_path
            graph.add_memlet_path(*path, dst_conn="index_%s_%d" % (arr_name, i), memlet=indexMemlet)

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
        code.format(arr='__ind_' + local_name, index=', '.join([symbolic.symstr(s) for s in newsubset])))

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
            src = graph.add_access(tmp_name, debuginfo=pvisitor.current_lineinfo)
        else:
            src = graph.add_read(tmp_name, debuginfo=pvisitor.current_lineinfo)
    elif dst is None:
        if end_dst:
            dst = graph.add_access(tmp_name, debuginfo=pvisitor.current_lineinfo)
        else:
            dst = graph.add_write(tmp_name, debuginfo=pvisitor.current_lineinfo)

    tmp_shape = storage.shape
    indirectRange = subsets.Range([(0, s - 1, 1) for s in tmp_shape])
    if ind_entry:  # Amend indirected range
        indirectRange = ','.join(["{} - {}".format(ind, r[0]) for ind, r in zip(ind_entry.map.params, mapped_rng)])

    # Create memlet that depends on the full array that we look up in
    fullRange = subsets.Range([(0, s - 1, 1) for s in array.shape])
    fullMemlet = Memlet.simple(memlet.data, fullRange, num_accesses=memlet.num_accesses)
    fullMemlet.dynamic = memlet.dynamic

    if output:
        if isinstance(dst, nodes.ExitNode):
            full_write_node = graph.add_write(memlet.data, debuginfo=pvisitor.current_lineinfo)
            path = out_base_path + [dst, full_write_node]
        elif isinstance(dst, nodes.AccessNode):
            path = out_base_path + [dst]
        else:
            raise Exception("Src node type for indirection is invalid.")
        if with_wcr:
            fullMemlet.wcr = memlet.wcr
        graph.add_memlet_path(*path, src_conn='__ind_' + local_name, memlet=fullMemlet)
    else:
        if isinstance(src, nodes.EntryNode):
            full_read_node = graph.add_read(memlet.data, debuginfo=pvisitor.current_lineinfo)
            path = [full_read_node, src] + inp_base_path
        elif isinstance(src, nodes.AccessNode):
            path = [src] + inp_base_path
        else:
            raise Exception("Src node type for indirection is invalid.")
        graph.add_memlet_path(*path, dst_conn='__ind_' + local_name, memlet=fullMemlet)

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
        graph.add_memlet_path(*path, src_conn=src_conn, dst_conn='lookup', memlet=valueMemlet)
        # Connect original source to the indirected-range-transient
        if start_src:
            if isinstance(start_src, nodes.AccessNode):
                src_conn = None
            else:
                src_conn = local_name
            graph.add_edge(start_src, src_conn, src, None, Memlet.from_array(tmp_name, storage))
    else:
        path = out_base_path + [dst]
        if isinstance(dst, nodes.AccessNode):
            dst_conn = None
        else:
            dst_conn = local_name
        graph.add_memlet_path(*path, src_conn='lookup', dst_conn=dst_conn, memlet=valueMemlet)
        # Connect original destination to the indirected-range-transient
        if end_dst:
            if isinstance(end_dst, nodes.AccessNode):
                dst_conn = None
            else:
                dst_conn = local_name
            graph.add_edge(dst, None, end_dst, dst_conn, Memlet.from_array(tmp_name, storage))

    return tmp_name


class TaskletTransformer(ExtNodeTransformer):
    """ A visitor that traverses a data-centric tasklet, removes memlet
        annotations and returns input and output memlets.
    """
    def __init__(self,
                 visitor,
                 defined,
                 sdfg: SDFG,
                 state: SDFGState,
                 filename: str,
                 lang=None,
                 location: dict = {},
                 nested: bool = False,
                 scope_arrays: Dict[str, data.Data] = dict(),
                 scope_vars: Dict[str, str] = dict(),
                 variables: Dict[str, str] = dict(),
                 accesses: Dict[Tuple[str, dace.subsets.Subset, str], str] = dict(),
                 symbols: Dict[str, "dace.symbol"] = dict()):
        """ Creates an AST parser for tasklets.
            :param sdfg: The SDFG to add the tasklet in (used for defined arrays and symbols).
            :param state: The SDFG state to add the tasklet to.
        """
        self.visitor = visitor
        self.sdfg = sdfg
        self.state = state
        self.defined = defined

        # For syntax errors
        self.filename = filename

        # Connectors generated from memlets
        self.inputs: Dict[str, Memlet] = {}
        self.outputs: Dict[str, Memlet] = {}

        self.extcode = None
        self.lang = lang
        self.globalcode = ''
        self.initcode = ''
        self.exitcode = ''
        self.location = location

        self.nested = nested
        self.scope_arrays = scope_arrays
        self.scope_vars = scope_vars
        self.variables = variables
        self.accesses = accesses

        self.sdfg_inputs: Dict[str, Tuple[Memlet, Set[int]]] = {}
        self.sdfg_outputs: Dict[str, Tuple[Memlet, Set[int]]] = {}

        # Tmp fix for missing state symbol propatation
        self.symbols = symbols

        # Disallow keywords
        for stmt in _DISALLOWED_STMTS:
            setattr(self, 'visit_' + stmt, lambda n: _disallow_stmt(self, n))

    def parse_tasklet(self, tasklet_ast: TaskletType, name: Optional[str] = None):
        """ Parses the AST of a tasklet and returns the tasklet node, as well as input and output memlets.
            :param tasklet_ast: The Tasklet's Python AST to parse.
            :param name: Optional name to use as prefix for tasklet.
            :return: 3-tuple of (Tasklet node, input memlets, output memlets).
            @rtype: Tuple[Tasklet, Dict[str, Memlet], Dict[str, Memlet]]
        """
        # Should return a tasklet object (with connectors)
        self.visit(tasklet_ast)

        # Location identifier
        locinfo = dtypes.DebugInfo(tasklet_ast.lineno, tasklet_ast.col_offset, tasklet_ast.body[-1].lineno,
                                   tasklet_ast.body[-1].col_offset, self.filename)

        # Determine tasklet name (either declared as a function or use line #)
        if name is not None:
            name += '_' + str(tasklet_ast.lineno)
        else:
            name = getattr(tasklet_ast, 'name', 'tasklet_%d' % tasklet_ast.lineno)

        if self.lang is None:
            self.lang = dtypes.Language.Python

        t = self.state.add_tasklet(name,
                                   set(self.inputs.keys()),
                                   set(self.outputs.keys()),
                                   self.extcode or tasklet_ast.body,
                                   language=self.lang,
                                   code_global=self.globalcode,
                                   code_init=self.initcode,
                                   code_exit=self.exitcode,
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
            var_name = "__tmp_{l}_{c}".format(l=target.lineno, c=target.col_offset)
        else:
            var_name = self.sdfg.temp_data_name()

        parent_name = self.scope_vars[name]
        parent_array = self.scope_arrays[parent_name]
        if _subset_has_indirection(rng):
            squeezed_rng = list(range(len(rng)))
            shape = parent_array.shape
            strides = [parent_array.strides[d] for d in squeezed_rng]
            # TODO: Why is squeezed_rng an index in the first place?
            squeezed_rng = subsets.Range([(i, i, 1) for i in squeezed_rng])
        else:
            ignore_indices = []
            sym_rng = []
            offset = []
            for i, r in enumerate(rng):
                repl_dict = {}
                for s, sr in self.symbols.items():
                    if s in symbolic.symlist(r).values():
                        ignore_indices.append(i)
                        sym_rng.append(sr)
                        # NOTE: Assume that the i-th index of the range is
                        # dependent on a local symbol s, i.e, rng[i] = f(s).
                        # Therefore, the i-th index will not be squeezed
                        # even if it has length equal to 1. However, it must
                        # still be offsetted by f(min(sr)), so that the indices
                        # for the squeezed connector start from 0.
                        # Example:
                        # Memlet range: [i+1, j, k+1]
                        # k: local symbol with range(1, 4)
                        # i,j: global symbols
                        # Squeezed range: [f(k)] = [k+1]
                        # Offset squeezed range: [f(k)-f(min(range(1, 4)))] =
                        #                        [f(k)-f(1)] = [k-1]
                        # NOTE: The code takes into account the case where an
                        # index is dependent on multiple symbols. See also
                        # tests/python_frontend/nested_name_accesses_test.py.
                        step = sr[0][2]
                        if (step < 0) == True:
                            repl_dict[s] = sr[0][1]
                        else:
                            repl_dict[s] = sr[0][0]
                offset.append(r[0].subs(repl_dict))

            if ignore_indices:
                tmp_memlet = Memlet.simple(parent_name, rng)
                use_dst = True if access_type == 'w' else False
                for s, r in self.symbols.items():
                    tmp_memlet = propagate_subset([tmp_memlet], parent_array, [s], r, use_dst=use_dst)

            to_squeeze_rng = rng
            if ignore_indices:
                to_squeeze_rng = rng.offset_new(offset, True)
            squeezed_rng = copy.deepcopy(to_squeeze_rng)
            non_squeezed = squeezed_rng.squeeze(ignore_indices)
            # TODO: Need custom shape computation here
            shape = squeezed_rng.size()
            for i, sr in zip(ignore_indices, sym_rng):
                iMin, iMax, step = sr.ranges[0]
                if (step < 0) == True:
                    iMin, iMax, step = iMax, iMin, -step
                ts = to_squeeze_rng.tile_sizes[i]
                sqz_idx = squeezed_rng.ranges.index(to_squeeze_rng.ranges[i])
                shape[sqz_idx] = ts * sympy.ceiling(((iMax.approx if isinstance(iMax, symbolic.SymExpr) else iMax) + 1 -
                                                     (iMin.approx if isinstance(iMin, symbolic.SymExpr) else iMin)) /
                                                    (step.approx if isinstance(step, symbolic.SymExpr) else step))
            # squeezed_rng = copy.deepcopy(rng)
            # non_squeezed = squeezed_rng.squeeze()
            # shape = squeezed_rng.size()
            if non_squeezed:
                strides = [parent_array.strides[d] for d in non_squeezed]
            else:
                strides = [1]
        dtype = parent_array.dtype

        if arr_type is None:
            arr_type = type(parent_array)
        if arr_type == data.Scalar:
            self.sdfg.add_scalar(var_name, dtype)
        elif arr_type in (data.Array, data.View):
            self.sdfg.add_array(var_name, shape, dtype, strides=strides)
        elif arr_type == data.Stream:
            self.sdfg.add_stream(var_name, dtype)
        else:
            raise NotImplementedError("Data type {} is not implemented".format(arr_type))

        self.accesses[(name, rng, access_type)] = (var_name, squeezed_rng)

        inner_indices = set()
        for n, r in reversed(list(enumerate(squeezed_rng))):
            if r == rng[n]:
                inner_indices.add(n)

        if access_type == 'r':
            if _subset_has_indirection(rng):
                self.sdfg_inputs[var_name] = (dace.Memlet.from_array(parent_name, parent_array), inner_indices)
            else:
                self.sdfg_inputs[var_name] = (dace.Memlet.simple(parent_name, rng), inner_indices)
        else:
            if _subset_has_indirection(rng):
                self.sdfg_outputs[var_name] = (dace.Memlet.from_array(parent_name, parent_array), inner_indices)
            else:
                self.sdfg_outputs[var_name] = (dace.Memlet.simple(parent_name, rng), inner_indices)

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
            # TODO: Does the TaskletTransformer need the double slice fix?
            new_name, new_rng = self._add_access(name, rng, 'r', target, new_name, arr_type)
            return (new_name, new_rng)
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

    def _get_range(self, node: Union[ast.Name, ast.Subscript, ast.Call], name: str):
        if isinstance(node, ast.Name):
            actual_node = copy.copy(node)
            actual_node.id = name
            expr: MemletExpr = ParseMemlet(self, {**self.sdfg.arrays, **self.scope_arrays, **self.defined}, actual_node)
            rng = expr.subset
        elif isinstance(node, ast.Subscript):
            actual_node = copy.copy(node)
            if isinstance(actual_node.value, ast.Call):
                actual_node.value = copy.copy(actual_node.value)
                actual_node.value.func = copy.copy(actual_node.value.func)
                actual_node.value.func.id = name
            else:
                actual_node.value = copy.copy(actual_node.value)
                actual_node.value.id = name
            expr: MemletExpr = ParseMemlet(self, {**self.sdfg.arrays, **self.scope_arrays, **self.defined}, actual_node)
            rng = expr.subset
        elif isinstance(node, ast.Call):
            rng = dace.subsets.Range.from_array({**self.sdfg.arrays, **self.scope_arrays}[name])
        else:
            raise NotImplementedError

        if isinstance(rng, subsets.Indices):
            rng = subsets.Range.from_indices(rng)

        return rng

    def _update_names(self, node: Union[ast.Name, ast.Subscript, ast.Call], name: str, name_subscript: bool = False):
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
                    squeezed_rng = None
                    if self.nested:
                        real_name = variables[name]
                        rng = self._get_range(target, real_name)
                        name, squeezed_rng = self._add_read_access(name, rng, target)
                        if squeezed_rng is not None:
                            name_sub = True
                    else:
                        if name in variables:
                            name = variables[name]
                    node.value.right = self._update_names(node.value.right, name, name_subscript=name_sub)
                    connector, memlet = parse_memlet(self, node.value.right, node.value.left, self.sdfg.arrays)
                    # Fix memlet with correct subset
                    if squeezed_rng is not None:
                        # TODO: Fix for `contains_sympy_functions`
                        # not liking ints
                        if isinstance(squeezed_rng, subsets.Indices):
                            memlet.subset = subsets.Range([(symbolic.pystr_to_symbolic(i),
                                                            symbolic.pystr_to_symbolic(i), 1)
                                                           for i in squeezed_rng.indices])
                        else:
                            memlet.subset = subsets.Range([
                                (symbolic.pystr_to_symbolic(b), symbolic.pystr_to_symbolic(e),
                                 symbolic.pystr_to_symbolic(s)) for b, e, s in squeezed_rng.ranges
                            ])
                    if self.nested and _subset_has_indirection(rng):
                        memlet = dace.Memlet.simple(memlet.data, rng)
                    if connector in self.inputs or connector in self.outputs:
                        raise DaceSyntaxError(self, node, 'Local variable is already a tasklet input or output')
                    self.inputs[connector] = memlet
                    return None  # Remove from final tasklet code
                elif isinstance(node.value.op, ast.RShift):
                    squeezed_rng = None
                    if self.nested:
                        real_name = variables[name]
                        rng = self._get_range(target, real_name)
                        name, squeezed_rng = self._add_write_access(name, rng, target)
                        if squeezed_rng is not None:
                            name_sub = True
                    else:
                        if name in variables:
                            name = variables[name]
                    node.value.right = self._update_names(node.value.right, name, name_subscript=name_sub)
                    connector, memlet = parse_memlet(self, node.value.left, node.value.right, self.sdfg.arrays)
                    # Fix memlet with correct subset
                    if squeezed_rng is not None:
                        # TODO: Fix for `contains_sympy_functions`
                        # not liking ints
                        if isinstance(squeezed_rng, subsets.Indices):
                            memlet.subset = subsets.Range([(symbolic.pystr_to_symbolic(i),
                                                            symbolic.pystr_to_symbolic(i), 1)
                                                           for i in squeezed_rng.indices])
                        else:
                            memlet.subset = subsets.Range([
                                (symbolic.pystr_to_symbolic(b), symbolic.pystr_to_symbolic(e),
                                 symbolic.pystr_to_symbolic(s)) for b, e, s in squeezed_rng.ranges
                            ])
                    if self.nested and _subset_has_indirection(rng):
                        memlet = dace.Memlet.simple(memlet.data, rng)
                    if self.nested and name in self.sdfg_outputs:
                        out_memlet = self.sdfg_outputs[name][0]
                        out_memlet.volume = memlet.volume
                        out_memlet.dynamic = memlet.dynamic
                        out_memlet.wcr = memlet.wcr
                        out_memlet.wcr_nonatomic = memlet.wcr_nonatomic
                    if connector in self.inputs or connector in self.outputs:
                        raise DaceSyntaxError(self, node, 'Local variable is already a tasklet input or output')
                    self.outputs[connector] = memlet
                    return None  # Remove from final tasklet code
        elif isinstance(node.value, ast.Str):
            return self.visit_TopLevelStr(node.value)

        return self.generic_visit(node)

    # Detect external tasklet code
    def visit_TopLevelStr(self, node: ast.Str):
        if self.extcode != None:
            raise DaceSyntaxError(self, node, 'Cannot provide more than one intrinsic implementation ' + 'for tasklet')
        self.extcode = node.s

        # TODO: Should get detected by _parse_Tasklet()
        if self.lang is None:
            self.lang = dtypes.Language.CPP

        return node

    def visit_Name(self, node: ast.Name):
        # If accessing a symbol, add it to the SDFG symbol list
        if (isinstance(node.ctx, ast.Load) and node.id in self.defined
                and isinstance(self.defined[node.id], symbolic.symbol)):
            if node.id not in self.sdfg.symbols:
                self.sdfg.add_symbol(node.id, self.defined[node.id].dtype)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        # Parsed objects are not allowed to be called from tasklets
        if hasattr(node.func, 'n') and isinstance(node.func.n, SDFGConvertible):
            node.func = node.func.oldnode.func

        fname = rname(node.func)
        if fname in self.defined:
            ftype = self.defined[fname].dtype
            if isinstance(ftype, dtypes.callback):
                if not ftype.is_scalar_function():
                    raise DaceSyntaxError(
                        self, node, 'Python callbacks that return arrays are not supported'
                        ' within `dace.tasklet` scopes. Please use function '
                        f'"{fname}" outside of a tasklet.')
        if fname in self.visitor.closure.callbacks:
            # TODO(later): When type/shape inference dives into tasklets
            raise DaceSyntaxError(
                self, node, 'Automatic Python callbacks are not yet '
                'supported within tasklets. Please define function '
                f'"{fname}" as a `dace.callback` explicitly and input it '
                'as a keyword argument to the function. Example:\n'
                '  addfunc = dace.symbol("addfunc", dace.callback(dace.float32, dace.float32, dace.float32))\n'
                '  @dace.program\n'
                '  def myprogram(...):\n'
                '    with dace.tasklet:\n'
                '      # ...\n'
                '      c = addfunc(a, b)\n'
                '  myprogram(..., addfunc=add)')
        return self.generic_visit(node)


class ProgramVisitor(ExtNodeVisitor):
    """ A visitor that traverses a data-centric Python program AST and
        constructs an SDFG.
    """
    progress_bar = None
    start_time: float = 0

    def __init__(self,
                 name: str,
                 filename: str,
                 line_offset: int,
                 col_offset: int,
                 global_vars: Dict[str, Any],
                 constants: Dict[str, Any],
                 scope_arrays: Dict[str, data.Data],
                 scope_vars: Dict[str, str],
                 map_symbols: Set[Union[str, symbolic.symbol]] = None,
                 annotated_types: Dict[str, data.Data] = None,
                 closure: SDFGClosure = None,
                 nested: bool = False,
                 tmp_idx: int = 0,
                 simplify: Optional[bool] = None):
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
            closure {SDFGClosure} -- The closure of this program
            simplify {bool} -- Whether to apply simplification pass after parsing nested dace programs

        Keyword Arguments:
            nested {bool} -- True, if SDFG is nested (default: {False})
            tmp_idx {int} -- First idx for tmp transient names (default: {0})
        """

        self.filename = filename
        self.src_line = line_offset
        self.src_col = col_offset
        self.orig_name = name
        if nested:
            self.name = "{n}_{l}_{c}".format(n=name, l=line_offset, c=col_offset)
        else:
            self.name = name

        self.globals = global_vars
        self.closure = closure
        self.nested = nested
        self.simplify = simplify

        # Keeps track of scope arrays, numbers, variables and accesses
        self.scope_arrays = OrderedDict()
        self.scope_arrays.update(scope_arrays)
        self.scope_vars = {k: k for k in scope_arrays.keys()}
        self.scope_vars.update(scope_vars)
        self.numbers = dict()  # Dict[str, str]
        self.variables = dict()  # Dict[str, str]
        self.accesses = dict()
        self.views: Dict[str, Tuple[str, Memlet]] = {}  # Keeps track of views
        self.nested_closure_arrays: Dict[str, Tuple[Any, data.Data]] = {}
        self.annotated_types: Dict[str, data.Data] = annotated_types or {}

        # Keep track of map symbols from upper scopes
        map_symbols = map_symbols or set()
        self.map_symbols = set()
        self.map_symbols.update(map_symbols)

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

        self.inputs: DependencyType = {}
        self.outputs: DependencyType = {}
        self.current_lineinfo = dtypes.DebugInfo(line_offset, col_offset, line_offset, col_offset, filename)

        self.modules = {k: v.__name__ for k, v in self.globals.items() if dtypes.ismodule(v)}

        # Add constants
        for cstname, cstval in constants.items():
            self.sdfg.add_constant(cstname, cstval)

        # Add symbols
        for arr in scope_arrays.values():
            self.scope_vars.update({str(k): self.globals[str(k)] for k in arr.free_symbols})

        # Disallow keywords
        for stmt in _DISALLOWED_STMTS:
            setattr(self, 'visit_' + stmt, lambda n: _disallow_stmt(self, n))

        # Loop status
        self.loop_idx = -1
        self.continue_states = []
        self.break_states = []

        # Tmp fix for missing state symbol propagation
        self.symbols = dict()

        # Indirections
        self.indirections = dict()

    @classmethod
    def progress_count(cls) -> int:
        """ Returns the number of parsed SDFGs so far within this run. """
        if cls.progress_bar is None:
            return 0
        if isinstance(cls.progress_bar, tuple):
            return cls.progress_bar[0]
        else:
            return cls.progress_bar.n

    @classmethod
    def increment_progress(cls, number=1):
        """ Adds a number of parsed SDFGs to the progress bar (whether visible or not). """
        if cls.progress_bar is not None:
            if isinstance(cls.progress_bar, tuple):
                i, t = cls.progress_bar
                cls.progress_bar = (i + number, t)
            else:
                cls.progress_bar.update(number)

    def visit(self, node: ast.AST):
        """Visit a node."""
        if hasattr(node, 'lineno'):
            self.current_lineinfo = dtypes.DebugInfo(node.lineno, node.col_offset, node.lineno, node.col_offset,
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

        # Set parents for nodes to access assignments from Calls
        program = astutils.AnnotateTopLevel().visit(program)
        self.program_ast = program

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
                if isinstance(self.sdfg.arrays[arrname], data.View):
                    # In case of a view, make a copy
                    # NOTE: If we are at the top level SDFG (not always clear),
                    # and it is a View of an input array, can we return a NumPy
                    # View directly?
                    desc = self.sdfg.arrays[arrname]
                    return_state = self._add_state()
                    r = return_state.add_read(arrname)
                    w = return_state.add_write(vname)
                    if vname not in self.sdfg.arrays:
                        self.sdfg.add_array(
                            vname,
                            desc.shape,
                            desc.dtype,
                            storage=desc.storage,
                            transient=False,
                            # NOTE: It seems that NumPy doesn't support creating
                            # non-contiguous arrays directly.
                            # strides=desc.strides,
                            # offset=desc.offset,
                            debuginfo=desc.debuginfo,
                            # total_size=desc.total_size,
                            allow_conflicts=desc.allow_conflicts)
                    return_state.add_nedge(r, w, Memlet(vname))
                else:
                    # Other cases can be replaced with return value directly
                    self.sdfg.replace(arrname, vname)
                    for k, (v, m) in self.views.items():
                        if v == arrname:
                            m.data = vname
                            self.views[k] = (vname, m)

        ####

        for arrname, arr in self.sdfg.arrays.items():
            # Return values become non-transient (accessible by the outside)
            if arrname.startswith('__return'):
                arr.transient = False
                self.outputs[arrname] = (None, Memlet.from_array(arrname, arr), [])

        def _views_to_data(state: SDFGState, nodes: List[dace.nodes.AccessNode]) -> List[dace.nodes.AccessNode]:
            new_nodes = []
            for vnode in nodes:
                if vnode.data in self.views:
                    if state.in_degree(vnode) == 0:
                        aname, m = self.views[vnode.data]
                        arr = self.sdfg.arrays[aname]
                        r = state.add_read(aname)
                        state.add_edge(r, None, vnode, 'views', copy.deepcopy(m))
                        new_nodes.append(r)
                    elif state.out_degree(vnode) == 0:
                        aname, m = self.views[vnode.data]
                        arr = self.sdfg.arrays[aname]
                        w = state.add_write(aname)
                        state.add_edge(vnode, 'views', w, None, copy.deepcopy(m))
                        new_nodes.append(w)
                    else:
                        raise ValueError(f'View "{vnode.data}" already has both incoming and outgoing edges')
            return new_nodes

        # Map view access nodes to their respective data
        for state in self.sdfg.nodes():
            # NOTE: We need to support views of views
            nodes = list(state.data_nodes())
            while nodes:
                nodes = _views_to_data(state, nodes)

        # Try to replace transients with their python-assigned names
        for pyname, arrname in self.variables.items():
            if arrname in self.sdfg.arrays:
                if self.sdfg.arrays[arrname].transient:
                    if (pyname and dtypes.validate_name(pyname) and pyname not in self.sdfg.arrays):
                        self.sdfg.replace(arrname, pyname)

        propagate_states(self.sdfg)
        for state, memlet, inner_indices in itertools.chain(self.inputs.values(), self.outputs.values()):
            if state is not None and state.dynamic_executions:
                memlet.dynamic = True

        return self.sdfg, self.inputs, self.outputs, self.symbols

    @property
    def defined(self):
        # Check parent SDFG arrays first
        result = {}
        result.update({k: v for k, v in self.globals.items() if isinstance(v, symbolic.symbol)})
        result.update({k: self.sdfg.arrays[v] for k, v in self.scope_vars.items() if v in self.sdfg.arrays})
        result.update({k: self.scope_arrays[v] for k, v in self.scope_vars.items() if v in self.scope_arrays})
        result.update({k: self.sdfg.arrays[v] for k, v in self.variables.items() if v in self.sdfg.arrays})
        result.update({v: self.sdfg.arrays[v] for _, v in self.variables.items() if v in self.sdfg.arrays})
        # TODO: Is there a case of a variable-symbol?
        result.update({k: self.sdfg.symbols[v] for k, v in self.variables.items() if v in self.sdfg.symbols})

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
            # TODO: Refactor to return proper symbols and not strings.
            rng = dace.subsets.Range(astutils.subscript_to_slice(arg, self.sdfg.arrays)[1])
            repldict = dict()
            for sname in rng.free_symbols:
                if sname in self.defined:
                    repldict[sname] = self.defined[sname]
            if repldict:
                rng.replace(repldict)
            result = rng.string_list()
            if as_list is False and len(result) == 1:
                return result[0]
            return result

        return arg

    def _decorator_or_annotation_params(self, node: ast.FunctionDef) -> List[Tuple[str, Any]]:
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
            args = [self._parse_arg(arg.annotation, as_list=False) for arg in node.args.args]

        result = [(rname(arg), argval) for arg, argval in zip(node.args.args, args)]

        # Ensure all arguments are annotated
        if len(result) != len(node.args.args):
            raise DaceSyntaxError(self, node, 'All arguments in primitive %s must be annotated' % node.name)
        return result

    def _parse_subprogram(self, name, node, is_tasklet=False, extra_symbols=None, extra_map_symbols=None):
        extra_symbols = extra_symbols or {}
        extra_map_symbols = extra_map_symbols or set()
        map_symbols = self.map_symbols.union(extra_map_symbols)
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
                            map_symbols=map_symbols,
                            annotated_types=self.annotated_types,
                            closure=self.closure,
                            nested=True,
                            tmp_idx=self.sdfg._temp_transients + 1)

        return pv.parse_program(node, is_tasklet)

    def _symbols_from_params(self, params: List[Tuple[str, Union[str, dtypes.typeclass]]],
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
            if val.data in self.sdfg.arrays:
                datatype = self.sdfg.arrays[val.data].dtype
            else:
                datatype = self.scope_arrays[val.data].dtype
            dyn_inputs[name] = symbolic.symbol(name, datatype)
        result.update(dyn_inputs)

        for name, val in params:
            if isinstance(val, dtypes.typeclass):
                result[name] = symbolic.symbol(name, dtype=val)
            else:
                values = str(val).split(':')
                if len(values) == 1:
                    result[name] = symbolic.symbol(name, infer_expr_type(values[0], {**self.globals, **dyn_inputs}))
                elif len(values) == 2:
                    result[name] = symbolic.symbol(
                        name,
                        dtypes.result_type_of(infer_expr_type(values[0], {
                            **self.globals,
                            **dyn_inputs
                        }), infer_expr_type(values[1], {
                            **self.globals,
                            **dyn_inputs
                        })))
                elif len(values) == 3:
                    result[name] = symbolic.symbol(name, infer_expr_type(values[0], {**self.globals, **dyn_inputs}))
                else:
                    raise DaceSyntaxError(
                        self, None, "Invalid number of arguments in a range iterator. "
                        "You may use up to 3 arguments (start:stop:step).")

        return result

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Supported decorated function types: map, mapscope, consume,
        # consumescope, tasklet, program

        if len(node.decorator_list) > 1:
            raise DaceSyntaxError(self, node, 'Exactly one DaCe decorator is allowed on a function')
        if len(node.decorator_list) == 0:
            dec = 'dace.tasklet'
        else:
            dec_ast = node.decorator_list[0]
            dec_ast = preprocessing.ModuleResolver(self.modules, True).visit(dec_ast)
            dec = rname(dec_ast)

        # Create a new state for the statement
        state = self._add_state("s{l}_{c}".format(l=node.lineno, c=node.col_offset))

        # Define internal node for reconnection
        internal_node = None

        # Select primitive according to function type
        if dec == 'dace.tasklet':  # Tasklet
            internal_node, inputs, outputs, sdfg_inp, sdfg_out = self._parse_tasklet(state, node)

            # Add memlets
            inputs = {k: (state, v, set()) for k, v in inputs.items()}
            outputs = {k: (state, v, set()) for k, v in outputs.items()}

            self._add_dependencies(state, internal_node, None, None, inputs, outputs)
            self.inputs.update({k: (state, *v) for k, v in sdfg_inp.items()})
            self.outputs.update({k: (state, *v) for k, v in sdfg_out.items()})

        elif dec.startswith('dace.map') or dec.startswith('dace.consume'):  # Scope or scope+tasklet
            if 'map' in dec:
                params = self._decorator_or_annotation_params(node)
                params, map_inputs = self._parse_map_inputs(node.name, params, node)
                map_symbols = self._symbols_from_params(params, map_inputs)
                entry, exit = state.add_map(node.name, ndrange=params, debuginfo=self.current_lineinfo)
            elif 'consume' in dec:
                (stream_name, stream_elem, PE_tuple, condition, chunksize) = self._parse_consume_inputs(node)
                params = [PE_tuple, (stream_elem, self.sdfg.arrays[stream_name].dtype)]
                map_inputs = {}
                map_symbols = set()
                entry, exit = state.add_consume(node.name,
                                                PE_tuple,
                                                condition,
                                                chunksize=chunksize,
                                                debuginfo=self.current_lineinfo)

            if dec.endswith('scope'):  # @dace.mapscope or @dace.consumescope
                # TODO: Now that we return the nested for-loop symbols,
                # can we use them for something here?
                sdfg, inputs, outputs, _ = self._parse_subprogram(node.name,
                                                                  node,
                                                                  extra_symbols=self._symbols_from_params(
                                                                      params, map_inputs),
                                                                  extra_map_symbols=map_symbols)
            else:  # Scope + tasklet (e.g., @dace.map)
                name = "{}_body".format(entry.label)
                # TODO: Now that we return the nested for-loop symbols,
                # can we use them for something here?
                sdfg, inputs, outputs, _ = self._parse_subprogram(name,
                                                                  node,
                                                                  True,
                                                                  extra_symbols=self._symbols_from_params(
                                                                      params, map_inputs),
                                                                  extra_map_symbols=map_symbols)

            internal_node = state.add_nested_sdfg(sdfg,
                                                  self.sdfg,
                                                  set(inputs.keys()),
                                                  set(outputs.keys()),
                                                  debuginfo=self.current_lineinfo)
            self._add_nested_symbols(internal_node)

            # If consume scope, inject stream inputs to the internal SDFG
            if 'consume' in dec:
                free_symbols_before = copy.copy(sdfg.free_symbols)
                self._inject_consume_memlets(dec, entry, inputs, internal_node, sdfg, state, stream_elem, stream_name)
                # Remove symbols defined after injection
                syms_to_remove = free_symbols_before - sdfg.free_symbols
                syms_to_remove.add(stream_elem)
                for sym in syms_to_remove:
                    del internal_node.symbol_mapping[sym]
                    del sdfg.symbols[sym]

            # Connect internal node with scope/access nodes
            self._add_dependencies(state, internal_node, entry, exit, inputs, outputs, map_inputs)

        elif dec == 'dace.program':  # Nested SDFG
            raise DaceSyntaxError(self, node, 'Nested programs must be defined outside existing programs')
        else:
            raise DaceSyntaxError(self, node, 'Unsupported function decorator')

    def _inject_consume_memlets(self, dec, entry, inputs, internal_node, sdfg, state, stream_elem, stream_name):
        """ Inject stream inputs to subgraph when creating a consume scope. """

        # Inject element to internal SDFG arrays
        ntrans = sdfg.temp_data_name()
        sdfg.add_array(ntrans, [1], self.sdfg.arrays[stream_name].dtype)
        internal_memlet = dace.Memlet.simple(ntrans, subsets.Indices([0]))
        external_memlet = dace.Memlet.simple(stream_name, subsets.Indices([0]), num_accesses=-1)

        # Inject to internal tasklet
        if not dec.endswith('scope'):
            injected_node_count = 0
            for s in sdfg.nodes():
                for n in s.nodes():
                    if isinstance(n, nodes.Tasklet):
                        n.add_in_connector(stream_elem)
                        rnode = s.add_read(ntrans, debuginfo=self.current_lineinfo)
                        s.add_edge(rnode, None, n, stream_elem, internal_memlet)
                        injected_node_count += 1
            assert injected_node_count == 1

        # Inject to nested SDFG node
        internal_node.add_in_connector(ntrans)
        stream_node = state.add_read(stream_name, debuginfo=self.current_lineinfo)
        state.add_edge_pair(entry,
                            internal_node,
                            stream_node,
                            external_memlet,
                            scope_connector='stream',
                            internal_connector=ntrans)

        # Mark as input so that no extra edges are added
        inputs[ntrans] = (state, None, set())

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
            raise DaceSyntaxError(self, node, "Target of ast.For must be a name or a tuple")

        if isinstance(node, ast.Name):
            elts = (node, )
        else:
            elts = node.elts

        indices = []
        for n in elts:
            if not isinstance(n, ast.Name):
                raise DaceSyntaxError(self, n, "For-loop iterator must be ast.Name")
            idx_id = n.id
            if idx_id in indices:
                raise DaceSyntaxError(self, n, "Duplicate index id ({}) in for-loop".format(idx_id))
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
            return str(self.visit(node))

    def _parse_slice(self, node: ast.Slice):
        """Parses a range

        Arguments:
            node {ast.Slice} -- Slice node

        Returns:
            Tuple[str] -- Range in (from, to, step) format
        """

        return (self._parse_value(node.lower), self._parse_value(node.upper),
                self._parse_value(node.step) if node.step is not None else "1")

    def _parse_index_as_range(self, node: Union[ast.Index, ast.Tuple]):
        """
        Parses an index as range
        :param node: Index node
        :return: Range in (from, to, step) format
        """
        if isinstance(node, ast.Index):
            val = self._parse_value(node.value)
        elif isinstance(node, ast.Tuple):
            val = self._parse_value(node.elts)
        else:
            val = self._parse_value(node)
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
            Tuple[str, List[str], List[ast.AST]] -- Iterator type, iteration 
                                                    ranges, and AST versions of
                                                    the ranges
        """

        if not isinstance(node, (ast.Call, ast.Subscript)):
            raise DaceSyntaxError(self, node, "Iterator of ast.For must be a function or a subscript")

        iter_name = preprocessing.ModuleResolver(self.modules, True).visit(node)
        iterator = rname(iter_name)

        ast_ranges = []

        if iterator not in {'range', 'prange', 'parrange', 'dace.map'}:
            raise DaceSyntaxError(self, node, "Iterator {} is unsupported".format(iterator))
        elif iterator in ['range', 'prange', 'parrange']:
            # AST nodes for common expressions
            zero = ast.parse('0').body[0]
            one = ast.parse('1').body[0]

            def visit_ast_or_value(arg):
                ast_res = self._visit_ast_or_value(arg)
                val_res = self._parse_value(ast_res)
                return val_res, ast_res

            if len(node.args) == 1:  # (par)range(stop)
                valr, astr = visit_ast_or_value(node.args[0])
                ranges = [('0', valr, '1')]
                ast_ranges = [(zero, astr, one)]
            elif len(node.args) == 2:  # (par)range(start, stop)
                valr0, astr0 = visit_ast_or_value(node.args[0])
                valr1, astr1 = visit_ast_or_value(node.args[1])
                ranges = [(valr0, valr1, '1')]
                ast_ranges = [(astr0, astr1, one)]
            elif len(node.args) == 3:  # (par)range(start, stop, step)
                valr0, astr0 = visit_ast_or_value(node.args[0])
                valr1, astr1 = visit_ast_or_value(node.args[1])
                valr2, astr2 = visit_ast_or_value(node.args[2])
                ranges = [(valr0, valr1, valr2)]
                ast_ranges = [(astr0, astr1, astr2)]
            else:
                raise DaceSyntaxError(self, node, 'Invalid number of arguments for "%s"' % iterator)
            if iterator in ('prange', 'parrange'):
                iterator = 'dace.map'
        else:
            ranges = []
            if isinstance(node.slice, (ast.Tuple, ast.ExtSlice)):
                for s in node.slice.dims:
                    ranges.append(self._parse_slice(s))
            elif isinstance(node.slice, ast.Slice):
                ranges.append(self._parse_slice(node.slice))
            else:  # isinstance(node.slice, ast.Index) is True
                ranges.append(self._parse_index_as_range(node.slice))

        return (iterator, ranges, ast_ranges)

    def _parse_map_inputs(self, name: str, params: List[Tuple[str, str]],
                          node: ast.AST) -> Tuple[Dict[str, str], Dict[str, Memlet]]:
        """ Parse map parameters for data-dependent inputs, modifying the
            parameter dictionary and returning relevant memlets.
            :return: A 2-tuple of (parameter dictionary, mapping from connector
                     name to memlet).
        """
        new_params = []
        map_inputs = {}
        ctr = 0
        for k, v in params:
            vsp = list(v.split(':'))
            for i, (val, vid) in enumerate(zip(vsp, 'best')):
                # Walk through expression, find functions and replace with
                # variables
                repldict = {}
                symval = pystr_to_symbolic(val)

                for atom in symval.free_symbols:
                    if symbolic.issymbolic(atom, self.sdfg.constants):
                        # Check for undefined variables
                        atomstr = str(atom)
                        if atomstr not in self.defined:
                            raise DaceSyntaxError(self, node, 'Undefined variable "%s"' % atom)
                        # Add to global SDFG symbols

                        # If scalar, should add dynamic map connector
                        candidate = atomstr
                        if candidate in self.variables and self.variables[candidate] in self.sdfg.arrays:
                            candidate = self.variables[candidate]

                        if candidate in self.sdfg.arrays and isinstance(self.sdfg.arrays[candidate], data.Scalar):
                            newvar = '__%s_%s%d' % (name, vid, ctr)
                            repldict[atomstr] = newvar
                            map_inputs[newvar] = Memlet.from_array(candidate, self.sdfg.arrays[candidate])
                            ctr += 1
                        elif candidate not in self.sdfg.symbols:
                            self.sdfg.add_symbol(atomstr, self.defined[candidate].dtype)

                for expr in symbolic.swalk(symval):
                    if symbolic.is_sympy_userfunction(expr):
                        # If function contains a function
                        if any(symbolic.contains_sympy_functions(a) for a in expr.args):
                            raise DaceSyntaxError(self, node, 'Indirect accesses not supported in map ranges')
                        arr = expr.func.__name__
                        newvar = '__%s_%s%d' % (name, vid, ctr)
                        repldict[arr] = newvar
                        # Create memlet
                        args = ','.join([str(a) for a in expr.args])
                        if arr in self.variables:
                            arr = self.variables[arr]
                        if not isinstance(arr, str) or arr not in self.sdfg.arrays:
                            rng = subsets.Range.from_string(args)
                            args = str(rng)
                        map_inputs[newvar] = Memlet.simple(arr, args)
                        # ','.join([str(a) for a in expr.args]))
                        ctr += 1
                # Replace functions with new variables
                for find, replace in repldict.items():
                    val = re.sub(r"%s\(.*?\)" % find, val, replace)
                vsp[i] = val

            new_params.append((k, ':'.join(vsp)))

        return new_params, map_inputs

    def _parse_consume_inputs(self, node: ast.FunctionDef) -> Tuple[str, str, Tuple[str, str], str, str]:
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
            raise DaceSyntaxError(self, node, 'Consume scope decorator must contain at least two arguments')

        # Parse function
        if len(node.args.args) != 2:
            raise DaceSyntaxError(self, node, 'Consume scope function must contain two arguments')

        stream_elem, PE_index = tuple(a.arg for a in node.args.args)

        return (stream_name, stream_elem, (PE_index, num_PEs), condition, chunksize)

    def _find_access(self, name: str, rng: subsets.Range, mode: str):
        for n, r, m in self.accesses:
            if n == name and m == mode:
                if r == rng:
                    return True
                elif r.covers(rng):
                    print("WARNING: New access {n}[{rng}] already covered by {n}[{r}]".format(n=name, rng=rng, r=r))
                elif rng.covers(r):
                    print("WARNING: New access {n}[{rng}] covers previous access {n}[{r}]".format(n=name, rng=rng, r=r))
                return False

    def _get_array_or_closure(self, name: str) -> data.Data:
        if name in self.sdfg.arrays:
            return self.sdfg.arrays[name]
        elif name in self.scope_arrays:
            return self.scope_arrays[name]
        elif name in self.closure.closure_arrays:
            return self.closure.closure_arrays[name][1]

        raise NameError(f'Array "{name}" not found in outer scope or closure')

    def _add_dependencies(self,
                          state: SDFGState,
                          internal_node: nodes.CodeNode,
                          entry_node: nodes.EntryNode,
                          exit_node: nodes.ExitNode,
                          inputs: DependencyType,
                          outputs: DependencyType,
                          map_inputs: Dict[str, Memlet] = None,
                          symbols: Dict[str, 'dace.symbol'] = dict()):

        # Parse map inputs (for memory-based ranges)
        if map_inputs:
            for conn, memlet in map_inputs.items():
                if self.nested:
                    # TODO: Make this work nested for-loops
                    new_name, _ = self._add_read_access(memlet.data, memlet.subset, None)
                    memlet = Memlet.from_array(new_name, self.sdfg.arrays[new_name])
                else:
                    new_name = memlet.data

                read_node = state.add_read(new_name, debuginfo=self.current_lineinfo)
                entry_node.add_in_connector(conn)
                state.add_edge(read_node, None, entry_node, conn, memlet)

        # Parse internal node inputs and indirect memory accesses
        if inputs:
            for conn, v in inputs.items():
                inner_state, memlet_or_node, inner_indices = v
                if memlet_or_node is None:  # Input already handled outside
                    continue

                if isinstance(memlet_or_node, nodes.Tasklet):
                    tasklet: nodes.Tasklet = memlet_or_node
                    # Create a code->code node
                    new_scalar = self.sdfg.temp_data_name()
                    if isinstance(internal_node, nodes.NestedSDFG):
                        dtype = internal_node.sdfg.arrays[conn].dtype
                    else:
                        raise SyntaxError('Cannot determine connector type for tasklet input dependency')
                    self.sdfg.add_scalar(new_scalar, dtype, transient=True)
                    accessnode = state.add_access(new_scalar)
                    state.add_edge(tasklet, conn, accessnode, None, dace.Memlet.simple(new_scalar, '0'))
                    state.add_edge(accessnode, None, internal_node, conn, dace.Memlet.simple(new_scalar, '0'))
                    if entry_node is not None:
                        state.add_edge(entry_node, None, tasklet, None, dace.Memlet())
                    continue

                memlet: Memlet = copy.deepcopy(memlet_or_node)

                arr = self._get_array_or_closure(memlet.data)

                for s, r in symbols.items():
                    memlet = propagate_subset([memlet], arr, [s], r, use_dst=False, defined_variables=set())
                if _subset_has_indirection(memlet.subset, self):
                    read_node = entry_node
                    if entry_node is None:
                        read_node = state.add_read(memlet.data, debuginfo=self.current_lineinfo)
                    add_indirection_subgraph(self.sdfg, state, read_node, internal_node, memlet, conn, self)
                    continue
                if memlet.data not in self.sdfg.arrays:
                    if entry_node:
                        scope_memlet = propagate_memlet(state, memlet, entry_node, True, arr)
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
                    # Avoid the case where all indices are outer,
                    # i.e., the whole array is carried through the nested SDFG levels.
                    if len(outer_indices) < len(irng) or irng.num_elements() == 1:
                        irng.pop(outer_indices)
                        orng.pop(outer_indices)
                        irng.offset(orng, True)
                    if (memlet.data, scope_memlet.subset, 'w') in self.accesses:
                        vname = self.accesses[(memlet.data, scope_memlet.subset, 'w')][0]
                        memlet = Memlet.simple(vname, str(irng))
                    elif (memlet.data, scope_memlet.subset, 'r') in self.accesses:
                        vname = self.accesses[(memlet.data, scope_memlet.subset, 'r')][0]
                        memlet = Memlet.simple(vname, str(irng))
                    elif memlet.data in self.closure.closure_arrays:
                        self.sdfg.add_datadesc(memlet.data, copy.deepcopy(arr))
                        vname = memlet.data
                        self.inputs[vname] = (state, memlet, [])
                    else:
                        name = memlet.data
                        vname = "{c}_in_from_{s}{n}".format(c=conn,
                                                            s=self.sdfg.nodes().index(state),
                                                            n=('_%s' % state.node_id(entry_node) if entry_node else ''))
                        self.accesses[(name, scope_memlet.subset, 'r')] = (vname, orng)
                        orig_shape = orng.size()
                        shape = [d for i, d in enumerate(orig_shape) if d != 1 or i in inner_indices]
                        strides = [i for j, i in enumerate(arr.strides) if j not in outer_indices]
                        strides = [
                            s for i, (d, s) in enumerate(zip(orig_shape, strides)) if d != 1 or i in inner_indices
                        ]
                        if not shape:
                            shape = [1]
                            strides = [1]
                        # TODO: Formulate this better
                        if not strides:
                            strides = [arr.strides[-1]]
                        dtype = arr.dtype
                        if isinstance(arr, data.Stream):
                            self.sdfg.add_stream(vname, dtype)
                        else:
                            self.sdfg.add_array(vname, shape, dtype, strides=strides)
                        self.inputs[vname] = (state, scope_memlet, inner_indices)
                        memlet.data = vname
                        # memlet.subset.offset(memlet.subset, True, outer_indices)
                else:
                    vname = memlet.data

                read_node = state.add_read(vname, debuginfo=self.current_lineinfo)

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
                inner_state, memlet, inner_indices = v
                if memlet is None:  # Output already handled outside
                    continue

                arr = self._get_array_or_closure(memlet.data)

                for s, r in symbols.items():
                    memlet = propagate_subset([memlet], arr, [s], r, use_dst=True, defined_variables=set())
                if _subset_has_indirection(memlet.subset, self):
                    write_node = exit_node
                    if exit_node is None:
                        write_node = state.add_write(memlet.data, debuginfo=self.current_lineinfo)
                    add_indirection_subgraph(self.sdfg, state, internal_node, write_node, memlet, conn, self, True)
                    continue
                inner_memlet = memlet
                if memlet.data not in self.sdfg.arrays:
                    if entry_node:
                        scope_memlet = propagate_memlet(state, memlet, entry_node, True, arr)
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
                    # Avoid the case where all indices are outer,
                    # i.e., the whole array is carried through the nested SDFG levels.
                    if len(outer_indices) < len(irng) or irng.num_elements() == 1:
                        irng.pop(outer_indices)
                        orng.pop(outer_indices)
                        irng.offset(orng, True)
                    if self._find_access(memlet.data, scope_memlet.subset, 'w'):
                        vname = self.accesses[(memlet.data, scope_memlet.subset, 'w')][0]
                        inner_memlet = Memlet.simple(vname, str(irng))
                        inner_memlet.num_accesses = memlet.num_accesses
                        inner_memlet.dynamic = memlet.dynamic
                    elif memlet.data in self.closure.closure_arrays:
                        self.sdfg.add_datadesc(memlet.data, copy.deepcopy(arr))
                        vname = memlet.data
                        self.outputs[vname] = (state, memlet, [])
                    else:
                        name = memlet.data
                        vname = "{c}_out_of_{s}{n}".format(c=conn,
                                                           s=self.sdfg.nodes().index(state),
                                                           n=('_%s' % state.node_id(exit_node) if exit_node else ''))
                        self.accesses[(name, scope_memlet.subset, 'w')] = (vname, orng)
                        orig_shape = orng.size()
                        shape = [d for d in orig_shape if d != 1]
                        shape = [d for i, d in enumerate(orig_shape) if d != 1 or i in inner_indices]
                        strides = [i for j, i in enumerate(arr.strides) if j not in outer_indices]
                        strides = [
                            s for i, (d, s) in enumerate(zip(orig_shape, strides)) if d != 1 or i in inner_indices
                        ]
                        if not shape:
                            shape = [1]
                            strides = [1]
                        # TODO: Formulate this better
                        if not strides:
                            strides = [arr.strides[-1]]
                        dtype = arr.dtype
                        if isinstance(arr, data.Stream):
                            self.sdfg.add_stream(vname, dtype)
                        else:
                            self.sdfg.add_array(vname, shape, dtype, strides=strides)
                        self.outputs[vname] = (state, scope_memlet, inner_indices)
                        inner_memlet.data = vname
                        # memlet.subset.offset(memlet.subset, True, outer_indices)
                else:
                    vname = memlet.data
                write_node = state.add_write(vname, debuginfo=self.current_lineinfo)
                if exit_node is not None:
                    state.add_memlet_path(internal_node,
                                          exit_node,
                                          write_node,
                                          memlet=inner_memlet,
                                          src_conn=conn,
                                          dst_conn=None)
                else:
                    state.add_edge(internal_node, conn, write_node, None, inner_memlet)
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
                if sym.name not in self.sdfg.symbols:
                    if (sym.name in self.globals and isinstance(self.globals[sym.name], symbolic.symbol)):
                        self.sdfg.add_symbol(sym.name, self.globals[sym.name].dtype)
                    elif sym.name in self.closure.callbacks:
                        self.sdfg.add_symbol(sym.name, nsdfg_node.sdfg.symbols[sym.name])

    def _recursive_visit(self, body: List[ast.AST], name: str, lineno: int, last_state=True, extra_symbols=None):
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

    def _replace_with_global_symbols(self, expr: sympy.Expr) -> sympy.Expr:
        repldict = dict()
        for s in expr.free_symbols:
            if s.name in self.defined:
                repldict[s] = self.defined[s.name]
        return expr.subs(repldict)

    def visit_For(self, node: ast.For):
        # We allow three types of for loops:
        # 1. `for i in range(...)`: Creates a looping state
        # 2. `for i in parrange(...)`: Creates a 1D map
        # 3. `for i,j,k in dace.map[0:M, 0:N, 0:K]`: Creates an ND map
        # print(ast.dump(node))
        indices = self._parse_for_indices(node.target)
        iterator, ranges, ast_ranges = self._parse_for_iterator(node.iter)

        if len(indices) != len(ranges):
            raise DaceSyntaxError(self, node, "Number of indices and ranges of for-loop do not match")

        if iterator == 'dace.map':
            if node.orelse:
                raise DaceSyntaxError(self, node, '"else" clause not supported on DaCe maps')

            state = self._add_state('MapState')
            params = [(k, ':'.join([str(t) for t in v])) for k, v in zip(indices, ranges)]
            params, map_inputs = self._parse_map_inputs('map_%d' % node.lineno, params, node)
            me, mx = state.add_map(name='%s_%d' % (self.name, node.lineno),
                                   ndrange=params,
                                   debuginfo=self.current_lineinfo)
            # body = SDFG('MapBody')
            body, inputs, outputs, symbols = self._parse_subprogram(
                self.name,
                node,
                extra_symbols=self._symbols_from_params(params, map_inputs),
                extra_map_symbols=self._symbols_from_params(params, map_inputs))
            tasklet = state.add_nested_sdfg(body,
                                            self.sdfg,
                                            inputs.keys(),
                                            outputs.keys(),
                                            debuginfo=self.current_lineinfo)
            self._add_nested_symbols(tasklet)
            self._add_dependencies(state, tasklet, me, mx, inputs, outputs, map_inputs, symbols)
        elif iterator == 'range':
            # Create an extra typed symbol for the loop iterate
            from dace.codegen.tools.type_inference import infer_expr_type

            sym_name = indices[0]
            integer = True
            nonnegative = None
            positive = None

            start = self._replace_with_global_symbols(symbolic.pystr_to_symbolic(ranges[0][0]))
            stop = self._replace_with_global_symbols(symbolic.pystr_to_symbolic(ranges[0][1]))
            step = self._replace_with_global_symbols(symbolic.pystr_to_symbolic(ranges[0][2]))
            eoff = -1
            if (step < 0) == True:
                eoff = 1
            try:
                conditions = [s >= 0 for s in (start, stop, step)]
                if (conditions == [True, True, True] or (start > stop and step < 0)):
                    nonnegative = True
                    if start != 0:
                        positive = True
            except:
                pass

            sym_obj = symbolic.symbol(indices[0],
                                      dtypes.result_type_of(infer_expr_type(ranges[0][0], self.sdfg.symbols),
                                                            infer_expr_type(ranges[0][1], self.sdfg.symbols),
                                                            infer_expr_type(ranges[0][2], self.sdfg.symbols)),
                                      integer=integer,
                                      nonnegative=nonnegative,
                                      positive=positive)

            # TODO: What if two consecutive loops use the same symbol
            # but different ranges?
            if sym_name in self.sdfg.symbols.keys():
                for k, v in self.symbols.items():
                    if (str(k) == sym_name and v != subsets.Range([(start, stop + eoff, step)])):
                        warnings.warn("Two for-loops using the same variable ({}) but "
                                      "different ranges in the same nested SDFG level. "
                                      "This may work but is not officially supported."
                                      "".format(sym_name))
                        break
            else:
                self.sdfg.add_symbol(sym_name, sym_obj.dtype)

            extra_syms = {sym_name: sym_obj}

            self.symbols[sym_obj] = subsets.Range([(start, stop + eoff, step)])

            # Add range symbols as necessary
            for rng in ranges[0]:
                symrng = pystr_to_symbolic(rng)
                for atom in symrng.free_symbols:
                    if symbolic.issymbolic(atom, self.sdfg.constants):
                        astr = str(atom)
                        # Check for undefined variables
                        if astr not in self.defined:
                            raise DaceSyntaxError(self, node, 'Undefined variable "%s"' % atom)
                        # Add to global SDFG symbols if not a scalar
                        if (astr not in self.sdfg.symbols and not (astr in self.variables or astr in self.sdfg.arrays)):
                            self.sdfg.add_symbol(astr, atom.dtype)

            # Add an initial loop state with a None last_state (so as to not
            # create an interstate edge)
            self.loop_idx += 1
            self.continue_states.append([])
            self.break_states.append([])
            laststate, first_loop_state, last_loop_state = self._recursive_visit(node.body,
                                                                                 'for',
                                                                                 node.lineno,
                                                                                 extra_symbols=extra_syms)
            end_loop_state = self.last_state

            # Add loop to SDFG
            loop_cond = '>' if ((pystr_to_symbolic(ranges[0][2]) < 0) == True) else '<'
            incr = {indices[0]: '%s + %s' % (indices[0], astutils.unparse(ast_ranges[0][2]))}
            _, loop_guard, loop_end = self.sdfg.add_loop(
                laststate, first_loop_state, end_loop_state, indices[0], astutils.unparse(ast_ranges[0][0]),
                '%s %s %s' % (indices[0], loop_cond, astutils.unparse(ast_ranges[0][1])), incr[indices[0]],
                last_loop_state)

            # Handle else clause
            if node.orelse:
                # Continue visiting body
                for stmt in node.orelse:
                    self.visit(stmt)

                # The state that all "break" edges go to
                loop_end = self._add_state(f'postloop_{node.lineno}')

            body_states = set()
            to_visit = [first_loop_state]
            while to_visit:
                state = to_visit.pop(0)
                for _, dst, _ in self.sdfg.out_edges(state):
                    if dst not in body_states and dst is not loop_guard:
                        to_visit.append(dst)
                body_states.add(state)
            
            continue_states = self.continue_states.pop()
            while continue_states:
                next_state = continue_states.pop()
                out_edges = self.sdfg.out_edges(next_state)
                for e in out_edges:
                    self.sdfg.remove_edge(e)
                self.sdfg.add_edge(next_state, loop_guard, dace.InterstateEdge(assignments=incr))
            break_states = self.break_states.pop()
            while break_states:
                next_state = break_states.pop()
                out_edges = self.sdfg.out_edges(next_state)
                for e in out_edges:
                    self.sdfg.remove_edge(e)
                self.sdfg.add_edge(next_state, loop_end, dace.InterstateEdge())
            self.loop_idx -= 1

            for state in body_states:
                if not nx.has_path(self.sdfg.nx, loop_guard, state):
                    for e in self.sdfg.all_edges(state):
                        self.sdfg.remove_edge(e)
                    self.sdfg.remove_node(state)
        else:
            raise DaceSyntaxError(self, node, 'Unsupported for-loop iterator "%s"' % iterator)

    def _is_test_simple(self, node: ast.AST):
        # Fix for scalar promotion tests
        # TODO: Maybe those tests should use the SDFG API instead of the
        # Python frontend which can change how it handles conditions.
        simple_ast_nodes = (ast.Constant, ast.Name, ast.NameConstant, ast.Num)
        is_test_simple = isinstance(node, simple_ast_nodes)
        if not is_test_simple:
            if isinstance(node, ast.Compare):
                is_left_simple = isinstance(node.left, simple_ast_nodes)
                is_right_simple = (len(node.comparators) == 1 and isinstance(node.comparators[0], simple_ast_nodes))
                if is_left_simple and is_right_simple:
                    return True
            elif isinstance(node, ast.BoolOp):
                return all(self._is_test_simple(value) for value in node.values)
        return is_test_simple

    def _visit_test(self, node: ast.Expr):
        is_test_simple = self._is_test_simple(node)

        # Visit test-condition
        if not is_test_simple:
            parsed_node = self.visit(node)
            if isinstance(parsed_node, str) and parsed_node in self.sdfg.arrays:
                datadesc = self.sdfg.arrays[parsed_node]
                if isinstance(datadesc, data.Array):
                    parsed_node += '[0]'
        else:
            parsed_node = astutils.unparse(node)

        # Generate conditions
        cond = astutils.unparse(parsed_node)
        cond_else = astutils.unparse(astutils.negate_expr(parsed_node))

        return cond, cond_else

    def visit_While(self, node: ast.While):
        # Get loop condition expression
        begin_guard = self._add_state("while_guard")
        loop_cond, _ = self._visit_test(node.test)
        end_guard = self.last_state

        # Parse body
        self.loop_idx += 1
        self.continue_states.append([])
        self.break_states.append([])
        laststate, first_loop_state, last_loop_state = \
            self._recursive_visit(node.body, 'while', node.lineno)
        end_loop_state = self.last_state

        assert (laststate == end_guard)

        # Add symbols from test as necessary
        symcond = pystr_to_symbolic(loop_cond)
        if symbolic.issymbolic(symcond):
            for atom in symcond.free_symbols:
                if symbolic.issymbolic(atom, self.sdfg.constants):
                    astr = str(atom)
                    # Check for undefined variables
                    if astr not in self.defined:
                        raise DaceSyntaxError(self, node, 'Undefined variable "%s"' % atom)
                    # Add to global SDFG symbols if not a scalar
                    if (astr not in self.sdfg.symbols and astr not in self.variables):
                        self.sdfg.add_symbol(astr, atom.dtype)

        # Add loop to SDFG
        _, loop_guard, loop_end = self.sdfg.add_loop(laststate, first_loop_state, end_loop_state, None, None, loop_cond,
                                                     None, last_loop_state)

        # Connect the correct while-guard state
        # Current state:
        # begin_guard -> ... -> end_guard/laststate -> loop_guard -> first_loop
        # Desired state:
        # begin_guard -> ... -> end_guard/laststate -> first_loop
        for e in list(self.sdfg.in_edges(loop_guard)):
            if e.src != laststate:
                self.sdfg.add_edge(e.src, begin_guard, e.data)
            self.sdfg.remove_edge(e)
        for e in list(self.sdfg.out_edges(loop_guard)):
            self.sdfg.add_edge(end_guard, e.dst, e.data)
            self.sdfg.remove_edge(e)
        self.sdfg.remove_node(loop_guard)

        # Handle else clause
        if node.orelse:
            # Continue visiting body
            for stmt in node.orelse:
                self.visit(stmt)

            # The state that all "break" edges go to
            loop_end = self._add_state(f'postwhile_{node.lineno}')
        
        body_states = set()
        to_visit = [first_loop_state]
        while to_visit:
            state = to_visit.pop(0)
            for _, dst, _ in self.sdfg.out_edges(state):
                if dst not in body_states and dst is not loop_guard:
                    to_visit.append(dst)
            body_states.add(state)

        continue_states = self.continue_states.pop()
        while continue_states:
            next_state = continue_states.pop()
            out_edges = self.sdfg.out_edges(next_state)
            for e in out_edges:
                self.sdfg.remove_edge(e)
            self.sdfg.add_edge(next_state, begin_guard, dace.InterstateEdge())
        break_states = self.break_states.pop()
        while break_states:
            next_state = break_states.pop()
            out_edges = self.sdfg.out_edges(next_state)
            for e in out_edges:
                self.sdfg.remove_edge(e)
            self.sdfg.add_edge(next_state, loop_end, dace.InterstateEdge())
        self.loop_idx -= 1

        for state in body_states:
            if not nx.has_path(self.sdfg.nx, end_guard, state):
                for e in self.sdfg.all_edges(state):
                    self.sdfg.remove_edge(e)
                self.sdfg.remove_node(state)

    def visit_Break(self, node: ast.Break):
        if self.loop_idx < 0:
            error_msg = "'break' is only supported inside for and while loops "
            if self.nested:
                error_msg += ("('break' is not supported in Maps and cannot be "
                              " used in nested DaCe program calls to break out "
                              " of loops of outer scopes)")
            raise DaceSyntaxError(self, node, error_msg)
        self.break_states[self.loop_idx].append(self.last_state)

    def visit_Continue(self, node: ast.Continue):
        if self.loop_idx < 0:
            error_msg = ("'continue' is only supported inside for and while loops ")
            if self.nested:
                error_msg += ("('continue' is not supported in Maps and cannot "
                              " be used in nested DaCe program calls to "
                              " continue loops of outer scopes)")
            raise DaceSyntaxError(self, node, error_msg)
        self.continue_states[self.loop_idx].append(self.last_state)

    def visit_If(self, node: ast.If):
        # Add a guard state
        self._add_state('if_guard')

        # Generate conditions
        cond, cond_else = self._visit_test(node.test)

        # Visit recursively
        laststate, first_if_state, last_if_state = \
            self._recursive_visit(node.body, 'if', node.lineno)
        end_if_state = self.last_state

        # Connect the states
        self.sdfg.add_edge(laststate, first_if_state, dace.InterstateEdge(cond))
        self.sdfg.add_edge(last_if_state, end_if_state, dace.InterstateEdge())

        # Process 'else'/'elif' statements
        if len(node.orelse) > 0:
            # Visit recursively
            _, first_else_state, last_else_state = \
                self._recursive_visit(node.orelse, 'else', node.lineno, False)

            # Connect the states
            self.sdfg.add_edge(laststate, first_else_state, dace.InterstateEdge(cond_else))
            self.sdfg.add_edge(last_else_state, end_if_state, dace.InterstateEdge())
            self.last_state = end_if_state
        else:
            self.sdfg.add_edge(laststate, end_if_state, dace.InterstateEdge(cond_else))

    def _parse_tasklet(self, state: SDFGState, node: TaskletType, name=None):

        # Looking for the first argument in a tasklet annotation: @dace.tasklet(STRING HERE)
        langInf = None
        if isinstance(node, ast.FunctionDef) and \
            hasattr(node, 'decorator_list') and \
            isinstance(node.decorator_list, list) and \
            len(node.decorator_list) > 0 and \
            hasattr(node.decorator_list[0], 'args') and \
            isinstance(node.decorator_list[0].args, list) and \
            len(node.decorator_list[0].args) > 0 and \
            hasattr(node.decorator_list[0].args[0], 'value'):

            langArg = node.decorator_list[0].args[0].value
            langInf = dtypes.Language[langArg]

        ttrans = TaskletTransformer(self,
                                    self.defined,
                                    self.sdfg,
                                    state,
                                    self.filename,
                                    lang=langInf,
                                    nested=self.nested,
                                    scope_arrays=self.scope_arrays,
                                    scope_vars=self.scope_vars,
                                    variables=self.variables,
                                    accesses=self.accesses,
                                    symbols=self.symbols)
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
                        op: Optional[str] = None,
                        boolarr: Optional[str] = None):
        # TODO: Refactor these if/else blocks. Maybe
        # the subset should never be None?
        if isinstance(target, tuple):
            target_name, target_subset = target
            if target_subset is None:
                target_array = self.sdfg.arrays[target_name]
                target_subset = subsets.Range.from_array(target_array)
        else:
            target_name = target
            target_array = self.sdfg.arrays[target_name]
            target_subset = subsets.Range.from_array(target_array)
        if isinstance(operand, tuple):
            op_name, op_subset = operand
            if op_subset is None:
                op_array = self.sdfg.arrays[op_name]
                op_subset = subsets.Range.from_array(op_array)
        elif isinstance(operand, str) and operand in self.sdfg.arrays:
            op_name = operand
            op_array = self.sdfg.arrays[op_name]
            op_subset = subsets.Range.from_array(op_array)
        else:
            op_name = None
            op_array = None
            op_subset = subsets.Range([(0, 0, 1)])
            if symbolic.issymbolic(operand):
                for sym in operand.free_symbols:
                    if str(sym) not in self.sdfg.symbols:
                        self.sdfg.add_symbol(str(sym), self.globals[str(sym)].dtype)
                operand = symbolic.symstr(operand)

        tasklet_code = ''
        input_memlets = {}
        if target_subset.num_elements() != 1:
            target_index = ','.join(['__i%d' % i for i in range(len(target_subset))])
        else:
            target_index = '0'

        # Handle boolean array access
        if boolarr is not None:
            if isinstance(boolarr, str) and boolarr in self.sdfg.arrays:  # Array
                input_memlets['__in_cond'] = Memlet(f'{boolarr}[{target_index}]')
                tasklet_code += 'if __in_cond:\n    '
            else:  # Constant
                tasklet_code += f'if {boolarr}[{target_index}]:\n    '

        state = self._add_state("assign_{l}_{c}".format(l=node.lineno, c=node.col_offset))

        if target_subset.num_elements() != 1:
            if op_subset.num_elements() != 1:
                squeezed = copy.deepcopy(target_subset)
                squeezed.squeeze(offset=False)
                squeezed_op = copy.deepcopy(op_subset)
                squeezed_op.squeeze(offset=False)

                ssize = squeezed.size()
                osize = squeezed_op.size()

                if len(ssize) != len(osize) or any(inequal_symbols(s, o) for s, o in zip(ssize, osize)) or op:

                    _, all_idx_tuples, _, _, inp_idx = _broadcast_to(squeezed.size(), op_subset.size())

                    idx = iter(i for i, _ in all_idx_tuples)
                    target_index = ','.join(
                        next(idx) if size != 1 else str(target_subset.ranges[i][0])
                        for i, size in enumerate(target_subset.size()))

                    inp_idx = inp_idx.split(',')
                    # create a fake subset that would be the input subset broadcasted to the correct size
                    missing_dimensions = squeezed.ranges[:len(all_idx_tuples) - len(inp_idx)]
                    op_dimensions = op_subset.ranges

                    fake_subset = dace.subsets.Range(missing_dimensions + op_dimensions)

                    # use this fake subset to calculate the offset
                    fake_subset.offset(squeezed, True)

                    # we access the inp subset using the computed offset
                    # since the inp_subset may be missing leading dimensions, we reverse-zip-reverse
                    idx_and_subset = reversed(list(zip(reversed(inp_idx), reversed(fake_subset))))

                    inp_memlet = Memlet("{a}[{s}]".format(a=op_name,
                                                          s=','.join(
                                                              [f'{idx} + {s}' for idx, (s, _, _) in idx_and_subset])))
                    out_memlet = Memlet("{a}[{s}]".format(a=target_name, s=target_index))
                    if op:
                        out_memlet.wcr = LambdaProperty.from_string('lambda x, y: x {} y'.format(op))

                    if boolarr is not None:
                        inp_memlet.dynamic = True
                        out_memlet.dynamic = True

                    tasklet_code += '__out = __inp'
                    state.add_mapped_tasklet(state.label, {
                        '__i%d' % i: '%s:%s+1:%s' % (start, end, step)
                        for i, (start, end, step) in enumerate(squeezed)
                    }, {
                        '__inp': inp_memlet,
                        **input_memlets
                    },
                                             tasklet_code, {'__out': out_memlet},
                                             external_edges=True,
                                             debuginfo=self.current_lineinfo)

                else:
                    if boolarr is not None:
                        raise NotImplementedError

                    op1 = state.add_read(op_name, debuginfo=self.current_lineinfo)
                    op2 = state.add_write(target_name, debuginfo=self.current_lineinfo)
                    memlet = Memlet("{a}[{s}]".format(a=target_name, s=target_subset))
                    memlet.other_subset = op_subset
                    if op:
                        memlet.wcr = LambdaProperty.from_string('lambda x, y: x {} y'.format(op))
                    state.add_nedge(op1, op2, memlet)
            else:
                memlet = Memlet("{a}[{s}]".format(a=target_name,
                                                  s=','.join(['__i%d' % i for i in range(len(target_subset))])))
                if op:
                    memlet.wcr = LambdaProperty.from_string('lambda x, y: x {} y'.format(op))
                if op_name:
                    inp_memlet = {'__inp': Memlet("{a}[{s}]".format(a=op_name, s=op_subset))}
                    tasklet_code += '__out = __inp'
                else:
                    inp_memlet = dict()
                    tasklet_code += '__out = {}'.format(operand)

                if boolarr is not None:
                    for m in inp_memlet.values():
                        m.dynamic = True
                    memlet.dynamic = True

                inp_memlet.update(input_memlets)
                state.add_mapped_tasklet(state.label, {
                    '__i%d' % i: '%s:%s+1:%s' % (start, end, step)
                    for i, (start, end, step) in enumerate(target_subset)
                },
                                         inp_memlet,
                                         tasklet_code, {'__out': memlet},
                                         external_edges=True,
                                         debuginfo=self.current_lineinfo)
        else:
            if op_subset.num_elements() != 1:
                raise DaceSyntaxError(self, node, "Incompatible subsets %s and %s" % (target_subset, op_subset))
            if op_name:
                op1 = state.add_read(op_name, debuginfo=self.current_lineinfo)
                inp_conn = {'__inp'}
                tasklet_code += '__out = __inp'
            else:
                inp_conn = set()
                tasklet_code += '__out = {}'.format(operand)
            inp_conn |= set(input_memlets.keys())
            op2 = state.add_write(target_name, debuginfo=self.current_lineinfo)
            tasklet = state.add_tasklet(name=state.label,
                                        inputs=inp_conn,
                                        outputs={'__out'},
                                        code=tasklet_code,
                                        debuginfo=self.current_lineinfo)
            if op_name:
                inp_memlet = Memlet.simple(op_name, '%s' % op_subset)
                if boolarr is not None:
                    inp_memlet.dynamic = True
                state.add_edge(op1, None, tasklet, '__inp', inp_memlet)

            out_memlet = Memlet.simple(target_name, '%s' % target_subset)
            if boolarr is not None:
                out_memlet.dynamic = True

            for cname, memlet in input_memlets.items():
                r = state.add_read(memlet.data)
                state.add_edge(r, None, tasklet, cname, memlet)

            if op:
                out_memlet.wcr = LambdaProperty.from_string('lambda x, y: x {} y'.format(op))

            state.add_edge(tasklet, '__out', op2, None, out_memlet)

    def _add_aug_assignment(self,
                            node: Union[ast.Assign, ast.AugAssign],
                            rtarget: Union[str, Tuple[str, subsets.Range]],
                            wtarget: Union[str, Tuple[str, subsets.Range]],
                            operand: Union[str, Tuple[str, subsets.Range]],
                            op: str,
                            boolarr: Optional[str] = None):

        # TODO: Refactor these if/else blocks. Maybe
        # the subset should never be None?
        if isinstance(rtarget, tuple):
            rtarget_name, rtarget_subset = rtarget
            if rtarget_subset is None:
                rtarget_array = self.sdfg.arrays[rtarget_name]
                rtarget_subset = subsets.Range.from_array(rtarget_array)
        else:
            rtarget_name = rtarget
            rtarget_array = self.sdfg.arrays[rtarget_name]
            rtarget_subset = subsets.Range.from_array(rtarget_array)
        if isinstance(wtarget, tuple):
            wtarget_name, wtarget_subset = wtarget
            if wtarget_subset is None:
                wtarget_array = self.sdfg.arrays[wtarget_name]
                wtarget_subset = subsets.Range.from_array(wtarget_array)
        else:
            wtarget_name = wtarget
            wtarget_array = self.sdfg.arrays[wtarget_name]
            wtarget_subset = subsets.Range.from_array(wtarget_array)
        if isinstance(operand, tuple):
            op_name, op_subset = operand
            if op_subset is None:
                op_array = self.sdfg.arrays[op_name]
                op_subset = subsets.Range.from_array(op_array)
        elif isinstance(operand, str) and operand in self.sdfg.arrays:
            op_name = operand
            op_array = self.sdfg.arrays[op_name]
            op_subset = subsets.Range.from_array(op_array)
        else:
            op_name = None
            op_array = None
            op_subset = subsets.Range([(0, 0, 1)])
            if symbolic.issymbolic(operand):
                for sym in operand.free_symbols:
                    if str(sym) not in self.sdfg.symbols:
                        self.sdfg.add_symbol(str(sym), self.globals[str(sym)].dtype)
                operand = symbolic.symstr(operand)

        tasklet_code = ''
        input_memlets = {}
        if wtarget_subset.num_elements() != 1:
            wtarget_index = ','.join(['__i%d' % i for i in range(len(wtarget_subset))])
        else:
            wtarget_index = '0'

        # Handle boolean array access
        if boolarr is not None:
            if isinstance(boolarr, str) and boolarr in self.sdfg.arrays:  # Array
                input_memlets['__in_cond'] = Memlet(f'{boolarr}[{wtarget_index}]')
                tasklet_code += 'if __in_cond:\n    '
            else:  # Constant
                tasklet_code += f'if {boolarr}[{wtarget_index}]:\n    '

        state = self._add_state("augassign_{l}_{c}".format(l=node.lineno, c=node.col_offset))

        if wtarget_subset.num_elements() != 1:
            if op_subset.num_elements() != 1:
                sqz_osub = copy.deepcopy(op_subset)
                oidx = sqz_osub.squeeze()
                sqz_wsub = copy.deepcopy(wtarget_subset)
                widx = sqz_wsub.squeeze()
                sqz_rsub = copy.deepcopy(rtarget_subset)
                ridx = sqz_rsub.squeeze()
                if (sqz_wsub.size() == sqz_osub.size() and sqz_wsub.size() == sqz_rsub.size()):
                    r_to_w = {i: j for i, j in zip(ridx, widx)}
                    o_to_w = {i: j for i, j in zip(oidx, widx)}
                    # NOTE: Since 'sqz_wsub is squeezed, 'start' should be
                    # equal to 0
                    map_range = {
                        f'__i{widx[i]}': f'{start}:{end} + 1:{step}'
                        for i, (start, end, step) in enumerate(sqz_wsub)
                    }
                    in1_memlet = Memlet.simple(
                        rtarget_name, ','.join([
                            f'__i{r_to_w[i]} + {s}' if i in ridx else str(s)
                            for i, (s, _, _) in enumerate(rtarget_subset)
                        ]))
                    in2_memlet = Memlet.simple(
                        op_name, ','.join([
                            f'__i{o_to_w[i]} + {s}' if i in oidx else str(s) for i, (s, _, _) in enumerate(op_subset)
                        ]))
                    out_memlet = Memlet.simple(
                        wtarget_name, ','.join(
                            [f'__i{i} + {s}' if i in widx else str(s) for i, (s, _, _) in enumerate(wtarget_subset)]))
                    if boolarr is not None:
                        in1_memlet.dynamic = True
                        out_memlet.dynamic = True
                    tasklet_code += '__out = __in1 {op} __in2'.format(op=op)
                    state.add_mapped_tasklet(state.label,
                                             map_range, {
                                                 '__in1': in1_memlet,
                                                 '__in2': in2_memlet,
                                                 **input_memlets,
                                             },
                                             tasklet_code, {'__out': out_memlet},
                                             external_edges=True,
                                             debuginfo=self.current_lineinfo)
                else:
                    if boolarr is not None:
                        raise NotImplementedError
                    op1 = state.add_read(op_name, debuginfo=self.current_lineinfo)
                    op2 = state.add_write(wtarget_name, debuginfo=self.current_lineinfo)
                    memlet = Memlet.simple(wtarget_name, wtarget_subset)
                    memlet.other_subset = op_subset
                    if op is not None:
                        memlet.wcr = LambdaProperty.from_string('lambda x, y: x {} y'.format(op))
                        memlet.wcr_nonatomic = True
                    state.add_nedge(op1, op2, memlet)
            else:
                in1_subset = copy.deepcopy(rtarget_subset)
                in1_subset.offset(wtarget_subset, True)
                in1_memlet = Memlet.simple(rtarget_name,
                                           ','.join(['__i%d + %s' % (i, s) for i, (s, _, _) in enumerate(in1_subset)]))
                if op_name:
                    in2_memlet = Memlet.simple(op_name, '%s' % op_subset)
                    inp_memlets = {'__in1': in1_memlet, '__in2': in2_memlet}
                    tasklet_code += '__out = __in1 {op} __in2'.format(op=op)
                else:
                    inp_memlets = {'__in1': in1_memlet}
                    tasklet_code += '__out = __in1 {op} {n}'.format(op=op, n=operand)
                inp_memlets.update(input_memlets)
                out_memlet = Memlet.simple(wtarget_name, ','.join(['__i%d' % i for i in range(len(wtarget_subset))]))
                if boolarr is not None:
                    in1_memlet.dynamic = True
                    out_memlet.dynamic = True
                state.add_mapped_tasklet(state.label, {
                    '__i%d' % i: '%s:%s+1:%s' % (start, end, step)
                    for i, (start, end, step) in enumerate(wtarget_subset)
                },
                                         inp_memlets,
                                         tasklet_code, {'__out': out_memlet},
                                         external_edges=True,
                                         debuginfo=self.current_lineinfo)
        else:
            if op_subset.num_elements() != 1:
                raise DaceSyntaxError(
                    self, node, "Incompatible subsets %s, %s and %s" % (rtarget_subset, op_subset, wtarget_subset))
            else:
                op1 = state.add_read(rtarget_name, debuginfo=self.current_lineinfo)
                if op_name:
                    op2 = state.add_read(op_name, debuginfo=self.current_lineinfo)
                    inp_conns = {'__in1', '__in2'}
                    tasklet_code += '__out = __in1 {op} __in2'.format(op=op)
                else:
                    inp_conns = {'__in1'}
                    tasklet_code += '__out = __in1 {op} {n}'.format(op=op, n=operand)
                inp_conns |= set(input_memlets.keys())
                op3 = state.add_write(wtarget_name, debuginfo=self.current_lineinfo)
                tasklet = state.add_tasklet(name=state.label,
                                            inputs=inp_conns,
                                            outputs={'__out'},
                                            code=tasklet_code,
                                            debuginfo=self.current_lineinfo)
                in1_memlet = Memlet.simple(rtarget_name, '%s' % rtarget_subset)
                if op_name:
                    in2_memlet = Memlet.simple(op_name, '%s' % op_subset)
                    state.add_edge(op2, None, tasklet, '__in2', in2_memlet)
                for cname, memlet in input_memlets.items():
                    r = state.add_read(memlet.data)
                    state.add_edge(r, None, tasklet, cname, memlet)

                out_memlet = Memlet.simple(wtarget_name, '%s' % wtarget_subset)
                if boolarr is not None:
                    in1_memlet.dynamic = True
                    out_memlet.dynamic = True
                state.add_edge(op1, None, tasklet, '__in1', in1_memlet)
                state.add_edge(tasklet, '__out', op3, None, out_memlet)

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
            var_name = "__tmp_{l}_{c}_{a}".format(l=target.lineno, c=target.col_offset, a=access_type)
        else:
            var_name = self.sdfg.temp_data_name()

        parent_name = self.scope_vars[name]
        parent_array = self.scope_arrays[parent_name]

        has_indirection = (_subset_has_indirection(rng, self) or _subset_is_local_symbol_dependent(rng, self))
        if has_indirection:
            # squeezed_rng = list(range(len(rng)))
            shape = parent_array.shape
            # strides = [parent_array.strides[d] for d in squeezed_rng]
            # # TODO: Why is squeezed_rng an index in the first place?
            # squeezed_rng = subsets.Range([(i, i, 1) for i in squeezed_rng])
            squeezed_rng = subsets.Range.from_array(parent_array)
            non_squeezed = list(range(len(rng)))
        else:
            ignore_indices = []
            sym_rng = []
            offset = []
            for i, r in enumerate(rng):
                repl_dict = {}
                for s, sr in self.symbols.items():
                    if s in symbolic.symlist(r).values():
                        ignore_indices.append(i)
                        if any(t in self.sdfg.arrays for t in sr.free_symbols):
                            sym_rng.append(subsets.Range([(0, parent_array.shape[i] - 1, 1)]))
                        else:
                            sym_rng.append(sr)
                            # NOTE: Assume that the i-th index of the range is
                            # dependent on a local symbol s, i.e, rng[i] = f(s).
                            # Therefore, the i-th index will not be squeezed
                            # even if it has length equal to 1. However, it must
                            # still be offsetted by f(min(sr)), so that the indices
                            # for the squeezed connector start from 0.
                            # Example:
                            # Memlet range: [i+1, j, k+1]
                            # k: local symbol with range(1, 4)
                            # i,j: global symbols
                            # Squeezed range: [f(k)] = [k+1]
                            # Offset squeezed range: [f(k)-f(min(range(1, 4)))] =
                            #                        [f(k)-f(1)] = [k-1]
                            # NOTE: The code takes into account the case where an
                            # index is dependent on multiple symbols. See also
                            # tests/python_frontend/nested_name_accesses_test.py.
                            step = sr[0][2]
                            if (step < 0) == True:
                                repl_dict[s] = sr[0][1]
                            else:
                                repl_dict[s] = sr[0][0]
                if repl_dict:
                    offset.append(r[0].subs(repl_dict))
                else:
                    offset.append(0)

            if ignore_indices:
                tmp_memlet = Memlet.simple(parent_name, rng)
                use_dst = True if access_type == 'w' else False
                for s, r in self.symbols.items():
                    tmp_memlet = propagate_subset([tmp_memlet], parent_array, [s], r, use_dst=use_dst)
            to_squeeze_rng = rng
            if ignore_indices:
                to_squeeze_rng = rng.offset_new(offset, True)
            squeezed_rng = copy.deepcopy(to_squeeze_rng)
            non_squeezed = squeezed_rng.squeeze(ignore_indices)
            # TODO: Need custom shape computation here
            shape = squeezed_rng.size()
            for i, sr in zip(ignore_indices, sym_rng):
                iMin, iMax, step = sr.ranges[0]
                if (step < 0) == True:
                    iMin, iMax, step = iMax, iMin, -step
                ts = to_squeeze_rng.tile_sizes[i]
                sqz_idx = squeezed_rng.ranges.index(to_squeeze_rng.ranges[i])
                shape[sqz_idx] = ts * sympy.ceiling(((iMax.approx if isinstance(iMax, symbolic.SymExpr) else iMax) + 1 -
                                                     (iMin.approx if isinstance(iMin, symbolic.SymExpr) else iMin)) /
                                                    (step.approx if isinstance(step, symbolic.SymExpr) else step))
        dtype = parent_array.dtype

        if arr_type is None:
            arr_type = type(parent_array)
            # Size (1,) slice of NumPy array returns scalar value
            if arr_type != data.Stream and (shape == [1] or shape == (1, )):
                arr_type = data.Scalar
        if arr_type == data.Scalar:
            self.sdfg.add_scalar(var_name, dtype)
        elif arr_type in (data.Array, data.View):
            if non_squeezed:
                strides = [parent_array.strides[d] for d in non_squeezed]
            else:
                strides = [1]
            self.sdfg.add_array(var_name, shape, dtype, strides=strides)
        elif arr_type == data.Stream:
            self.sdfg.add_stream(var_name, dtype)
        else:
            raise NotImplementedError("Data type {} is not implemented".format(arr_type))

        self.accesses[(name, rng, access_type)] = (var_name, squeezed_rng)

        inner_indices = set(non_squeezed)

        state = self.last_state

        if access_type == 'r':
            if has_indirection:
                self.inputs[var_name] = (state, dace.Memlet.from_array(parent_name, parent_array), inner_indices)
            else:
                self.inputs[var_name] = (state, dace.Memlet.simple(parent_name, rng), inner_indices)
        else:
            if has_indirection:
                self.outputs[var_name] = (state, dace.Memlet.from_array(parent_name, parent_array), inner_indices)
            else:
                self.outputs[var_name] = (state, dace.Memlet.simple(parent_name, rng), inner_indices)

        self.variables[var_name] = var_name
        return (var_name, squeezed_rng)

    def _add_read_access(self,
                         name: str,
                         rng: subsets.Range,
                         target: Union[ast.Name, ast.Subscript],
                         new_name: str = None,
                         arr_type: data.Data = None):
        if name in self.sdfg.arrays:
            return (name, None)
        elif (name, rng, 'w') in self.accesses:
            return self.accesses[(name, rng, 'w')]
        elif (name, rng, 'r') in self.accesses:
            return self.accesses[(name, rng, 'r')]
        elif name in self.variables:
            return (self.variables[name], None)
        elif name in self.scope_vars:
            new_name, new_rng = self._add_access(name, rng, 'r', target, new_name, arr_type)
            full_rng = subsets.Range.from_array(self.sdfg.arrays[new_name])
            if (_subset_has_indirection(rng, self) or _subset_is_local_symbol_dependent(rng, self)):
                new_name, new_rng = self.make_slice(new_name, rng)
            elif full_rng != new_rng:
                new_name, new_rng = self.make_slice(new_name, new_rng)
            return (new_name, new_rng)
        else:
            raise NotImplementedError

    def _add_write_access(self,
                          name: str,
                          rng: subsets.Range,
                          target: Union[ast.Name, ast.Subscript],
                          new_name: str = None,
                          arr_type: data.Data = None):

        if name in self.sdfg.arrays:
            return (name, None)
        if (name, rng, 'w') in self.accesses:
            return self.accesses[(name, rng, 'w')]
        elif name in self.variables:
            return (self.variables[name], None)
        elif (name, rng, 'r') in self.accesses or name in self.scope_vars:
            return self._add_access(name, rng, 'w', target, new_name, arr_type)
        else:
            raise NotImplementedError

    def visit_NamedExpr(self, node):  # node : ast.NamedExpr
        self._visit_assign(node, node.target, None)

    def visit_Assign(self, node: ast.Assign):
        # Compute first target
        self._visit_assign(node, node.targets[0], None)

        # Then, for other targets make copies
        for target in node.targets[1:]:
            assign_from_first = ast.copy_location(ast.Assign(targets=[target], value=node.targets[0]), node)
            self._visit_assign(assign_from_first, target, None)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        type_name = rname(node.annotation)
        try:
            dtype = eval(astutils.unparse(node.annotation), self.globals, self.defined)
            if isinstance(dtype, data.Data):
                simple_type = dtype.dtype
            else:
                simple_type = dtype
            if not isinstance(simple_type, dtypes.typeclass):
                raise TypeError
        except:
            dtype = None
            warnings.warn('typeclass {} is not supported'.format(type_name))
        if node.value is None and dtype is not None:  # Annotating type without assignment
            self.annotated_types[rname(node.target)] = dtype
            return
        self._visit_assign(node, node.target, None, dtype=dtype)

    def _visit_assign(self, node, node_target, op, dtype=None, is_return=False):
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
                results.extend(self._gettype(n))
        else:
            results.extend(self._gettype(node.value))

        if len(results) != len(elts):
            raise DaceSyntaxError(self, node, 'Function returns %d values but %d provided' % (len(results), len(elts)))

        defined_vars = {**self.variables, **self.scope_vars}
        defined_arrays = {**self.sdfg.arrays, **self.scope_arrays}

        for target, (result, _) in zip(elts, results):

            name = rname(target)
            true_name = None
            if name in defined_vars:
                true_name = defined_vars[name]
                true_array = defined_arrays[true_name]

            # If type was already annotated
            if dtype is None and name in self.annotated_types:
                dtype = self.annotated_types[name]

            if (isinstance(target, ast.Attribute) and until(name, '.') in self.globals):
                raise DaceSyntaxError(
                    self, target, f'Cannot assign value to global attribute or field "{name}". '
                    'Please define it prior to calling the function/method.')

            if (not is_return and isinstance(target, ast.Name) and true_name and not op
                    and not isinstance(true_array, data.Scalar) and not (true_array.shape == (1, ))):
                if (isinstance(result, str) and result in self.sdfg.arrays
                        and self.sdfg.arrays[result].is_equivalent(true_array)):
                    # Skip error if the arrays are defined exactly in the same way
                    true_name = None
                else:
                    raise DaceSyntaxError(self, target, 'Cannot reassign value to variable "{}"'.format(name))

            if is_return and true_name:
                if (isinstance(result, str) and result in self.sdfg.arrays
                        and not _is_equivalent(self.sdfg.arrays[result], true_array)):
                    raise DaceSyntaxError(
                        self, target, 'Return values of a data-centric function must always '
                        'have the same type and shape')

            if not true_name and (op or isinstance(target, ast.Subscript)):
                raise DaceSyntaxError(self, target, 'Variable "{}" used before definition'.format(name))

            new_data, rng = None, None
            dtype_keys = tuple(dtypes.DTYPE_TO_TYPECLASS.keys())
            if not (symbolic.issymbolic(result) or isinstance(result, dtype_keys) or
                    (isinstance(result, str) and result in self.sdfg.arrays)):
                raise DaceSyntaxError(
                    self, node, "In assignments, the rhs may only be "
                    "data, numerical/boolean constants "
                    "and symbols")
            if not true_name:
                if (symbolic.issymbolic(result) or isinstance(result, dtype_keys)):
                    if symbolic.issymbolic(result):
                        rtype = _sym_type(result)
                    else:
                        rtype = type(result)
                    if name.startswith('__return'):
                        true_name, new_data = self.sdfg.add_temp_transient([1], rtype)
                    else:
                        true_name = self.sdfg.temp_data_name()
                        if dtype:
                            ttype = dtype
                        else:
                            ttype = rtype
                        _, new_data = self.sdfg.add_scalar(true_name, ttype, transient=True)
                    self.variables[name] = true_name
                    defined_vars[name] = true_name
                elif isinstance(result, str) and result in self.sdfg.arrays:
                    result_data = self.sdfg.arrays[result]
                    if (name.startswith('__return') and isinstance(result_data, data.Scalar)):
                        true_name, new_data = self.sdfg.add_temp_transient([1], result_data.dtype)
                        self.variables[name] = true_name
                        defined_vars[name] = true_name
                    elif (not name.startswith('__return')
                          and (isinstance(result_data, data.View) or
                               (not result_data.transient and isinstance(result_data, data.Array)))):
                        true_name, new_data = self.sdfg.add_view(result,
                                                                 result_data.shape,
                                                                 result_data.dtype,
                                                                 result_data.storage,
                                                                 result_data.strides,
                                                                 result_data.offset,
                                                                 find_new_name=True)
                        self.views[true_name] = (result, Memlet.from_array(result, result_data))
                        self.variables[name] = true_name
                        defined_vars[name] = true_name
                        continue
                    elif not result_data.transient or result in self.sdfg.constants_prop:
                        true_name, new_data = _add_transient_data(self.sdfg, result_data, dtype)
                        self.variables[name] = true_name
                        defined_vars[name] = true_name
                    else:
                        self.variables[name] = result
                        defined_vars[name] = result
                        continue

            boolarr = None
            if new_data:
                rng = rng or dace.subsets.Range.from_array(new_data)
            else:
                true_target = copy.copy(target)
                nslice = None
                if isinstance(target, ast.Name):
                    true_target.id = true_name
                elif isinstance(target, ast.Subscript):
                    true_target.value = copy.copy(true_target.value)
                    true_target.value.id = true_name

                    # Visit slice contents
                    nslice = self._parse_subscript_slice(true_target.slice)

                defined_arrays = {**self.sdfg.arrays, **self.scope_arrays, **self.defined}
                expr: MemletExpr = ParseMemlet(self, defined_arrays, true_target, nslice)
                rng = expr.subset
                if isinstance(rng, subsets.Indices):
                    rng = subsets.Range.from_indices(rng)

                # Figure out whether the target subcript is an array-index
                # indirection or a boolean array
                array_indirection = False
                if expr.arrdims:
                    for arr in expr.arrdims.values():
                        if isinstance(arr, str):
                            if arr in self.sdfg.arrays:
                                desc = self.sdfg.arrays[arr]
                                if desc.dtype == dtypes.bool:
                                    boolarr = arr
                            elif arr in self.sdfg.constants:
                                desc = self.sdfg.constants[arr]
                                if desc.dtype == numpy.bool_:
                                    boolarr = arr
                            else:
                                raise IndexError(f'Array index "{arr}" undefined')
                        elif isinstance(arr, (list, tuple)):
                            if numpy.array(arr).dtype == numpy.bool_:
                                carr = numpy.array(arrname, dtype=dtypes.typeclass(int).type)
                                cname = self.sdfg.find_new_constant(f'__ind{i}_{aname}')
                                self.sdfg.add_constant(cname, carr)
                                boolarr = cname

                        array_indirection = boolarr is None

                if array_indirection:
                    raise NotImplementedError('Array indexing as assignment target not yet implemented')
                if boolarr is not None and _subset_has_indirection(rng, self):
                    raise IndexError('Boolean array indexing cannot be combined with indirect access')

            if self.nested and not new_data:
                new_name, new_rng = self._add_write_access(name, rng, target)
                # Local symbol or local data dependent
                if _subset_is_local_symbol_dependent(rng, self):
                    new_rng = rng
            else:
                new_name, new_rng = true_name, rng

            # Self-copy check
            if result in self.views and new_name == self.views[result][1].data:
                read_rng = self.views[result][1].subset
                try:
                    needs_copy = not (new_rng.intersects(read_rng) == False)
                except TypeError:
                    needs_copy = True
                if needs_copy:
                    view = self.sdfg.arrays[result]
                    cname, carr = self.sdfg.add_transient(result, view.shape, view.dtype, find_new_name=True)
                    self._add_state(f'copy_from_view_{node.lineno}')
                    rnode = self.last_state.add_read(result, debuginfo=self.current_lineinfo)
                    wnode = self.last_state.add_read(cname, debuginfo=self.current_lineinfo)
                    self.last_state.add_nedge(rnode, wnode, Memlet.from_array(cname, carr))
                    result = cname

            # Strict independent access check for augmented assignments
            if op:
                independent = False
                if Config.get_bool('frontend', 'avoid_wcr') and not _subset_is_local_symbol_dependent(new_rng, self):
                    independent = True
                    waccess = inverse_dict_lookup(self.accesses, (new_name, new_rng))
                    if self.map_symbols and waccess:
                        for s in self.map_symbols:
                            if s not in waccess[1].free_symbols:
                                independent = False
                                break

            # Handle output indirection
            output_indirection = None
            if _subset_has_indirection(rng, self):
                output_indirection = self.sdfg.add_state('wslice_%s_%d' % (new_name, node.lineno))
                wnode = output_indirection.add_write(new_name, debuginfo=self.current_lineinfo)
                memlet = Memlet.simple(new_name, str(rng))
                # Dependent augmented assignments need WCR in the
                # indirection edge.
                with_wcr = False
                if op and not independent:
                    memlet.wcr = LambdaProperty.from_string('lambda x, y: x {} y'.format(op))
                    with_wcr = True
                    # WCR not needed in the assignment edge any longer.
                    op = None
                tmp = self.sdfg.temp_data_name()
                ind_name = add_indirection_subgraph(self.sdfg,
                                                    output_indirection,
                                                    None,
                                                    wnode,
                                                    memlet,
                                                    tmp,
                                                    self,
                                                    True,
                                                    with_wcr=with_wcr)
                wtarget = ind_name
            else:
                wtarget = (new_name, new_rng)

            # Handle augassign input indirection
            # (only needed for independent augmented assignments)
            if op and independent:
                if _subset_has_indirection(rng, self):
                    self._add_state('rslice_%s_%d' % (new_name, node.lineno))
                    rnode = self.last_state.add_read(new_name, debuginfo=self.current_lineinfo)
                    memlet = Memlet.simple(new_name, str(rng))
                    tmp = self.sdfg.temp_data_name()
                    ind_name = add_indirection_subgraph(self.sdfg, self.last_state, rnode, None, memlet, tmp, self)
                    rtarget = ind_name
                else:
                    rtarget = (new_name, new_rng)

            # Generate subgraph for assignment
            if op and independent:
                self._add_aug_assignment(node, rtarget, wtarget, result, op, boolarr)
            else:
                self._add_assignment(node, wtarget, result, op, boolarr)

            # Connect states properly when there is output indirection
            if output_indirection:
                self.sdfg.add_edge(self.last_state, output_indirection, dace.sdfg.InterstateEdge())
                self.last_state = output_indirection

    def visit_AugAssign(self, node: ast.AugAssign):
        self._visit_assign(node, node.target, augassign_ops[type(node.op).__name__])

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

        raise DaceSyntaxError(self, keywords, "Keyword {} not found".format(arg))

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
                raise DaceSyntaxError(self, node, "Attribute {} is not shape".format(rname(node)))
            shape = self.scope_arrays[node.value.id].shape
        else:
            raise DaceSyntaxError(
                self, node, "Array shape must either be a list of dimension lengths or "
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
                raise DaceSyntaxError(self, node, "Attribute {} is not dtype".format(rname(node)))
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
            dtype_node = self._get_keyword_value(node.keywords, "dtype")
            dtype = self._parse_dtype(dtype_node)
        elif num_args == 1:
            shape_node = node.args[0]
            shape = self._parse_shape(shape_node)
            dtype_node = self._get_keyword_value(node.keywords, "dtype")
            dtype = self._parse_dtype(dtype_node)
        elif num_args >= 2:
            shape_node = node.args[0]
            shape = self._parse_shape(shape_node)
            dtype_node = node.args[1]
            dtype = self._parse_dtype(dtype_node)

        return (shape, dtype)

    def _parse_function_arg(self, arg: ast.AST):
        # Obtain a string representation
        result = self.visit(arg)
        if isinstance(result, (list, tuple)):
            if len(result) == 1 and isinstance(result[0], (str, slice)):
                return result[0]
        return result

    def _is_inputnode(self, sdfg: SDFG, name: str):
        visited_data = set()
        for state in sdfg.nodes():
            visited_state_data = set()
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data == name:
                    visited_state_data.add(node.data)
                    if (node.data not in visited_data and state.in_degree(node) == 0):
                        return True
            visited_data = visited_data.union(visited_state_data)

    def _is_outputnode(self, sdfg: SDFG, name: str):
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data == name:
                    if state.in_degree(node) > 0:
                        return True

    def _get_sdfg(self, value: Any, args: Tuple[Any], kwargs: Dict[str, Any]) -> SDFG:
        if isinstance(value, SDFG):  # Already an SDFG
            return value
        if hasattr(value, '__sdfg__'):  # Object that can be converted to SDFG
            return value.__sdfg__(*args, **kwargs)
        return None

    def _has_sdfg(self, value: Any) -> bool:
        return isinstance(value, SDFG) or hasattr(value, '__sdfg__')

    def _eval_arg(self, arg: Union[str, Any]) -> Any:
        if not isinstance(arg, str):
            return arg
        if arg in self.defined:
            return self.defined[arg]
        if arg in self.sdfg.arrays:
            return self.sdfg.arrays[arg]
        if arg in self.sdfg.symbols:
            return self.sdfg.symbols[arg]
        return arg

    def _assert_arg_constant(self, node: ast.Call, aname: str, aval: Union[ast.AST, Any], parsed: Tuple[str, Any]):
        """
        Checks if given argument is constant. If not, raises a DaceSyntaxError exception.
        :param node: AST node of the call (used for exception).
        :param aname: Argument name.
        :param aval: AST (or visited) value of the argument.
        :param parsed: A 2-tuple of the parsed argument.
        :raises: DaceSyntaxError if argument is not constant.
        """
        # If constant in AST
        if sys.version_info < (3, 8):
            if isinstance(aval, (ast.Str, ast.Num, ast.Bytes, ast.NameConstant, ast.Ellipsis)):
                return
        else:
            if isinstance(aval, ast.Constant):
                return
        # If a constant value (non-AST) is given during parsing
        if not isinstance(parsed[1], str) or parsed[0] != parsed[1]:
            return

        raise DaceSyntaxError(self, node,
                              f'Argument "{aname}" was defined as dace.compiletime but was not given a constant')

    def _parse_sdfg_call(self, funcname: str, func: Union[SDFG, SDFGConvertible], node: ast.Call):
        # Avoid import loops
        from dace.frontend.python.common import SDFGConvertible
        from dace.frontend.python.parser import DaceProgram

        if func is None:
            raise TypeError('Tried to parse a None function')
        if isinstance(func, SDFG):
            sdfg = copy.deepcopy(func)
            funcname = sdfg.name
            posargs = [(aname, self._parse_function_arg(arg)) for aname, arg in zip(sdfg.arg_names, node.args)]
            kwargs = [(arg.arg, self._parse_function_arg(arg.value)) for arg in node.keywords]
            args = posargs + kwargs
            required_args = [a for a in sdfg.arglist().keys() if a not in sdfg.symbols and not a.startswith('__return')]
            all_args = required_args
            closure_arrays = {}
            self.increment_progress()
        elif isinstance(func, SDFGConvertible) or self._has_sdfg(func):
            argnames, constant_args = func.__sdfg_signature__()
            posargs = [(aname, self._parse_function_arg(arg)) for aname, arg in zip(argnames, node.args)]
            kwargs = [(arg.arg, self._parse_function_arg(arg.value)) for arg in node.keywords]
            required_args = argnames
            args = posargs + kwargs

            # Check for proper constant arguments
            for aname, arg, parsed in zip(argnames, node.args, posargs):
                if aname in constant_args:
                    self._assert_arg_constant(node, aname, arg, parsed)
            for arg, parsed in zip(node.keywords, kwargs):
                if arg.arg in constant_args:
                    self._assert_arg_constant(node, arg.arg, arg.value, parsed)

            # fcopy = copy.copy(func)
            fcopy = func
            if hasattr(fcopy, 'global_vars'):
                fcopy.global_vars = {**self.globals, **func.global_vars}

            cnt = self.progress_count()
            try:
                fargs = tuple(self._eval_arg(arg) for _, arg in posargs)
                fkwargs = {k: self._eval_arg(arg) for k, arg in kwargs}

                if isinstance(fcopy, DaceProgram):
                    fcopy.signature = copy.deepcopy(func.signature)
                    sdfg = fcopy.to_sdfg(*fargs, **fkwargs, simplify=self.simplify, save=False)
                else:
                    sdfg = fcopy.__sdfg__(*fargs, **fkwargs)

                    # Filter out parsed/omitted arguments
                    posargs = [(k, v) for k, v in posargs if k in required_args]
                    kwargs = [(k, v) for k, v in kwargs if k in required_args]
                    args = posargs + kwargs

                # Handle parsing progress bar for non-dace-program SDFG convertibles
                if cnt == self.progress_count():
                    self.increment_progress()

            except Exception as ex:  # Parsing failure
                # If error should propagate outwards, do not try to parse as callback
                if getattr(ex, '__noskipcall__', False):
                    raise

                # If parsing fails in an auto-parsed context, exit silently
                if hasattr(node.func, 'oldnode'):
                    raise SkipCall
                else:
                    # Propagate error outwards
                    if Config.get_bool('frontend', 'raise_nested_parsing_errors'):
                        ex.__noskipcall__ = True
                    raise ex

            funcname = sdfg.name
            all_args = required_args
            # Try to promote args of kind `sym = scalar`
            args = [(a, self._parse_subscript_slice(v)) if a in sdfg.symbols and str(v) not in self.sdfg.symbols else
                    (a, v) for a, v in args]
            required_args = [k for k, _ in args if k in sdfg.arg_names]
            # Filter out constant and None-constant arguments
            req = sdfg.arglist().keys()
            args = [(k, v) for k, v in args if k not in constant_args and (v is not None or k in req)]

            # Handle nested closure
            closure_arrays = getattr(fcopy, '__sdfg_closure__', lambda *args: {})()
            closure_arrays.update({k: closure_arrays[v] for k, v in sdfg.callback_mapping.items()})
            for aname, arr in closure_arrays.items():
                if aname in sdfg.symbols:
                    outer_name = self.sdfg.find_new_symbol(aname)
                    self.sdfg.add_symbol(outer_name, sdfg.symbols[aname])
                    args.append((aname, outer_name))
                    required_args.append(aname)
                    self.nested_closure_arrays[outer_name] = (arr, sdfg.symbols[aname])
                    continue

                desc = data.create_datadescriptor(arr)
                if isinstance(desc, data.Scalar) and isinstance(desc.dtype, dtypes.callback):
                    # If the symbol is a callback, but is not used in the nested SDFG, skip it
                    continue

                outer_name = self.sdfg.add_datadesc(aname, desc, find_new_name=True)
                if not desc.transient:
                    self.nested_closure_arrays[outer_name] = (arr, desc)
                    # Add closure arrays as function arguments
                    args.append((aname, outer_name))
                    required_args.append(aname)
        else:
            raise DaceSyntaxError(self, node,
                                  'Unrecognized SDFG type "%s" in call to "%s"' % (type(func).__name__, funcname))

        # Avoid import loops
        from dace.frontend.python.parser import infer_symbols_from_datadescriptor

        # Map internal SDFG symbols by adding keyword arguments
        # symbols = set(sdfg.symbols.keys())
        symbols = sdfg.free_symbols
        try:
            mapping = infer_symbols_from_datadescriptor(
                sdfg, {k: self.sdfg.arrays[v]
                       for k, v in args if v in self.sdfg.arrays},
                set(sym.arg for sym in node.keywords if sym.arg in symbols))
        except ValueError as ex:
            raise DaceSyntaxError(self, node, str(ex))
        if len(mapping) == 0:  # Default to same-symbol mapping
            mapping = None

        # Add undefined symbols to required arguments
        if mapping:
            required_args.extend([sym for sym in symbols if sym not in mapping])
        else:
            required_args.extend(symbols)
        required_args = dtypes.deduplicate(required_args)

        # Argument checks
        for aname, arg in kwargs:
            # Skip explicit return values
            if aname.startswith('__return'):
                required_args.append(aname)
                continue
            if aname not in required_args and aname not in all_args:
                raise DaceSyntaxError(self, node, 'Invalid keyword argument "%s" in call to '
                                      '"%s"' % (aname, funcname))
        if len(args) != len(required_args):
            gargs = set(a[0] for a in args)
            if len(args) > len(required_args):
                extra = set(gargs) - set(required_args)
                raise DaceSyntaxError(
                    self, node, 'Argument number mismatch in'
                    ' call to "%s" (expected %d,'
                    ' got %d). Extra arguments provided: %s' % (funcname, len(required_args), len(args), extra))
            else:
                missing = set(required_args) - set(gargs)
                raise DaceSyntaxError(
                    self, node, 'Argument number mismatch in'
                    ' call to "%s" (expected %d,'
                    ' got %d). Missing arguments: %s' % (funcname, len(required_args), len(args), missing))

        # Remove newly-defined symbols from arguments
        if mapping is not None:
            symbols -= set(mapping.keys())
        # if len(symbols) > 0:
        #     mapping = mapping or {}
        # TODO: Why above the None fix was applied when there were symbols?
        mapping = mapping or {}
        args_to_remove = []
        for i, (aname, arg) in enumerate(args):
            if aname in symbols:
                args_to_remove.append(args[i])
                mapping[aname] = arg
        for arg in args_to_remove:
            args.remove(arg)

        # Change connector names
        updated_args = []
        arrays_before = list(sdfg.arrays.items())
        names_to_replace: Dict[str, str] = {}
        for i, (conn, arg) in enumerate(args):
            if (conn in self.scope_vars or conn in self.sdfg.arrays or conn in self.sdfg.symbols):
                if self.sdfg._temp_transients > sdfg._temp_transients:
                    new_conn = self.sdfg.temp_data_name()
                else:
                    new_conn = sdfg.temp_data_name()
                # warnings.warn("Renaming nested SDFG connector {c} to "
                #               "{n}".format(c=conn, n=new_conn))
                names_to_replace[conn] = new_conn
                updated_args.append((new_conn, arg))
                # Rename the connector's Views
                for arrname, array in arrays_before:
                    if (isinstance(array, data.View) and len(arrname) > len(conn)
                            and arrname[:len(conn) + 1] == f'{conn}_'):
                        new_name = f'{new_conn}{arrname[len(conn):]}'
                        names_to_replace[arrname] = new_name
            else:
                updated_args.append((conn, arg))
        args = updated_args

        # Change transient names
        arrays_before = list(sdfg.arrays.items())
        for arrname, array in arrays_before:
            if array.transient and arrname[:5] == '__tmp':
                if int(arrname[5:]) < self.sdfg._temp_transients:
                    if self.sdfg._temp_transients > sdfg._temp_transients:
                        new_name = self.sdfg.temp_data_name()
                    else:
                        new_name = sdfg.temp_data_name()
                    names_to_replace[arrname] = new_name
        self.sdfg._temp_transients = max(self.sdfg._temp_transients, sdfg._temp_transients)
        sdfg._temp_transients = self.sdfg._temp_transients
        replace_datadesc_names(sdfg, names_to_replace)

        # TODO: This workaround needs to be formalized (pass-by-assignment)
        slice_state = None
        output_slices = set()
        for arg in itertools.chain(node.args, [kw.value for kw in node.keywords]):
            if isinstance(arg, ast.Subscript):
                slice_state = self.last_state
                break

        # Make sure that any scope vars in the arguments are substituted
        # by an access.
        for i, (aname, arg) in enumerate(args):
            if not isinstance(arg, str) or arg not in self.sdfg.arrays:
                if isinstance(arg, str) and arg in self.scope_arrays:
                    # TODO: Do we need to do something with the sqz range?
                    newarg, _ = self._add_read_access(arg, subsets.Range.from_array(self.scope_arrays[arg]), node)
                else:
                    newarg = arg
                args[i] = (aname, newarg)

        state = self._add_state('call_%s_%d' % (funcname, node.lineno))
        argdict = {conn: Memlet.from_array(arg, self.sdfg.arrays[arg]) for conn, arg in args if arg in self.sdfg.arrays}
        # Handle scalar inputs to nested SDFG calls
        for conn, arg in args:
            if ((not isinstance(arg, str) or arg not in self.sdfg.arrays) and conn not in mapping.keys() | symbols):
                argdict[conn] = state.add_tasklet('scalar', {}, {conn},
                                                  '%s = %s' % (conn, arg),
                                                  debuginfo=self.current_lineinfo)

        # Handle scalar inputs that become symbols in the nested SDFG
        for sym, local in mapping.items():
            if isinstance(local, str) and local in self.sdfg.arrays:
                # Add assignment state and inter-state edge
                symassign_state = self.sdfg.add_state_before(state)
                isedge = self.sdfg.edges_between(symassign_state, state)[0]
                newsym = self.sdfg.find_new_symbol(f'sym_{local}')
                desc = self.sdfg.arrays[local]
                self.sdfg.add_symbol(newsym, desc.dtype)
                if isinstance(desc, data.Array):
                    isedge.data.assignments[newsym] = f'{local}[0]'
                else:
                    isedge.data.assignments[newsym] = local

                # Replace mapping with symbol
                mapping[sym] = newsym

        inputs = {k: v for k, v in argdict.items() if self._is_inputnode(sdfg, k)}
        outputs = {
            k: copy.deepcopy(v) if k in inputs else v
            for k, v in argdict.items() if self._is_outputnode(sdfg, k)
        }

        # Add closure to global inputs/outputs (e.g., if processed as part of a map)
        for arrname in closure_arrays.keys():
            if arrname not in names_to_replace:
                continue
            narrname = names_to_replace[arrname]

            if narrname in inputs:
                self.inputs[arrname] = (state, inputs[narrname], [])
            if narrname in outputs:
                self.outputs[arrname] = (state, outputs[narrname], [])

        # If an argument does not register as input nor as output,
        # put it in the inputs.
        # This may happen with input argument that are used to set
        # a promoted scalar.
        for k, v in argdict.items():
            if k not in inputs.keys() and k not in outputs.keys():
                inputs[k] = v
        # Unset parent inputs/read accesses that
        # turn out to be outputs/write accesses.
        for memlet in outputs.values():
            aname = memlet.data
            rng = memlet.subset
            access_value = (aname, rng)
            access_key = inverse_dict_lookup(self.accesses, access_value)
            # NOTE: `memlet in inputs.values()` doesn't work because
            # it only looks at the subset and not the data
            # isinput = memlet in inputs.values()
            isinput = False
            for other in inputs.values():
                if not isinstance(other, dace.Memlet):
                    continue
                if memlet == other and memlet.data == other.data:
                    isinput = True
                    break
            if access_key:
                # Delete read access and create write access and output
                vname = aname[:-1] + 'w'
                name, rng, atype = access_key
                if atype == 'r':
                    if not isinput:
                        del self.accesses[access_key]
                    access_value = self._add_write_access(name, rng, node, new_name=vname)
                    memlet.data = vname
                # Delete the old read descriptor
                if not isinput:
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
            if not isinput and aname in self.inputs:
                # Delete input
                del self.inputs[aname]
            # Delete potential input slicing
            if not isinput and slice_state:
                for n in slice_state.nodes():
                    if isinstance(n, nodes.AccessNode) and n.data == aname:
                        for e in slice_state.in_edges(n):
                            sub = None
                            for s in itertools.chain(node.args, [kw.value for kw in node.keywords]):
                                if isinstance(s, ast.Subscript):
                                    if s.value.id == e.src.data:
                                        sub = s
                                        break
                            if not sub:
                                raise KeyError("Did not find output subscript")
                            output_slices.add((sub, ast.Name(id=aname)))
                            slice_state.remove_edge(e)
                            slice_state.remove_node(e.src)
                        slice_state.remove_node(n)
                        break

        # Add return values as additional outputs
        rets = []
        given_args = set(a for a, _ in args)
        for arrname, arr in sdfg.arrays.items():
            if (arrname.startswith('__return') and not arr.transient and arrname not in given_args):
                # Add a transient to the current SDFG
                new_arrname = '%s_ret_%d' % (sdfg.name, len(rets))
                newarr = copy.deepcopy(arr)
                newarr.transient = True

                # Substitute symbol mapping to get actual shape/strides
                if mapping is not None:
                    # Two-step replacement (N -> __dacesym_N --> mapping[N])
                    # to avoid clashes
                    symbolic.safe_replace(mapping, lambda m: sd.replace_properties_dict(newarr, m))

                new_arrname = self.sdfg.add_datadesc(new_arrname, newarr, find_new_name=True)

                # Create an output entry for the connectors
                outputs[arrname] = dace.Memlet.from_array(new_arrname, newarr)
                rets.append(new_arrname)

        # Update strides
        inv_mapping = {v: k for k, v in mapping.items() if symbolic.issymbolic(v) or isinstance(v, str)}
        for a, m in itertools.chain(inputs.items(), outputs.items()):
            # NOTE: This is more complicated than it should because we allow passing
            # arguments to a nested SDFG with incompatible shapes. For an example,
            # see 'tests/tranformations/redundant_reshape_views_test::test_inline_reshape_views_work'
            if not isinstance(m, Memlet):
                continue
            outer_data = self.sdfg.arrays[m.data]
            if outer_data.shape == (1, ):
                continue
            strides = tuple(outer_data.strides[i] for i, sz in enumerate(m.subset.size()) if sz != 1)
            if len(strides) == len(sdfg.arrays[a].shape):
                sdfg.arrays[a]._strides = strides
                if inv_mapping:
                    symbolic.safe_replace(inv_mapping, lambda m: sd.replace_properties_dict(sdfg.arrays[a], m))
            else:
                if strides and (strides[-1] != 1 or sdfg.arrays[a].strides[-1] != 1):
                    warnings.warn(f'Incompatible strides: inner {sdfg.arrays[a].strides} - outer {strides}')

        nsdfg = state.add_nested_sdfg(sdfg,
                                      self.sdfg,
                                      inputs.keys(),
                                      outputs.keys(),
                                      mapping,
                                      debuginfo=self.current_lineinfo)
        self._add_nested_symbols(nsdfg)
        inputs = {k: (state, v, set()) for k, v in inputs.items()}
        outputs = {k: (state, v, set()) for k, v in outputs.items()}
        self._add_dependencies(state, nsdfg, None, None, inputs, outputs)

        # If __pystate is detected within nested SDFG, map to local Python state
        if '__pystate' in sdfg.arrays:
            sdfg.arrays['__pystate'].transient = False
            self._connect_pystate(nsdfg, state)

        if output_slices:
            if len(rets) > 0:
                raise DaceSyntaxError(self, node, 'Both return values and output slices unsupported')

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
            return self._visit_assign(assign_node, assign_node.targets, None)

        # Return SDFG return values, if exist
        if len(rets) == 1:
            return rets[0]
        return rets

    def create_callback(self, node: ast.Call, create_graph=True):
        funcname = astutils.rname(node)
        if funcname not in self.closure.callbacks:
            raise DaceSyntaxError(f'Cannot find appropriate Python callback for {funcname}')
        func: Callable[..., Any]
        _, func, _ = self.closure.callbacks[funcname]

        # Infer the type of the function arguments and return value
        argtypes = []
        args = []
        outargs = []
        allargs = []
        kwargs = [kw.value for kw in node.keywords]
        for arg in itertools.chain(node.args, kwargs):
            parsed_args = self._parse_function_arg(arg)

            # Flatten literal arguments in call (will be unflattened in callback,
            # see ``flatten_callback`` in preprocessing)
            if isinstance(parsed_args, (list, tuple)):
                pass  # If already list or tuple, keep as-is
            elif isinstance(parsed_args, dict):
                # Keep dictionary entries in order of call
                parsed_args = list(parsed_args.values())
            else:  # If a standard argument
                parsed_args = [parsed_args]

            for parsed_arg in parsed_args:
                if parsed_arg in self.defined:
                    atype = self.defined[parsed_arg]
                    args.append(parsed_arg)
                    if isinstance(atype, data.Array):
                        outargs.append(parsed_arg)
                        allargs.append(f'__out_{parsed_arg}')
                    elif isinstance(atype, data.Scalar):
                        allargs.append(f'__in_{parsed_arg}')
                    else:
                        allargs.append(parsed_arg)
                else:
                    if isinstance(parsed_arg, (Number, numpy.number, type(None))):
                        atype = data.create_datadescriptor(type(parsed_arg))
                    else:
                        atype = data.create_datadescriptor(parsed_arg)

                    if isinstance(parsed_arg, str):
                        # Special case for strings
                        parsed_arg = f'"{astutils.escape_string(parsed_arg)}"'
                    allargs.append(parsed_arg)

                argtypes.append(atype)

        # Return type inference
        return_type = None

        # Get the parent node of this AST node
        parent_is_toplevel = True
        parent: ast.AST = None
        for anode in ast.walk(self.program_ast):
            if parent is not None:
                break
            for child in ast.iter_child_nodes(anode):
                if child is node:
                    parent = anode
                    parent_is_toplevel = getattr(anode, 'toplevel', False)
                    break
                if hasattr(child, 'func') and hasattr(child.func, 'oldnode'):
                    # Check if the AST node is part of a failed parse
                    if child.func.oldnode is node:
                        parent = anode
                        parent_is_toplevel = getattr(anode, 'toplevel', False)
                        break
        if parent is None:
            raise DaceSyntaxError(self, node, f'Cannot obtain parent AST node for callback "{funcname}"')

        # If the parent is a top level expression, the return value is unused
        if isinstance(parent, ast.Expr) and parent_is_toplevel:
            return_type = dtypes.typeclass(None)
        elif isinstance(parent, ast.AnnAssign):
            return_names = []
            # NOTE: Python doesn't currently allow multiple return values in
            # annotated assignments
            try:
                return_type = eval(astutils.unparse(parent.annotation), self.globals, self.defined)
            except:
                # TODO: Use a meaningful exception
                pass
            return_type = data.create_datadescriptor(return_type)
            aname, _ = self.sdfg.add_temp_transient_like(return_type)
            return_names = [aname]
            outargs.extend(return_names)
            allargs.extend([f'__out_{n}' for n in return_names])

        elif isinstance(parent, (ast.Assign, ast.AugAssign)):
            defined_vars = {**self.variables, **self.scope_vars}
            defined_arrays = {**self.sdfg.arrays, **self.scope_arrays}

            return_names = []
            return_type = []

            def parse_target(t: Union[ast.Name, ast.Subscript]):
                name = rname(t)
                if name in defined_vars:
                    tname = defined_vars[name]
                    tarr = defined_arrays[tname]
                    if isinstance(t, ast.Subscript):
                        dtype, shape = self.visit_Subscript(copy.deepcopy(t), True)
                        n, arr = self.sdfg.add_temp_transient(shape, dtype)
                    else:
                        if isinstance(tarr, data.Scalar):
                            n, arr = self.sdfg.add_temp_transient((1, ), tarr.dtype)
                        else:
                            n, arr = self.sdfg.add_temp_transient_like(tarr)
                elif name in self.annotated_types:
                    dtype = self.annotated_types[name]
                    if isinstance(dtype, data.Data):
                        n, arr = self.sdfg.add_temp_transient_like(dtype)
                    elif isinstance(dtype, dtypes.typeclass):
                        n, arr = self.sdfg.add_temp_transient((1, ), dtype)
                    else:
                        n, arr = None, None
                else:
                    n, arr = None, None
                return n, arr

            for target in parent.targets:
                if isinstance(target, ast.Tuple):
                    for actual_target in target.elts:
                        n, arr = parse_target(actual_target)
                        if not arr:
                            return_type = None
                            break
                        return_names.append(n)
                        return_type.append(arr)
                else:
                    n, arr = parse_target(target)
                    if not arr:
                        return_type = None
                        break
                    return_names.append(n)
                    return_type.append(arr)

            outargs.extend(return_names)
            allargs.extend([f'__out_{n}' for n in return_names])

        # TODO(later): A proper type/shape inference pass can uncover
        #              return values if in e.g., nested calls: f(g(a))

        # If not annotated, nor the array didn't exist,
        # raise a syntax error with an example of how to do it
        if return_type is None:
            raise DaceSyntaxError(
                self, node, f'Cannot infer return type of function call "{funcname}". '
                'To ensure that the return types can be inferred, try to '
                'extract the call to a separate statement and annotate the '
                'return values. For example:\n'
                '  a: dace.int32\n'
                '  b: dace.float64[N]\n'
                '  a, b = call(c, d)')

        # Create a matching callback symbol from function type
        if (not isinstance(return_type, (list, tuple)) and return_type == dtypes.typeclass(None)):
            return_type = None
        callback_type = dace.callback(return_type, *argtypes)

        if funcname not in self.sdfg.symbols:
            self.sdfg.add_symbol(funcname, callback_type)
        else:
            # If callback signature mismatches
            symtype = self.sdfg.symbols[funcname]
            if symtype != callback_type:
                new_funcname = self.sdfg.find_new_symbol(funcname)
                self.sdfg.add_symbol(new_funcname, callback_type)
                self.sdfg.callback_mapping[new_funcname] = funcname
                funcname = new_funcname

        # Create the graph that calls the callback
        if not create_graph:
            return []

        # Create a state with a tasklet and the right arguments
        self._add_state('callback_%d' % node.lineno)
        self.last_state.set_default_lineinfo(self.current_lineinfo)

        if callback_type.is_scalar_function() and len(callback_type.return_types) > 0:
            call_args = ', '.join(str(s) for s in allargs[:-1])
            tasklet = self.last_state.add_tasklet(f'callback_{node.lineno}', {f'__in_{name}'
                                                                              for name in args} | {'__istate'},
                                                  {f'__out_{name}'
                                                   for name in outargs} | {'__ostate'},
                                                  f'__out_{outargs[0]} = {funcname}({call_args})',
                                                  side_effects=True)
        else:
            call_args = ', '.join(str(s) for s in allargs)
            tasklet = self.last_state.add_tasklet(f'callback_{node.lineno}', {f'__in_{name}'
                                                                              for name in args} | {'__istate'},
                                                  {f'__out_{name}'
                                                   for name in outargs} | {'__ostate'},
                                                  f'{funcname}({call_args})',
                                                  side_effects=True)

        # Avoid cast of output pointers to scalars in code generation
        for cname in outargs:
            if (cname in self.sdfg.arrays and tuple(self.sdfg.arrays[cname].shape) == (1, )):
                tasklet._out_connectors[f'__out_{cname}'] = dtypes.pointer(self.sdfg.arrays[cname].dtype)

        # Setup arguments in graph
        for arg in dtypes.deduplicate(args):
            r = self.last_state.add_read(arg)
            self.last_state.add_edge(r, None, tasklet, f'__in_{arg}', Memlet(arg))

        for arg in dtypes.deduplicate(outargs):
            w = self.last_state.add_write(arg)
            self.last_state.add_edge(tasklet, f'__out_{arg}', w, None, Memlet(arg))

        # Connect Python state
        self._connect_pystate(tasklet, self.last_state, '__istate', '__ostate')

        if return_type is None:
            return []
        else:
            return return_names

    def _connect_pystate(self,
                         tasklet: nodes.CodeNode,
                         state: SDFGState,
                         inp_conn: str = '__pystate',
                         out_conn: str = '__pystate',
                         arr_name: str = '__pystate'):
        '''
        Create and connect a __pystate variable that blocks reordering
        optimizations to a given tasklet.
        '''
        if arr_name not in self.sdfg.arrays:
            self.sdfg.add_scalar(arr_name, dace.int32, transient=True)
        rs = state.add_read(arr_name)
        ws = state.add_write(arr_name)
        tasklet.add_in_connector(inp_conn, dace.int32, force=True)
        state.add_edge(rs, None, tasklet, inp_conn, Memlet(arr_name))
        tasklet.add_out_connector(out_conn, dace.int32, force=True)
        state.add_edge(tasklet, out_conn, ws, None, Memlet(arr_name))

    def visit_Call(self, node: ast.Call, create_callbacks=False):
        func = None
        funcname = None
        # If the call directly refers to an SDFG or dace-compatible program
        if isinstance(node.func, ast.Num):
            if self._has_sdfg(node.func.n):
                func = node.func.n
        elif isinstance(node.func, ast.Constant):
            if self._has_sdfg(node.func.value):
                func = node.func.value

        if func is None:
            funcname = rname(node)
            # Check if the function exists as an SDFG in a different module
            modname = until(funcname, '.')
            if ('.' in funcname and len(modname) > 0 and modname in self.globals
                    and dtypes.ismodule(self.globals[modname])):
                try:
                    func = getattr(self.globals[modname], funcname[len(modname) + 1:])
                except AttributeError:
                    func = None

                # Not an SDFG, ignore (might be a recognized function, see below)
                if not self._has_sdfg(func):
                    func = None
                else:
                    # An SDFG, replace dots in name with underscores
                    funcname = funcname.replace('.', '_')

            # If the function is a callable object
            elif funcname in self.globals and callable(self.globals[funcname]):
                fobj = self.globals[funcname]
                fcall = getattr(fobj, '__call__', False)
                if self._has_sdfg(fcall):
                    func = fcall
                    funcname = fcall.name
                elif isinstance(fobj, numpy.ufunc):
                    modname = 'numpy'
                    funcname = f'numpy.{fobj.__name__}'
                # Try to evaluate function directly
                elif (callable(fobj) and hasattr(fobj, '__module__') and hasattr(fobj, '__name__')):
                    module = fobj.__module__
                    if (module is None or module == str.__class__.__module__ or module == '__main__'):
                        candidate = fobj.__name__
                    else:
                        candidate = fobj.__module__ + '.' + fobj.__name__

                    if oprepo.Replacements.get(candidate) is not None:
                        funcname = candidate

        # If the function exists as a global SDFG or @dace.program, use it
        if func is not None:
            try:
                return self._parse_sdfg_call(funcname, func, node)
            except SkipCall as ex:
                # Re-parse call with non-parsed information, trying
                # to create callbacks instead
                try:
                    return self.visit_Call(node.func.oldnode, create_callbacks=True)
                except Exception:  # Anything could happen here
                    # Raise original exception instead
                    raise ex.__context__

        # Set arguments
        args = []

        # NumPy ufunc support
        found_ufunc = False
        if modname == "numpy" and len(funcname) > 6:
            name = funcname[len(modname) + 1:]
            npfuncname = until(name, '.')
            func = getattr(self.globals[modname], npfuncname)
            if isinstance(func, numpy.ufunc):
                ufunc_name = npfuncname
                if len(funcname) > len(modname) + len(npfuncname) + 1:
                    method_name = funcname[len(modname) + len(npfuncname) + 2:]
                else:
                    method_name = None
                func = oprepo.Replacements.get_ufunc(method_name)
                if ufunc_name in replacements.ufuncs.keys() and func:
                    found_ufunc = True

        # Check if this is a method called on an object
        if ('.' in funcname and len(modname) > 0 and modname in self.defined):
            methodname = funcname[len(modname) + 1:]
            cls = type(self.defined[modname])
            classname = cls.__name__
            func = oprepo.Replacements.get_method(cls, methodname)
            if func is None:
                nm = rname(node)
                if create_callbacks and nm in self.closure.callbacks:
                    warnings.warn('Performance warning: Automatically creating '
                                  f'callback to Python interpreter from method "{funcname}" '
                                  f'in class "{classname}". If you would like to know why '
                                  'parsing failed, please place a @dace.method decorator on the method. '
                                  'If a DaCe method cannot be provided (for example, due to '
                                  'recursion), register a replacement through '
                                  '"dace.frontend.common.op_repository".')
                    return self.create_callback(node)
                raise DaceSyntaxError(self, node,
                                      'Method "%s" is not registered for object type "%s"' % (methodname, classname))
            # Add object as first argument
            if modname in self.variables.keys():
                arg = self.variables[modname]
            else:
                arg = self.scope_vars[modname]
            args.append(arg)
        # Otherwise, try to find a default implementation for the SDFG
        elif not found_ufunc:
            func = oprepo.Replacements.get(funcname)
            if func is None:
                nm = rname(node)
                if nm in self.closure.callbacks:
                    warnings.warn('Performance warning: Automatically creating '
                                  f'callback to Python interpreter from method "{funcname}". '
                                  f'If you would like to know why parsing failed, please '
                                  'place a @dace.program decorator on the function. '
                                  'If a DaCe function cannot be provided (for example, due to '
                                  'recursion), register a replacement through '
                                  '"dace.frontend.common.op_repository".')
                    return self.create_callback(node)
                raise DaceSyntaxError(self, node, 'Function "%s" is not registered with an SDFG '
                                      'implementation' % funcname)

        # NOTE: Temporary fix for MPI library-node replacements
        # Parsing the arguments with `_parse_function_arg` will generate
        # slices even for the output arguments.
        # We make a special exception for MPI calls (`dace.comm` namespace)
        # and we pass instead the array names and the ranges accessed.
        # The replacement functions are responsible for generating the correct
        # subgraph/memlets.
        if funcname.startswith("dace.comm"):
            mpi_args = []
            for arg in node.args:
                # We are only looking for subscripts on arrays of the current SDFG.
                # If it is not a subscript, then we just pass the array pointer directly.
                # If it is not an array of the current SDFG, then the normal
                # argument parsing will create a connector, i.e. a pointer.
                if (isinstance(arg, ast.Subscript) and
                    (rname(arg) in self.sdfg.arrays.keys() or
                     (rname(arg) in self.variables.keys() and self.variables[rname(arg)] in self.sdfg.arrays.keys()))):
                    arg.slice = self.visit(arg.slice)
                    expr: MemletExpr = ParseMemlet(self, {**self.sdfg.arrays, **self.defined}, arg)
                    if isinstance(expr.subset, subsets.Indices):
                        expr.subset = subsets.Range.from_indices(expr.subset)
                    name = rname(arg)
                    if name in self.variables.keys():
                        name = self.variables[name]
                    mpi_args.append((name, expr.subset))
                else:
                    mpi_args.append(self._parse_function_arg(arg))
            args.extend(mpi_args)
        else:
            args.extend([self._parse_function_arg(arg) for arg in node.args])
        keywords = {arg.arg: self._parse_function_arg(arg.value) for arg in node.keywords}

        self._add_state('call_%d' % node.lineno)
        self.last_state.set_default_lineinfo(self.current_lineinfo)

        if found_ufunc:
            result = func(self, node, self.sdfg, self.last_state, ufunc_name, args, keywords)
        else:
            result = func(self, self.sdfg, self.last_state, *args, **keywords)

        self.last_state.set_default_lineinfo(None)

        if isinstance(result, tuple) and type(result[0]) is nested_call.NestedCall:
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
            if isinstance(src_expr.subset, subsets.Indices):
                src_expr.subset = subsets.Range.from_indices(src_expr.subset)
            if isinstance(dst_expr.subset, subsets.Indices):
                dst_expr.subset = subsets.Range.from_indices(dst_expr.subset)
            if src_expr.arrdims or dst_expr.arrdims:
                raise NotImplementedError('Copying with array indices only allowed through assignment '
                                          'expressions ("A[...] = B[...]")')
            src_name = src_expr.name
            src_rng = None
            if src_name not in self.sdfg.arrays:
                src_name, src_rng = self._add_read_access(src_name, src_expr.subset, None)
            dst_name = dst_expr.name
            dst_rng = None
            if dst_name not in self.sdfg.arrays:
                dst_name, dst_rng = self._add_write_access(dst_name, dst_expr.subset, None)

            rnode = state.add_read(src_name, debuginfo=self.current_lineinfo)
            wnode = state.add_write(dst_name, debuginfo=self.current_lineinfo)
            if isinstance(self.sdfg.arrays[dst_name], data.Stream):
                dst_rng = dst_rng or subsets.Range.from_array(self.sdfg.arrays[dst_name])
                mem = Memlet.simple(dst_name, dst_rng, num_accesses=dst_expr.accesses, wcr_str=dst_expr.wcr)
            else:
                src_rng = src_rng or subsets.Range.from_array(self.sdfg.arrays[src_name])
                mem = Memlet.simple(src_name, src_rng, num_accesses=src_expr.accesses, wcr_str=dst_expr.wcr)
            state.add_nedge(rnode, wnode, mem)
            return

        # Calling reduction or other SDFGs / functions
        elif isinstance(node.value, ast.Call):
            # Handles reduction and calling other SDFGs / DaCe programs
            # self._add_state('call_%d' % node.lineno)
            self.visit_Call(node.value)
            return

        elif (sys.version_info.major == 3 and sys.version_info.minor >= 8 and isinstance(node.value, ast.NamedExpr)):
            self.visit_NamedExpr(node.value)
            return

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        # Modify node value to become an expression
        new_node = ast.copy_location(ast.Expr(value=node.value), node)

        # Return values can either be tuples or a single object
        if isinstance(node.value, (ast.Tuple, ast.List)):
            ast_tuple = ast.copy_location(
                ast.parse('(%s,)' % ','.join('__return_%d' % i for i in range(len(node.value.elts)))).body[0].value,
                node)
            self._visit_assign(new_node, ast_tuple, None, is_return=True)
        else:
            ast_name = ast.copy_location(ast.Name(id='__return'), node)
            self._visit_assign(new_node, ast_name, None, is_return=True)

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
                inputs = {k: (state, v, set()) for k, v in inputs.items()}
                outputs = {k: (state, v, set()) for k, v in outputs.items()}
                self._add_dependencies(state, tasklet, None, None, inputs, outputs)
                self.inputs.update({k: (state, *v) for k, v in sdfg_inp.items()})
                self.outputs.update({k: (state, *v) for k, v in sdfg_out.items()})
                return

        raise DaceSyntaxError(self, node, 'General "with" statements disallowed in DaCe programs')

    def visit_AsyncWith(self, node):
        return self.visit_With(node, is_async=True)

    def _visitname(self, name: str, node: ast.AST):
        if isinstance(name, (sympy.Symbol, symbolic.symbol)):
            name = str(name)
        elif symbolic.issymbolic(name, self.sdfg.constants):
            raise TypeError('Symbolic expression found instead of variable name')

        if not isinstance(name, str):
            return name

        # First, if it is defined in the parser, use the definition
        if name in self.variables:
            return self.variables[name]

        # TODO: Why if the following code-block is moved after the code-block
        # looking for `name` in `self.sdfg.symbols`, a lot of tests break?
        # If an allowed global, use directly
        if name in self.globals:
            result = inner_eval_ast(self.globals, node)
            # If a symbol, add to symbols
            if (isinstance(result, symbolic.symbol) and name not in self.sdfg.symbols.keys()):
                self.sdfg.add_symbol(result.name, result.dtype)
            return result

        if name in self.sdfg.arrays:
            return name

        if name in self.sdfg.symbols:
            return name

        if name not in self.scope_vars:
            raise DaceSyntaxError(self, node, 'Use of undefined variable "%s"' % name)
        rname = self.scope_vars[name]
        if rname in self.scope_arrays:
            rng = subsets.Range.from_array(self.scope_arrays[rname])
            rname, _ = self._add_read_access(rname, rng, node)
        return rname

    #### Visitors that return arrays
    def visit_Str(self, node: ast.Str):
        # A string constant returns itself
        return node.s

    def visit_Num(self, node: ast.Num):
        if isinstance(node.n, bool):
            return dace.bool_(node.n)
        if isinstance(node.n, (int, float, complex)):
            return dtypes.DTYPE_TO_TYPECLASS[type(node.n)](node.n)
        return node.n

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool):
            return dace.bool_(node.value)
        if isinstance(node.value, (int, float, complex)):
            return dtypes.DTYPE_TO_TYPECLASS[type(node.value)](node.value)
        return node.value

    def visit_Name(self, node: ast.Name):
        # If visiting a name, check if it is a defined variable or a global
        return self._visitname(node.id, node)

    def visit_NameConstant(self, node: ast.NameConstant):
        return self.visit_Constant(node)

    def visit_Attribute(self, node: ast.Attribute):
        # If visiting an attribute, return attribute value if it's of an array or global
        name = until(astutils.unparse(node), '.')
        result = self._visitname(name, node)
        if isinstance(result, str) and result in self.sdfg.arrays:
            arr = self.sdfg.arrays[result]
        elif isinstance(result, str) and result in self.scope_arrays:
            arr = self.scope_arrays[result]
        else:
            return result

        # Try to find sub-SDFG attribute
        func = oprepo.Replacements.get_attribute(type(arr), node.attr)
        if func is not None:
            return func(self, self.sdfg, self.last_state, result)

        # Otherwise, try to find compile-time attribute (such as shape)
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

    def visit_Set(self, node: ast.Set):
        # Recursively loop over elements
        return set(self.visit(a) for a in node.elts)

    def visit_Dict(self, node: ast.Dict):
        # Recursively loop over elements and return an ordered dictionary (for callback consistency)
        return OrderedDict([(self.visit(k), self.visit(v)) for k, v in zip(node.keys, node.values)])

    def visit_Lambda(self, node: ast.Lambda):
        # Return a string representation of the function
        return astutils.unparse(node)

    ############################################################

    def _gettype(self, opnode: ast.AST) -> List[Tuple[str, str]]:
        """ Returns an operand and its type as a 2-tuple of strings. """
        if isinstance(opnode, ast.AST):
            operands = self.visit(opnode)
        else:
            operands = opnode

        if isinstance(operands, (list, tuple)):
            if len(operands) == 0:
                raise DaceSyntaxError(self, opnode, 'Operand has no return value')
        else:
            operands = [operands]

        result = []
        for operand in operands:
            if isinstance(operand, str) and operand in self.sdfg.arrays:
                result.append((operand, type(self.sdfg.arrays[operand])))
            elif isinstance(operand, str) and operand in self.scope_arrays:
                result.append((operand, type(self.scope_arrays[operand])))
            elif isinstance(operand, tuple(dtypes.DTYPE_TO_TYPECLASS.keys())):
                if isinstance(operand, (bool, numpy.bool_)):
                    result.append((operand, 'BoolConstant'))
                else:
                    result.append((operand, 'NumConstant'))
            elif isinstance(operand, sympy.Basic):
                result.append((operand, 'symbol'))
            else:
                result.append((operand, type(operand)))

        return result

    def _visit_op(self, node: Union[ast.UnaryOp, ast.BinOp, ast.BoolOp], op1: ast.AST, op2: ast.AST):
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

        func = oprepo.Replacements.getop(op1type, opname, otherclass=op2type)
        if func is None:
            # Check for SDFG as fallback
            func = oprepo.Replacements.getop(op1type, opname, otherclass=op2type)
            if func is None:
                op1name = getattr(op1type, '__name__', op1type)
                op2name = getattr(op2type, '__name__', op2type)
                raise DaceSyntaxError(self, node, f'Operator {opname} is not defined for types {op1name} and {op2name}')

        self._add_state('%s_%d' % (type(node).__name__, node.lineno))
        self.last_state.set_default_lineinfo(self.current_lineinfo)
        try:
            result = func(self, self.sdfg, self.last_state, operand1, operand2)
        except SyntaxError as ex:
            raise DaceSyntaxError(self, node, str(ex))
        if not isinstance(result, (list, tuple)):
            results = [result]
        else:
            results = result
        for r in results:
            if isinstance(r, str) and r in self.sdfg.arrays:
                if r in self.variables.keys():
                    raise DaceSyntaxError(self, node, "Variable {v} has been already defined".format(v=r))
                self.variables[r] = r

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

    def _add_read_slice(self, array: str, node: ast.Subscript, expr: MemletExpr):

        arrobj = self.sdfg.arrays[array]

        # Consider array dims (rhs expression)
        has_array_indirection = False
        for dim, arrname in expr.arrdims.items():
            # Boolean arrays only allowed as lhs
            if (isinstance(arrname, str) and self.sdfg.arrays[arrname].dtype == dtypes.bool):
                raise IndexError('Boolean array indexing is only supported for '
                                 'assignment targets (e.g., "A[A > 5] += 1")')
            has_array_indirection = True

        # Add slicing state
        self._add_state('slice_%s_%d' % (array, node.lineno))
        if has_array_indirection:
            # Make copy slicing state
            rnode = self.last_state.add_read(array, debuginfo=self.current_lineinfo)
            return self._array_indirection_subgraph(rnode, expr)
        else:
            is_index = False
            if isinstance(expr.subset, subsets.Indices):
                is_index = True
                other_subset = subsets.Range([(i, i, 1) for i in expr.subset])
            else:
                other_subset = copy.deepcopy(expr.subset)
            strides = list(arrobj.strides)

            # Make new axes and squeeze for scalar subsets (as per numpy behavior)
            # For example: A[0, np.newaxis, 5:7] results in a 1x2 ndarray
            new_axes = []
            if expr.new_axes:
                new_axes = other_subset.unsqueeze(expr.new_axes)
                for i in new_axes:
                    strides.insert(i, 1)
            length = len(other_subset)
            nsqz = other_subset.squeeze(ignore_indices=new_axes)
            sqz = [i for i in range(length) if i not in nsqz]
            for i in reversed(sqz):
                strides.pop(i)
            if not strides:
                strides = None

            if is_index:
                tmp = self.sdfg.temp_data_name()
                tmp, tmparr = self.sdfg.add_scalar(tmp, arrobj.dtype, arrobj.storage, transient=True)
            else:
                tmp, tmparr = self.sdfg.add_view(array,
                                                 other_subset.size(),
                                                 arrobj.dtype,
                                                 storage=arrobj.storage,
                                                 strides=strides,
                                                 find_new_name=True)
                self.views[tmp] = (array,
                                   Memlet(f'{array}[{expr.subset}]->{other_subset}', volume=expr.accesses,
                                          wcr=expr.wcr))
            self.variables[tmp] = tmp
            if not isinstance(tmparr, data.View):
                rnode = self.last_state.add_read(array, debuginfo=self.current_lineinfo)
                wnode = self.last_state.add_write(tmp, debuginfo=self.current_lineinfo)
                self.last_state.add_nedge(
                    rnode, wnode, Memlet(f'{array}[{expr.subset}]->{other_subset}', volume=expr.accesses, wcr=expr.wcr))
            return tmp

    def _parse_subscript_slice(self,
                               s: ast.AST,
                               multidim: bool = False) -> Union[Any, Tuple[Union[Any, str, symbolic.symbol]]]:
        """ Parses the slice attribute of an ast.Subscript node.
            Scalar data are promoted to symbols.
        """
        def _promote(node: ast.AST) -> Union[Any, str, symbolic.symbol]:
            node_str = astutils.unparse(node)
            sym = None
            if node_str in self.indirections:
                sym = self.indirections[node_str]
            if isinstance(node, str):
                scalar = node_str
            else:
                scalar = self.visit(node)
            if isinstance(scalar, str) and scalar in self.sdfg.arrays:
                desc = self.sdfg.arrays[scalar]
                if isinstance(desc, data.Scalar):
                    if not sym:
                        sym = dace.symbol(f'__sym_{scalar}', dtype=desc.dtype)
                        self.indirections[node_str] = sym
                    state = self._add_state(f'promote_{scalar}_to_{str(sym)}')
                    edge = self.sdfg.in_edges(state)[0]
                    edge.data.assignments = {str(sym): scalar}
                    return sym
            return scalar

        if isinstance(s, (Number, bool, numpy.bool_, sympy.Basic)):
            res = s
        elif isinstance(s, ast.Constant):  # 1D index (since Python 3.9)
            # Special case for Python slice objects
            if isinstance(s.value, slice):
                res = self.visit(s)
            else:
                res = self._visit_ast_or_value(s)
        elif isinstance(s, ast.Index):
            res = self._parse_subscript_slice(s.value)
        elif isinstance(s, ast.Slice):
            lower = s.lower
            if isinstance(lower, ast.AST):
                lower = _promote(lower)
            upper = s.upper
            if isinstance(upper, ast.AST):
                upper = _promote(upper)
            step = s.step
            if isinstance(step, ast.AST):
                step = _promote(step)
            if multidim:
                res = (lower, upper, step)
            else:
                res = ((lower, upper, step), )
        elif isinstance(s, ast.Tuple):
            res = tuple(self._parse_subscript_slice(d, multidim=True) for d in s.elts)
        elif isinstance(s, ast.ExtSlice):
            res = tuple(self._parse_subscript_slice(d, multidim=True) for d in s.dims)
        else:
            res = _promote(s)
        # Unpack tuple of a single Python slice object
        if (isinstance(res, (list, tuple)) and len(res) == 1 and isinstance(res[0], slice)):
            res = res[0]
        return res

    ### Subscript (slicing) handling
    def visit_Subscript(self, node: ast.Subscript, inference: bool = False):

        if self.nested:

            defined_vars = {**self.variables, **self.scope_vars}
            defined_arrays = {**self.sdfg.arrays, **self.scope_arrays, **self.defined}

            name = rname(node)
            true_name = defined_vars[name]

            # If this subscript originates from an external array, create the
            # subset in the edge going to the connector, as well as a local
            # reference to the subset
            if (true_name not in self.sdfg.arrays and isinstance(node.value, ast.Name)):
                true_node = copy.deepcopy(node)
                true_node.value.id = true_name

                # Visit slice contents
                nslice = self._parse_subscript_slice(node.slice)

                # Try to construct memlet from subscript
                expr: MemletExpr = ParseMemlet(self, defined_arrays, true_node, nslice)
                rng = expr.subset
                if isinstance(rng, subsets.Indices):
                    rng = subsets.Range.from_indices(rng)
                if inference:
                    rng.offset(rng, True)
                    return self.sdfg.arrays[true_name].dtype, rng.size()
                new_name, new_rng = self._add_read_access(name, rng, node)
                new_arr = self.sdfg.arrays[new_name]
                full_rng = subsets.Range.from_array(new_arr)
                if new_rng.ranges == full_rng.ranges:
                    return new_name
                else:
                    new_name, _ = self.make_slice(new_name, new_rng)
                    return new_name

        # Obtain array/tuple
        node_parsed = self._gettype(node.value)

        if len(node_parsed) > 1:
            # If the value is a tuple of constants (e.g., array.shape) and the
            # slice is constant, return the value itself
            nslice = self.visit(node.slice)
            if isinstance(nslice, (ast.Index, Number)):
                if isinstance(nslice, ast.Index):
                    v = self._parse_value(nslice.value)
                else:
                    v = nslice
                try:
                    value, valtype = node_parsed[int(v)]
                    return value
                except (TypeError, ValueError):
                    pass  # Passthrough to exception

            raise DaceSyntaxError(self, node.value, 'Subscripted object cannot be a tuple')
        array, arrtype = node_parsed[0]
        if arrtype == 'str' or arrtype in dtypes._CTYPES:
            raise DaceSyntaxError(self, node, 'Type "%s" cannot be sliced' % arrtype)
        if arrtype == 'NumConstant':
            return array

        # Visit slice contents
        # TODO: Maybe we actually want to do scalar promotion even in inference
        # mode
        nslice = self._parse_subscript_slice(node.slice)

        # Try to construct memlet from subscript
        node.value = ast.Name(id=array)
        expr: MemletExpr = ParseMemlet(self, {**self.sdfg.arrays, **self.defined}, node, nslice)

        if inference:
            rng = expr.subset
            rng.offset(rng, True)
            return self.sdfg.arrays[array].dtype, rng.size()

        return self._add_read_slice(array, node, expr)

    def _visit_ast_or_value(self, node: ast.AST) -> Any:
        result = self.visit(node)
        newnode = None
        if result is None:
            return node
        if isinstance(result, (list, tuple)):
            res_num = len(result)
        else:
            res_num = 1
            result = [result]
        out = []
        for i, r in enumerate(result):
            if isinstance(r, ast.AST):
                newnode = r
            elif isinstance(r, (Number, numpy.bool_)):
                # Compatibility check since Python changed their AST nodes
                if sys.version_info >= (3, 8):
                    newnode = ast.Constant(value=r, kind='')
                else:
                    newnode = ast.Num(n=r)
            else:
                newnode = ast.Name(id=r)
            ast.copy_location(newnode, node)
            out.append(newnode)
        if res_num == 1:
            out = out[0]
        return out

    def visit_Index(self, node: ast.Index) -> Any:
        if isinstance(node.value, ast.Tuple):
            for i, elt in enumerate(node.value.elts):
                node.value.elts[i] = self._visit_ast_or_value(elt)
            return node
        node.value = self._visit_ast_or_value(node.value)
        return node

    def visit_ExtSlice(self, node: ast.ExtSlice) -> Any:
        for i, dim in enumerate(node.dims):
            node.dims[i] = self._visit_ast_or_value(dim)

        return node

    def make_slice(self, arrname: str, rng: subsets.Range):

        array = arrname
        arrobj = self.sdfg.arrays[arrname]

        # Add slicing state
        # TODO: naming issue, we don't have the linenumber here
        self._add_state('slice_%s' % (array))
        rnode = self.last_state.add_read(array, debuginfo=self.current_lineinfo)
        other_subset = copy.deepcopy(rng)
        other_subset.squeeze()
        if _subset_has_indirection(rng, self):
            memlet = Memlet.simple(array, rng)
            tmp = self.sdfg.temp_data_name()
            tmp = add_indirection_subgraph(self.sdfg, self.last_state, rnode, None, memlet, tmp, self)
        else:
            tmp, tmparr = self.sdfg.add_temp_transient(other_subset.size(), arrobj.dtype, arrobj.storage)
            wnode = self.last_state.add_write(tmp, debuginfo=self.current_lineinfo)
            self.last_state.add_nedge(
                rnode, wnode, Memlet.simple(array, rng, num_accesses=rng.num_elements(), other_subset_str=other_subset))
        return tmp, other_subset

    def _array_indirection_subgraph(self, rnode: nodes.AccessNode, expr: MemletExpr) -> str:
        aname = rnode.data
        idesc = self.sdfg.arrays[aname]

        if expr.new_axes:
            # NOTE: Matching behavior with numpy would be to append all new
            # axes in the end
            raise IndexError('New axes unsupported when array indices are used')

        # Create output shape dimensions based on the sizes of the arrays
        output_shape = None
        constant_indices: Dict[int, str] = {}
        for i, arrname in expr.arrdims.items():
            if isinstance(arrname, str):  # Array or constant
                if arrname in self.sdfg.arrays:
                    desc = self.sdfg.arrays[arrname]
                elif arrname in self.sdfg.constants:
                    desc = self.sdfg.constants[arrname]
                    constant_indices[i] = arrname
                else:
                    raise NameError(f'Array "{arrname}" used in indexing "{aname}" not found')
                shape = desc.shape
            else:  # Literal list or tuple, add as constant and use shape
                arrname = [v if isinstance(v, Number) else self._parse_value(v) for v in arrname]
                carr = numpy.array(arrname, dtype=dtypes.typeclass(int).type)
                cname = self.sdfg.find_new_constant(f'__ind{i}_{aname}')
                self.sdfg.add_constant(cname, carr)
                constant_indices[i] = cname
                shape = carr.shape

            if output_shape is not None and tuple(shape) != output_shape:
                raise IndexError(f'Mismatch in array index shapes in access of '
                                 f'"{aname}": {arrname} (shape {shape}) '
                                 f'does not match existing shape {output_shape}')
            elif output_shape is None:
                output_shape = tuple(shape)

        # Check subset shapes for matching the array shapes
        input_index = []
        i0 = symbolic.pystr_to_symbolic('__i0')
        for i, elem in enumerate(expr.subset.size()):
            if i in expr.arrdims:
                input_index.append((0, elem - 1, 1))
                continue
            if len(output_shape) > 1:
                raise IndexError('Combining multidimensional array indices and '
                                 'numeric subsets is unsupported (array '
                                 f'"{aname}").')
            if (elem, ) != output_shape:
                # TODO(later): Properly broadcast multiple (and missing) shapes
                raise IndexError(f'Mismatch in array index shapes in access of '
                                 f'"{aname}": Subset {expr.subset[i]} '
                                 f'does not match existing shape {output_shape}')

            # Since there can only be one-dimensional outputs if arrays and
            # subsets are both involved, express memlet as a function of _i0
            rb, _, rs = expr.subset[i]
            input_index.append((rb + i0 * rs, rb + i0 * rs, 1))

        outname, _ = self.sdfg.add_temp_transient(output_shape, idesc.dtype)

        # Make slice subgraph - input shape dimensions are len(expr.subset) and
        # output shape dimensions are len(output_shape)

        # Make map with output shape
        state: SDFGState = self.last_state
        wnode = state.add_write(outname)
        maprange = [(f'__i{i}', f'0:{s}') for i, s in enumerate(output_shape)]
        me, mx = state.add_map('indirect_slice', maprange, debuginfo=self.current_lineinfo)

        # Make indirection tasklet for array-index dimensions
        array_indices = set(expr.arrdims.keys()) - set(constant_indices.keys())
        output_str = ', '.join(ind for ind, _ in maprange)
        access_str = ', '.join(
            [f'__inp{i}' if i in array_indices else f'{cname}[{output_str}]' for i in expr.arrdims.keys()])
        t = state.add_tasklet('indirection', {'__arr'} | set(f'__inp{i}' for i in array_indices), {'__out'},
                              f'__out = __arr[{access_str}]')

        # Offset input memlet according to offset and stride if fixed, or
        # entire array with volume 1 if array-index
        input_subset = subsets.Range(input_index)
        state.add_edge_pair(me, t, rnode, Memlet(data=aname, subset=input_subset, volume=1), internal_connector='__arr')
        # Add array-index memlets
        for dim in array_indices:
            arrname = expr.arrdims[dim]
            arrnode = state.add_read(arrname)
            state.add_edge_pair(me,
                                t,
                                arrnode,
                                Memlet(data=arrname, subset=subsets.Range([(ind, ind, 1) for ind, _ in maprange])),
                                internal_connector=f'__inp{dim}')

        # Output matches the output shape exactly
        output_index = subsets.Range([(ind, ind, 1) for ind, _ in maprange])
        state.add_edge_pair(mx,
                            t,
                            wnode,
                            Memlet(data=outname, subset=output_index),
                            external_memlet=Memlet(data=outname),
                            internal_connector='__out')

        return outname

    ##################################
