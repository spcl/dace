import ast
import astor
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import dace
from dace.frontend.python import astutils
from dace.frontend.python.astutils import ExtNodeVisitor, ExtNodeTransformer, rname
from dace import SDFG, SDFGState, data, dtypes
from dace.graph import nodes
from dace.symbolic import pystr_to_symbolic

# class Replacements:
#     _rep = {}
#     @staticmethod
#     def get(name):
#         return Replacements._rep[name]

# def replaces(name, func):
#     Replacements._rep[name] = func

# @replaces('numpy.ndarray')
# def fff(shape, dtype):
#     return ArrayNode()


class DaceSyntaxError(Exception):
    def __init__(self, visitor, node: ast.AST, message: str):
        self.visitor = visitor
        self.node = node
        self.message = message

    def __str__(self):
        # Try to recover line and column
        try:
            line = self.node.lineno
            col = self.node.col_offset
        except AttributeError:
            line = 0
            col = 0

        return (self.message + "\n  in File " + str(self.visitor.filename) +
                ", line " + str(line) + ":" + str(col))


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
    'TryFinally', 'ExceptHandler', 'With', 'AsyncWith', 'Starred', 'Ellipsis',
    'ClassDef', 'AsyncFor', 'Await', 'Bytes', 'Set', 'Dict', 'ListComp',
    'GeneratorExp', 'SetComp', 'DictComp', 'comprehension'
]

MemletType = Union[ast.Call, ast.Attribute, ast.Subscript, ast.Name]
TaskletType = Union[ast.FunctionDef, ast.With, ast.For]


def _disallow_stmt(visitor, node):
    raise DaceSyntaxError(visitor, node,
                          'Keyword "%s" disallowed' % (type(node).__name__))


###############################################################
# Parsing functions
###############################################################


def _pyexpr_to_symbolic(defined_arrays_and_symbols: Dict[str, Any],
                        expr_ast: ast.AST):
    """ Converts a Python AST expression to a DaCe symbolic expression
        with error checks (raises `SyntaxError` on failure).
        @param defined_arrays_and_symbols: Defined arrays and symbols
               in the context of this expression.
        @param expr_ast: The Python AST expression to convert.
        @return: Symbolic expression.
    """
    raise NotImplementedError


def _ndslice_to_subset(ndslice):
    is_tuple = [isinstance(x, tuple) for x in ndslice]
    if not any(is_tuple):
        return subsets.Indices(ndslice)
    else:
        if not all(is_tuple):
            # If a mix of ranges and indices is found, convert to range
            for i in range(len(ndslice)):
                if not is_tuple[i]:
                    ndslice[i] = (ndslice[i], ndslice[i], 1)
        return subsets.Range(ndslice)


def _fill_missing_slices(das, ast_ndslice, array, indices):
    # Filling ndslice with default values from array dimensions
    # if ranges not specified (e.g., of the form "A[:]")
    ndslice = [None] * len(ast_ndslice)
    ndslice_size = 1
    offsets = []
    idx = 0
    for i, dim in enumerate(ast_ndslice):
        if isinstance(dim, tuple):
            rb = _pyexpr_to_symbolic(das, dim[0])
            re = _pyexpr_to_symbolic(das, dim[1])
            if re is not None:
                re -= 1
            rs = _pyexpr_to_symbolic(das, dim[2])
            if rb is None: rb = 0
            if re is None: re = array.shape[indices[idx]] - 1
            if rs is None: rs = 1
            ndslice[i] = (rb, re, rs)
            offsets.append(i)
            idx += 1
        else:
            ndslice[i] = _pyexpr_to_symbolic(das, dim)

    return ndslice, offsets


# Parses a memlet statement
def ParseMemlet(defined_arrays_and_symbols: Dict[str, Any], local_name: str,
                rhsnode: MemletType):
    rhs = rname(rhsnode)
    if rhs.find('.') >= 0:  # attribute, form G.out_edges[:]
        arrname = rhs[:rhs.find('.')]
        arrattr = rhs[rhs.find('.') + 1:]
    else:  # normal memlet, form A(1)[i,j]
        arrname = rhs
        arrattr = None

    das = defined_arrays_and_symbols
    array = das[arrname]

    # Determine number of accesses to the memlet (default is the slice size)
    num_accesses = None
    write_conflict_resolution = None
    wcr_identity = None
    # Detects expressions of the form "A(2)[...]", "A(300)", "A(1, sum)[:]"
    if isinstance(rhsnode, ast.Call):
        if len(rhsnode.args) < 1 or len(rhsnode.args) > 3:
            raise SyntaxError(
                'Number of accesses in memlet must be a number, symbolic '
                'expression, or -1 (dynamic)')
        num_accesses = _pyexpr_to_symbolic(das, rhsnode.args[0])
        if len(rhsnode.args) >= 2:
            write_conflict_resolution = rhsnode.args[1]
    elif isinstance(rhsnode, ast.Subscript) and isinstance(
            rhsnode.value, ast.Call):
        if len(rhsnode.value.args) < 1 or len(rhsnode.value.args) > 3:
            raise SyntaxError(
                'Number of accesses in memlet must be a number, symbolic '
                'expression, or -1 (dynamic)')
        num_accesses = _pyexpr_to_symbolic(das, rhsnode.value.args[0])
        if len(rhsnode.value.args) >= 2:
            write_conflict_resolution = rhsnode.value.args[1]

    array_dependencies = {}

    # Get memlet range
    ndslice = [(0, s - 1, 1) for s in array.shape]
    if isinstance(rhsnode, ast.Subscript):
        # Parse and evaluate ND slice(s) (possibly nested)
        ast_ndslices = subscript_to_ast_slice_recursive(rhsnode)
        offsets = list(range(len(array.shape)))

        # Loop over nd-slices (A[i][j][k]...)
        subset_array = []
        for ast_ndslice in ast_ndslices:
            # Loop over the N dimensions
            ndslice, offsets = _fill_missing_slices(ast_ndslice, array,
                                                    offsets)
            subset_array.append(_ndslice_to_subset(ndslice))

        subset = subset_array[0]

        # Compose nested indices, e.g., of the form "A[i,:,j,:][k,l]"
        for i in range(1, len(subset_array)):
            subset = subset.compose(subset_array[i])

        # Compute additional array dependencies (as a result of
        # indirection)
        for dim in subset:
            if not isinstance(dim, tuple): dim = [dim]
            for r in dim:
                for expr in symbolic.swalk(r):
                    if symbolic.is_sympy_userfunction(expr):
                        arr = expr.func.__name__
                        array_dependencies[arr] = self.curnode.globals[arr]

    else:  # Use entire range
        subset = self._ndslice_to_subset(ndslice)

    # If undefined, default number of accesses is the slice size
    if num_accesses is None:
        num_accesses = subset.num_elements()

    # This is a valid DaCe load/store, register it
    return astnodes._Memlet(array, arrname, arrattr, num_accesses,
                            write_conflict_resolution, wcr_identity, subset, 1,
                            local_name, rhsnode, array_dependencies)


class TaskletTransformer(ExtNodeTransformer):
    """ A visitor that traverses a data-centric tasklet, removes memlet 
        annotations and returns input and output memlets. 
    """

    def __init__(self,
                 sdfg: SDFG,
                 state: SDFGState,
                 filename: str,
                 lang=dtypes.Language.Python,
                 location: str = '-1'):
        """ Creates an AST parser for tasklets. 
            @param sdfg: The SDFG to add the tasklet in (used for defined arrays and symbols).
            @param state: The SDFG state to add the tasklet to.
        """
        self.sdfg = sdfg
        self.state = state

        # For syntax errors
        self.filename = filename

        # Connectors generated from memlets
        self.inputs = {}  # type: Dict[str, Memlet]
        self.outputs = {}  # type: Dict[str, Memlet]

        self.extcode = None
        self.lang = lang
        self.globalcode = ''
        self.initcode = ''
        self.exitcode = ''
        self.location = location

        # Disallow keywords
        for stmt in _DISALLOWED_STMTS:
            setattr(self, 'visit_' + stmt, lambda n: _disallow_stmt(self, n))

    def parse_tasklet(self, tasklet_ast: TaskletType):
        """ Parses the AST of a tasklet and returns the tasklet node, as well as input and output memlets. 
            @param tasklet_ast: The Tasklet's Python AST to parse.
            @return: 3-tuple of (Tasklet node, input memlets, output memlets).
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
        name = getattr(tasklet_ast, 'name', 'tasklet_%d' % tasklet_ast.lineno)

        t = self.state.add_tasklet(
            name,
            set(self.inputs.keys()),
            set(self.outputs.keys()),
            self.extcode or tasklet_ast,
            language=self.lang,
            code_global=self.globalcode,
            code_init=self.initcode,
            code_exit=self.exitcode,
            location=self.location,
            debuginfo=locinfo)

        return t, self.inputs, self.outputs

    def visit_TopLevelExpr(self, node):
        if isinstance(node.value, ast.BinOp):
            if (isinstance(node.value.op, ast.LShift)
                    or isinstance(node.value.op, ast.RShift)):
                ParseMemlet(..., ..., ...)
                return None
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


class ProgramVisitor(ExtNodeVisitor):
    """ A visitor that traverses a data-centric Python program AST and 
        constructs an SDFG.
    """

    def __init__(self,
                 name: str,
                 filename: str,
                 lineoffset: int,
                 arrays: Dict[str, data.Data],
                 global_vars: Dict[str, Any],
                 nested: bool = False):
        self.curnode = None
        self.filename = filename
        self.lineoffset = lineoffset
        self.globals = global_vars
        self.nested = nested

        self.global_arrays = OrderedDict()  # type: Dict[str, data.Data]
        self.global_arrays.update(arrays)

        # Entry point to the program
        self.program = None
        self.sdfg = SDFG(name)
        self.last_state = None
        if not self.nested:
            self.sdfg.arrays.update(arrays)

        # Keep track of variables and scopes
        self.variables = {}  # type: Dict[str, data.Data]
        self.inputs = {}
        self.outputs = {}

        # Disallow keywords
        for stmt in _DISALLOWED_STMTS:
            setattr(self, 'visit_' + stmt, lambda n: _disallow_stmt(self, n))

    def parse_program(self, program: ast.FunctionDef):
        for stmt in program.body:
            self.visit_TopLevel(stmt)
        if len(self.sdfg.nodes()) == 0:
            self.sdfg.add_state("EmptyState")
        return self.sdfg, self.inputs, self.outputs

    def _parse_arg(self, arg: Any):
        """ Parse possible values to slices or objects that can be used in
            the SDFG API. """
        if isinstance(arg, ast.Subscript) and rname(arg) == '_':
            return astutils.subscript_to_slice(arg, self.sdfg.arrays)[1]
        return arg

    def _decorator_or_annotation_params(
            self, node: ast.FunctionDef) -> List[Tuple[str, Any]]:
        """ Returns a list of parameters, either from the function parameters
            and decorator arguments or parameters and their annotations (type
            hints).
            @param node The given function definition node.
            @return A list of 2-tuples (name, value).
        """
        # If the arguments are defined in the decorator
        dec = node.decorator_list[0]
        if 'args' in dir(dec) and len(dec.args) > 0:
            # If it's one argument of the form of ND range, e.g., "_[0:M, 0:N]"
            parg0 = self._parse_arg(dec.args[0])
            if isinstance(parg0, list) and len(parg0) > 1:
                args = parg0
            else:
                args = [self._parse_arg(arg) for arg in dec.args]
        else:  # Otherwise, use annotations
            args = [arg.annotation for arg in node.args.args]

        result = [(rname(arg), argval)
                  for arg, argval in zip(node.args.args, args)]

        # Ensure all arguments are annotated
        if len(result) != len(node.args.args):
            raise DaceSyntaxError(
                self, node,
                'All arguments in primitive %s must be annotated' % node.name)
        return result

    def _parse_mapscope(self, node):
        return None, None

    def _parse_subprogram(self, name, node):
        pv = ProgramVisitor(name, self.filename, self.lineoffset,
                            self.global_arrays, self.globals, True)
        # sdfg, inputs, outputs = pv.parse_program(node.body)
        # return sdfg, inputs, outputs

        return pv.parse_program(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Supported decorated function types: map, mapscope, consume,
        # consumescope, tasklet, program

        if len(node.decorator_list) != 1:
            raise DaceSyntaxError(
                self, node,
                'Exactly one DaCe decorator is allowed on a function')
        dec = rname(node.decorator_list[0])

        # Create a new state for the statement
        self.state = self.sdfg.add_state('s' + str(self.lineoffset))

        # Select primitive according to function type
        if dec == 'dace.tasklet':  # Tasklet
            t, i, o = TaskletTransformer(self.sdfg, self.state,
                                         self.filename).parse_tasklet(node)

        elif dec.startswith('dace.map'):  # Map scope or map+tasklet
            mapparams = self._decorator_or_annotation_params(node)
            mapentry, mapexit = self._parse_mapscope(node)

            if dec == 'dace.mapscope':
                print('MAP SCOPE!', node.name)
                #
                # TODO: Parse scope as nested SDFG, get inputs and outputs
                self._parse_subprogram()
            else:
                print('MAP and TASKLET', node.name)
                #self.state.add_map(node.name, )
                t, i, o = TaskletTransformer(self.sdfg, self.state,
                                             self.filename).parse_tasklet(node)
        elif dec.startswith('dace.consume'):  # Consume scope or tasklet
            if dec == 'dace.consumescope':
                print('CONSUME SCOPE!', node.name)
                #self.state.add_consume(node.name, )
                # TODO: Parse scope as nested SDFG, get inputs and outputs
            else:
                print('CONSUME and TASKLET', node.name)
                #self.state.add_consume(node.name, )
                t, i, o = TaskletTransformer(self.sdfg, self.state,
                                             self.filename).parse_tasklet(node)
        elif dec == 'dace.program':  # Nested SDFG
            raise DaceSyntaxError(
                self, node, 'Nested programs must be '
                'defined outside existing programs')
        else:
            raise DaceSyntaxError(self, node, 'Unsupported function decorator')

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

        if not isinstance(node, ast.Tuple):
            raise DaceSyntaxError(self, node,
                                  "Target of ast.For must be ast.Tuple")

        indices = []
        for n in node.elts:
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

    def _parse_value(self, node: Union[ast.Name, ast.Num]):
        """Parses a value
        
        Arguments:
            node {Union[ast.Name, ast.Num]} -- Value node
        
        Raises:
            DaceSyntaxError: If node is not ast.Name or ast.Num
        
        Returns:
            str -- Value id or number as string
        """

        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Num):
            return str(node.n)
        else:
            raise DaceSyntaxError(self, node,
                                  "Expected ast.Name or ast.Num as value")

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

        if not isinstance(node, ast.Subscript):
            raise DaceSyntaxError(self, node,
                                  "Iterator of ast.For must be ast.Subscript")

        iterator = rname(node.value)
        if iterator not in {'range', 'parrange', 'dace.map'}:
            raise DaceSyntaxError(
                self, node, "Iterator {} is unsupported".format(iterator))
        if iterator in {'range', 'parrange'}:
            raise NotImplementedError

        ranges = []
        if isinstance(node.slice, ast.ExtSlice):
            for s in node.slice.dims:
                ranges.append(self._parse_slice(s))
        elif isinstance(node.slice, ast.Slice):
            ranges.append(self._parse_slice(node.slice))
        else:  # isinstance(node.slice, ast.Index) is True
            ranges.append(self._parse_index_as_range(node.slice))

        return (iterator, ranges)

    def visit_For(self, node: ast.For):
        # We allow three types of for loops:
        # 1. `for i in range(...)`: Creates a looping state
        # 2. `for i in parrange(...)`: Creates a 1D map
        # 3. `for i,j,k in dace.map[0:M, 0:N, 0:K]`: Creates an ND map
        # print(ast.dump(node))
        indices = self._parse_for_indices(node.target)
        print(indices)
        iterator, ranges = self._parse_for_iterator(node.iter)
        print(iterator)
        print(ranges)

        if len(indices) != len(ranges):
            raise DaceSyntaxError(
                self, node,
                "Number of indices and ranges of for-loop do not match")

        if iterator == 'dace.map':
            state = self.sdfg.add_state('MapState')
            if self.last_state:
                self.sdfg.add_edge(self.last_state, state,
                                   dace.InterstateEdge())
                self.last_state = state
            me, mx = state.add_map(
                name='Map',
                ndrange={k: ':'.join(v)
                         for k, v in zip(indices, ranges)})
            # body = SDFG('MapBody')
            body, inputs, outputs = self._parse_subprogram('MapBody', node)
            tasklet = state.add_nested_sdfg(body, self.sdfg, inputs.keys(),
                                            outputs.keys())
            if inputs:
                for conn, (arr, rng) in inputs.items():
                    read_node = state.add_read(arr)
                    state.add_memlet_path(
                        read_node,
                        me,
                        tasklet,
                        memlet=dace.Memlet.simple(arr, str(rng)),
                        src_conn=None,
                        dst_conn=conn)
            else:
                state.add_nedge(me, tasklet, dace.EmptyMemlet())
            if outputs:
                for conn, (arr, rng) in outputs.items():
                    write_node = state.add_write(arr)
                    state.add_memlet_path(
                        tasklet,
                        mx,
                        write_node,
                        memlet=dace.Memlet.simple(arr, str(rng)),
                        src_conn=conn,
                        dst_conn=None)
            else:
                state.add_nedge(tasklet, mx, dace.EmptyMemlet())

    def visit_While(self, node: ast.While):
        pass

    def visit_If(self, node: ast.If):
        pass

    def _parse_index(self, node: ast.Index):

        indices = []
        for idx in node.value.elts:
            indices.append(self._parse_value(idx))

        return indices

    def _add_tasklet(self, target, value_node):

        value = self._parse_value(value_node)
        print(target, value)

        state = self.sdfg.add_state("TaskletState")
        if self.last_state:
            self.sdfg.add_edge(self.last_state, state, dace.InterstateEdge())
            self.last_state = state
        write_node = state.add_write(target)
        tasklet_node = state.add_tasklet(
            name="Tasklet",
            inputs={},
            outputs={"out"},
            code="out = {}".format(value))
        state.add_edge(
            tasklet_node, "out", write_node, None,
            dace.Memlet.from_array(target, write_node.desc(self.sdfg)))

    def visit_Assign(self, node: ast.Assign):
        # Validate assignment targets
        # for target in _targets(node):
        #     if isinstance(target, ast.Name) and target.id in self.global_arrays:
        #         raise DaceSyntaxError(
        #             self, target,
        #             'Cannot reassign value to parameter "%s"' % target.id)

        if len(node.targets) > 1:
            raise DaceSyntaxError(
                self, node,
                "Only 1 target per assignment is currently supported")

        target = node.targets[0]

        # Array creation (?)
        if isinstance(target, ast.Name):
            if target.id in self.global_arrays:
                raise DaceSyntaxError(
                    self, target,
                    'Cannot reassign value to parameter "%s"' % target.id)

            result = self.visit(node.value)
            name = target.id
            self.sdfg.add_array(name, result[0], result[1], transient=True)
            self.global_arrays[name] = self.sdfg.arrays[name]
        # Tasklet creation (?)
        elif isinstance(target, ast.Subscript):
            print(ast.dump(target))
            name = target.value.id
            if name not in self.global_arrays:
                raise DaceSyntaxError(
                    self, name, "Array {} has not been defined".format(name))
            if name not in self.sdfg.arrays:
                postfix = 0
                new_name = "{n}_{p}".format(n=name, p=postfix)
                while new_name in self.global_arrays:
                    postfix += 1
                    new_name = "{n}_{p}".format(n=name, p=postfix)
                rng = dace.subsets.Range(
                    astutils.subscript_to_slice(target, self.global_arrays)[1])
                shape = rng.size()
                dtype = self.global_arrays[name].dtype
                self.sdfg.add_array(new_name, shape, dtype)
                self.global_arrays[new_name] = self.sdfg.arrays[new_name]
                self.outputs[new_name] = (name, rng)
            else:
                new_name = name
            self._add_tasklet(new_name, node.value)

    def visit_AugAssign(self, node: ast.AugAssign):
        print('hi', ast.dump(node))
        pass

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
                    self, node, "Attribute {} is not shape".format(
                        rname(node)))
            shape = self.global_arrays[node.value.id].shape
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
                    self, node, "Attribute {} is not dtype".format(
                        rname(node)))
            else:
                dtype = self.global_arrays[node.value.id].dtype
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

    def visit_Call(self, node):
        print("AAAAA", rname(node))
        print("LO", ast.dump(node))

        func_id = rname(node)

        if func_id == "numpy.ndarray":
            return self._parse_ndarray(node)

    def _parse_memlet(self, src: MemletType, dst: MemletType):
        pass

    def _parse_memlet_subset(self, node: ast.Subscript):
        pass

    # Used for memlet expressions, otherwise ignored
    def visit_TopLevelExpr(self, node: ast.Expr):
        if isinstance(node.value, ast.BinOp):
            # Add two access nodes and a memlet (the arrays must already exist)
            if (isinstance(node.value.op, ast.LShift)):
                src = node.value.right
                dst = node.value.left
            elif (isinstance(node.value.op, ast.RShift)):
                src = node.value.left
                dst = node.value.right
            else:
                # Top-level binary operator that is not a memlet, does nothing
                self.generic_visit(node)
                return

            srcnode = self.state.add_read(rname(src))
            dstnode = self.state.add_write(rname(dst))
            memlet = self.parse_memlet(src, dst)
            self.state.add_nedge(srcnode, dstnode, memlet)
        # Calling reduction or other SDFGs / functions
        elif isinstance(node.value, ast.Call):
            # TODO: Handle reduction
            # TODO: Handle calling other SDFGs / DacePrograms
            raise NotImplementedError("CALLING")

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        if isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                # arrays['return'] = this
                pass
        else:
            pass
            # arrays['return'] = this
        pass
