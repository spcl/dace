import ast
from collections import OrderedDict, namedtuple
import copy
from typing import Any, Dict, List, Tuple, Union, Callable

import dace
from dace import data, dtypes, subsets
from dace.config import Config
from dace.frontend.common import op_impl
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import astutils
from dace.frontend.python.astutils import ExtNodeVisitor, ExtNodeTransformer, rname
from dace.graph import nodes
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.symbolic import pystr_to_symbolic

# A type that defines assignment information
AssignmentInfo = Tuple[SDFG, SDFGState, Tuple[str]]


@oprepo.replaces('dace.define_local')
@oprepo.replaces('numpy.ndarray')
def _define_local(sdfg: SDFG, state: SDFGState, shape, dtype):
    name, _ = sdfg.add_temp_transient(shape, dtype)
    return name


def _unop(sdfg: SDFG, state: SDFGState, op1: str, opcode: str, opname: str):
    """ Implements a general element-wise unary operator. """
    arr1 = sdfg.arrays[op1]

    name, _ = sdfg.add_temp_transient(arr1.shape, arr1.dtype, arr1.storage)
    state.add_mapped_tasklet(
        "_%s_" % opname,
        {'__i%d' % i: '0:%s' % s
         for i, s in enumerate(arr1.shape)}, {
             'in1':
             Memlet.simple(
                 op1, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))
         },
        'out = %s in1' % opcode, {
            'out':
            Memlet.simple(
                name, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))
        },
        external_edges=True)
    return name


def _binop(sdfg: SDFG, state: SDFGState, op1: str, op2: str, opcode: str,
           opname: str):
    """ Implements a general element-wise binary operator. """
    arr1 = sdfg.arrays[op1]
    arr2 = sdfg.arrays[op2]
    if (len(arr1.shape) != len(arr2.shape)
            or any(s1 != s2 for s1, s2 in zip(arr1.shape, arr2.shape))):
        raise SyntaxError('Array sizes must match')

    name, _ = sdfg.add_temp_transient(arr1.shape, arr1.dtype, arr1.storage)
    state.add_mapped_tasklet(
        "_%s_" % opname,
        {'__i%d' % i: '0:%s' % s
         for i, s in enumerate(arr1.shape)}, {
             'in1':
             Memlet.simple(
                 op1, ','.join(['__i%d' % i for i in range(len(arr1.shape))])),
             'in2':
             Memlet.simple(
                 op2, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))
         },
        'out = in1 %s in2' % opcode, {
            'out':
            Memlet.simple(
                name, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))
        },
        external_edges=True)
    return name


# Defined as a function in order to include the op and the opcode in the closure
def _makeunop(op, opcode):
    @oprepo.replaces_operator('Array', op)
    def _op(sdfg: SDFG, state: SDFGState, op1: str, op2=None):
        return _unop(sdfg, state, op1, opcode, op)


def _makebinop(op, opcode):
    @oprepo.replaces_operator('Array', op)
    def _op(sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _binop(sdfg, state, op1, op2, opcode, op)


# Define all standard Python unary operators
for op, opcode in [('UAdd', '+'), ('USub', '-'), ('Not', '!'), ('Invert',
                                                                '~')]:
    _makeunop(op, opcode)

# Define all standard Python binary operators
# NOTE: ('MatMult', '@') is defined separately
for op, opcode in [('Add', '+'), ('Sub', '-'), ('Mult', '*'), ('Div', '/'),
                   ('FloorDiv', '//'), ('Mod', '%'), ('Pow', '**'), ('LShift',
                                                                     '<<'),
                   ('RShift', '>>'), ('BitOr', '|'), ('BitXor',
                                                      '^'), ('BitAnd', '&'),
                   ('And', '&&'), ('Or', '||'), ('Eq', '=='), ('NotEq', '!='),
                   ('Lt', '<'), ('LtE', '<='), ('Gt', '>'), ('GtE', '>=')]:
    _makebinop(op, opcode)

# @oprepo.replaces('dace.define_stream')
# def _define_stream(sdfg: SDFG, state: SDFGState, dtype=dtypes.float32, buffer_size=0):
#     name, _ = sdfg.add_temp_transient(shape, dtype)
#     return name
#
# @oprepo.replaces('dace.define_streamarray')
# def _define_streamarray(sdfg: SDFG, state: SDFGState, dimensions, dtype=dtypes.float32, buffer_size=0):
#     name, _ = sdfg.add_temp_transient(shape, dtype)
#     return name


def until(val, substr):
    """ Helper function that returns the substring of a string until a certain pattern. """
    if substr not in val:
        return val
    return val[:val.find(substr)]


@oprepo.replaces('dace.reduce')
def _reduce(sdfg: SDFG,
            state: SDFGState,
            redfunction: Callable[[Any, Any], Any],
            input: str,
            output=None,
            axis=None,
            identity=None):
    # TODO(later): If output is None, derive the output size from the input and create a new node
    if output is None:
        raise NotImplementedError
    inarr = until(input, '[')
    outarr = until(output, '[')

    # Convert axes to tuple
    if axis is not None and not isinstance(axis, (tuple, list)):
        axis = (axis, )
    if axis is not None:
        axis = tuple(pystr_to_symbolic(a) for a in axis)

    # Compute memlets
    input_subset = _parse_memlet_subset(sdfg.arrays[inarr],
                                        ast.parse(input).body[0].value, {})
    input_memlet = Memlet(inarr, input_subset.num_elements(), input_subset, 1)
    output_subset = _parse_memlet_subset(sdfg.arrays[outarr],
                                         ast.parse(output).body[0].value, {})
    output_memlet = Memlet(outarr, output_subset.num_elements(), output_subset,
                           1)

    # Create reduce subgraph
    inpnode = state.add_read(inarr)
    rednode = state.add_reduce(redfunction, axis, identity)
    outnode = state.add_write(outarr)
    state.add_nedge(inpnode, rednode, input_memlet)
    state.add_nedge(rednode, outnode, output_memlet)

    return []


############################################


def parse_dace_program(f, argtypes, global_vars, modules):
    """ Parses a `@dace.program` function into a _ProgramNode object.
        @param f: A Python function to parse.
        @param argtypes: An dictionary of (name, type) for the given
                         function's arguments, which may pertain to data
                         nodes or symbols (scalars).
        @param global_vars: A dictionary of global variables in the closure
                            of `f`.
        @param modules: A dictionary from an imported module name to the
                        module itself.
        @return: Hierarchical tree of `astnodes._Node` objects, where the top
                 level node is an `astnodes._ProgramNode`.
        @rtype: astnodes._ProgramNode
    """
    src_ast, src_file, src_line, src = astutils.function_to_ast(f)

    src_ast = ModuleResolver(modules).visit(src_ast)
    # Convert modules to after resolution
    for mod, modval in modules.items():
        if mod == 'builtins':
            continue
        newmod = global_vars[mod]
        del global_vars[mod]
        global_vars[modval] = newmod

    pv = ProgramVisitor(
        f.__name__,
        src_file,
        src_line,
        # astutils.get_argtypes(src_ast.body[0], global_vars),
        argtypes,
        global_vars)

    sdfg, _, _ = pv.parse_program(src_ast.body[0])
    sdfg.set_sourcecode(src, 'python')

    return sdfg


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
    'TryFinally', 'ExceptHandler', 'Starred', 'Ellipsis', 'ClassDef',
    'AsyncFor', 'Await', 'Bytes', 'Set', 'Dict', 'ListComp', 'GeneratorExp',
    'SetComp', 'DictComp', 'comprehension'
]

MemletType = Union[ast.Call, ast.Attribute, ast.Subscript, ast.Name]
TaskletType = Union[ast.FunctionDef, ast.With, ast.For]


def _disallow_stmt(visitor, node):
    raise DaceSyntaxError(visitor, node,
                          'Keyword "%s" disallowed' % (type(node).__name__))


###############################################################
# Parsing functions
###############################################################


def _inner_eval_ast(defined, node, additional_syms=None):
    if isinstance(node, ast.AST):
        code = astutils.unparse(node)
    else:
        return node

    syms = {}
    syms.update(defined)
    if additional_syms is not None:
        syms.update(additional_syms)

    # First try to evaluate normally
    try:
        return eval(code, syms)
    except:  # Literally anything can happen here
        # If doesn't work, try to evaluate as a sympy expression
        # Replace subscript expressions with function calls (sympy support)
        code = code.replace('[', '(')
        code = code.replace(']', ')')
        return pystr_to_symbolic(code)


def _pyexpr_to_symbolic(defined_arrays_and_symbols: Dict[str, Any],
                        expr_ast: ast.AST):
    """ Converts a Python AST expression to a DaCe symbolic expression
        with error checks (raises `SyntaxError` on failure).
        @param defined_arrays_and_symbols: Defined arrays and symbols
               in the context of this expression.
        @param expr_ast: The Python AST expression to convert.
        @return: Symbolic expression.
    """
    # TODO!
    return _inner_eval_ast(defined_arrays_and_symbols, expr_ast)


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
            rb = _pyexpr_to_symbolic(das, dim[0] or 0)
            re = _pyexpr_to_symbolic(das, dim[1]
                                     or array.shape[indices[idx]]) - 1
            rs = _pyexpr_to_symbolic(das, dim[2] or 1)
            ndslice[i] = (rb, re, rs)
            offsets.append(i)
            idx += 1
        else:
            ndslice[i] = _pyexpr_to_symbolic(das, dim)

    return ndslice, offsets


MemletExpr = namedtuple('MemletExpr',
                        ['name', 'accesses', 'wcr', 'wcr_identity', 'subset'])


def _parse_memlet_subset(array: data.Data,
                         node: Union[ast.Name, ast.Subscript],
                         das: Dict[str, Any]):
    array_dependencies = {}

    # Get memlet range
    ndslice = [(0, s - 1, 1) for s in array.shape]
    if isinstance(node, ast.Subscript):
        # Parse and evaluate ND slice(s) (possibly nested)
        ast_ndslices = astutils.subscript_to_ast_slice_recursive(node)
        offsets = list(range(len(array.shape)))

        # Loop over nd-slices (A[i][j][k]...)
        subset_array = []
        for ast_ndslice in ast_ndslices:
            # Loop over the N dimensions
            ndslice, offsets = _fill_missing_slices(das, ast_ndslice, array,
                                                    offsets)
            subset_array.append(_ndslice_to_subset(ndslice))

        subset = subset_array[0]

        # Compose nested indices, e.g., of the form "A[i,:,j,:][k,l]"
        for i in range(1, len(subset_array)):
            subset = subset.compose(subset_array[i])

        # Compute additional array dependencies (as a result of
        # indirection)
        # for dim in subset:
        #     if not isinstance(dim, tuple): dim = [dim]
        #     for r in dim:
        #         for expr in symbolic.swalk(r):
        #             if symbolic.is_sympy_userfunction(expr):
        #                 arr = expr.func.__name__
        #                 array_dependencies[arr] = self.curnode.globals[arr]

    else:  # Use entire range
        subset = _ndslice_to_subset(ndslice)

    return subset


# Parses a memlet statement
def ParseMemlet(visitor, defined_arrays_and_symbols: Dict[str, Any],
                node: MemletType):
    das = defined_arrays_and_symbols
    arrname = rname(node)
    array = das[arrname]

    # Determine number of accesses to the memlet (default is the slice size)
    num_accesses = None
    write_conflict_resolution = None
    wcr_identity = None
    # Detects expressions of the form "A(2)[...]", "A(300)", "A(1, sum)[:]"
    if isinstance(node, ast.Call):
        if len(node.args) < 1 or len(node.args) > 3:
            raise DaceSyntaxError(
                visitor, node,
                'Number of accesses in memlet must be a number, symbolic '
                'expression, or -1 (dynamic)')
        num_accesses = _pyexpr_to_symbolic(das, node.args[0])
        if len(node.args) >= 2:
            write_conflict_resolution = node.args[1]
    elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call):
        if len(node.value.args) < 1 or len(node.value.args) > 3:
            raise DaceSyntaxError(
                visitor, node,
                'Number of accesses in memlet must be a number, symbolic '
                'expression, or -1 (dynamic)')
        num_accesses = _pyexpr_to_symbolic(das, node.value.args[0])
        if len(node.value.args) >= 2:
            write_conflict_resolution = node.value.args[1]

    subset = _parse_memlet_subset(array, node, das)

    # If undefined, default number of accesses is the slice size
    if num_accesses is None:
        num_accesses = subset.num_elements()

    return MemletExpr(arrname, num_accesses, write_conflict_resolution,
                      wcr_identity, subset)


def _parse_memlet(visitor, src: MemletType, dst: MemletType,
                  defined_arrays_and_symbols: Dict[str, data.Data]):
    srcexpr, dstexpr, localvar = None, None, None
    if isinstance(src,
                  ast.Name) and rname(src) not in defined_arrays_and_symbols:
        localvar = rname(src)
    else:
        srcexpr = ParseMemlet(visitor, defined_arrays_and_symbols, src)
    if isinstance(dst,
                  ast.Name) and rname(dst) not in defined_arrays_and_symbols:
        if localvar is not None:
            raise DaceSyntaxError(
                visitor, src,
                'Memlet source and destination cannot both be local variables')
        localvar = rname(dst)
    else:
        dstexpr = ParseMemlet(visitor, defined_arrays_and_symbols, dst)

    if srcexpr is not None and dstexpr is not None:
        # Create two memlets
        raise NotImplementedError
    elif srcexpr is not None:
        expr = srcexpr
    else:
        expr = dstexpr

    return localvar, Memlet(
        expr.name,
        expr.accesses,
        expr.subset,
        1,
        wcr=expr.wcr,
        wcr_identity=expr.wcr_identity)


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
                 location: str = '-1'):
        """ Creates an AST parser for tasklets. 
            @param sdfg: The SDFG to add the tasklet in (used for defined arrays and symbols).
            @param state: The SDFG state to add the tasklet to.
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
            if isinstance(node.value.op, (ast.LShift, ast.RShift)):
                if isinstance(node.value.op, ast.LShift):
                    connector, memlet = _parse_memlet(
                        self, node.value.right, node.value.left, self.defined)
                    if connector in self.inputs or connector in self.outputs:
                        raise DaceSyntaxError(
                            self, node,
                            'Local variable is already a tasklet input or output'
                        )
                    self.inputs[connector] = memlet
                elif isinstance(node.value.op, ast.RShift):
                    connector, memlet = _parse_memlet(
                        self, node.value.left, node.value.right, self.defined)
                    if connector in self.inputs or connector in self.outputs:
                        raise DaceSyntaxError(
                            self, node,
                            'Local variable is already a tasklet input or output'
                        )
                    self.outputs[connector] = memlet
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


# TODO: Take care of recursive SDFG generation w.r.t. temporary transient creation (maybe there
#  is no need if the temporary transients from the parent SDFG are added to the current SDFG arrays)


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
                 nested: bool = False,
                 parent_arrays: Dict[str, data.Data] = dict(),
                 variables: Dict[str, str] = dict()):
        self.curnode = None
        self.filename = filename
        self.lineoffset = lineoffset
        self.globals = global_vars
        self.nested = nested

        self.global_arrays = OrderedDict()  # type: Dict[str, data.Data]
        self.global_arrays.update(arrays)

        self.parent_arrays = parent_arrays

        # Entry point to the program
        self.program = None
        self.sdfg = SDFG(name)
        self.last_state = None  # self.sdfg.add_state('init', is_start_state=True)
        # if not self.nested:
        self.sdfg.arrays.update(parent_arrays)
        self.sdfg.arrays.update(arrays)
        self.inputs = {}
        self.outputs = {}

        # Keep track of variables and scopes
        self.variables = {k: k for k in arrays.keys()}  # type: Dict[str, str]
        self.variables.update(variables)

        # Add symbols. TODO: more elegant way
        for arr in arrays.values():
            for dim in arr.shape:
                if not hasattr(dim, 'free_symbols'): continue
                self.variables.update(
                    {str(k): self.globals[str(k)]
                     for k in dim.free_symbols})

        # Disallow keywords
        for stmt in _DISALLOWED_STMTS:
            setattr(self, 'visit_' + stmt, lambda n: _disallow_stmt(self, n))

    def parse_program(self, program: ast.FunctionDef):
        for stmt in program.body:
            self.visit_TopLevel(stmt)
        if len(self.sdfg.nodes()) == 0:
            self.sdfg.add_state("EmptyState")

        # TODO: Incremental, union-of-subset creation of self.inputs and self.outputs
        inputs = {}
        outputs = {}
        for state in self.sdfg.nodes():
            inputs.update({
                n.data: state.out_edges(n)[0].data
                for n in state.source_nodes()
                if isinstance(n, nodes.AccessNode) and (
                    self.sdfg.arrays[n.data].transient == False
                    or n.data in self.parent_arrays)
            })
        for state in self.sdfg.nodes():
            outputs.update({
                n.data: state.in_edges(n)[0].data
                for n in state.sink_nodes()
                if isinstance(n, nodes.AccessNode) and (
                    self.sdfg.arrays[n.data].transient == False
                    or n.data in self.parent_arrays)
            })

        return self.sdfg, inputs, outputs

    @property
    def defined(self):
        # Check parent SDFG arrays first
        # result = {
        #     k: self.parent_arrays[v]
        #     for k, v in self.variables.items() if v in self.parent_arrays
        # }
        result = {}
        result.update({
            k: self.sdfg.arrays[v]
            for k, v in self.variables.items() if v in self.sdfg.arrays
        })

        return result

    def _add_state(self, label=None):
        state = self.sdfg.add_state(label)
        if self.last_state is not None:
            self.sdfg.add_edge(self.last_state, state, dace.InterstateEdge())
        self.last_state = state
        return state

    def _parse_arg(self, arg: Any):
        """ Parse possible values to slices or objects that can be used in
            the SDFG API. """
        if isinstance(arg, ast.Subscript) and rname(arg) == '_':
            return [
                ':'.join([str(d) for d in dim]) for dim in
                astutils.subscript_to_slice(arg, self.sdfg.arrays)[1]
            ]
        return arg

    def _decorator_or_annotation_params(
            self, node: ast.FunctionDef) -> List[Tuple[str, Any]]:
        """ Returns a list of parameters, either from the function parameters
            and decorator arguments or parameters and their annotations (type
            hints).
            @param node: The given function definition node.
            @return: A list of 2-tuples (name, value).
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
                            self.global_arrays, self.globals, True,
                            self.sdfg.arrays, self.variables)

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
        state = self._add_state('s' + str(self.lineoffset))

        # Define internal node for reconnection
        internal_node = None

        # Select primitive according to function type
        if dec == 'dace.tasklet':  # Tasklet
            internal_node, inputs, outputs = self._parse_tasklet(state, node)

        elif dec.startswith('dace.map') or dec.startswith(
                'dace.consume'):  # Scope or scope+tasklet
            params = self._decorator_or_annotation_params(node)
            if 'map' in dec:
                entry, exit = state.add_map(node.name, ndrange=params)
            elif 'consume' in dec:
                entry, exit = state.add_consume(node.name, **params)

            if dec.endswith('scope'):  # @dace.mapscope or @dace.consumescope
                sdfg, inputs, outputs = self._parse_subprogram(node.name, node)
                internal_node = nodes.NestedSDFG(node.name, sdfg,
                                                 set(inputs.keys()),
                                                 set(outputs.keys()))
            else:  # Scope + tasklet (e.g., @dace.map)
                internal_node, inputs, outputs = self._parse_tasklet(
                    state, node)

            # Connect internal node with scope/access nodes
            self._add_dependencies(state, internal_node, entry, exit, inputs,
                                   outputs)

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

        if not isinstance(node, (ast.Call, ast.Subscript)):
            raise DaceSyntaxError(
                self, node,
                "Iterator of ast.For must be a function or a subscript")

        iterator = rname(node)
        if iterator not in {'range', 'parrange', 'dace.map'}:
            raise DaceSyntaxError(
                self, node, "Iterator {} is unsupported".format(iterator))
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

    def _add_dependencies(
            self, state: SDFGState, internal_node: nodes.CodeNode,
            entry_node: nodes.EntryNode, exit_node: nodes.ExitNode,
            inputs: Dict[str, Memlet], outputs: Dict[str, Memlet]):
        if inputs:
            for conn, memlet in inputs.items():
                read_node = state.add_read(memlet.data)
                state.add_memlet_path(
                    read_node,
                    entry_node,
                    internal_node,
                    memlet=memlet,
                    src_conn=None,
                    dst_conn=conn)
        else:
            state.add_nedge(entry_node, internal_node, dace.EmptyMemlet())
        if outputs:
            for conn, memlet in outputs.items():
                write_node = state.add_write(memlet.data)
                state.add_memlet_path(
                    internal_node,
                    exit_node,
                    write_node,
                    memlet=memlet,
                    src_conn=conn,
                    dst_conn=None)
        else:
            state.add_nedge(internal_node, exit_node, dace.EmptyMemlet())

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
            me, mx = state.add_map(
                name='Map',
                ndrange={k: ':'.join(v)
                         for k, v in zip(indices, ranges)})
            # body = SDFG('MapBody')
            body, inputs, outputs = self._parse_subprogram('MapBody', node)
            tasklet = state.add_nested_sdfg(body, self.sdfg, inputs.keys(),
                                            outputs.keys())
            self._add_dependencies(state, tasklet, me, mx, inputs, outputs)
        elif iterator == 'range':
            # Add an initial loop state with a None last_state (so as to not
            # create an interstate edge)
            laststate = self.last_state
            self.last_state = None
            first_loop_state = self._add_state('for_%d' % node.lineno)

            # Recursive loop processing
            for stmt in node.body:
                self.visit_TopLevel(stmt)

            # Create the next state
            last_loop_state = self.last_state
            self.last_state = None
            end_loop_state = self._add_state('endfor_%d' % node.lineno)

            # Add loop to SDFG
            loop_cond = '>' if ((pystr_to_symbolic(ranges[0][2]) <
                                 0) == True) else '<'
            self.sdfg.add_loop(
                laststate, first_loop_state, end_loop_state, indices[0],
                ranges[0][0],
                '%s %s %s' % (indices[0], loop_cond, ranges[0][1]),
                '%s + %s' % (indices[0], ranges[0][2]), last_loop_state)

    def visit_While(self, node: ast.While):
        pass

    def visit_If(self, node: ast.If):
        pass

    def _parse_index(self, node: ast.Index):

        indices = []
        for idx in node.value.elts:
            indices.append(self._parse_value(idx))

        return indices

    def _parse_tasklet(self, state: SDFGState, node: TaskletType):
        ttrans = TaskletTransformer(self.defined, self.sdfg, state,
                                    self.filename)
        node, inputs, outputs = ttrans.parse_tasklet(node)

        # Convert memlets to their actual data nodes
        for i in inputs.values():
            i.data = self.variables[i.data]
        for o in outputs.values():
            o.data = self.variables[o.data]
        return node, inputs, outputs

    def _add_tasklet(self, target, value_node):
        value = self._parse_value(value_node)
        print(target, value)

        if isinstance(target, tuple):
            name, subset = target
            memlet = dace.Memlet(name, subset.num_elements(), subset, 1)
        else:
            name = target
            memlet = dace.Memlet.from_array(name, self.sdfg.arrays[name])

        state = self._add_state("TaskletState")
        write_node = state.add_write(name)
        tasklet_node = state.add_tasklet(
            name="Tasklet",
            inputs={},
            outputs={"out"},
            code="out = {}".format(value))
        state.add_edge(tasklet_node, "out", write_node, None, memlet)
        # dace.Memlet.from_array(name, write_node.desc(self.sdfg)))

    def _get_variable_name(self, node, name):
        if name not in self.variables:
            raise DaceSyntaxError(self, node,
                                  'Array "%s" used before definition' % name)
        else:
            new_name = self.variables[name]
            if new_name not in self.sdfg.arrays:
                arr = self.parent_arrays[new_name]
                self.sdfg.arrays[new_name] = arr

        return new_name

    def visit_Assign(self, node: ast.Assign):
        # Validate assignment targets
        # for target in _targets(node):
        #     if isinstance(target, ast.Name) and target.id in self.global_arrays:
        #         raise DaceSyntaxError(
        #             self, target,
        #             'Cannot reassign value to parameter "%s"' % target.id)
        #
        # if len(node.targets) > 1:
        #     raise DaceSyntaxError(
        #         self, node,
        #         "Only 1 target per assignment is currently supported")

        if not isinstance(node.targets[0], ast.Tuple):
            elts = [node.targets[0]]
        else:
            elts = node.targets[0].elts

        state = self._add_state("assign_%d" % node.lineno)

        # Return result (UnaryOp, BinOp, BoolOp, Call, etc.)
        results = self.visit(node.value)
        if not isinstance(results, (tuple, list)):
            results = [results]
        if len(results) != len(elts):
            raise DaceSyntaxError(
                self, node,
                'Function returns %d values but %d provided' % (len(results),
                                                                len(elts)))

        for target, result in zip(elts, results):
            # Variable assignment
            if isinstance(target, ast.Name):
                if target.id in self.global_arrays:
                    raise DaceSyntaxError(
                        self, target,
                        'Cannot reassign value to parameter "%s"' % target.id)

                name = target.id
                self.variables[name] = result

            # Variable broadcast
            elif isinstance(target, ast.Subscript):
                print(ast.dump(target))
                name = self._get_variable_name(node, target.value.id)
                arr = self.sdfg.arrays[name]
                self.outputs[name] = (name, dace.subsets.Range.from_array(arr))
                # postfix = 0
                # new_name = "{n}_{p}".format(n=name, p=postfix)
                # while new_name in self.global_arrays:
                #     postfix += 1
                #     new_name = "{n}_{p}".format(n=name, p=postfix)
                rng = dace.subsets.Range(
                    astutils.subscript_to_slice(target, self.global_arrays)[1])
                # shape = rng.size()
                # dtype = self.global_arrays[name].dtype
                # self.sdfg.add_array(new_name, shape, dtype)
                # self.global_arrays[new_name] = self.sdfg.arrays[new_name]
                # self.outputs[new_name] = (name, rng)

                self._add_tasklet((name, rng), node.value)

    def visit_AugAssign(self, node: ast.AugAssign):

        print(ast.dump(node))

        state = self._add_state("AugAssignState")

        if isinstance(node.target, ast.Name):
            name = self._get_variable_name(node, node.target.id)
            output_node = state.add_write(name)
        else:
            raise NotImplementedError

        if isinstance(node.value, ast.Name):
            name = self._get_variable_name(node, node.value.id)
            input_node = state.add_read(name)
            constant = 1
        elif isinstance(node.value, ast.BinOp):
            if isinstance(node.value.left, ast.Name):
                name = self._get_variable_name(node, node.value.left.id)
                input_node = state.add_read(name)
            elif isinstance(node.value.right, ast.Name):
                name = self._get_variable_name(node, node.value.right.id)
                input_node = state.add_read(name)
            else:
                raise NotImplementedError
            if isinstance(node.value.left, ast.Num):
                constant = node.value.left.n
            elif isinstance(node.value.right, ast.Num):
                constant = node.value.right.n
            else:
                raise NotImplementedError

        if isinstance(node.op, ast.Add):
            pass
        elif isinstance(node.op, ast.Sub):
            constant = -constant
        else:
            raise NotImplementedError

        op_impl.constant_array_multiplication(
            state,
            constant,
            input_node,
            input_node,
            output_node,
            output_node,
            accumulate=True,
            label="AugAssign")

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

    def _parse_function_arg(self, arg: ast.AST):
        # Obtain a string representation
        return self.visit(arg)

    def visit_Call(self, node):
        default_impl = Config.get('frontend', 'implementation')
        funcname = rname(node)
        func = oprepo.Replacements.get(funcname, default_impl)
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
                % (funcname, default_impl))

        result = func(
            self.sdfg, self.last_state,
            *(self._parse_function_arg(arg) for arg in node.args), **{
                arg.arg: self._parse_function_arg(arg.value)
                for arg in node.keywords
            })
        if not isinstance(result, (tuple, list)):
            return [result]
        return result

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

            state = self._add_state()
            srcnode = state.add_read(rname(src))
            dstnode = state.add_write(rname(dst))
            memlet = self.parse_memlet(src, dst)
            state.add_nedge(srcnode, dstnode, memlet)
            return

        # Calling reduction or other SDFGs / functions
        elif isinstance(node.value, ast.Call):
            # Handles reduction and calling other SDFGs / DaCe programs
            self._add_state('call_%d' % node.lineno)
            self.visit_Call(node.value)
            return

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

    def visit_With(self, node, is_async=False):
        # "with dace.tasklet" syntax
        if len(node.items) == 1:
            dec = node.items[0].context_expr
            funcname = rname(dec)
            if funcname == 'dace.tasklet':
                # Parse as tasklet
                state = self._add_state('with_%d' % node.lineno)
                tasklet, inputs, outputs = self._parse_tasklet(state, node)

                # Add memlets
                for connector, memlet in inputs.items():
                    accessnode = state.add_read(memlet.data)
                    state.add_edge(accessnode, None, tasklet, connector,
                                   memlet)
                for connector, memlet in outputs.items():
                    accessnode = state.add_write(memlet.data)
                    state.add_edge(tasklet, connector, accessnode, None,
                                   memlet)
                return

        raise DaceSyntaxError(
            self, node,
            'General "with" statements disallowed in DaCe programs')

    def visit_AsyncWith(self, node):
        return self.visit_With(node, is_async=True)

    def _visitname(self, name: str, node: ast.AST):
        # If an allowed global, use directly
        if name in self.globals:
            return _inner_eval_ast(self.globals, node)

        if name not in self.variables:
            raise DaceSyntaxError(self, node,
                                  'Use of undefined variable "%s"' % name)
        return self.variables[name]

    #### Visitors that return arrays
    def visit_Str(self, node: ast.Str):
        # A string constant returns itself
        return node.s

    def visit_Num(self, node: ast.Str):
        # A constant returns itself
        return node.n

    def visit_Name(self, node: ast.Name):
        # If visiting a name, check if it is a defined variable or a global
        return self._visitname(node.id, node)

    def visit_Attribute(self, node: ast.Attribute):
        # If visiting an attribute, return attribute value if it's of an array or global
        name = until(astutils.unparse(node), '.')
        result = self._visitname(name, node)
        try:
            return getattr(self.sdfg.arrays[result], node.attr)
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

    def _gettype(self, opnode: ast.AST):
        """ Returns an operand and its type as a 2-tuple of strings. """
        operand = self.visit(opnode)
        if isinstance(operand, (list, tuple)) and len(operand) != 1:
            raise DaceSyntaxError(self, opnode, 'Operand cannot be a tuple')
        operand = operand[0]

        if isinstance(operand, str) and operand in self.sdfg.arrays:
            return operand, type(self.sdfg.arrays[operand]).__name__
        else:
            return operand, type(operand).__name__

    def _visit_op(self, node: Union[ast.UnaryOp, ast.BinOp, ast.BoolOp],
                  op1: ast.AST, op2: ast.AST):
        default_impl = Config.get('frontend', 'implementation')
        opname = type(node.op).__name__

        # Parse operands
        operand1, op1type = self._gettype(op1)
        if op2 is not None:
            operand2, op2type = self._gettype(op2)
        else:
            operand2, op2type = None, None

        func = oprepo.Replacements.getop(
            op1type, opname, implementation=default_impl, otherclass=op2type)
        if func is None:
            # Check for SDFG as fallback
            func = oprepo.Replacements.getop(
                op1type, opname, otherclass=op2type)
            if func is None:
                raise DaceSyntaxError(
                    self, node,
                    'Operator "%s" is not defined for types %s and %s' %
                    (opname, op1type, op2type))
            print(
                'WARNING: Operator "%s" is not registered with an %s implementation for'
                'types %s and %s, falling back to SDFG' %
                (opname, default_impl, op1type, op2type))

        self._add_state('%s_%d' % (type(node).__name__, node.lineno))
        result = func(self.sdfg, self.last_state, operand1, operand2)
        if not isinstance(result, (tuple, list)):
            return [result]
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
