from __future__ import print_function
import ast
import astunparse
from collections import OrderedDict
import copy
from functools import wraps
import inspect

from dace import data, subsets, symbolic, types
from dace.config import Config
from dace.frontend.python import astnodes, astutils
from dace.frontend.python.astutils import *


def function_to_ast(f):
    """ Obtain the source code of a Python function and create an AST. 
        @param f: Python function.
        @return: A 4-tuple of (AST, function filename, function line-number,
                               source code as string).
    """
    try:
        src = inspect.getsource(f)
    # TypeError: X is not a module, class, method, function, traceback, frame,
    # or code object; OR OSError: could not get source code
    except (TypeError, OSError):
        raise TypeError('cannot obtain source code for dace program')

    src_file = inspect.getfile(f)
    _, src_line = inspect.findsource(f)
    src_ast = ast.parse(_remove_outer_indentation(src))
    ast.increment_lineno(src_ast, src_line)

    return src_ast, src_file, src_line, src


def _remove_outer_indentation(src: str):
    """ Removes extra indentation from a source Python function.
        @param src: Source code (possibly indented).
        @return: Code after de-indentation.
    """
    lines = src.split('\n')
    indentation = len(lines[0]) - len(lines[0].lstrip())
    return '\n'.join([line[indentation:] for line in lines])


class FindLocals(ast.NodeVisitor):
    """ Python AST node visitor that recovers all left-hand-side (stored)
        locals. """

    def __init__(self):
        self.locals = {}

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.locals[node.id] = node


def parse_dace_program(f, argtypes, global_vars, modules):
    """ Parses a `@dace.program` function into a _ProgramNode object. 
        @param f: A Python function to parse.
        @param argtypes: An iterable of tuples (name, type) for the given
                         function's arguments.
        @param global_vars: A dictionary of global variables in the closure
                            of `f`.
        @param modules: A dictionary from an imported module name to the
                        module itself.
        @return: Hierarchical tree of `astnodes._Node` objects, where the top
                 level node is an `astnodes._ProgramNode`.
        @rtype: astnodes._ProgramNode
    """
    src_ast, src_file, src_line, src = function_to_ast(f)

    # Find local variables
    local_finder = FindLocals()
    local_finder.visit(src_ast)
    local_vars = local_finder.locals

    # 1. Inline all "dace.call"ed functions
    inliner = FunctionInliner(global_vars, modules, local_vars)
    inliner.visit(src_ast)

    # 2. resolve all the symbols in the AST
    allowed_globals = global_vars.copy()
    allowed_globals.update(argtypes)
    symresolver = SymbolResolver(allowed_globals)
    symresolver.visit(src_ast)

    # 3. Parse the DaCe program to a hierarchical dependency representation
    ast_parser = ParseDaCe(src_file, src_line, argtypes, global_vars, modules,
                           symresolver)
    ast_parser.visit(src_ast)
    pdp = ast_parser.program
    pdp.source = src
    pdp.filename = src_file
    pdp.param_syms = sorted(symbolic.getsymbols(argtypes.values()).items())
    pdp.argtypes = argtypes

    return pdp


class MemletRemover(ExtNodeTransformer):
    """ A Python AST transformer that removes memlet expressions of the type
        `a << b[c]` and `d >> e(f)[g]`. """

    def visit_TopLevelExpr(self, node):
        # This is a DaCe shift, omit it
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.op, ast.LShift) or isinstance(
                    node.value.op, ast.RShift):
                return None
        return self.generic_visit(node)


class ModuleInliner(ExtNodeTransformer):
    """ A Python AST transformer that renames modules from their imported alias
        to their actual name. """

    def __init__(self, modules):
        self.modules = modules

    def visit_Attribute(self, node):
        attrname = rname(node)
        module_name = attrname[:attrname.rfind('.')]
        if module_name in self.modules:  # math or equivalent modules
            modname = self.modules[module_name]
            node.value = ast.copy_location(
                ast.Name(id=(modname), ctx=ast.Load), node.value)
            return node
        return self.generic_visit(node)


# Parses a DaCe program
class ParseDaCe(ExtNodeVisitor):
    """ A Python AST visitor that creates DaCe program trees.
        @see: parse_dace_program
    """

    def __init__(self, filename, lineoffset, argtypes, global_vars, modules,
                 symresolver):
        self.curnode = None
        self.program_name = None
        self.filename = filename
        self.lineoffset = lineoffset
        self.argtypes = argtypes
        self.modules = modules
        self.globals = global_vars
        self.symresolver = symresolver

        # Maps: {array name: data.Data)}
        self.global_arrays = OrderedDict()
        self.global_arrays.update(argtypes)

        # Entry point to the program
        self.program = None

    ###############################################################
    # Helper functions
    ###############################################################
    def _get_module(self, node):
        try:
            fullmodname = inspect.getmodule(eval(unparse(node),
                                                 self.globals)).__name__
        except NameError:
            fullmodname = ''
        # Only use the top-level module
        if fullmodname.find('.') >= 0:
            return fullmodname[:fullmodname.find('.')]
        return fullmodname

    def _inner_eval_ast(self, node, additional_syms=None):
        code = unparse(node)
        syms = {}
        syms.update(self.curnode.globals)
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
            return symbolic.pystr_to_symbolic(code)

    def _compile_ast(self, node_body, line_offset, filename):
        self.symresolver.visit(node_body)
        wrapper = ast.Module(body=[node_body])

        if line_offset is not None:
            for node in ast.walk(wrapper):
                node.lineno = line_offset
                node.col_offset = 0

        codeobj = compile(wrapper, filename, 'exec')
        gen_module = {}
        gen_module.update(self.globals)
        exec(codeobj, gen_module)
        return gen_module[node_body.name]

    def _eval_ast(self, node):
        if node is None:
            return None
        elif isinstance(node, ast.Call):
            # Only work on allowed functions and external functions according to
            # decision flowchart for intra-program function evaluation:
            # 1. Does it exist in the same program + already parsed?
            # 2. Is it a @dace.external_function?
            # 3. Is it one of the standard functions from the allowed module?
            # 4. If neither of the previous, fail
            func = rname(node)

            # Function call to a tasklet defined within the same program
            if func in self.curnode.globals and isinstance(
                    self.curnode.globals[func], ast.FunctionDef):
                # Since the function is never compiled by Python, we need to
                # do so ourselves
                compiled_func = self._compile_ast(
                    self.curnode.globals[func], self.lineoffset, self.filename)
                return self._inner_eval_ast(node, {func: compiled_func})

            # Standard function call, e.g., int(), math.sin()
            elif self._get_module(node.func) in self.modules:
                return self._inner_eval_ast(node)

            # External function calls
            elif func in self.globals:
                if isinstance(self.globals[func], types._external_function):
                    # External function needs to be recompiled with current
                    # symbols
                    src_ast, src_file, src_line, src = function_to_ast(
                        self.globals[func].func)
                    compiled_func = self._compile_ast(src_ast.body[0],
                                                      src_line, src_file)
                    return self._inner_eval_ast(node, {func: compiled_func})
                else:
                    return self._inner_eval_ast(node)

            else:
                return self._inner_eval_ast(node)
        elif isinstance(node, ast.FunctionDef):
            compiled_sdfg = self._compile_ast(node, node.lineno, self.filename)
            return compiled_sdfg.to_sdfg()
        else:
            # Not a function, try to evaluate
            return self._inner_eval_ast(node)

    # Track local variables
    def _set_locals(self):
        if self.curnode.parent is None:
            # Handle parameters (first set all to symbols, then set type
            # descriptors for arrays)
            self.curnode.globals.update(
                {k: symbolic.symbol(k)
                 for k in self.curnode.params})
            self.curnode.globals.update(self.globals)
            self.curnode.globals.update(self.global_arrays)
        else:
            self.curnode.globals.update(self.curnode.parent.globals)
            self.curnode.globals.update(
                {k: symbolic.symbol(k)
                 for k in self.curnode.params})

    # Helper function to find the dtype of an array, either as a keyword or
    # as the last parameter
    def getarg_or_kwarg(self, node, argoff, argname):
        if len(node.args) > argoff:
            return node.args[argoff]
        for k in node.keywords:
            if rname(k) == argname:
                return k.value
        return None

    ###############################################################
    # Parsing functions
    ###############################################################

    def _ndslice_to_subset(self, ndslice):
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

    def _fill_missing_slices(self, ast_ndslice, array, indices):
        # Filling ndslice with default values from array dimensions
        # if ranges not specified (e.g., of the form "A[:]")
        ndslice = [None] * len(ast_ndslice)
        ndslice_size = 1
        offsets = []
        idx = 0
        for i, dim in enumerate(ast_ndslice):
            if isinstance(dim, tuple):
                rb = self._eval_ast(dim[0])
                re = self._eval_ast(dim[1])
                if re is not None:
                    re -= 1
                rs = self._eval_ast(dim[2])
                if rb is None: rb = 0
                if re is None: re = array.shape[indices[idx]] - 1
                if rs is None: rs = 1
                ndslice[i] = (rb, re, rs)
                offsets.append(i)
                idx += 1
            else:
                ndslice[i] = self._eval_ast(dim)

        return ndslice, offsets

    # Parses a memlet statement
    def ParseMemlet(self, local_name, rhsnode):
        rhs = rname(rhsnode)
        if rhs.find('.') >= 0:  # attribute, form G.out_edges[:]
            arrname = rhs[:rhs.find('.')]
            arrattr = rhs[rhs.find('.') + 1:]
        else:  # normal memlet, form A(1)[i,j]
            arrname = rhs
            arrattr = None

        array = self.curnode.globals[arrname]

        # Determine number of accesses to the memlet (default is the slice size)
        num_accesses = None
        write_conflict_resolution = None
        wcr_identity = None
        # Detects expressions of the form "A(2)[...]", "A(300)", "A(1, sum)[:]"
        if isinstance(rhsnode, ast.Call):
            if len(rhsnode.args) < 1 or len(rhsnode.args) > 3:
                raise DaCeSyntaxError(
                    self, rhsnode,
                    'Number of accesses in memlet must be a number, symbolic '
                    'expression, or -1')
            num_accesses = self._eval_ast(rhsnode.args[0])
            if len(rhsnode.args) >= 2:
                write_conflict_resolution = rhsnode.args[1]
            if len(rhsnode.args) >= 3:
                wcr_identity = ast.literal_eval(rhsnode.args[2])
        elif isinstance(rhsnode, ast.Subscript) and isinstance(
                rhsnode.value, ast.Call):
            if len(rhsnode.value.args) < 1 or len(rhsnode.value.args) > 3:
                raise DaCeSyntaxError(
                    self, rhsnode,
                    'Number of accesses in memlet must be a number, symbolic '
                    'expression, or -1')
            num_accesses = self._eval_ast(rhsnode.value.args[0])
            if len(rhsnode.value.args) >= 2:
                write_conflict_resolution = rhsnode.value.args[1]
            if len(rhsnode.value.args) >= 3:
                wcr_identity = ast.literal_eval(rhsnode.value.args[2])

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
                ndslice, offsets = self._fill_missing_slices(
                    ast_ndslice, array, offsets)
                subset_array.append(self._ndslice_to_subset(ndslice))

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
        return astnodes._Memlet(
            array, arrname, arrattr, num_accesses, write_conflict_resolution,
            wcr_identity, subset, 1, local_name, rhsnode, array_dependencies)

    # Helper function: parses DaCe array statement
    def ParseArrayStatement(self, node, bInput):
        if self.curnode is None:
            raise DaCeSyntaxError(
                self, node,
                'DaCe load/store statement declared outside function bounds')

        lhs = rname(node.value.left)
        rhs = rname(node.value.right)

        if rhs.find('.') >= 0:  # attribute, form G.out_edges[:]
            arrname = rhs[:rhs.find('.')]
            arrattr = rhs[rhs.find('.') + 1:]
        else:  # normal memlet, form A(1)[i,j]
            arrname = rhs
            arrattr = None

        arrays = self.curnode.arrays()

        # If this is not an undefined symbol (and the rhs is not a DaCe array),
        # this is just a regular shift
        if lhs in self.curnode.locals:
            if arrname not in arrays:
                return
            else:
                raise DaCeSyntaxError(
                    self, node,
                    'Cannot load/store DaCe variable using an existing symbol')

        if arrname not in arrays:
            raise DaCeSyntaxError(
                self, node, 'Cannot load/store DaCe variable "' + arrname +
                '" from a non-DaCe array')

        lhs_name = lhs
        if lhs in arrays:
            lhs = arrays[lhs]

        # Make sure the DaCe assignment is unique
        if lhs in self.curnode.inputs:
            raise DaCeSyntaxError(
                self, node, 'Variable already assigned to another input')
        if lhs in self.curnode.outputs:
            raise DaCeSyntaxError(
                self, node, 'Variable already assigned to another output')

        ########################
        # Determine the properties of the memlet
        memlet = self.ParseMemlet(lhs_name, node.value.right)

        if bInput:
            self.curnode.inputs[lhs_name] = memlet
        else:
            self.curnode.outputs[lhs_name] = memlet

    def ParseCallAssignment(self, node, target):
        funcname = rname(node.func)
        modname = self._get_module(node.func)

        ######################################
        # Handle DaCe-specific calls
        if modname == 'dace':  # modname is already the real name of the module
            # Do not allow instantiation of ND arrays and DaCe scalars
            if funcname == "ndarray" or funcname == "scalar":
                raise DaCeSyntaxError(
                    self, node,
                    'Cannot define a DaCe array within a program, try using '
                    'dace.define_local or dace.define_local_scalar')

            # Handle transient variables
            if funcname.endswith(".define_local"):
                if len(node.args) < 1:
                    raise DaCeSyntaxError(
                        self, node,
                        'Invalid call to define_local, at least 1 parameter '
                        'is required')
                if self.getarg_or_kwarg(node, 1, 'dtype') is None:
                    raise DaCeSyntaxError(
                        self, node,
                        'Transient variable declaration must specify type')

                # Construct type descriptor
                shape = self._eval_ast(node.args[0])
                dtype = self._eval_ast(self.getarg_or_kwarg(node, 1, 'dtype'))
                allow_conflicts = self._eval_ast(
                    self.getarg_or_kwarg(node, 2, 'allow_conflicts'))
                allow_conflicts = False if allow_conflicts is None else True
                try:
                    tdesc = data.Array(
                        dtype,
                        shape,
                        transient=True,
                        allow_conflicts=allow_conflicts)
                except TypeError as ex:
                    raise DaCeSyntaxError(self, node, str(ex))

                self.curnode.transients[rname(target)] = tdesc
                self.curnode.globals[rname(target)] = tdesc
                return None

            elif funcname.endswith(".define_local_scalar"):
                if self.getarg_or_kwarg(node, 0, 'dtype') is None:
                    raise DaCeSyntaxError(
                        self, node,
                        'Transient variable declaration must specify type')

                # Construct type descriptor
                dtype = self._eval_ast(self.getarg_or_kwarg(node, 0, 'dtype'))
                allow_conflicts = self._eval_ast(
                    self.getarg_or_kwarg(node, 1, 'allow_conflicts'))
                allow_conflicts = False if allow_conflicts is None else True

                tdesc = data.Scalar(
                    dtype, transient=True, allow_conflicts=allow_conflicts)

                self.curnode.transients[rname(target)] = tdesc
                self.curnode.globals[rname(target)] = tdesc
                return None
            elif funcname.endswith(".define_stream") or funcname.endswith(
                    ".define_streamarray"):
                argOffset = 0
                if funcname.endswith('array'):
                    # Defined stream array, expecting shape
                    shape = self._eval_ast(
                        self.getarg_or_kwarg(node, 0, 'shape'))
                    argOffset += 1
                else:
                    shape = [1]

                dtype = self._eval_ast(
                    self.getarg_or_kwarg(node, argOffset, 'dtype'))

                # Optional parameters
                internal_size = self._eval_ast(
                    self.getarg_or_kwarg(node, argOffset + 1, 'buffer_size'))

                tdesc = data.Stream(
                    dtype, 1, internal_size, shape=shape, transient=True)

                self.curnode.transients[rname(target)] = tdesc
                self.curnode.globals[rname(target)] = tdesc
                return None
            elif (funcname.rfind('.') != -1
                  and funcname[funcname.rfind('.') +
                               1:] in types.TYPECLASS_STRINGS):
                return node
            else:
                raise DaCeSyntaxError(
                    self, node, 'Unrecognized function call \'%s\'' % funcname)
        ######################################
        # Other calls are treated as memlet functions (independent of arrays,
        # inline-able)
        else:
            return node

    def _add_possible_inputs(self, nodes, prim):
        if not isinstance(nodes, list):
            nodes = [nodes]
        extended_nodes = []
        final_nodes = []

        # Extract values from lists, tuples and subsets
        for node in nodes:
            if isinstance(node, tuple):
                final_nodes.extend(list(node))
            elif isinstance(node, subsets.Range):
                for dim in node.ranges:
                    final_nodes.extend(list(dim))
            elif isinstance(node, subsets.Indices):
                final_nodes.extend(list(node))

        # Find AST names
        for node in extended_nodes:
            if isinstance(node, ast.AST):
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Name):
                        final_nodes.append(subnode.id)
            else:
                final_nodes.append(node)
        nodeset = set()
        for n in final_nodes:
            if symbolic.issymbolic(n):
                nodeset.update(str(s) for s in n.free_symbols)
            elif isinstance(n, str):
                nodeset.add(n)

        arrs = self.curnode.arrays()
        for input in nodeset:
            if input in arrs:
                inName = '__DACEIN_' + input
                prim.inputs[inName] = astnodes._Memlet(
                    arrs[input], input, None, 1, None, None,
                    subsets.Indices([0]), 1, None, None, {})

    ###############################################################
    # AST visiting functions
    ###############################################################

    def visit_FunctionDef(self, node, is_async=False):
        # Obtain function name
        parent_node = self.curnode
        curprim = None

        arrays = OrderedDict()
        if self.curnode is not None:
            arrays = self.curnode.arrays()

        # Obtain program/primitive name (only one program is allowed)
        if (len(node.decorator_list) > 0):
            if (len(node.decorator_list) > 1):
                raise DaCeSyntaxError(self, node,
                                      'Only one DaCe decorator is allowed')

            # Make sure that the module is DaCe
            dec = node.decorator_list[0]
            decname = rname(dec)
            if isinstance(dec, ast.Call):
                modname = self._get_module(dec.func)
            else:
                modname = self._get_module(dec)
            if modname not in self.modules.values() or modname != 'dace':
                raise DaCeSyntaxError(
                    self, node,
                    'Decorators from module \'%s\' not allowed' % modname)
            #####################################

            # Create DaCe program node
            if decname.endswith('.program'):
                if self.program is not None:
                    # Parse internal program separately as an SDFG of its own
                    sdfg = self._eval_ast(node)
                    curprim = astnodes._NestedSDFGNode(node.name, node, sdfg)

                    # Inherit I/O from immediate parent
                    curprim.inputs = copy.copy(parent_node.inputs)
                    curprim.outputs = copy.copy(parent_node.outputs)
                    # Cancel parent node's relevant I/O
                    parent_node.inputs.clear()
                    parent_node.outputs.clear()

                    # Set children of parent primitive, if it is a primitive
                    if parent_node is not None and curprim is not None:
                        parent_node.children.append(curprim)
                    curprim.parent = parent_node

                    # Exit so that child AST nodes will not be parsed
                    return

                self.program = astnodes._ProgramNode(node.name, node)
                curprim = self.program

            # Parse primitives
            # Dataflow primitives
            elif decname.endswith('map'):
                curprim = astnodes._MapNode(node.name, node)

                # If the arguments are defined in the decorator
                if 'args' in dir(dec) and len(dec.args) > 0:
                    curprim.range = subsets.Range(
                        subscript_to_slice(dec.args[0], arrays)[1])
                else:
                    try:
                        curprim.range = subsets.Range([
                            subscript_to_slice(arg.annotation, arrays)[1][0]
                            for arg in node.args.args
                        ])
                    except (AttributeError, TypeError, ValueError):
                        raise DaCeSyntaxError(
                            self, node,
                            'All arguments in DaCe primitive %s must be annotated with a range'
                            % node.name)
                self._add_possible_inputs(curprim.range, curprim)

            elif decname.endswith('consume'):
                curprim = astnodes._ConsumeNode(node.name, node)

                # If the arguments are defined in the decorator
                if 'args' in dir(dec) and len(dec.args) > 0:
                    if dec.args[0].id not in self.curnode.globals:
                        raise DaCeSyntaxError(
                            self, node, 'Undefined stream %s in consume %s' %
                            (dec.args[0].id, node.name))
                    curprim.stream = self.curnode.globals[rname(dec.args[0])]
                    ast_memlet = self.ParseMemlet(node.args.args[0].arg,
                                                  dec.args[0])
                    ast_memlet.num_accesses = -1
                    curprim.inputs[node.args.args[0].arg] = ast_memlet
                    if len(dec.args) < 2:
                        raise DaCeSyntaxError(
                            self, node,
                            'Consume %s missing required argument: '
                            'number of processing elements' % node.name)
                    curprim.num_pes = symbolic.pystr_to_symbolic(
                        unparse(dec.args[1]))
                    if len(dec.args) > 2:
                        curprim.condition = unparse(dec.args[2])
                    else:
                        curprim.condition = None
                else:
                    raise DaCeSyntaxError(
                        self, node,
                        'Consume syntax only supports parameters at the '
                        'decorator')
                self._add_possible_inputs(curprim.stream, curprim)

            elif decname.endswith('tasklet'):
                # Parse arguments
                lang = None
                gcode = None
                if isinstance(dec, ast.Call):
                    lang = self._eval_ast(
                        self.getarg_or_kwarg(dec, 0, 'language'))
                    gcode = self._eval_ast(
                        self.getarg_or_kwarg(dec, 1, 'global_code'))

                if lang is None:
                    lang = types.Language.Python
                else:
                    try:
                        lang = types.Language[lang]
                    except KeyError:
                        raise DaCeSyntaxError(
                            self, node,
                            'Unrecognized tasklet language "%s"' % lang)
                if gcode is None:
                    gcode = ''

                curprim = astnodes._TaskletNode(node.name, node, lang, gcode)

            # Control flow primitives
            elif decname.endswith('iterate'):
                if isinstance(parent_node, astnodes._DataFlowNode):
                    raise DaCeSyntaxError(
                        self, node, 'Control flow within data flow disallowed')

                curprim = astnodes._IterateNode(node.name, node)

                if 'args' in dir(dec) and len(
                        dec.args
                ) > 0:  # If the arguments are defined in the decorator
                    curprim.range = subsets.Range(
                        subscript_to_slice(dec.args[0], arrays)[1])
                else:
                    try:
                        curprim.range = subsets.Range([
                            subscript_to_slice(arg.annotation, arrays)[1][0]
                            for arg in node.args.args
                        ])
                    except (AttributeError, TypeError, ValueError):
                        raise SyntaxError(
                            'All arguments in DaCe primitive %s must be annotated with a range'
                            % node.name)
                self._add_possible_inputs(curprim.range, curprim)

            elif decname.endswith('loop'):
                if isinstance(parent_node, astnodes._DataFlowNode):
                    raise DaCeSyntaxError(
                        self, node, 'Control flow within data flow disallowed')

                curprim = astnodes._LoopNode(node.name, node)

                if 'args' in dir(dec) and len(
                        dec.args
                ) > 0:  # If the arguments are defined in the decorator
                    curprim.condition = dec.args[0]
                else:
                    raise SyntaxError(
                        'Condition must be given as argument to decorator in DaCe primitive %s'
                        % node.name)
                self._add_possible_inputs(curprim.condition, curprim)

            elif decname.endswith('conditional'):
                if isinstance(parent_node, astnodes._DataFlowNode):
                    raise DaCeSyntaxError(
                        self, node, 'Control flow within data flow disallowed')

                curprim = astnodes._ConditionalNode(node.name, node)

                if 'args' in dir(dec) and len(
                        dec.args
                ) > 0:  # If the arguments are defined in the decorator
                    curprim.condition = dec.args[0]
                else:
                    raise SyntaxError(
                        'Condition must be given as argument to decorator in DaCe primitive %s'
                        % node.name)
                self._add_possible_inputs(curprim.condition, curprim)

            else:
                raise DaCeSyntaxError(self, node,
                                      'Unrecognized primitive ' + decname)

            if '.async_' in decname or is_async:
                curprim.is_async = True
        # End of program/primitive name

        # If this is a primitive
        if curprim is not None:
            # If function definition contains arguments
            if 'args' in dir(node):
                for arg in node.args.args:

                    # If it is not the program, add locals as symbols
                    if self.program != node.name:
                        curprim.globals[rname(arg)] = symbolic.symbol(
                            rname(arg))
                    if curprim is not None:
                        curprim.params.append(rname(arg))

            # Set children of parent primitive, if it is a primitive
            if parent_node is not None and curprim is not None:
                parent_node.children.append(curprim)
            curprim.parent = parent_node

            self.curnode = curprim

            # Track local variables
            self._set_locals()

            # Mandatory (to keep visiting children)
            for stmt in node.body:
                self.visit_TopLevel(stmt)

            # After traversing the function, pop "function name stack"
            self.curnode = parent_node
        else:  # Not a primitive
            self.curnode.locals[node.name] = node
            self.curnode.globals[node.name] = node

            # Mandatory (to keep visiting children)
            for stmt in node.body:
                self.visit_TopLevel(stmt)

    def visit_AsyncFunctionDef(self, node):
        # Treat as a plain function
        self.visit_FunctionDef(node, is_async=True)

    def visit_Call(self, node):
        if (not isinstance(node.func, ast.Attribute)
                or node.func.value.id not in self.modules
                or self.modules[node.func.value.id] != 'dace'):
            self.generic_visit(node)
            return

        # Reduce call
        if node.func.attr.endswith('reduce'):
            dec = node
            # Mandatory arguments
            wcr = dec.args[0]
            src = dec.args[1]
            dst = dec.args[2]
            # In case the axis argument is given without explicit kwarg
            # notation
            axisarg = dec.args[3] if len(dec.args) > 3 else None
            identityarg = dec.args[4] if len(dec.args) > 4 else None

            curprim = astnodes._ReduceNode('reduce', wcr)
            curprim.axes = get_tuple(self, getkwarg(dec, 'axis', axisarg))
            curprim.identity = get_tuple(
                self, getkwarg(dec, 'identity', identityarg))
            if curprim.identity is not None:
                curprim.identity = curprim.identity[0]
            curprim.inputs['input'] = self.ParseMemlet('input', src)
            curprim.outputs['output'] = self.ParseMemlet('output', dst)

            # Set children of parent primitive, if it is a primitive
            self.curnode.children.append(curprim)
            curprim.parent = self.curnode

    def visit_TopLevelExpr(self, node):
        if isinstance(node.value, ast.BinOp):
            if (isinstance(node.value.op, ast.LShift)):
                self.ParseArrayStatement(node, True)
                return
            if (isinstance(node.value.op, ast.RShift)):
                self.ParseArrayStatement(node, False)
                return
        elif isinstance(node.value, ast.Str):
            self.visit_TopLevelStr(node.value)
            return

        self.generic_visit(node)

    def visit_TopLevelStr(self, node):
        if isinstance(self.curnode, astnodes._TaskletNode):
            if self.curnode.extcode != None:
                raise DaCeSyntaxError(
                    self, node,
                    'Cannot provide more than one intrinsic implementation ' +
                    'for tasklet')
            self.curnode.extcode = node.s
            return

        self.generic_visit(node)

    # Detect locals and transient variables
    def visit_Assign(self, node):
        # Don't allow assignment to tuples (for now)
        if len(node.targets) > 1:
            raise DaCeSyntaxError(self, node,
                                  'Assignment to tuples not supported (yet)')
        target = node.targets[0]
        if isinstance(target, ast.Tuple):
            if len(target.elts) > 1:
                raise DaCeSyntaxError(
                    self, node, 'Assignment to tuples not supported (yet)')
            target = target.elts[0]

        # Tasklet code
        if self.curnode is not None:
            if isinstance(node.value, ast.Call) and\
               not isinstance(self.curnode, astnodes._TaskletNode):
                retval = self.ParseCallAssignment(node.value, target)
                if retval is not None:
                    self.curnode.locals[rname(target)] = retval
                    self.curnode.globals[rname(target)] = retval

                # No need to further visit the node's children
                return
            else:
                if isinstance(self.curnode, astnodes._DataFlowNode):
                    self.curnode.locals[rname(target)] = None
                    self.curnode.globals[rname(target)] = None
                else:
                    retval = self._eval_ast(node.value)
                    self.curnode.locals[rname(target)] = retval
                    self.curnode.globals[rname(target)] = retval

                # No need to further visit the node's children
                return

        self.generic_visit(node)

    # Visit statements that define locals
    def visit_Name(self, node):
        if self.curnode is None:
            arrays = self.global_arrays
        else:
            arrays = self.curnode.arrays()

        if node.id in arrays and (not isinstance(arrays[node.id], data.Scalar)
                                  or arrays[node.id].transient):
            if isinstance(node.ctx, ast.Load) or isinstance(
                    node.ctx, ast.Store):
                raise DaCeSyntaxError(
                    self, node,
                    'Directly reading from and writing to arrays is not '
                    'allowed. Please use memlet notation (a << A[i])')

        self.generic_visit(node)

    # Control flow blocks
    #########################
    def visit_For(self, node):
        # Syntax: Only accept for loops without 'else'; only accept for loops
        # with structure 'for <x> in range(<y>)'
        if len(node.orelse) > 0:
            raise DaCeSyntaxError(
                self, node,
                'Loops with \'else\' footer are not allowed in DaCe programs')

        if self.curnode is not None:
            # Verify syntax
            ########################################################
            # We allow only three types of for loops:
            # 1. `for i in range(...)`: Creates a looping state
            # 2. `for i in parrange(...)`: Creates a 1D map
            # 3. `for i,j,k in dace.map[0:M, 0:N, 0:K]`: Creates an ND map

            if isinstance(node.iter, ast.Call):
                funcname = rname(node.iter.func)
                modname = self._get_module(node.iter.func)
            elif isinstance(node.iter, ast.Subscript):
                funcname = rname(node.iter.value)
                modname = self._get_module(node.iter.value)
            else:
                funcname, modname = None, None

            # Case 1: Iterate
            if (isinstance(node.target, ast.Name)
                    and isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == 'range'):
                # If we are inside a dataflow construct, ignore
                if isinstance(self.curnode, astnodes._DataFlowNode):
                    self.generic_visit(node)
                    return

                # Obtain parameters
                varname = node.target.id
                nargs = len(node.iter.args)
                var_rb = 0 if nargs < 2 else symbolic.pystr_to_symbolic(
                    unparse(node.iter.args[0]))
                var_re = (symbolic.pystr_to_symbolic(
                    unparse(node.iter.args[1]))
                          if nargs > 1 else symbolic.pystr_to_symbolic(
                              unparse(node.iter.args[0]))) - 1
                var_rs = 1 if nargs < 3 else symbolic.pystr_to_symbolic(
                    unparse(node.iter.args[2]))

                # Create node
                curprim = astnodes._IterateNode('iterate_' + str(node.lineno),
                                                node)
                curprim.range = [(var_rb, var_re, var_rs)]
                curprim.params = [varname]
                self._add_possible_inputs(curprim.range, curprim)
                self.curnode.children.append(curprim)
                curprim.parent = self.curnode

                # Traverse into loop
                oldnode = self.curnode
                self.curnode = curprim
                self._set_locals()
                for stmt in node.body:
                    self.visit(stmt)
                self.curnode = oldnode
                ####################
                return

            # Case 2: 1D map (for i in parrange(...))
            elif (isinstance(node.target, ast.Name)
                  and isinstance(node.iter, ast.Call)
                  and isinstance(node.iter.func, ast.Name)
                  and node.iter.func.id == 'parrange'):
                curprim = astnodes._MapNode('map_' + str(node.lineno), node)

                # Get arguments for range
                maprange = []
                if len(node.iter.args) == 1:  # end only
                    maprange = [(None, node.iter.args[0], None)]
                elif len(node.iter.args) == 2:  # begin, end
                    maprange = [(node.iter.args[0], node.iter.args[1], None)]
                elif len(node.iter.args) == 3:  # begin, end, skip
                    maprange = [(node.iter.args[0], node.iter.args[1],
                                 node.iter.args[2])]
                else:
                    raise DaCeSyntaxError(
                        self, node,
                        'Invalid number of arguments for "parrange"')

                curprim.range = subsets.Range(
                    astrange_to_symrange(maprange, self.curnode.arrays()))
                curprim.params = [rname(node.target)]

                self._add_possible_inputs(curprim.range, curprim)
                self.curnode.children.append(curprim)
                curprim.parent = self.curnode

                # Traverse into loop
                oldnode = self.curnode
                self.curnode = curprim
                self._set_locals()
                for stmt in node.body:
                    self.visit(stmt)
                self.curnode = oldnode
                ####################

                return

            # Case 3: ND map
            elif (isinstance(node.target, ast.Tuple)
                  and isinstance(node.iter, ast.Subscript)
                  and isinstance(node.iter.value, ast.Attribute)
                  and modname == 'dace' and node.iter.value.attr == 'map'):
                curprim = astnodes._MapNode('map_' + str(node.lineno), node)

                # Get range from array subscript, check for length mismatch
                _, range_values = subscript_to_slice(node.iter,
                                                     self.curnode.arrays())
                range_keys = [rname(n) for n in node.target.elts]
                if len(range_keys) != len(range_values):
                    raise DaCeSyntaxError(
                        self, node,
                        'Map range must match tuple length in for loop')
                curprim.params = range_keys
                curprim.range = subsets.Range(range_values)

                self._add_possible_inputs(curprim.range, curprim)
                self.curnode.children.append(curprim)
                curprim.parent = self.curnode

                # Traverse into loop
                oldnode = self.curnode
                self.curnode = curprim
                self._set_locals()
                for stmt in node.body:
                    self.visit(stmt)
                self.curnode = oldnode
                ####################

                return

            # No match
            else:
                raise DaCeSyntaxError(
                    self, node, 'Invalid loop syntax. Supported options are:\n'
                    '    for <var> in range(<value>)\n'
                    '    for <var> in parrange(<value>)\n'
                    '    for <vars> in dace.map[ranges]')
            #######################################################

        self.generic_visit(node)

    def visit_While(self, node):
        # Syntax: Only accept while loops without 'else'
        if len(node.orelse) > 0:
            raise DaCeSyntaxError(
                self, node,
                'Loops with \'else\' footer are not allowed in DaCe programs')

        if self.curnode is not None:
            # If we are inside a dataflow construct, ignore
            if not isinstance(self.curnode, astnodes._DataFlowNode):
                # Obtain parameters
                cond = node.test

                # Create node
                curprim = astnodes._LoopNode('while_' + str(node.lineno), node)
                curprim.condition = cond
                self._add_possible_inputs(curprim.condition, curprim)
                self.curnode.children.append(curprim)
                curprim.parent = self.curnode

                # Traverse into loop
                oldnode = self.curnode
                self.curnode = curprim
                self._set_locals()
                for stmt in node.body:
                    self.visit(stmt)
                self.curnode = oldnode
                ####################
                return

        self.generic_visit(node)

    def visit_If(self, node):
        if self.curnode is not None:
            # If we are inside a dataflow construct, ignore
            if not isinstance(self.curnode, astnodes._DataFlowNode):
                # Obtain parameters
                cond = node.test

                # Create node
                curprim = astnodes._IfNode('if_' + str(node.lineno), node)
                curprim.condition = cond
                self._add_possible_inputs(curprim.condition, curprim)
                self.curnode.children.append(curprim)
                curprim.parent = self.curnode

                # Traverse into condition
                oldnode = self.curnode
                self.curnode = curprim
                self._set_locals()
                for stmt in node.body:
                    self.visit(stmt)
                self.curnode = oldnode

                # Process 'else'/'elif' statements
                if len(node.orelse) > 0:
                    # Create node
                    curprim = astnodes._ElseNode(
                        'else_' + str(node.orelse[0].lineno), node)
                    # Negate condition
                    curprim.condition = astutils.negate_expr(cond)
                    self.curnode.children.append(curprim)
                    curprim.parent = self.curnode

                    # Traverse into condition
                    oldnode = self.curnode
                    self.curnode = curprim
                    self._set_locals()
                    for stmt in node.orelse:
                        self.visit(stmt)
                    self.curnode = oldnode

                return

        self.generic_visit(node)

    def visit_With(self, node, is_async=False):
        # "with dace.tasklet" syntax
        if len(node.items) == 1:
            dec = node.items[0].context_expr
            if isinstance(dec, ast.Call):
                funcname = rname(dec.func)
                modname = self._get_module(dec.func)
            elif isinstance(dec, ast.Attribute):
                funcname = rname(dec)
                modname = self._get_module(dec)
            else:
                funcname, modname = None, None

            if modname == 'dace' and funcname.endswith('.tasklet'):
                # Parse as tasklet
                # NOTE: This is almost a direct copy of the tasklet parser
                # above.
                lang = None
                gcode = None
                if isinstance(dec, ast.Call):
                    lang = self._eval_ast(
                        self.getarg_or_kwarg(dec, 0, 'language'))
                    gcode = self._eval_ast(
                        self.getarg_or_kwarg(dec, 1, 'global_code'))

                if lang is None:
                    lang = types.Language.Python
                else:
                    try:
                        lang = types.Language[lang]
                    except KeyError:
                        raise DaCeSyntaxError(
                            self, node,
                            'Unrecognized tasklet language "%s"' % lang)
                if gcode is None:
                    gcode = ''

                curprim = astnodes._TaskletNode('tasklet_' + str(node.lineno),
                                                node, lang, gcode)
                if self.curnode is not None:
                    self.curnode.children.append(curprim)
                curprim.parent = self.curnode

                # Traverse into tasklet
                oldnode = self.curnode
                self.curnode = curprim
                self._set_locals()
                for stmt in node.body:
                    self.visit_TopLevel(stmt)
                self.curnode = oldnode
                return

        raise DaCeSyntaxError(
            self, node,
            'General "with" statements disallowed in DaCe programs')

    #########################

    ## Disallowed statements
    def visit_Global(self, node):
        raise DaCeSyntaxError(
            self, node, '"global" statements disallowed in DaCe sub-programs')

    def visit_Delete(self, node):
        raise DaCeSyntaxError(self, node,
                              '"del" statements disallowed in DaCe programs')

    def visit_Import(self, node):
        raise DaCeSyntaxError(self, node,
                              'imports disallowed in DaCe programs')

    def visit_ImportFrom(self, node):
        raise DaCeSyntaxError(self, node,
                              'imports disallowed in DaCe programs')

    def visit_Assert(self, node):
        raise DaCeSyntaxError(
            self, node, '"assert" statements disallowed in DaCe programs')

    def visit_Pass(self, node):
        raise DaCeSyntaxError(self, node,
                              '"pass" statements disallowed in DaCe programs')

    def visit_Exec(self, node):
        raise DaCeSyntaxError(self, node,
                              '"exec" statements disallowed in DaCe programs')

    def visit_Print(self, node):
        raise DaCeSyntaxError(
            self, node, '"print" statements disallowed in DaCe programs')

    def visit_Nonlocal(self, node):
        raise DaCeSyntaxError(
            self, node, '"nonlocal" statements disallowed in DaCe programs')

    def visit_Yield(self, node):
        raise DaCeSyntaxError(
            self, node, '"yield" statements disallowed in DaCe programs')

    def visit_YieldFrom(self, node):
        raise DaCeSyntaxError(
            self, node, '"yield" statements disallowed in DaCe programs')

    def visit_Raise(self, node):
        raise DaCeSyntaxError(self, node,
                              'exceptions disallowed in DaCe programs')

    def visit_Try(self, node):
        raise DaCeSyntaxError(self, node,
                              'exceptions disallowed in DaCe programs')

    def visit_TryExcept(self, node):
        raise DaCeSyntaxError(self, node,
                              'exceptions disallowed in DaCe programs')

    def visit_TryFinally(self, node):
        raise DaCeSyntaxError(self, node,
                              'exceptions disallowed in DaCe programs')

    def visit_ExceptHandler(self, node):
        raise DaCeSyntaxError(self, node,
                              'exceptions disallowed in DaCe programs')

    def visit_AsyncWith(self, node):
        self.visit_With(node, is_async=True)

    def visit_Starred(self, node):
        raise DaCeSyntaxError(
            self, node, 'starred statements disallowed in DaCe programs')

    def visit_Ellipsis(self, node):
        raise DaCeSyntaxError(self, node,
                              '"..." statements disallowed in DaCe programs')

    # disallowed for now
    def visit_ClassDef(self, node):
        raise DaCeSyntaxError(self, node,
                              'classes disallowed (for now) in DaCe programs')

    def visit_AsyncFor(self, node):
        raise DaCeSyntaxError(
            self, node,
            'asynchronous loops disallowed (for now) in DaCe programs')

    def visit_Await(self, node):
        raise DaCeSyntaxError(self, node,
                              'await disallowed (for now) in DaCe programs')

    #Data structures
    def visit_Bytes(self, node):
        raise DaCeSyntaxError(
            self, node, 'bytestrings disallowed (for now) in DaCe programs')

    def visit_Set(self, node):
        raise DaCeSyntaxError(self, node,
                              'sets disallowed (for now) in DaCe programs')

    def visit_Dict(self, node):
        raise DaCeSyntaxError(
            self, node, 'dictionaries disallowed (for now) in DaCe programs')

    #Comprehensions
    def visit_ListComp(self, node):
        raise DaCeSyntaxError(self, node,
                              'comprehensions disallowed in DaCe programs')

    def visit_GeneratorExp(self, node):
        raise DaCeSyntaxError(self, node,
                              'comprehensions disallowed in DaCe programs')

    def visit_SetComp(self, node):
        raise DaCeSyntaxError(self, node,
                              'comprehensions disallowed in DaCe programs')

    def visit_DictComp(self, node):
        raise DaCeSyntaxError(self, node,
                              'comprehensions disallowed in DaCe programs')

    def visit_comprehension(self, node):
        raise DaCeSyntaxError(self, node,
                              'comprehensions disallowed in DaCe programs')

    def visit_ImportFrom(self, node):
        raise DaCeSyntaxError(self, node,
                              'imports disallowed in DaCe programs')


class ASTFindAndReplace(ast.NodeTransformer):
    """ A Python AST transformer utility that finds and replaces names. """

    def __init__(self, replacements, skip_subscripts=True):
        self.replacement_dict = replacements
        self.skip_subscripts = skip_subscripts

    def visit_Subscript(self, node):
        # Do not visit subscripts that contain a replacement
        if rname(node) in self.replacement_dict and self.skip_subscripts:
            return node
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.replacement_dict:
            return ast.copy_location(
                ast.Name(id=self.replacement_dict[node.id], ctx=node.ctx),
                node)

        return self.generic_visit(node)


class SymbolResolver(astutils.ExtNodeTransformer):
    """ Python AST transformer that resolves symbols to their name or 
        value. """

    def __init__(self, symbols):
        self.symbols = symbols
        self.locals = {}
        self.top_function = True

    def resolve(self, node):
        if node is None:
            return None
        if isinstance(node, tuple):
            return tuple(self.resolve(n) for n in node)
        return unparse(self.visit(node))

    def visit_FunctionDef(self, node):
        oldlocals = {}
        oldlocals.update(self.locals)
        oldtop = self.top_function

        # Register parameters as locals
        if not self.top_function:
            for arg in node.args.args:
                self.locals[rname(arg)] = arg

        self.top_function = False
        result = self.generic_visit(node)
        self.top_function = oldtop

        self.locals = oldlocals

        return result

    def visit_TopLevelExpr(self, node):
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.op, ast.LShift) or isinstance(
                    node.value.op, ast.RShift):
                self.locals[rname(node.value.left)] = node.value.left

                node.value.right = self.visit(node.value.right)
                return node

        return self.generic_visit(node)

    def visit_Name(self, node):
        # Defining a local
        if isinstance(node.ctx, ast.Store):
            # TODO(later): Scope management
            # Example:
            # n = 5
            # @dace.program
            # def prog():
            #     def inner():
            #         n = dace.define_local(...)
            #         use n (should be "n")
            #     use n (should be 5)

            self.locals[node.id] = node
            return node

        if node.id not in self.symbols:
            return node
        if node.id in self.locals:
            return node

        sym = self.symbols[node.id]
        if isinstance(sym, symbolic.symbol):
            return ast.copy_location(ast.Name(id=sym.name, ctx=node.ctx), node)
        elif isinstance(sym, types.typeclass):
            # Find dace module name
            dacemodule = next(
                k for k, v in self.symbols.items()
                if isinstance(v, type(types)) and v.__name__ == 'dace')

            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id=dacemodule, ctx=ast.Load()),
                    attr=sym.to_string(),
                    ctx=node.ctx), node)
        elif types.isconstant(sym):
            return ast.copy_location(ast.Num(n=sym, ctx=node.ctx), node)
        elif isinstance(sym, ast.Name):
            return ast.copy_location(ast.Name(id=sym.id, ctx=node.ctx), node)
        elif isinstance(sym, ast.AST):
            return ast.copy_location(sym, node)
        else:
            return node


##########################################################################
# Function inlining


class CounterDict(object):
    """ Dictionary object that counts how many times a value was added to 
        it. """

    def __init__(self):
        self.values = {}

    def get(self, key):
        if key in self.values:
            return self.values[key]
        else:
            return 0

    def add(self, key, count=1):
        if key not in self.values:
            self.values[key] = count
        else:
            self.values[key] += count


class FunctionInliner(ExtNodeTransformer):
    """ A Python AST transformer that inlines functions called (e.g., with 
        "dace.call") in an existing AST. """

    def __init__(self, global_vars, modules, local_vars={}):
        self.globals = global_vars
        self.locals = local_vars
        self.modules = modules
        self.function_inline_counter = CounterDict()

    def visit_Call(self, node):
        cnode = node

        # First, visit arguments and (possibly) inline them. This takes care
        # of "dace.call(func, dace.call(f2, arg), ...)" cases
        node = self.generic_visit(node)

        # Only accept "dace.call" calls
        if isinstance(cnode.func, ast.Attribute) and cnode.func.attr == 'call':
            # Verify that the module is DaCe
            if self.modules[cnode.func.value.id] == 'dace':
                # INLINE
                if len(cnode.args) < 1:
                    raise SyntaxError(
                        'dace.call must have at least one parameter')
                return self.inline_function(cnode, cnode.args[0])

        return node

    # Inline top-level calls as well
    def visit_TopLevelExpr(self, node):
        if isinstance(node.value, ast.Call):
            node.value = self.visit_TopLevelCall(node.value)
            return node
        return self.generic_visit(node)

    def _fname_and_module(self, funcnode):
        funcmodule = None
        if isinstance(funcnode, ast.Attribute):
            funcmodule = funcnode.value.id
            funcname = funcnode.attr
        else:
            funcname = funcnode.id
        return (funcmodule, funcname)

    def visit_TopLevelCall(self, node):
        # If dace.call(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'call':
            return self.visit_Call(node)

        funcmodule, funcname = self._fname_and_module(node.func)
        if funcmodule is None and funcname in self.globals:
            # First, visit arguments and (possibly) inline them. This takes care
            # of "dace.call(func, dace.call(f2, arg), ...)" cases
            node = self.generic_visit(node)

            return self.inline_function(node, node.func)

        return self.generic_visit(node)

    def _transients_from_ast(self, src_ast):
        results = set()
        for astnode in ast.walk(src_ast):
            if (isinstance(astnode, ast.Assign)
                    and isinstance(astnode.value, ast.Call)):
                modulename, _ = self._fname_and_module(astnode.value.func)
                if (modulename is not None
                        and self.modules[modulename] == 'dace'):
                    # Don't allow assignment to tuples (for now)
                    if len(astnode.targets) > 1:
                        raise DaCeSyntaxError(
                            self, astnode,
                            'Assignment to tuples not supported (yet)')
                    target = astnode.targets[0]
                    if isinstance(target, ast.Tuple):
                        if len(target.elts) > 1:
                            raise DaCeSyntaxError(
                                self, node,
                                'Assignment to tuples not supported (yet)')
                        target = target.elts[0]

                    results.add(rname(target))
        return results

    def inline_function(self, cnode, funcnode):
        funcmodule, funcname = self._fname_and_module(funcnode)
        if funcmodule is None and funcname not in self.globals:
            raise SyntaxError(
                'Function %s not found (is it declared as @dace.external_function?)'
                % funcname)
        if funcmodule is not None:
            raise SyntaxError('External DaCe functions should be' +
                              ' imported directly using "from ' +
                              '<MODULE> import ..."')

        self.function_inline_counter.add(funcname)

        # Obtain the function object
        f = None
        if isinstance(self.globals[funcname], types._external_function):
            f = self.globals[funcname].func
        else:
            f = self.globals[funcname]

        # Parse that function's AST
        src_ast, src_file, src_line, src = function_to_ast(f)

        # Inline the function's intenal dace.calls recursively
        for astnode in ast.walk(src_ast):
            if isinstance(astnode, ast.Call):
                src_ast = FunctionInliner(self.globals,
                                          self.modules).visit(src_ast)
                break

        # Replace the function's parameters with the values in arguments
        func_args = src_ast.body[0].args.args

        if cnode.func == funcnode:  # In case of calling a function directly
            call_args = cnode.args[:]
        else:  # In case of calling a function through dace.call
            call_args = cnode.args[1:]
        if len(func_args) != len(call_args):
            raise SyntaxError(
                'Mismatch in arguments to call %s. Expecting %d, got %d' %
                (f.__name__, len(func_args), len(call_args)))

        replacement_map = {  # parameter replacement map
            rname(k): v
            for k, v in zip(func_args, call_args)
        }

        # Obtain and rename transients as well. "tmp" --> "func0_tmp"
        local_replacement_map = {
            k: ast.Name(
                ctx=ast.Load(),
                id='%s%d_%s' % (funcname,
                                self.function_inline_counter.get(funcname), k))
            for k in self._transients_from_ast(src_ast)
        }
        for replacement in local_replacement_map.values():
            for repl_ast in ast.walk(replacement):
                if isinstance(repl_ast, ast.Name):
                    if (repl_ast.id in self.globals
                            or repl_ast.id in self.locals):
                        raise SyntaxError(
                            ('Cannot name a symbol %s due to function ' +
                             'inlining, please choose another name') %
                            repl_ast.id)
        replacement_map.update(local_replacement_map)

        src_ast = SymbolResolver(replacement_map).visit(src_ast)

        # If the function has a return statement, then we need to
        # evaluate the AST instead
        if any(isinstance(stmt, ast.Return) for stmt in ast.walk(src_ast)):
            if len(src_ast.body[0].body) > 1:
                raise NotImplementedError(
                    "Functions with return value and more than one statement are not implemented"
                )

            # Inline the function by replacing the return value
            return src_ast.body[0].body[0].value

        return src_ast.body[0].body
