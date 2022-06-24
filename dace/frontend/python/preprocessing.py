# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import collections
import copy
from dataclasses import dataclass
import inspect
import numbers
import numpy
import re
import sympy
import sys
import warnings

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import dace
from dace import data, dtypes, subsets, symbolic, sdfg as sd
from dace.config import Config
from dace.sdfg import SDFG
from dace.frontend.python import astutils
from dace.frontend.python.common import (DaceSyntaxError, SDFGConvertible, SDFGClosure)


class DaceRecursionError(Exception):
    """
    Exception that indicates a recursion in a data-centric parsed context.
    The exception includes the id of the topmost function as a stopping
    condition for parsing.
    """
    def __init__(self, fid: int):
        self.fid = fid

    def __str__(self) -> str:
        return ('Non-analyzable recursion detected, function cannot be parsed ' 'as data-centric')


@dataclass
class PreprocessedAST:
    """
    Python AST and metadata of a preprocessed @dace.program/method, for use
    in parsing. 
    """
    filename: str
    src_line: int
    src: str
    preprocessed_ast: ast.AST
    program_globals: Dict[str, Any]


class StructTransformer(ast.NodeTransformer):
    """ A Python AST transformer that replaces `Call`s to create structs with
        the custom StructInitializer AST node. """
    def __init__(self, gvars):
        super().__init__()
        self._structs = {k: v for k, v in gvars.items() if isinstance(v, dtypes.struct)}

    def visit_Call(self, node: ast.Call):
        # Struct initializer
        name = astutils.unparse(node.func)
        if name not in self._structs:
            return self.generic_visit(node)

        # Parse name and fields
        struct = self._structs[name]
        name = struct.name
        fields = {astutils.rname(arg.arg): arg.value for arg in node.keywords}
        if tuple(sorted(fields.keys())) != tuple(sorted(struct.fields.keys())):
            raise SyntaxError('Mismatch in fields in struct definition')

        # Create custom node
        #new_node = astutils.StructInitializer(name, fields)
        #return ast.copy_location(new_node, node)

        node.func = ast.copy_location(ast.Name(id='__DACESTRUCT_' + name, ctx=ast.Load()), node.func)

        return node


# Replaces instances of modules Y imported with "import X as Y" by X
class ModuleResolver(ast.NodeTransformer):
    def __init__(self, modules: Dict[str, str], always_replace=False):
        self.modules = modules
        self.should_replace = False
        self.always_replace = always_replace

    def visit_Call(self, node) -> Any:
        self.should_replace = True
        node.func = self.visit(node.func)
        self.should_replace = False
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if not self.should_replace and not self.always_replace:
            return self.generic_visit(node)
        # Traverse AST until reaching the top-level value (could be a name
        # or a function)
        cnode = node
        while isinstance(cnode.value, ast.Attribute):
            cnode = cnode.value

        if (isinstance(cnode.value, ast.Name) and cnode.value.id in self.modules):
            cnode.value.id = self.modules[cnode.value.id]

        return self.generic_visit(node)


class RewriteSympyEquality(ast.NodeTransformer):
    """ 
    Replaces symbolic equality checks by ``sympy.{Eq,Ne}``. 
    This is done because a test ``if x == 0`` where ``x`` is a symbol would
    result in False, even in indeterminate cases.
    """
    def __init__(self, globals: Dict[str, Any]) -> None:
        super().__init__()
        self.globals = globals

    def visit_Compare(self, node: ast.Compare) -> Any:
        if len(node.comparators) != 1:
            return self.generic_visit(node)
        left = astutils.evalnode(self.visit(node.left), self.globals)
        right = astutils.evalnode(self.visit(node.comparators[0]), self.globals)
        if (isinstance(left, sympy.Basic) or isinstance(right, sympy.Basic)):
            if isinstance(node.ops[0], ast.Eq):
                return sympy.Eq(left, right)
            elif isinstance(node.ops[0], ast.NotEq):
                return sympy.Ne(left, right)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, numpy.bool_):
            node.value = bool(node.value)
        elif isinstance(node.value, numpy.number):
            node.value = node.value.item()
        return self.generic_visit(node)

    # Compatibility for Python 3.7
    def visit_Num(self, node):
        if isinstance(node.n, numpy.bool_):
            node.n = bool(node.n)
        elif isinstance(node.n, numpy.number):
            node.n = node.n.item()
        return self.generic_visit(node)


class ConditionalCodeResolver(ast.NodeTransformer):
    """ 
    Replaces if conditions by their bodies if can be evaluated at compile time.
    """
    def __init__(self, globals: Dict[str, Any]):
        super().__init__()
        self.globals_and_locals = copy.copy(globals)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self.globals_and_locals[node.id] = SyntaxError
        return self.generic_visit(node)

    def visit_If(self, node: ast.If) -> Any:
        node = self.generic_visit(node)
        try:
            test = RewriteSympyEquality(self.globals_and_locals).visit(node.test)
            result = astutils.evalnode(test, self.globals_and_locals)

            # Check symbolic conditions separately
            if isinstance(result, sympy.Basic):
                if result == True:
                    # Only return "if" body
                    return node.body
                elif result == False:
                    # Only return "else" body
                    return node.orelse
                else:
                    # Any other case is indeterminate, fall back to generic visit
                    return node
            else:  # If not symbolic, check value directly
                if result:
                    # Only return "if" body
                    return node.body
                elif not result:
                    # Only return "else" body
                    return node.orelse

        except SyntaxError:
            # Cannot evaluate if condition at compile time
            pass

        return node

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        return self.visit_If(node)


class _FindBreakContinueStmts(ast.NodeVisitor):
    """
    Find control statements in the given loop (break / continue), without
    traversing into nested loops.
    """
    def __init__(self) -> None:
        super().__init__()
        self.has_cflow = False

    def visit_For(self, node):
        # Skip visiting contents
        return

    def visit_AsyncFor(self, node):
        # Skip visiting contents
        return

    def visit_While(self, node):
        # Skip visiting contents
        return

    def visit_Break(self, node):
        self.has_cflow = True
        return self.generic_visit(node)

    def visit_Continue(self, node):
        self.has_cflow = True
        return self.generic_visit(node)


class DeadCodeEliminator(ast.NodeTransformer):
    """ Removes any code within scope after return/break/continue/raise. """
    def generic_visit(self, node: ast.AST):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                # Scope fields
                scope_field = field in ('body', 'orelse')

                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                        elif (scope_field and isinstance(value, (ast.Return, ast.Break, ast.Continue, ast.Raise))):
                            # Any AST node after this one is unreachable and
                            # not parsed by this transformer
                            new_values.append(value)
                            break
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


def has_replacement(callobj: Callable, parent_object: Optional[Any] = None, node: Optional[ast.AST] = None) -> bool:
    """
    Returns True if the function/operator replacement repository
    has a registered replacement for the called function described by
    a live object.
    """
    from dace.frontend.common import op_repository as oprepo

    # Nothing from the `dace` namespace needs preprocessing
    mod = None
    try:
        mod = callobj.__module__
    except AttributeError:
        try:
            mod = parent_object.__module__
        except AttributeError:
            pass
    if mod and (mod == 'dace' or mod.startswith('dace.') or mod == 'math' or mod.startswith('math.')):
        return True

    # Attributes and methods
    classname = None
    if parent_object is not None:
        classname = type(parent_object).__name__
        attrname = callobj.__name__
        repl = oprepo.Replacements.get_attribute(classname, attrname)
        if repl is not None:
            return True
        repl = oprepo.Replacements.get_method(classname, attrname)
        if repl is not None:
            return True

    # NumPy ufuncs
    if (isinstance(callobj, numpy.ufunc) or isinstance(parent_object, numpy.ufunc)):
        return True

    # Functions
    # Special case: Constructor method (e.g., numpy.ndarray)
    if classname == "type":
        cbqualname = astutils.rname(node)
        if oprepo.Replacements.get(cbqualname) is not None:
            return True
    full_func_name = callobj.__module__ + '.' + callobj.__qualname__
    if oprepo.Replacements.get(full_func_name) is not None:
        return True

    # Also try the function as it is called in the AST
    return oprepo.Replacements.get(astutils.rname(node)) is not None


def _create_unflatten_instruction(arg: ast.AST) -> Tuple[Callable, int]:
    """
    Creates a lambda function for recreating the original Python object and returns the number of 
    arguments to increment.
    """
    if isinstance(arg, ast.List):
        return (list, len(arg.elts))
    elif isinstance(arg, ast.Tuple):
        return (tuple, len(arg.elts))
    elif isinstance(arg, ast.Set):
        return (set, len(arg.elts))
    elif isinstance(arg, ast.Dict):
        # Use two levels of functions to preserve keyword names
        def make_remake(kwnames):
            def remake_dict(args):
                return {k: a for k, a in zip(kwnames, args)}

            return remake_dict

        # Remake keyword argument names from AST
        kwarg_names = []
        for kw in arg.keys:
            if isinstance(kw, ast.Num):
                kwarg_names.append(kw.n)
            elif isinstance(kw, (ast.Str, ast.Bytes)):
                kwarg_names.append(kw.s)
            elif isinstance(kw, ast.NameConstant):
                kwarg_names.append(kw.value)
            elif sys.version_info >= (3, 8) and isinstance(kw, ast.Constant):
                kwarg_names.append(kw.value)
            else:
                raise NotImplementedError(f'Key type {type(kw).__name__} is not supported')

        return (make_remake(kwarg_names), len(arg.keys))
    return (None, 1)


def flatten_callback(func: Callable, node: ast.Call):
    """
    Creates a version of the function that has only marshallable arguments and no keyword arguments.
    Arguments in callback matches the number of arguments used exactly.
    Used for creating callbacks from C to Python with keyword arguments or other Pythonic structures 
    (such as literal lists).
    """

    # Find out if any Python arguments should be flattened
    unflatten_instructions: Dict[int, Tuple[Callable, int]] = collections.OrderedDict()
    curarg = 0
    instructions_exist = False
    for arg in node.args:
        call, inc = _create_unflatten_instruction(arg)
        if call is not None:
            instructions_exist = True
        else:
            call = lambda x: x[0]
        unflatten_instructions[curarg] = call, inc
        curarg += inc
    for kw in node.keywords:
        call, inc = _create_unflatten_instruction(kw.value)
        if call is not None:
            instructions_exist = True
        else:
            call = lambda x: x[0]
        unflatten_instructions[curarg] = call, inc
        curarg += inc

    # Nothing to do, early exit
    if not node.keywords and not instructions_exist:
        return func

    keywords = [kw.arg for kw in node.keywords]
    poscount = len(node.args)

    # Using two levels of functions to ensure keywords are stored with the callback
    if instructions_exist:
        # If unflattening is necessary, have one version of the callback
        def make_cb(keywords, poscount, instructions):
            def cb_func(*all_args):
                # Create an unflattened version of the original arguments
                unflattened = []
                for i, (unflatten, skip) in instructions.items():
                    unflattened.append(unflatten(all_args[i:i + skip]))

                args = unflattened[:poscount]
                kwargs = {kw: arg for kw, arg in zip(keywords, unflattened[poscount:])}
                return func(*args, **kwargs)

            return cb_func
    else:

        def make_cb(keywords, poscount, _):
            def cb_func(*all_args):
                args = all_args[:poscount]
                kwargs = {kw: arg for kw, arg in zip(keywords, all_args[poscount:])}
                return func(*args, **kwargs)

            return cb_func

    return make_cb(keywords, poscount, unflatten_instructions)


class GlobalResolver(astutils.ExtNodeTransformer, astutils.ASTHelperMixin):
    """ Resolves global constants and lambda expressions if not
        already defined in the given scope. """
    def __init__(self, globals: Dict[str, Any], resolve_functions: bool = False, default_args: Set[str] = None):
        self._globals = globals
        self.resolve_functions = resolve_functions
        self.default_args = default_args or set()
        self.current_scope = set()
        self.toplevel_function = True
        self.do_not_detect_callables = False
        self.ignore_node_ctx = False

        self.closure = SDFGClosure()

    @property
    def globals(self):
        return {k: v for k, v in self._globals.items() if k not in self.current_scope}

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

    def _qualname_to_array_name(self, qualname: str, prefix: str = '__g_') -> str:
        """ Converts a Python qualified attribute name to an SDFG array name. """
        # We only support attributes and subscripts for now
        sanitized = re.sub(r'[\.\[\]\'\",]', '_', qualname)
        if not dtypes.validate_name(sanitized):
            raise NameError(f'Variable name "{sanitized}" is not sanitized '
                            'properly during parsing. Please report this issue.')
        return f"{prefix}{sanitized}"

    def global_value_to_node(self,
                             value,
                             parent_node,
                             qualname,
                             recurse=False,
                             detect_callables=False,
                             keep_object=False):
        # if recurse is false, we don't allow recursion into lists
        # this should not happen anyway; the globals dict should only contain
        # single "level" lists
        if not recurse and isinstance(value, (list, tuple)):
            # bail after more than one level of lists
            return None

        if isinstance(value, list):
            elts = [
                self.global_value_to_node(v, parent_node, qualname + f'[{i}]', detect_callables=detect_callables)
                for i, v in enumerate(value)
            ]
            if any(e is None for e in elts):
                return None
            newnode = ast.List(elts=elts, ctx=parent_node.ctx)
        elif isinstance(value, tuple):
            elts = [
                self.global_value_to_node(v, parent_node, qualname + f'[{i}]', detect_callables=detect_callables)
                for i, v in enumerate(value)
            ]
            if any(e is None for e in elts):
                return None
            newnode = ast.Tuple(elts=elts, ctx=parent_node.ctx)
        elif isinstance(value, symbolic.symbol):
            # Symbols resolve to the symbol name
            newnode = ast.Name(id=value.name, ctx=ast.Load())
        elif (dtypes.isconstant(value) or isinstance(value, SDFG) or hasattr(value, '__sdfg__')):
            # Could be a constant, an SDFG, or SDFG-convertible object
            if isinstance(value, SDFG) or hasattr(value, '__sdfg__'):
                self.closure.closure_sdfgs[id(value)] = (qualname, value)
            else:
                # If this is a function call to a None function, do not add its result to the closure
                if isinstance(parent_node, ast.Call):
                    fqname = getattr(parent_node.func, 'qualname', astutils.rname(parent_node.func))
                    if fqname in self.closure.closure_constants and self.closure.closure_constants[fqname] is None:
                        return None
                    if hasattr(parent_node.func, 'n') and parent_node.func.n is None:
                        return None

                self.closure.closure_constants[qualname] = value

            # Compatibility check since Python changed their AST nodes
            newnode = astutils.create_constant(value)
            newnode.qualname = qualname

        elif detect_callables and hasattr(value, '__call__') and hasattr(value.__call__, '__sdfg__'):
            return self.global_value_to_node(value.__call__, parent_node, qualname, recurse, detect_callables)
        elif dtypes.is_array(value):
            # Arrays need to be stored as a new name and fed as an argument
            if id(value) in self.closure.array_mapping:
                arrname = self.closure.array_mapping[id(value)]
            else:
                arrname = self._qualname_to_array_name(qualname)
                desc = data.create_datadescriptor(value)
                if keep_object:
                    self.closure.closure_arrays[arrname] = (qualname, desc, lambda: value, False)
                else:
                    self.closure.closure_arrays[arrname] = (qualname, desc, lambda: eval(qualname, self.globals), False)
                self.closure.array_mapping[id(value)] = arrname

            newnode = ast.Name(id=arrname, ctx=ast.Load())
        elif detect_callables and callable(value):
            # Try parsing the function as a dace function/method
            newnode = None
            try:
                from dace.frontend.python import parser  # Avoid import loops

                parent_object = None
                if hasattr(value, '__self__'):
                    parent_object = value.__self__

                # If it is a callable object
                if (not inspect.isfunction(value) and not inspect.ismethod(value) and not inspect.isbuiltin(value)
                        and hasattr(value, '__call__')):
                    parent_object = value
                    value = value.__call__

                # Replacements take precedence over auto-parsing
                try:
                    if has_replacement(value, parent_object, parent_node):
                        return None
                except Exception:
                    pass

                # Store the handle to the original callable, in case parsing fails
                if isinstance(parent_node, ast.Call):
                    cbqualname = astutils.unparse(parent_node.func)
                else:
                    cbqualname = astutils.rname(parent_node)
                cbname = self._qualname_to_array_name(cbqualname, prefix='')

                # Make a version of the callback without keyword arguments or Python literal objects (list, tuple, ...)
                cb_func = flatten_callback(value, parent_node)

                # If the callback already exists, and the details differ (e.g., different kwarg names), make new
                if cbname in self.closure.callbacks and cb_func is not self.closure.callbacks[cbname][1]:
                    cbname = data.find_new_name(cbname, self.closure.callbacks)

                self.closure.callbacks[cbname] = (cbqualname, cb_func, False)

                # From this point on, any failure will result in a callback
                newnode = ast.Name(id=cbname, ctx=ast.Load())

                # Decorated or functions with missing source code
                sast, _, _, _ = astutils.function_to_ast(value)
                if len(sast.body[0].decorator_list) > 0:
                    return newnode
                if find_disallowed_statements(sast):
                    return newnode

                parsed = parser.DaceProgram(value, [], {}, False, dtypes.DeviceType.CPU)
                # If method, add the first argument (which disappears due to
                # being a bound method) and the method's object
                if parent_object is not None:
                    parsed.methodobj = parent_object
                    parsed.objname = inspect.getfullargspec(value).args[0]

                res = self.global_value_to_node(parsed, parent_node, qualname, recurse, detect_callables)

                res.oldnode = astutils.copy_tree(parent_node)
                res.cbname = cbname

                # Keep callback in callbacks in case of parsing failure
                # del self.closure.callbacks[cbname]
                return res
            except Exception:  # Parsing failed (almost any exception can occur)
                return newnode
        elif keep_object:
            # General object, keep in globals and give a unique name
            objname = self._qualname_to_array_name(qualname, prefix='')
            objname = data.find_new_name(objname, self._globals.keys())
            self._globals[objname] = value
            newnode = ast.Name(id=objname, ctx=ast.Load())
        else:
            return None

        if parent_node is not None:
            return ast.copy_location(newnode, parent_node)
        else:
            return newnode

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        # Skip the top function definition (handled outside of the resolver)
        if self.toplevel_function:
            self.toplevel_function = False
            node.decorator_list = []  # Skip decorators
            return self.generic_visit(node)

        for arg in ast.walk(node.args):
            if isinstance(arg, ast.arg):
                # Skip unspecified default arguments
                if arg.arg in self.default_args:
                    continue

                # Skip ``dace.constant``-annotated arguments
                is_constant = False
                if arg.annotation is not None:
                    try:
                        ann = astutils.evalnode(arg.annotation, self.globals)
                        if ann is dace.constant:
                            is_constant = True
                    except SyntaxError:
                        pass
                if not is_constant:
                    self.current_scope.add(arg.arg)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        return self.visit_FunctionDef(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        # Node target in augassign is ast.Store, even though it is updating an existing value
        oldvalue = self.ignore_node_ctx
        self.ignore_node_ctx = True
        node.target = self.visit(node.target)
        self.ignore_node_ctx = oldvalue

        # Parse the rest of the fields
        return self.generic_visit_filtered(node, {'target'})

    def visit_Name(self, node: ast.Name):
        if not self.ignore_node_ctx and isinstance(node.ctx, ast.Store):
            self.current_scope.add(node.id)
        else:
            if node.id in self.current_scope:
                return node
            if node.id in self.globals:
                global_val = self.globals[node.id]
                newnode = self.global_value_to_node(global_val, parent_node=node, qualname=node.id, recurse=True)
                if newnode is None:
                    return node
                return newnode
        return node

    def visit_keyword(self, node: ast.keyword):
        if node.arg in self.globals and isinstance(self.globals[node.arg], symbolic.symbol):
            node.arg = self.globals[node.arg].name
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # Try to evaluate the expression with only the globals
        try:
            global_val = astutils.evalnode(node, self.globals)
        except SyntaxError:
            return node  # Do NOT use generic_visit here as it may modify the attribute value too soon

        if not isinstance(global_val, dtypes.typeclass):
            newnode = self.global_value_to_node(global_val,
                                                parent_node=node,
                                                qualname=astutils.unparse(node),
                                                recurse=True)
            if newnode is not None:
                return newnode
        return node  # Do NOT use generic_visit here as it may modify the attribute value too soon

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        # First visit the subscripted value alone, then the whole subscript
        node.value = self.visit(node.value)

        # Try to evaluate literal lists/dicts/tuples directly
        if isinstance(node.value, (ast.List, ast.Dict, ast.Tuple)):
            # First evaluate key
            try:
                gslice = astutils.evalnode(node.slice, self.globals)
            except SyntaxError:
                return self.generic_visit(node)

            # Then query for the right value
            if isinstance(node.value, ast.Dict):
                for k, v in zip(node.value.keys, node.value.values):
                    try:
                        gkey = astutils.evalnode(k, self.globals)
                    except SyntaxError:
                        continue
                    if gkey == gslice:
                        return self.visit_Attribute(v)
            else:  # List or Tuple
                return self.visit_Attribute(node.value.elts[gslice])

        return self.visit_Attribute(node)

    def visit_Call(self, node: ast.Call) -> Any:
        from dace.frontend.python.interface import in_program  # Avoid import loop

        if hasattr(node.func, 'n') and isinstance(node.func.n, SDFGConvertible):
            # Skip already-parsed calls
            return self.generic_visit(node)

        try:
            global_func = astutils.evalnode(node.func, self.globals)

            # Built-in functions are resolved directly
            if global_func is in_program:
                return self.global_value_to_node(True, parent_node=node, qualname=astutils.unparse(node), recurse=True)

            if self.resolve_functions:
                global_val = astutils.evalnode(node, self.globals)
            else:
                global_val = node
        except SyntaxError:
            return self.generic_visit(node)

        newnode = None
        if self.resolve_functions and global_val is not node:
            # Without this check, casts don't generate code
            if not isinstance(global_val, dtypes.typeclass):
                newnode = self.global_value_to_node(global_val,
                                                    parent_node=node,
                                                    qualname=astutils.unparse(node),
                                                    recurse=True)
                if newnode is not None:
                    return newnode
        elif not isinstance(global_func, dtypes.typeclass):
            callables = not self.do_not_detect_callables
            newnode = self.global_value_to_node(global_func,
                                                parent_node=node,
                                                qualname=astutils.unparse(node),
                                                recurse=True,
                                                detect_callables=callables)
            if newnode is not None:
                node.func = newnode
                return self.generic_visit(node)
        return self.generic_visit(node)

    def visit_For(self, node: ast.For):
        # Special case: for loop generators cannot be dace programs
        oldval = self.do_not_detect_callables
        self.do_not_detect_callables = True
        self.generic_visit_field(node, 'target')
        self.generic_visit_field(node, 'iter')
        self.do_not_detect_callables = oldval
        self.generic_visit_field(node, 'body')
        self.generic_visit_field(node, 'orelse')
        return node

    def visit_TopLevelExpr(self, node: ast.Expr):
        # Ignore memlet targets in tasklets
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.op, (ast.LShift, ast.RShift)):
                # Do not visit node.value.left
                node.value.right = self.visit(node.value.right)
                return node
        return self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> Any:
        # Try to evaluate assertion statically
        try:
            global_val = astutils.evalnode(node.test, self.globals)
            if global_val:  # Check condition the same way as assert does
                return None
            try:
                msg = astutils.evalnode(node.msg, self.globals)
                if msg is not None:
                    msg = '. Message: ' + msg
                else:
                    msg = '.'
            except SyntaxError:
                msg = ' (ERROR: could not statically evaluate message).'

            raise AssertionError('Assertion failed statically at line '
                                 f'{node.lineno} during compilation of DaCe '
                                 'program' + msg)
        except SyntaxError:
            warnings.warn(f'Runtime assertion at line {node.lineno} could not'
                          ' be checked in DaCe program, skipping check.')
        return None

    def visit_Raise(self, node: ast.Raise) -> Any:
        warnings.warn(f'Runtime exception at line {node.lineno} is not supported and will be skipped.')
        return None

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Any:
        try:
            global_val = astutils.evalnode(node, self.globals)
            return ast.copy_location(ast.Constant(kind='', value=global_val), node)
        except SyntaxError:
            warnings.warn(f'f-string at line {node.lineno} could not '
                          'be fully evaluated in DaCe program, converting to '
                          'partially-evaluated string.')
            visited = self.generic_visit(node)
            parsed = [
                not isinstance(v, ast.FormattedValue) or isinstance(v.value, ast.Constant) for v in visited.values
            ]
            values = [v.s if isinstance(v, ast.Str) else astutils.unparse(v.value) for v in visited.values]
            return ast.copy_location(
                ast.Constant(kind='', value=''.join(('{%s}' % v) if not p else v for p, v in zip(parsed, values))),
                node)


class ContextManagerInliner(ast.NodeTransformer, astutils.ASTHelperMixin):
    """
    Since exceptions are disabled in dace programs, AST with ``with`` statements
    can be replaced with appropriate calls to the ``__enter__``/``__exit__`` calls
    in the right places, i.e., at the end of the body or when the context is left due to
    a return statement, or top-level break/continue statements.
    """
    def __init__(self, globals: Dict[str, Any], filename: str, closure_resolver: GlobalResolver) -> None:
        super().__init__()
        self.with_statements: List[ast.With] = []
        self.context_managers: Dict[ast.With, List[Tuple[str, Any]]] = {}
        self.globals: Dict[str, Any] = globals
        self.filename = filename
        self.resolver = closure_resolver

    def _visit_node_with_body(self, node):
        node = self.generic_visit_filtered(node, {'body'})
        self.with_statements.append(node)
        node = self.generic_visit_field(node, 'body')
        self.with_statements.pop()
        return node

    def _register_callback(self, node: ast.AST, callable: Callable[..., Any]):
        # Store the handle to the original callable, in case parsing fails
        if isinstance(node, ast.Call):
            cbqualname = astutils.unparse(node.func)
        else:
            cbqualname = astutils.rname(node)
        newnode = self.resolver.global_value_to_node(callable, node, cbqualname, detect_callables=True)
        if isinstance(node, ast.Call):
            node.func = newnode
            return node
        return newnode

    def _add_exits(self, until_loop_end: bool, only_one: bool = False) -> List[ast.AST]:
        result = []
        if len(self.with_statements) == 0:
            return result
        for stmt in reversed(self.with_statements):
            if until_loop_end and not isinstance(stmt, (ast.With, ast.AsyncWith)):
                break
            for mgrname, mgr in reversed(self.context_managers[stmt]):
                # Call __exit__ (without exception management all three arguments are set to None)
                exit_call = ast.copy_location(ast.parse(f'{mgrname}.__exit__(None, None, None)').body[0], stmt)
                exit_call.value = self._register_callback(exit_call.value, mgr.__exit__)
                result.append(exit_call)
            if only_one:
                break

        return result

    def _add_entries(self, node: ast.With) -> List[ast.AST]:
        result = []
        ctx_mgr_names = []
        for i, item in enumerate(node.items):
            # Check if manager is parse-time evaluatable
            try:
                if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name):
                    fname = item.context_expr.func.id
                    if fname in self.resolver.closure.callbacks:
                        newglobals = copy.copy(self.globals)
                        newglobals[fname] = self.resolver.closure.callbacks[fname][1]
                        ctxmgr = astutils.evalnode(item.context_expr, newglobals)
                    else:
                        ctxmgr = astutils.evalnode(item.context_expr, self.globals)
                else:
                    ctxmgr = astutils.evalnode(item.context_expr, self.globals)
            except SyntaxError:
                raise ValueError(f'Cannot create context manager at {self.filename}:{node.lineno} - only compile-time '
                                 'evaluatable context managers are supported.')

            # Create manager as part of closure
            mgr_name = f'__with_{node.lineno}_{i}' if len(node.items) > 1 else f'__with_{node.lineno}'
            mgr = self.resolver.global_value_to_node(ctxmgr, node, mgr_name, keep_object=True)
            ctx_mgr_names.append((mgr.id, ctxmgr))

            # Call __enter__
            enter_call = ast.copy_location(ast.parse(f'{mgr.id}.__enter__()').body[0], node)
            enter_call.value = self._register_callback(enter_call.value, ctxmgr.__enter__)
            if item.optional_vars is not None:
                enter_call = ast.copy_location(
                    ast.Assign(targets=[item.optional_vars], value=enter_call.value, type_comment=None), node)
            result.append(enter_call)

        self.context_managers[node] = ctx_mgr_names

        return result

    def visit_With(self, node: ast.With):
        # Avoid parsing "with dace.tasklet"
        try:
            evald = astutils.evalnode(node.items[0].context_expr, self.globals)
            if evald is dace.tasklet or isinstance(evald, dace.tasklet):
                return self.generic_visit(node)
        except SyntaxError:
            pass

        # Beginning and end of body adds __enter__, __exit__ calls for each item
        self.with_statements.append(node)

        # Make empty block
        ifnode: ast.If = ast.parse('if True: pass').body[0]
        ifnode = ast.copy_location(ifnode, node)

        # Make enter calls
        entries = self._add_entries(node)
        ifnode.body = entries

        # Visit body
        node = self.generic_visit_field(node, 'body')
        ifnode.body += node.body

        # Make exit calls
        ifnode.body += self._add_exits(True, True)

        # Pop context manager information
        self.with_statements.pop()
        del self.context_managers[node]
        return ifnode

    def visit_AsyncWith(self, node):
        return self.visit_With(node)

    def visit_For(self, node):
        return self._visit_node_with_body(node)

    def visit_AsyncFor(self, node):
        return self._visit_node_with_body(node)

    def visit_While(self, node):
        return self._visit_node_with_body(node)

    def visit_Break(self, node):
        node = self.generic_visit(node)
        return self._add_exits(True) + [node]

    def visit_Continue(self, node):
        node = self.generic_visit(node)
        return self._add_exits(True) + [node]

    def visit_Return(self, node):
        node = self.generic_visit(node)
        return self._add_exits(False) + [node]


class LoopUnroller(ast.NodeTransformer):
    """ 
    Replaces loops by their unrolled bodies if generator can be evaluated at
    compile time and one of the following conditions apply:
        1. `dace.unroll` was explicitly called
        2. looping over compile-time constant tuples/lists/dictionaries
        3. generator is one of the predetermined "stateless generators"
        4. any generator with compile-time size that is lower than the "unroll_threshold" configuration
    """
    STATELESS_GENERATORS = [
        enumerate,
        zip,
        reversed,
        dict.values,
        dict.keys,
        dict.items,
    ]

    THRESHOLD_GENERATORS = [
        range,
    ]

    def __init__(self, globals: Dict[str, Any], filename: str, closure_resolver: GlobalResolver):
        super().__init__()
        self.globals = globals
        self.filename = filename
        self.threshold = int(Config.get('frontend', 'unroll_threshold'))
        self.resolver = closure_resolver

    def visit_For(self, node: ast.For) -> Any:
        # Avoid import loops
        EXPLICIT_GENERATORS = [
            range,  # Handled in ProgramVisitor
            dace.map,
            dace.consume,
        ]

        node = self.generic_visit(node)

        # If this node was already designated as a no-unroll node, continue
        if getattr(node, 'nounroll', False):
            return node

        # First, skip loops that contain break/continue that is part of this
        # for loop (rather than nested ones)
        cannot_unroll = False
        cflow_finder = _FindBreakContinueStmts()
        for stmt in node.body:
            cflow_finder.visit(stmt)
        if cflow_finder.has_cflow or node.orelse:
            cannot_unroll = True

        niter = node.iter

        # Find out if loop was explicitly requested to be unrolled with unroll,
        # and whether it should be done implicitly
        explicitly_requested = False
        if isinstance(niter, ast.Call):
            # Avoid import loop
            from dace.frontend.python.interface import nounroll, unroll

            try:
                genfunc = astutils.evalnode(niter.func, self.globals)
            except SyntaxError:
                genfunc = None

            if genfunc is unroll:
                explicitly_requested = True
                niter = niter.args[0]
            elif genfunc is nounroll:
                # Return the contents of the nounroll call
                node.nounroll = True
                node.iter = niter.args[0]
                return node

        if explicitly_requested and cannot_unroll:
            raise DaceSyntaxError(None, node, 'Cannot unroll loop due to "break", "continue", or "else" statements.')

        # Find out if unrolling should be done implicitly
        implicit = True
        # Anything not a call is implicitly allowed
        if isinstance(niter, (ast.Call, ast.Subscript)):
            if isinstance(niter, ast.Subscript):
                nfunc = niter.value
            else:
                nfunc = niter.func

            implicit = False
            # Try to see if it's one of the allowed stateless generators
            try:
                genfunc = astutils.evalnode(nfunc, self.globals)

                # If genfunc is a bound method, try to extract function from type
                if hasattr(genfunc, '__self__'):
                    genfunc = getattr(type(genfunc.__self__), genfunc.__name__, False)

                if (self.threshold >= 0
                        and (genfunc not in EXPLICIT_GENERATORS or genfunc in LoopUnroller.THRESHOLD_GENERATORS)):
                    implicit = True
                elif genfunc in LoopUnroller.STATELESS_GENERATORS:
                    implicit = True
                elif genfunc in EXPLICIT_GENERATORS:
                    implicit = False

            except SyntaxError:
                pass

        # Loop will not be unrolled
        if not implicit and not explicitly_requested:
            return node

        # Check if loop target is supported
        if isinstance(node.target, ast.Tuple):
            to_replace = node.target.elts
        elif isinstance(node.target, ast.Name):
            to_replace = [node.target]
        else:
            # Unsupported loop target
            return node

        if isinstance(niter, (ast.Tuple, ast.List, ast.Set)):
            # Check if a literal tuple/list/set
            generator = niter.elts
        elif isinstance(niter, ast.Dict):
            # If dict, take keys (Python compatible)
            generator = niter.keys
        # elif isinstance(iter, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
        #     # Check if a comprehension or generator expression
        #     pass
        else:
            # Check if the generator is compile-time constant
            try:
                generator = astutils.evalnode(niter, self.globals)
            except SyntaxError:
                # Cannot evaluate generator at compile time
                return node

        if self.threshold == 0:  # Unroll any loop
            explicitly_requested = True
        elif self.threshold > 0:
            generator = list(generator)
            if len(generator) > self.threshold:
                return node
            explicitly_requested = True

        # Too verbose?
        if implicit and not explicitly_requested:
            warnings.warn(f'Loop at {self.filename}:{node.lineno} will be implicitly unrolled.')

        ##########################################
        # Unroll loop
        new_body = []
        for eid, elem in enumerate(generator):
            # Paste loop body with replaced elements
            if not isinstance(elem, (list, tuple, set)):
                elem = [elem]
            else:
                elem = list(elem)

            # If an unknown/mutable object, add to closure
            for i, e in enumerate(elem):
                # Already AST
                if isinstance(e, ast.AST):
                    continue
                if isinstance(e, (numbers.Number, str)):
                    # Compatibility check since Python changed their AST nodes
                    newnode = astutils.create_constant(e)
                    elem[i] = newnode
                else:
                    # Augment closure with new value
                    eid_str = f'{eid}'
                    if len(elem) > 1:
                        eid_str = f'{eid}_{i}'
                    elem[i] = self.resolver.global_value_to_node(e,
                                                                 node,
                                                                 f'gen{node.lineno}_{eid_str}',
                                                                 True,
                                                                 keep_object=True)

            elembody = [astutils.copy_tree(stmt) for stmt in node.body]
            replace = astutils.ASTFindReplace({k: v for k, v in zip(to_replace, elem)})
            for stmt in elembody:
                new_body.append(replace.visit(stmt))

        return new_body

    def visit_AsyncFor(self, node) -> Any:
        return self.visit_For(node)


class CallTreeResolver(ast.NodeVisitor):
    def __init__(self, closure: SDFGClosure, globals: Dict[str, Any]) -> None:
        self.closure = closure
        self.seen_calls: Set[str] = set()
        self.globals = globals

    def _eval_args(self, node: ast.Call) -> Dict[str, Any]:
        res = {}

        # Evaluate positional arguments
        for i, arg in enumerate(node.args):
            try:
                val = astutils.evalnode(arg, self.globals)
                res[i] = val
            except SyntaxError:
                pass

        # Evaluate keyword arguments
        for kwarg in node.keywords:
            kwname = kwarg.arg
            kwval = kwarg.value
            try:
                val = astutils.evalnode(kwval, self.globals)
                res[kwname] = val
            except SyntaxError:
                pass

        return res

    def _get_given_args(self, node: ast.Call, function: 'DaceProgram') -> Set[str]:
        """ Returns a set of names of the given arguments from the positional and keyword arguments """
        from dace.frontend.python.parser import DaceProgram  # Avoid import loop

        posargs = node.args
        kwargs = [kwarg.arg for kwarg in node.keywords]
        result = set()

        if not isinstance(function, DaceProgram):
            # Make set of parameters from __sdfg_signature__
            parameters = [(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in function.__sdfg_signature__()[0]]
        else:
            parameters = [(aname, arg.kind) for aname, arg in function.signature.parameters.items()]

        # Handle "self" argument
        objname = getattr(function, 'objname', False)

        nargs = len(posargs)
        arg_ind = 0
        # Track both positional arguments and function signature together
        for aname, sig_kind in parameters:
            if aname == objname:
                # Skip "self" argument
                continue

            # Variable-length arguments: obtain from the remainder of given_*
            if sig_kind is inspect.Parameter.VAR_POSITIONAL:
                vargs = posargs[arg_ind:]
                result.update({f'__arg{j}' for j, _ in enumerate(vargs)})
                # Shift arg_ind to the end
                arg_ind = len(posargs)
            elif sig_kind is inspect.Parameter.VAR_KEYWORD:
                vargs = {k for k in kwargs.keys() if k not in result}
                result.update({f'__kwarg_{k}' for k in vargs.keys()})
            # END OF VARIABLE-LENGTH ARGUMENTS
            elif sig_kind is inspect.Parameter.POSITIONAL_ONLY:
                if arg_ind < nargs:
                    result.add(aname)
                    arg_ind += 1
            elif sig_kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                if arg_ind >= nargs:
                    if aname in kwargs:
                        result.add(aname)
                else:
                    result.add(aname)
                    arg_ind += 1
            elif sig_kind is inspect.Parameter.KEYWORD_ONLY:
                if aname in kwargs:
                    result.add(aname)

        return result

    def visit_Call(self, node: ast.Call):
        # Only parse calls to parsed SDFGConvertibles
        if not isinstance(node.func, (ast.Num, ast.Constant)):
            self.seen_calls.add(astutils.unparse(node.func))
            return self.generic_visit(node)
        if hasattr(node.func, 'oldnode'):
            if isinstance(node.func.oldnode, ast.Call):
                self.seen_calls.add(astutils.unparse(node.func.oldnode.func))
            else:
                self.seen_calls.add(astutils.rname(node.func.oldnode))
        if isinstance(node.func, ast.Num):
            value = node.func.n
        else:
            value = node.func.value

        if not hasattr(value, '__sdfg__') or isinstance(value, SDFG):
            return self.generic_visit(node)

        constant_args = self._eval_args(node)

        # Resolve nested closure as necessary
        qualname = None
        try:
            if id(value) in self.closure.closure_sdfgs:
                qualname, _ = self.closure.closure_sdfgs[id(value)]
            elif hasattr(node.func, 'qualname'):
                qualname = node.func.qualname
            self.seen_calls.add(qualname)
            if hasattr(value, 'closure_resolver'):
                # Get given arguments from signature and args/kwargs
                given_args = self._get_given_args(node, value)

                self.closure.nested_closures.append(
                    (qualname, value.closure_resolver(constant_args, given_args, self.closure)))
            else:
                self.closure.nested_closures.append((qualname, SDFGClosure()))
        except DaceRecursionError:  # Parsing failed in a nested context, raise
            raise
        except Exception as ex:  # Parsing failed (anything can happen here)
            optional_qname = ''
            if qualname is not None:
                optional_qname = f' ("{qualname}")'
            warnings.warn(
                f'Preprocessing SDFGConvertible {value}{optional_qname} failed with {type(ex).__name__}: {ex}')
            if Config.get_bool('frontend', 'raise_nested_parsing_errors'):
                raise
            if id(value) in self.closure.closure_sdfgs:
                del self.closure.closure_sdfgs[id(value)]
            # Return old call AST instead
            if not hasattr(node.func, 'oldnode'):
                raise

            # If callback exists, use callback name
            if hasattr(node.func, 'cbname'):
                newnode = ast.Name(id=node.func.cbname, ctx=ast.Load())
                newnode.oldnode = node.func.oldnode
                node.func = ast.copy_location(newnode, node.func)
            else:
                # Revert to old call AST
                node.func = node.func.oldnode.func

            return self.generic_visit(node)


class ArrayClosureResolver(ast.NodeVisitor):
    def __init__(self, closure: SDFGClosure):
        self.closure = closure
        self.arrays: Set[str] = set()

    def visit_Name(self, node: ast.Name):
        if node.id in self.closure.closure_arrays:
            self.arrays.add(node.id)
        self.generic_visit(node)


class AugAssignExpander(ast.NodeTransformer):
    def visit_AugAssign(self, node: ast.AugAssign) -> ast.Assign:
        target = self.generic_visit(node.target)
        value = self.generic_visit(node.value)
        newvalue = ast.copy_location(ast.BinOp(left=copy.deepcopy(target), op=node.op, right=value), value)
        return ast.copy_location(ast.Assign(targets=[target], value=newvalue), node)


def find_disallowed_statements(node: ast.AST):
    from dace.frontend.python.newast import DISALLOWED_STMTS  # Avoid import loop
    for subnode in ast.walk(node):
        # Found disallowed statement
        if type(subnode).__name__ in DISALLOWED_STMTS:
            return type(subnode).__name__
    return None


def preprocess_dace_program(f: Callable[..., Any],
                            argtypes: Dict[str, data.Data],
                            global_vars: Dict[str, Any],
                            modules: Dict[str, Any],
                            resolve_functions: bool = False,
                            parent_closure: Optional[SDFGClosure] = None,
                            default_args: Optional[Set[str]] = None) -> Tuple[PreprocessedAST, SDFGClosure]:
    """
    Preprocesses a ``@dace.program`` and all its nested functions, returning
    a preprocessed AST object and the closure of the resulting SDFG.
    :param f: A Python function to parse.
    :param argtypes: An dictionary of (name, type) for the given
                        function's arguments, which may pertain to data
                        nodes or symbols (scalars).
    :param global_vars: A dictionary of global variables in the closure
                        of `f`.
    :param modules: A dictionary from an imported module name to the
                    module itself.
    :param constants: A dictionary from a name to a constant value.
    :param resolve_functions: If True, treats all global functions defined
                                outside of the program as returning constant
                                values.
    :param parent_closure: If not None, represents the closure of the parent of
                           the currently processed function.
    :param default_args: If not None, defines a list of unspecified default arguments.
    :return: A 2-tuple of the AST and its reduced (used) closure.
    """
    src_ast, src_file, src_line, src = astutils.function_to_ast(f)

    # Resolve data structures
    src_ast = StructTransformer(global_vars).visit(src_ast)

    src_ast = ModuleResolver(modules).visit(src_ast)
    # Convert modules after resolution
    for mod, modval in modules.items():
        if mod == 'builtins':
            continue
        newmod = global_vars[mod]
        #del global_vars[mod]
        global_vars[modval] = newmod

    # Resolve constants to their values (if they are not already defined in this scope)
    # and symbols to their names
    resolved = {k: v for k, v in global_vars.items() if k not in (argtypes.keys() - default_args) and k != '_'}
    closure_resolver = GlobalResolver(resolved, resolve_functions, default_args=default_args)

    # Append element to call stack and handle max recursion depth
    if parent_closure is not None:
        fid = id(f)
        if fid in parent_closure.callstack:
            raise DaceRecursionError(fid)
        if len(parent_closure.callstack) > Config.get('frontend', 'implicit_recursion_depth'):
            raise TypeError('Implicit (automatically parsed) recursion depth '
                            'exceeded. Functions below this call will not be '
                            'parsed. To change this setting, modify the value '
                            '`frontend.implicit_recursion_depth` in .dace.conf')

        closure_resolver.closure.callstack = parent_closure.callstack + [fid]

    # Find disallowed AST nodes
    disallowed = find_disallowed_statements(src_ast)
    if disallowed:
        raise TypeError(f'Converting function "{f.__name__}" ({src_file}:{src_line}) to callback due to disallowed '
                        f'keyword: {disallowed}')

    passes = int(Config.get('frontend', 'preprocessing_passes'))
    if passes >= 0:
        gen = range(passes)
    else:  # Run until the code stops changing

        def check_code(src_ast):
            old_src = ast.dump(src_ast)
            i = 0
            while True:
                yield i
                new_src = ast.dump(src_ast)
                if new_src == old_src:
                    return
                old_src = new_src
                i += 1

        gen = check_code(src_ast)

    for pass_num in gen:
        try:
            closure_resolver.toplevel_function = True
            src_ast = closure_resolver.visit(src_ast)
            src_ast = LoopUnroller(resolved, src_file, closure_resolver).visit(src_ast)
            src_ast = ContextManagerInliner(resolved, src_file, closure_resolver).visit(src_ast)
            src_ast = ConditionalCodeResolver(resolved).visit(src_ast)
            src_ast = DeadCodeEliminator().visit(src_ast)
        except Exception:
            if Config.get_bool('frontend', 'verbose_errors'):
                print(f'VERBOSE: Failed to preprocess (pass #{pass_num}) the following program:')
                print(astutils.unparse(src_ast))
            raise

    try:
        ctr = CallTreeResolver(closure_resolver.closure, resolved)
        ctr.visit(src_ast)
    except DaceRecursionError as ex:
        if id(f) == ex.fid:
            raise TypeError('Parsing failed due to recursion in a data-centric ' 'context called from this function')
        else:
            raise ex
    used_arrays = ArrayClosureResolver(closure_resolver.closure)
    used_arrays.visit(src_ast)

    # Filter out arrays that are not used after dead code elimination
    closure_resolver.closure.closure_arrays = {
        k: v
        for k, v in closure_resolver.closure.closure_arrays.items() if k in used_arrays.arrays
    }

    # Filter out callbacks that were removed after dead code elimination
    closure_resolver.closure.callbacks = {
        k: v
        for k, v in closure_resolver.closure.callbacks.items() if k in ctr.seen_calls
    }

    # Filter remaining global variables according to type and scoping rules
    program_globals = {k: v for k, v in global_vars.items() if k not in argtypes}

    # Fill in data descriptors from closure arrays
    argtypes.update({arrname: v[1] for arrname, v in closure_resolver.closure.closure_arrays.items()})

    # Combine nested closures with the current one
    closure_resolver.closure.combine_nested_closures()

    past = PreprocessedAST(src_file, src_line, src, src_ast, program_globals)

    return past, closure_resolver.closure
