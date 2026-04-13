# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import asyncio
import collections
import copy
from dataclasses import dataclass
import functools
import inspect
import numbers
import numpy
import re
import sympy
import threading
import warnings

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import dace
from dace import data, dtypes, symbolic, sdfg
from dace.config import Config
from dace.sdfg import SDFG
from dace.frontend.python import astutils
from dace.frontend.python.common import (DaceSyntaxError, SDFGConvertible, SDFGClosure, StringLiteral)

if TYPE_CHECKING:
    from dace.frontend.python.parser import DaceProgram


class DaceRecursionError(Exception):
    """
    Exception that indicates a recursion in a data-centric parsed context.
    The exception includes the id of the topmost function as a stopping
    condition for parsing.
    """

    def __init__(self, fid: int):
        self.fid = fid

    def __str__(self) -> str:
        return ('Non-analyzable recursion detected, function cannot be parsed as data-centric')


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


def __dace_iterator_init(iterable):
    return iterable.__iter__()


def __dace_iterator_next(iterator, sentinel):
    try:
        return iterator.__next__()
    except StopIteration:
        return sentinel


class StructTransformer(ast.NodeTransformer):
    """
    A Python AST transformer that replaces ``Call`` nodes to create structs with
    the custom ``StructInitializer`` AST node.
    """

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


def _create_unflatten_instruction(arg: ast.AST, global_vars: Dict[str, Any]) -> Tuple[Callable, int]:
    """
    Creates a lambda function for recreating the original Python object and returns the number of
    arguments to increment.
    """
    try:
        # Constant-valued arguments stay as-is
        val = astutils.evalnode(arg, global_vars)
        if not symbolic.issymbolic(val):
            return (lambda *args: val), 0, True
    except SyntaxError:
        pass

    if isinstance(arg, ast.List):
        return (list, len(arg.elts), False)
    elif isinstance(arg, ast.Tuple):
        return (tuple, len(arg.elts), False)
    elif isinstance(arg, ast.Set):
        return (set, len(arg.elts), False)
    elif isinstance(arg, ast.Dict):
        # Use two levels of functions to preserve keyword names
        def make_remake(kwnames):

            def remake_dict(args):
                return {k: a for k, a in zip(kwnames, args)}

            return remake_dict

        # Remake keyword argument names from AST
        kwarg_names = []
        for kw in arg.keys:
            if not isinstance(kw, ast.Constant):
                raise NotImplementedError(f'Key type {type(kw).__name__} is not supported')

            kwarg_names.append(kw.value)

        return (make_remake(kwarg_names), len(arg.keys), False)
    return (None, 1, False)


def flatten_callback(func: Callable, node: ast.Call, global_vars: Dict[str, Any]):
    """
    Creates a version of the function that has only marshallable arguments and no keyword arguments.
    Arguments in callback matches the number of arguments used exactly.
    Used for creating callbacks from C to Python with keyword arguments or other Pythonic structures
    (such as literal lists).
    """

    # Find out if any Python arguments should be flattened
    unflatten_instructions: List[Tuple[int, Callable, int, bool]] = []
    curarg = 0
    instructions_exist = False

    # Constant [keyword] arguments to remove
    args_to_remove = []
    kwargs_to_remove = []

    for i, arg in enumerate(node.args):
        call, inc, constant = _create_unflatten_instruction(arg, global_vars)
        if call is not None:
            instructions_exist = True
        else:
            call = lambda x: x[0]
        if constant:
            args_to_remove.append(i)
        unflatten_instructions.append((curarg, call, inc, constant))
        curarg += inc
    for i, kw in enumerate(node.keywords):
        call, inc, constant = _create_unflatten_instruction(kw.value, global_vars)
        if call is not None:
            instructions_exist = True
        else:
            call = lambda x: x[0]
        if constant:
            kwargs_to_remove.append(i)
        unflatten_instructions.append((curarg, call, inc, constant))
        curarg += inc

    # Filter arguments from AST
    poscount = len(node.args)

    def _wrap_async_callback(callback: Callable) -> Callable:
        if not inspect.iscoroutinefunction(func):
            return callback

        @functools.wraps(callback)
        def _wrapped(*all_args):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(callback(*all_args))

            holder: Dict[str, Any] = {}

            def _runner() -> None:
                try:
                    holder['result'] = asyncio.run(callback(*all_args))
                except BaseException as ex:
                    holder['error'] = ex

            worker = threading.Thread(target=_runner)
            worker.start()
            worker.join()
            if 'error' in holder:
                raise holder['error']
            return holder.get('result')

        return _wrapped

    # Nothing to do, early exit
    if not node.keywords and not instructions_exist:
        return _wrap_async_callback(func)

    keywords = [kw.arg for kw in node.keywords]

    # Annotate that these arguments should not be visited during callback generation
    node.skip_args = args_to_remove
    node.skip_keywords = kwargs_to_remove

    # Using two levels of functions to ensure keywords are stored with the callback
    if instructions_exist:
        # If unflattening is necessary, have one version of the callback
        def make_cb(keywords, poscount, instructions):

            def cb_func(*all_args):
                # Create an unflattened version of the original arguments
                unflattened = []
                for i, unflatten, skip, constant in instructions:
                    if constant:
                        unflattened.append(unflatten())
                    else:
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

    return _wrap_async_callback(make_cb(keywords, poscount, unflatten_instructions))


class GlobalResolver(astutils.ExtNodeTransformer, astutils.ASTHelperMixin):
    """ Resolves global constants and lambda expressions if not
        already defined in the given scope. """

    def __init__(self,
                 globals: Dict[str, Any],
                 resolve_functions: bool = False,
                 default_args: Set[str] = None,
                 preserve_object_attributes: bool = False):
        self._globals = globals
        self.resolve_functions = resolve_functions
        self.default_args = default_args or set()
        self.preserve_object_attributes = preserve_object_attributes
        self.current_scope = set()
        self.toplevel_function = True
        self.do_not_detect_callables = False
        self.ignore_node_ctx = False

        self.closure = SDFGClosure()

    @property
    def globals(self):
        return {k: v for k, v in self._globals.items() if k not in self.current_scope}

    def _contains_preserved_attribute_access(self, node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Attribute) and self._should_preserve_attribute_access(child)
            for child in ast.walk(node))

    def _should_preserve_attribute_access(self, node: ast.Attribute) -> bool:
        if not self.preserve_object_attributes:
            return False

        try:
            base_value = astutils.evalnode(node.value, self.globals)
        except Exception:
            return False

        if self._is_native_attribute_base(base_value):
            return False

        # User objects should remain attribute accesses in the preprocessed AST.
        # The schedule-tree frontend can then decide whether to render them as
        # direct attributes or as explicit protocol calls (__get__/__set__/etc.).
        preserve_direct_attribute = True

        try:
            static_attr = inspect.getattr_static(base_value, node.attr)
        except AttributeError:
            static_attr = None

        if static_attr is not None and self._is_descriptor(static_attr):
            if isinstance(node.ctx, ast.Load) and hasattr(static_attr, '__get__'):
                return True
            if isinstance(node.ctx, (ast.Store, ast.Del)) and (hasattr(static_attr, '__set__')
                                                               or hasattr(static_attr, '__delete__')):
                return True

        objtype = type(base_value)
        if isinstance(node.ctx, ast.Load):
            if '__getattr__' in objtype.__dict__:
                return True
            getattribute = objtype.__dict__.get('__getattribute__')
            if getattribute is not None and getattribute is not object.__getattribute__:
                return True
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            setattr_method = objtype.__dict__.get('__setattr__')
            if setattr_method is not None and setattr_method is not object.__setattr__:
                return True

        return preserve_direct_attribute

    def _is_descriptor(self, value: Any) -> bool:
        return any(hasattr(value, attr) for attr in ('__get__', '__set__', '__delete__'))

    def _is_native_attribute_base(self, value: Any) -> bool:
        if dtypes.ismodule(value):
            return True
        if isinstance(value,
                      (dtypes.typeclass, symbolic.symbol, sympy.Basic, data.Data, SDFG, numpy.ndarray, numpy.generic)):
            return True

        module_name = getattr(type(value), '__module__', '')
        return module_name.startswith(('numpy', 'dace', 'sympy', 'builtins'))

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
        elif isinstance(value, sympy.Basic):  # Symbolic or constant expression
            newnode = ast.parse(symbolic.symstr(value)).body[0].value
        elif isinstance(value, ast.Name):
            newnode = ast.Name(id=value.id, ctx=ast.Load())
        elif (dtypes.isconstant(value) or isinstance(value, (StringLiteral, SDFG)) or hasattr(value, '__sdfg__')):
            # Could be a constant, an SDFG, or SDFG-convertible object
            if isinstance(value, SDFG) or hasattr(value, '__sdfg__'):
                self.closure.closure_sdfgs[id(value)] = (qualname, value)
            elif isinstance(value, StringLiteral):
                value = value.value
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
                # NumPy array dispatchers have an _implementation field and are NOT regarded as functions by Python
                if (not inspect.isfunction(value) and not inspect.ismethod(value) and not inspect.isbuiltin(value)
                        and hasattr(value, '__call__') and not hasattr(value, '_implementation')):
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
                    if hasattr(parent_node, 'qualname') and parent_node.qualname in self.closure.callbacks:
                        # Already parsed
                        return None

                    cbqualname = astutils.unparse(parent_node.func)
                else:
                    cbqualname = astutils.rname(parent_node)
                cbname = self._qualname_to_array_name(cbqualname, prefix='')

                # Make a version of the callback without keyword arguments or Python literal objects (list, tuple, ...)
                cb_func = flatten_callback(value, parent_node, self.globals)

                # If the callback already exists, and the details differ (e.g., different kwarg names), make new
                if cbname in self.closure.callbacks and cb_func is not self.closure.callbacks[cbname][1]:
                    cbname = data.find_new_name(cbname, self.closure.callbacks)

                self.closure.callbacks[cbname] = (cbqualname, cb_func, False)
                parent_node.qualname = cbname

                # From this point on, any failure will result in a callback
                newnode = ast.Name(id=cbname, ctx=ast.Load())
                if isinstance(parent_node, ast.Call):
                    newnode.oldnode = parent_node.func

                if inspect.iscoroutinefunction(value):
                    return newnode

                # Decorated or functions with missing source code
                sast, _, _, _ = astutils.function_to_ast(value)
                if len(sast.body[0].decorator_list) > 0:
                    return newnode
                if find_disallowed_statements(sast):
                    return newnode

                parsed = parser.DaceProgram(value, [], {}, False, dtypes.DeviceType.CPU, ignore_type_hints=True)
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

                # Skip ``dace.compiletime``-annotated arguments
                is_constant = False
                if arg.annotation is not None:
                    try:
                        ann = astutils.evalnode(arg.annotation, self.globals)
                        if ann is dace.compiletime:
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

    def _visit_potential_constant(self, node: ast.AST, recurse_on_fail: bool) -> Optional[ast.AST]:
        if self._contains_preserved_attribute_access(node):
            if recurse_on_fail:
                return self.generic_visit(node)
            return node

        # Try to evaluate the expression with only the globals
        try:
            global_val = astutils.evalnode(node, self.globals)
        except SyntaxError:
            return self.generic_visit(node)

        if not isinstance(global_val, dtypes.typeclass):
            newnode = self.global_value_to_node(global_val,
                                                parent_node=node,
                                                qualname=astutils.unparse(node),
                                                recurse=True)
            if newnode is not None:
                return newnode

        # Failure
        if recurse_on_fail:
            return self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # Do not visit node recursilve on fail, it may modify the attribute value too soon
        return self._visit_potential_constant(node, False)

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
            if isinstance(node.value, ast.Dict):  # Dict
                for k, v in zip(node.value.keys, node.value.values):
                    try:
                        gkey = astutils.evalnode(k, self.globals)
                    except SyntaxError:
                        continue
                    if gkey == gslice:
                        return self._visit_potential_constant(v, True)
            elif isinstance(node.value, (ast.List, ast.Tuple)):  # List & Tuple
                # Loop over the list if slicing makes it a list
                if isinstance(node.value.elts[gslice], List):
                    visited_list = astutils.copy_tree(node.value)
                    visited_list.elts.clear()
                    for v in node.value.elts[gslice]:
                        visited_cst = self._visit_potential_constant(v, True)
                        visited_list.elts.append(visited_cst)
                    node.value = visited_list
                    return node
                else:
                    return self._visit_potential_constant(node.value.elts[gslice], True)
            else:  # Catch-all
                return self._visit_potential_constant(node, True)

        return self._visit_potential_constant(node, True)

    def visit_Call(self, node: ast.Call) -> Any:
        from dace.frontend.python.interface import in_program, inline  # Avoid import loop

        if hasattr(node.func, 'value') and isinstance(node.func.value, SDFGConvertible):
            # Skip already-parsed calls
            return self.generic_visit(node)

        try:
            global_func = astutils.evalnode(node.func, self.globals)

            # Built-in functions are resolved directly
            if global_func is in_program:
                return self.global_value_to_node(True, parent_node=node, qualname=astutils.unparse(node), recurse=True)
            # Inline contents are kept as-is
            if global_func is inline:
                return node

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
                if hasattr(newnode, 'oldnode'):
                    node.oldnode = newnode.oldnode
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
            values = [astutils.unparse(v.value) for v in visited.values]
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
        self.names: Set[str] = set()

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
            elif not until_loop_end and isinstance(stmt, (ast.For, ast.While)):
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
            mgr_name = data.find_new_name(
                f'__with_{item.context_expr.qualname if hasattr(item.context_expr, "qualname") else item.context_expr.id}',
                self.names)
            mgr = self.resolver.global_value_to_node(ctxmgr, node, mgr_name, keep_object=True)
            ctx_mgr_names.append((mgr.id, ctxmgr))
            self.names.add(mgr_name)

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
            if evald is dace.tasklet or evald is dace.named or isinstance(evald, (dace.tasklet, dace.named)):
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
        from dace.frontend.python.interface import MapGenerator

        EXPLICIT_GENERATORS = [
            range,  # Handled in ProgramVisitor
            dace.map,
            dace.consume,
            MapGenerator
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

        # Special case for map with @ operator
        if isinstance(niter, ast.BinOp) and isinstance(niter.op, ast.MatMult):
            try:
                geniter = astutils.evalnode(niter.left, self.globals)
                if isinstance(geniter, MapGenerator):
                    niter = niter.left
            except SyntaxError:
                pass
        # End of special case

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


class IteratorForLoopNormalizer(ast.NodeTransformer):
    """
    Rewrites non-range/map for-loops into simpler control-flow that the direct
    schedule-tree frontend can lower. Array-like iteration, zip, and enumerate
    are normalized to index-based loops; remaining iterators fall back to an
    explicit iterator protocol while-loop.
    """

    def __init__(self, globals: Dict[str, Any], argtypes: Dict[str, data.Data], closure_resolver: GlobalResolver):
        super().__init__()
        self.globals = globals
        self.argtypes = argtypes
        self.resolver = closure_resolver
        self._counter = 0

    def visit_For(self, node: ast.For) -> Any:
        node = self.generic_visit(node)

        if self._is_structured_iterator(node.iter):
            return node

        rewritten = self._normalize_indexed_iteration(node)
        if rewritten is not None:
            return rewritten

        rewritten = self._normalize_zip_iteration(node)
        if rewritten is not None:
            return rewritten

        rewritten = self._normalize_enumerate_iteration(node)
        if rewritten is not None:
            return rewritten

        return self._normalize_generic_iteration(node)

    def _is_structured_iterator(self, iterator: ast.AST) -> bool:
        schedule_target = iterator.left if isinstance(iterator, ast.BinOp) and isinstance(iterator.op,
                                                                                          ast.MatMult) else iterator
        if isinstance(schedule_target, ast.Call):
            return astutils.rname(schedule_target.func) in {'range', 'prange', 'parrange'}
        if isinstance(schedule_target, ast.Subscript):
            return astutils.rname(schedule_target.value) == 'dace.map'
        return False

    def _normalize_indexed_iteration(self, node: ast.For) -> Optional[ast.For]:
        length_expr = self._indexed_iterator_length(node.iter)
        if length_expr is None:
            return None

        index_name = self._fresh_name('iter_idx')
        yielded_value = self._indexed_iterator_value(node.iter, index_name, node)
        if yielded_value is None:
            return None
        replacements = self._target_replacements(node.target, yielded_value)
        if replacements is None:
            return None

        rewritten = ast.For(target=ast.Name(id=index_name, ctx=ast.Store()),
                            iter=self._make_range_call(length_expr),
                            body=self._rewrite_body(node.body, replacements),
                            orelse=[astutils.copy_tree(stmt) for stmt in node.orelse])
        return ast.fix_missing_locations(ast.copy_location(rewritten, node))

    def _normalize_zip_iteration(self, node: ast.For) -> Optional[Any]:
        if not isinstance(node.iter, ast.Call) or astutils.rname(node.iter.func) != 'zip' or len(node.iter.args) == 0:
            return None

        return self._normalize_generic_zip_iteration(node)

    def _normalize_enumerate_iteration(self, node: ast.For) -> Optional[Any]:
        if not isinstance(node.iter, ast.Call) or astutils.rname(node.iter.func) != 'enumerate' or len(
                node.iter.args) == 0:
            return None

        iterable = node.iter.args[0]
        start = astutils.copy_tree(node.iter.args[1]) if len(node.iter.args) > 1 else astutils.create_constant(0, node)

        return self._normalize_generic_iteration(node, enumerate_start=start)

    def _normalize_generic_zip_iteration(self, node: ast.For) -> Any:
        iterator_names = [self._fresh_name('iter') for _ in node.iter.args]
        value_names = [self._fresh_name('iter_value') for _ in node.iter.args]
        sentinel_id = self._sentinel_name(node)

        init_nodes: List[ast.AST] = []
        for iterator_name, value_name, arg in zip(iterator_names, value_names, node.iter.args):
            init_nodes.append(self._assign(iterator_name, self._helper_call('__dace_iterator_init', [arg]), node))
            init_nodes.append(
                self._assign(
                    value_name,
                    self._helper_call(
                        '__dace_iterator_next',
                        [ast.Name(id=iterator_name, ctx=ast.Load()),
                         ast.Name(id=sentinel_id, ctx=ast.Load())]), node))

        yielded_value = ast.Tuple(elts=[ast.Name(id=value_name, ctx=ast.Load()) for value_name in value_names],
                                  ctx=ast.Load())
        replacements = self._target_replacements(node.target, yielded_value)
        destructuring_setup = None
        if replacements is None:
            destructuring_setup = self._destructuring_setup(node.target, yielded_value, node)
            if destructuring_setup is None:
                return node
            replacements = {}

        test = ast.BoolOp(op=ast.And(),
                          values=[
                              ast.Compare(left=ast.Name(id=value_name, ctx=ast.Load()),
                                          ops=[ast.IsNot()],
                                          comparators=[ast.Name(id=sentinel_id, ctx=ast.Load())])
                              for value_name in value_names
                          ])
        body: List[ast.AST] = []
        if destructuring_setup is not None:
            body.append(destructuring_setup)
        body.extend(self._rewrite_body(node.body, replacements))
        for iterator_name, value_name in zip(iterator_names, value_names):
            body.append(
                self._assign(
                    value_name,
                    self._helper_call(
                        '__dace_iterator_next',
                        [ast.Name(id=iterator_name, ctx=ast.Load()),
                         ast.Name(id=sentinel_id, ctx=ast.Load())]), node))

        loop = ast.While(test=test, body=body, orelse=[astutils.copy_tree(stmt) for stmt in node.orelse])
        return [*init_nodes, ast.fix_missing_locations(ast.copy_location(loop, node))]

    def _normalize_generic_iteration(self, node: ast.For, enumerate_start: Optional[ast.AST] = None) -> Any:
        iterator_name = self._fresh_name('iter')
        value_name = self._fresh_name('iter_value')
        sentinel_id = self._sentinel_name(node)

        init_nodes: List[ast.AST] = [
            self._assign(iterator_name, self._helper_call('__dace_iterator_init', [node.iter]), node),
            self._assign(
                value_name,
                self._helper_call(
                    '__dace_iterator_next',
                    [ast.Name(id=iterator_name, ctx=ast.Load()),
                     ast.Name(id=sentinel_id, ctx=ast.Load())]), node),
        ]

        counter_name: Optional[str] = None
        if enumerate_start is not None:
            counter_name = self._fresh_name('iter_index')
            init_nodes.append(self._assign(counter_name, enumerate_start, node))
            yielded_value: ast.AST = ast.Tuple(
                elts=[ast.Name(id=counter_name, ctx=ast.Load()),
                      ast.Name(id=value_name, ctx=ast.Load())],
                ctx=ast.Load())
        else:
            yielded_value = ast.Name(id=value_name, ctx=ast.Load())

        replacements = self._target_replacements(node.target, yielded_value)
        destructuring_setup = None
        if replacements is None:
            destructuring_setup = self._destructuring_setup(node.target, yielded_value, node)
            if destructuring_setup is None:
                return node
            replacements = {}

        body: List[ast.AST] = []
        if destructuring_setup is not None:
            body.append(destructuring_setup)
        body.extend(self._rewrite_body(node.body, replacements))
        if counter_name is not None:
            body.append(
                self._assign(
                    counter_name,
                    ast.BinOp(left=ast.Name(id=counter_name, ctx=ast.Load()),
                              op=ast.Add(),
                              right=astutils.create_constant(1, node)), node))
        body.append(
            self._assign(
                value_name,
                self._helper_call(
                    '__dace_iterator_next',
                    [ast.Name(id=iterator_name, ctx=ast.Load()),
                     ast.Name(id=sentinel_id, ctx=ast.Load())]), node))

        test = ast.Compare(left=ast.Name(id=value_name, ctx=ast.Load()),
                           ops=[ast.IsNot()],
                           comparators=[ast.Name(id=sentinel_id, ctx=ast.Load())])
        loop = ast.While(test=test, body=body, orelse=[astutils.copy_tree(stmt) for stmt in node.orelse])
        return [*init_nodes, ast.fix_missing_locations(ast.copy_location(loop, node))]

    def _is_indexable_expr(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id in self.argtypes:
            descriptor = self.argtypes[node.id]
            return hasattr(descriptor, 'shape') and not isinstance(descriptor, data.Scalar)
        try:
            value = astutils.evalnode(node, self.globals)
        except SyntaxError:
            return False
        return dtypes.is_array(value) or (hasattr(value, '__len__') and hasattr(value, '__getitem__'))

    def _indexed_iterator_length(self, iterator: ast.AST) -> Optional[ast.AST]:
        if self._is_indexable_expr(iterator):
            return self._make_len_call(iterator)

        if isinstance(iterator, ast.Call):
            call_name = astutils.rname(iterator.func)
            if call_name == 'zip' and iterator.args:
                lengths = [self._indexed_iterator_length(arg) for arg in iterator.args]
                if any(length is None for length in lengths):
                    return None
                return self._make_min_call(lengths)
            if call_name == 'enumerate' and iterator.args:
                return self._indexed_iterator_length(iterator.args[0])

        return None

    def _indexed_iterator_value(self, iterator: ast.AST, index_name: str, location: ast.AST) -> Optional[ast.AST]:
        if self._is_indexable_expr(iterator):
            return self._make_subscript(iterator, index_name)

        if isinstance(iterator, ast.Call):
            call_name = astutils.rname(iterator.func)
            if call_name == 'zip' and iterator.args:
                values = [self._indexed_iterator_value(arg, index_name, location) for arg in iterator.args]
                if any(value is None for value in values):
                    return None
                return ast.Tuple(elts=values, ctx=ast.Load())
            if call_name == 'enumerate' and iterator.args:
                inner_value = self._indexed_iterator_value(iterator.args[0], index_name, location)
                if inner_value is None:
                    return None
                start = astutils.copy_tree(iterator.args[1]) if len(iterator.args) > 1 else astutils.create_constant(
                    0, location)
                counter = ast.BinOp(left=start, op=ast.Add(), right=ast.Name(id=index_name, ctx=ast.Load()))
                return ast.Tuple(elts=[counter, inner_value], ctx=ast.Load())

        return None

    def _sentinel_name(self, node: ast.AST) -> str:
        sentinel = self.resolver.global_value_to_node(object(), node, self._fresh_name('iter_end'), keep_object=True)
        return sentinel.id

    def _fresh_name(self, prefix: str) -> str:
        name = f'__dace_{prefix}_{self._counter}'
        self._counter += 1
        return name

    def _make_range_call(self, stop: ast.AST) -> ast.Call:
        return ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[stop], keywords=[])

    def _make_len_call(self, value: ast.AST) -> ast.Call:
        return ast.Call(func=ast.Name(id='len', ctx=ast.Load()), args=[astutils.copy_tree(value)], keywords=[])

    def _make_min_call(self, values: List[ast.AST]) -> ast.Call:
        return ast.Call(func=ast.Name(id='min', ctx=ast.Load()), args=values, keywords=[])

    def _make_subscript(self, value: ast.AST, index_name: str) -> ast.Subscript:
        return ast.Subscript(value=astutils.copy_tree(value),
                             slice=ast.Name(id=index_name, ctx=ast.Load()),
                             ctx=ast.Load())

    def _helper_call(self, helper_name: str, args: List[ast.AST]) -> ast.Call:
        return ast.Call(func=ast.Name(id=helper_name, ctx=ast.Load()),
                        args=[astutils.copy_tree(arg) for arg in args],
                        keywords=[])

    def _assign(self, target_name: str, value: ast.AST, location: ast.AST) -> ast.Assign:
        return ast.fix_missing_locations(
            ast.copy_location(ast.Assign(targets=[ast.Name(id=target_name, ctx=ast.Store())], value=value), location))

    def _target_replacements(self, target: ast.AST, value: ast.AST) -> Optional[Dict[str, ast.AST]]:
        result: Dict[str, ast.AST] = {}

        def _collect(current_target: ast.AST, current_value: ast.AST) -> bool:
            if isinstance(current_target, ast.Name):
                result[current_target.id] = current_value
                return True

            if isinstance(current_target, (ast.Tuple, ast.List)):
                if not isinstance(current_value, (ast.Tuple, ast.List)):
                    return False
                if len(current_target.elts) != len(current_value.elts):
                    return False
                return all(
                    _collect(sub_target, sub_value)
                    for sub_target, sub_value in zip(current_target.elts, current_value.elts))

            return False

        if not _collect(target, value):
            return None
        return result

    def _rewrite_body(self, body: List[ast.AST], replacements: Dict[str, ast.AST]) -> List[ast.AST]:
        rewritten: List[ast.AST] = []
        for stmt in body:
            copied = astutils.copy_tree(stmt)
            replace = astutils.ASTFindReplace({name: astutils.copy_tree(value) for name, value in replacements.items()})
            rewritten.append(ast.fix_missing_locations(replace.visit(copied)))
        return rewritten

    def _destructuring_setup(self, target: ast.AST, value: ast.AST, location: ast.AST) -> Optional[ast.Assign]:
        if not isinstance(target, (ast.Tuple, ast.List)):
            return None
        setup = ast.Assign(targets=[astutils.copy_tree(target)], value=astutils.copy_tree(value))
        return ast.fix_missing_locations(ast.copy_location(setup, location))


class ExpressionInliner(ast.NodeTransformer):
    """
    Replaces dace.inline() expressions by their bodies if they can be
    compile-time evaluated.
    """

    def __init__(self, globals: Dict[str, Any], filename: str, closure_resolver: GlobalResolver):
        super().__init__()
        self.globals = globals
        self.filename = filename
        self.resolver = closure_resolver

    def visit_Call(self, node: ast.Call) -> Any:
        # Avoid import loop
        from dace.frontend.python.interface import inline

        node = self.generic_visit(node)

        try:
            nfunc = astutils.evalnode(node.func, self.globals)
        except SyntaxError:
            nfunc = None

        if nfunc is not inline:
            return node

        if len(node.args) != 1:
            raise DaceSyntaxError(None, node, 'dace.inline must be called with one argument')

        # Try to inline the expression on the current AST
        try:
            contents = astutils.evalnode(node.args[0], self.globals)
        except SyntaxError:
            raise DaceSyntaxError(
                None, node, 'Cannot inline expression with dace.inline, it '
                'cannot be evaluated at compile time.')

        ##########################################

        # Already AST
        def _convert_to_ast(contents: Any):
            if isinstance(contents, ast.AST):
                newnode = contents
            elif isinstance(contents, (numbers.Number, str)):
                # Compatibility check since Python changed their AST nodes
                newnode = astutils.create_constant(contents)
            elif isinstance(contents, (list, tuple, set)):
                newnode = ast.copy_location(ast.Tuple(elts=[_convert_to_ast(c) for c in contents], ctx=ast.Load()),
                                            node)
            else:
                # Augment closure with new value
                newnode = self.resolver.global_value_to_node(contents,
                                                             node,
                                                             f'inlined_{id(contents)}',
                                                             True,
                                                             keep_object=True)
            return newnode

        return _convert_to_ast(contents)


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

    def _get_given_args(self, node: ast.Call, function) -> Set[str]:
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
        if not isinstance(node.func, ast.Constant):
            self.seen_calls.add(astutils.unparse(node.func))
            return self.generic_visit(node)
        if hasattr(node.func, 'oldnode'):
            if isinstance(node.func.oldnode, ast.Call):
                self.seen_calls.add(astutils.unparse(node.func.oldnode.func))
            else:
                self.seen_calls.add(astutils.rname(node.func.oldnode))
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


class DisallowedAssignmentChecker(ast.NodeVisitor):
    """
    Tests a pre-processed program for disallowed assignments to compile-time constants, and raises a
    ``DaceSyntaxError`` exception if one is found.
    """

    def __init__(self, filename: str) -> None:
        super().__init__()
        self.visitor = collections.namedtuple('Visitor', 'filename')
        self.visitor.filename = filename

    def _check_assignment_target(self, node: ast.expr, parent_node: ast.AST):
        if hasattr(node, 'qualname'):
            raise DaceSyntaxError(
                self.visitor, parent_node, f'Trying to assign to a compile-time constant "{node.qualname}", which is '
                'disallowed. Refer to the Frequently Asked Questions in the documentation on how to avoid this issue.')

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            self._check_assignment_target(target, node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._check_assignment_target(node.target, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._check_assignment_target(node.target, node)
        self.generic_visit(node)

    def visit_NamedExpr(self, node):
        self._check_assignment_target(node.target, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if any(k.arg is None for k in node.keywords):
            raise DaceSyntaxError(
                self.visitor, node, 'Double-starred (dictionary unpacking, e.g., `**a`) arguments are '
                'currently unsupported.')


class NamedExprDesugarer(ast.NodeTransformer):
    """Lifts walrus operator (NamedExpr / :=) assignments out of expressions.

    ``if (x := f()): body`` becomes::

        x = f()
        if x: body

    ``while (x := f()): body`` becomes::

        x = f()
        while x:
            body
            x = f()
    """

    def _extract_named_exprs(self, node: ast.AST):
        """Find NamedExpr nodes in an expression and return (assignments, rewritten_expr)."""
        assignments = []

        class _Replacer(ast.NodeTransformer):

            def visit_NamedExpr(self, ne: ast.NamedExpr) -> ast.AST:
                # Recurse into the value first
                ne.value = self.visit(ne.value)
                assign = ast.Assign(targets=[copy.deepcopy(ne.target)], value=ne.value)
                ast.copy_location(assign, ne)
                assignments.append(assign)
                replacement = ast.Name(id=ne.target.id, ctx=ast.Load())
                return ast.copy_location(replacement, ne)

        rewritten = _Replacer().visit(copy.deepcopy(node))
        return assignments, rewritten

    def _has_named_expr(self, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.NamedExpr):
                return True
        return False

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if not self._has_named_expr(node.test):
            return node
        assignments, new_test = self._extract_named_exprs(node.test)
        node.test = new_test
        ast.fix_missing_locations(node)
        return assignments + [node]

    def visit_While(self, node: ast.While) -> ast.AST:
        self.generic_visit(node)
        if not self._has_named_expr(node.test):
            return node
        assignments, new_test = self._extract_named_exprs(node.test)
        node.test = new_test
        # Add re-evaluation at end of loop body
        for assign in assignments:
            node.body.append(copy.deepcopy(assign))
        ast.fix_missing_locations(node)
        return assignments + [node]

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if not self._has_named_expr(node.value):
            return node
        assignments, new_value = self._extract_named_exprs(node.value)
        node.value = new_value
        ast.fix_missing_locations(node)
        return assignments + [node]

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        self.generic_visit(node)
        if not self._has_named_expr(node.value):
            return node
        assignments, new_value = self._extract_named_exprs(node.value)
        node.value = new_value
        ast.fix_missing_locations(node)
        return assignments + [node]

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if node.value is None or not self._has_named_expr(node.value):
            return node
        assignments, new_value = self._extract_named_exprs(node.value)
        node.value = new_value
        ast.fix_missing_locations(node)
        return assignments + [node]


class ComprehensionDesugarer(ast.NodeTransformer):
    """Desugars all comprehensions and generator expressions to explicit loops.

    ``[expr for x in iter if cond]`` becomes::

        __comp_tmp_N = []
        for x in iter:
            if cond:
                __comp_tmp_N.append(expr)

    Set and dict comprehensions are handled similarly.
    Generator expressions consumed by a call (e.g. ``sum(x for x in ...)``)
    are desugared to list comprehensions then wrapped in the call.
    """

    def __init__(self):
        self._counter = 0

    def _fresh_name(self) -> str:
        self._counter += 1
        return f'__comp_tmp_{self._counter}'

    def _build_loop_nest(self, generators, body_stmt, target_node) -> list:
        """Build nested for/if statements from comprehension generators."""
        stmts = body_stmt
        # Build inside-out
        for gen in reversed(generators):
            # Wrap with if-filters
            for if_clause in reversed(gen.ifs):
                if_node = ast.If(test=if_clause, body=stmts if isinstance(stmts, list) else [stmts], orelse=[])
                ast.copy_location(if_node, target_node)
                stmts = [if_node]
            # Wrap with for-loop
            for_node = ast.For(target=gen.target,
                               iter=gen.iter,
                               body=stmts if isinstance(stmts, list) else [stmts],
                               orelse=[])
            ast.copy_location(for_node, target_node)
            stmts = [for_node]
        return stmts if isinstance(stmts, list) else [stmts]

    def _desugar_listcomp(self, node: ast.ListComp, target_node: ast.AST) -> Tuple[str, list]:
        name = self._fresh_name()
        # __comp_tmp = []
        init = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load()))
        ast.copy_location(init, target_node)
        # __comp_tmp.append(elt)
        append_call = ast.Expr(
            value=ast.Call(func=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr='append', ctx=ast.Load()),
                           args=[node.elt],
                           keywords=[]))
        ast.copy_location(append_call, target_node)
        loops = self._build_loop_nest(node.generators, [append_call], target_node)
        return name, [init] + loops

    def _desugar_setcomp(self, node: ast.SetComp, target_node: ast.AST) -> Tuple[str, list]:
        name = self._fresh_name()
        # __comp_tmp = set()
        init = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                          value=ast.Call(func=ast.Name(id='set', ctx=ast.Load()), args=[], keywords=[]))
        ast.copy_location(init, target_node)
        # __comp_tmp.add(elt)
        add_call = ast.Expr(
            value=ast.Call(func=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr='add', ctx=ast.Load()),
                           args=[node.elt],
                           keywords=[]))
        ast.copy_location(add_call, target_node)
        loops = self._build_loop_nest(node.generators, [add_call], target_node)
        return name, [init] + loops

    def _desugar_dictcomp(self, node: ast.DictComp, target_node: ast.AST) -> Tuple[str, list]:
        name = self._fresh_name()
        # __comp_tmp = {}
        init = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Dict(keys=[], values=[]))
        ast.copy_location(init, target_node)
        # __comp_tmp[key] = value
        assign_stmt = ast.Assign(
            targets=[ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()), slice=node.key, ctx=ast.Store())],
            value=node.value)
        ast.copy_location(assign_stmt, target_node)
        loops = self._build_loop_nest(node.generators, [assign_stmt], target_node)
        return name, [init] + loops

    def _desugar_generatorexp(self, node: ast.GeneratorExp, target_node: ast.AST) -> Tuple[str, list]:
        # Desugar generator expressions as list comprehensions
        listcomp = ast.ListComp(elt=node.elt, generators=node.generators)
        ast.copy_location(listcomp, target_node)
        return self._desugar_listcomp(listcomp, target_node)

    def _find_and_desugar(self, node: ast.AST) -> Tuple[list, ast.AST]:
        """Walk an expression, desugar any comprehensions found, return (prefix_stmts, rewritten_expr)."""
        prefix_stmts = []

        outer_self = self

        class _Replacer(ast.NodeTransformer):

            def visit_ListComp(self, lc: ast.ListComp) -> ast.AST:
                # Recurse into sub-expressions first
                lc = self.generic_visit(lc)
                name, stmts = outer_self._desugar_listcomp(lc, lc)
                prefix_stmts.extend(stmts)
                return ast.Name(id=name, ctx=ast.Load())

            def visit_SetComp(self, sc: ast.SetComp) -> ast.AST:
                sc = self.generic_visit(sc)
                name, stmts = outer_self._desugar_setcomp(sc, sc)
                prefix_stmts.extend(stmts)
                return ast.Name(id=name, ctx=ast.Load())

            def visit_DictComp(self, dc: ast.DictComp) -> ast.AST:
                dc = self.generic_visit(dc)
                name, stmts = outer_self._desugar_dictcomp(dc, dc)
                prefix_stmts.extend(stmts)
                return ast.Name(id=name, ctx=ast.Load())

            def visit_GeneratorExp(self, ge: ast.GeneratorExp) -> ast.AST:
                ge = self.generic_visit(ge)
                name, stmts = outer_self._desugar_generatorexp(ge, ge)
                prefix_stmts.extend(stmts)
                return ast.Name(id=name, ctx=ast.Load())

        rewritten = _Replacer().visit(copy.deepcopy(node))
        return prefix_stmts, rewritten

    def _has_comprehension(self, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                return True
        return False

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if not self._has_comprehension(node.value):
            return node
        prefix, new_value = self._find_and_desugar(node.value)
        node.value = new_value
        ast.fix_missing_locations(node)
        result = prefix + [node]
        for stmt in result:
            ast.fix_missing_locations(stmt)
        return result

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        self.generic_visit(node)
        if not self._has_comprehension(node.value):
            return node
        prefix, new_value = self._find_and_desugar(node.value)
        node.value = new_value
        ast.fix_missing_locations(node)
        result = prefix + [node]
        for stmt in result:
            ast.fix_missing_locations(stmt)
        return result

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if node.value is None or not self._has_comprehension(node.value):
            return node
        prefix, new_value = self._find_and_desugar(node.value)
        node.value = new_value
        ast.fix_missing_locations(node)
        result = prefix + [node]
        for stmt in result:
            ast.fix_missing_locations(stmt)
        return result

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if not self._has_comprehension(node.test):
            return node
        prefix, new_test = self._find_and_desugar(node.test)
        node.test = new_test
        ast.fix_missing_locations(node)
        result = prefix + [node]
        for stmt in result:
            ast.fix_missing_locations(stmt)
        return result

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        if not self._has_comprehension(node.iter):
            return node
        prefix, new_iter = self._find_and_desugar(node.iter)
        node.iter = new_iter
        ast.fix_missing_locations(node)
        result = prefix + [node]
        for stmt in result:
            ast.fix_missing_locations(stmt)
        return result

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        if not self._has_comprehension(node.value):
            return node
        prefix, new_value = self._find_and_desugar(node.value)
        node.value = new_value
        ast.fix_missing_locations(node)
        result = prefix + [node]
        for stmt in result:
            ast.fix_missing_locations(stmt)
        return result


class AugAssignExpander(ast.NodeTransformer):

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.Assign:
        target = self.generic_visit(node.target)
        value = self.generic_visit(node.value)
        newvalue = ast.copy_location(ast.BinOp(left=copy.deepcopy(target), op=node.op, right=value), value)
        return ast.copy_location(ast.Assign(targets=[target], value=newvalue), node)


def find_disallowed_statements(node: ast.AST, stmts=None):
    if stmts is None:
        from dace.frontend.python.newast import DISALLOWED_STMTS  # Avoid import loop
        stmts = DISALLOWED_STMTS
    # Skip everything until the function contents (in case there are disallowed statements in a decorator)
    if isinstance(node, ast.Module) and isinstance(node.body[0], ast.FunctionDef):
        nodes = node.body[0].body
    else:
        nodes = [node]

    for topnode in nodes:
        for subnode in ast.walk(topnode):
            # Found disallowed statement
            if type(subnode).__name__ in stmts:
                return type(subnode).__name__

            # Calls with double-starred arguments (**args)
            if isinstance(subnode, ast.Call):
                if any(k.arg is None for k in subnode.keywords):
                    return type(subnode).__name__
    return None


class MPIResolver(ast.NodeTransformer):
    """ Resolves mpi4py-related constants, e.g., mpi4py.MPI.COMM_WORLD. """

    def __init__(self, globals: Dict[str, Any]):
        from mpi4py import MPI
        self.globals = globals
        self.MPI = MPI
        self.parents = {}
        self.parent = None

    def visit(self, node):
        self.parents[node] = self.parent
        self.parent = node
        node = super().visit(node)
        if isinstance(node, ast.AST):
            self.parent = self.parents[node]
        return node

    def visit_Name(self, node: ast.Name) -> Union[ast.Name, ast.Attribute]:
        self.generic_visit(node)
        if node.id in self.globals:
            obj = self.globals[node.id]
            if isinstance(obj, self.MPI.Comm):
                lattr = ast.Attribute(ast.Name(id='mpi4py', ctx=ast.Load), attr='MPI')
                if obj is self.MPI.COMM_NULL:
                    newnode = ast.copy_location(ast.Attribute(value=lattr, attr='COMM_NULL'), node)
                    self.parents[newnode] = self.parents[node]
                    return newnode
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        self.generic_visit(node)
        if isinstance(node.attr, str) and node.attr == 'Request':
            try:
                val = astutils.evalnode(node, self.globals)
                if val is self.MPI.Request and not isinstance(self.parents[node], ast.Attribute):
                    newnode = ast.copy_location(
                        ast.Attribute(value=ast.Name(id='dace', ctx=ast.Load), attr='MPI_Request'), node)
                    self.parents[newnode] = self.parents[node]
                    return newnode
            except SyntaxError:
                pass
        return node


class ModuloConverter(ast.NodeTransformer):
    """ Converts a % b expressions to (a + b) % b for C/C++ compatibility. """

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        if isinstance(node.op, ast.Mod):
            left = self.generic_visit(node.left)
            right = self.generic_visit(node.right)
            newleft = ast.copy_location(ast.BinOp(left=left, op=ast.Add(), right=astutils.copy_tree(right)), left)
            node.left = newleft
            return node
        return self.generic_visit(node)


def preprocess_dace_program(f: Callable[..., Any],
                            argtypes: Dict[str, data.Data],
                            global_vars: Dict[str, Any],
                            modules: Dict[str, Any],
                            resolve_functions: bool = False,
                            parent_closure: Optional[SDFGClosure] = None,
                            default_args: Optional[Set[str]] = None,
                            normalize_generic_for_loops: bool = False,
                            preserve_object_attributes: bool = False,
                            disallowed_stmts: Optional[Set[str]] = None) -> Tuple[PreprocessedAST, SDFGClosure]:
    """
    Preprocesses a ``@dace.program`` and all its nested functions, returning
    a preprocessed AST object and the closure of the resulting SDFG.

    :param f: A Python function to parse.
    :param argtypes: An dictionary of (name, type) for the given
                        function's arguments, which may pertain to data
                        nodes or symbols (scalars).
    :param global_vars: A dictionary of global variables in the closure
                        of ``f``.
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

    try:
        src_ast = MPIResolver(global_vars).visit(src_ast)
    except (ImportError, ModuleNotFoundError, RuntimeError):
        pass
    src_ast = ModuloConverter().visit(src_ast)

    if normalize_generic_for_loops:
        global_vars['__dace_iterator_init'] = __dace_iterator_init
        global_vars['__dace_iterator_next'] = __dace_iterator_next

    # Resolve constants to their values (if they are not already defined in this scope)
    # and symbols to their names
    resolved = {k: v for k, v in global_vars.items() if k not in (argtypes.keys() - default_args) and k != '_'}
    closure_resolver = GlobalResolver(resolved,
                                      resolve_functions,
                                      default_args=default_args,
                                      preserve_object_attributes=preserve_object_attributes)

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
    if disallowed_stmts is None:
        disallowed = find_disallowed_statements(src_ast)
    elif disallowed_stmts:
        disallowed = find_disallowed_statements(src_ast, disallowed_stmts)
    else:
        disallowed = None  # Empty set means nothing is disallowed
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
            DisallowedAssignmentChecker(src_file).visit(src_ast)
            if normalize_generic_for_loops:
                src_ast = ComprehensionDesugarer().visit(src_ast)
            src_ast = LoopUnroller(resolved, src_file, closure_resolver).visit(src_ast)
            if normalize_generic_for_loops:
                src_ast = IteratorForLoopNormalizer(resolved, argtypes, closure_resolver).visit(src_ast)
            src_ast = ExpressionInliner(resolved, src_file, closure_resolver).visit(src_ast)
            src_ast = ContextManagerInliner(resolved, src_file, closure_resolver).visit(src_ast)
            src_ast = ConditionalCodeResolver(resolved).visit(src_ast)
            if normalize_generic_for_loops:
                src_ast = NamedExprDesugarer().visit(src_ast)
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
            raise TypeError('Parsing failed due to recursion in a data-centric context called from this function')
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
