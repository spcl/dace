
import ast
from collections import ChainMap

from ast import AST, NodeVisitor, dump, iter_fields
from copy import deepcopy

from .coverage import class_attrs_blacklist, global_class_attrs, global_module_types
from .ssa_nodes import PhiAssign, UniqueName, WhileCFG, SingleAssign
from .type_types import Any, Dict, FunctionReturn, List, Union, Type, Tuple, NoReturn, Optional, VariableFunction, TypeVar
from .type_types import TypeRef, DictOfTypes, ClassAttributes, NewType, DefiniteFunction, TypedArg, TypeDict, Undefined
from .type_helpers import TypeInferenceError, EnclosingFunction, type_of_type
from .type_inferrer_expr import ExpressionTyper


# Functions
# Args - Types & Positions
# Kwargs - Types & Positions
# *args
# **kwargs
# default values
# missing implementation (builtins & extensions)


# Ideas:
# Add ways to set what Type is used for widening, unknown attribute lookup etc
# Collect all possible return types for all function "overloads"
# Put side-effects into function return type -> create new "type"
# Toggle to assume / not assume side-effects for functions w/o explicit side-effect declaration
# -> can read common type annotations or .pyi files?

# Questions
# Should accessing an undefined variable throw an error? What if it's never reached anyway?
# How to convert Type -> str and str -> Type
# Do I really need an Undefined type?
# Is the class structure good?

# Problems:
# Classes / Types / Objects / Attributes / Methods

# ToDo
# Handle PyVersions
# Register Module import Types (somehow)

# Done
# Remove Multi-Assignments
# Remove Tuple Assignments
# Annotate all assignments to make type inference readable



class TypeInferrer(NodeVisitor):

    ignore_unknown_types = False

    unknown_type = Any
    widening_type = Any
    import_default = Any
    function_arg_default = Undefined

    #############################
    # Constructor & Main Methods
    #############################

    def __init__(self, expr_typer: Type[ExpressionTyper] = ExpressionTyper):

        self.expression_typer = expr_typer()
        self.class_attrs: ClassAttributes = {}
        self._update_assigns: bool = True

    def get_type_of_expr(self, expr: ast.AST, types: DictOfTypes) -> TypeRef:

        all_class_attrs = ChainMap(self.class_attrs, global_class_attrs)
        
        return self.expression_typer(expr, types, all_class_attrs)
    
    def is_subtype(self, subtype: TypeRef, supertype: TypeRef) -> bool:
        return self.expression_typer.is_definite_subtype(subtype, supertype)
    
    def visit(self, node: AST, types: DictOfTypes) -> DictOfTypes:
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, types)

    def visit_block(self, block: List[AST], types: DictOfTypes) -> TypeDict:
        
        new_types = TypeDict()
        all_types = ChainMap(new_types, types)

        for value in block:
            types = self.visit(value, types=all_types)
            new_types.update(types)
        
        return new_types
      
    def generic_visit(self, node: AST, types: DictOfTypes) -> TypeDict:
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        self.visit(item, types)
            elif isinstance(value, AST):
                self.visit(value, types)
          
    def visit_Module(self, node: ast.Module, types: DictOfTypes) -> TypeDict:
    #       | Module(stmt* body, type_ignore* type_ignores)

        block_types = self.visit_block(node.body, types)
        defined_types = {var: typ for var, typ in block_types.items() if typ is not Undefined}

        return defined_types

    #############################
    # Function & Class Statements
    #############################

    def visit_ClassDef(self, node: ast.ClassDef, types: DictOfTypes) -> TypeDict:
    #       | ClassDef(identifier name,
    #          expr* bases,
    #          keyword* keywords,
    #          stmt* body,
    #          expr* decorator_list)

        new_types = TypeDict()
        cls_name = node.name

        body_block = node.body

        body_types = self.visit_block(body_block, types)
        class_namespace = {var: typ for var, typ in body_types.items() if typ is not Undefined}

        unsupported_attrs = class_namespace.keys() & class_attrs_blacklist
        if unsupported_attrs:
            raise TypeInferenceError(f"Class '{cls_name}' defines unsupported attributes {unsupported_attrs}")

        # HACK (temporary)
        cls_type = NewType(cls_name, object)
        new_types[cls_name] = TypeRef(Type[TypeRef(cls_type)], class_namespace)

        self.class_attrs[cls_type] = class_namespace

        return new_types

    def visit_FunctionDef(self, node: ast.FunctionDef, types: DictOfTypes) -> TypeDict:
    #       | FunctionDef(identifier name, arguments args,
    #                    stmt* body, expr* decorator_list, expr? returns,
    #                    string? type_comment)

        args = node.args

        arg_converter = lambda x: self._arg_to_typed(x, types)
        
        # all the different possible arguments (in order)
        func_posonlyargs = [*map(arg_converter, args.posonlyargs)]
        func_args = [*map(arg_converter, args.args)]
        func_vararg = arg_converter(args.vararg) if args.vararg else args.vararg
        func_kwonlyargs = [*map(arg_converter, args.kwonlyargs)]
        func_kwarg = arg_converter(args.kwarg) if args.kwarg else args.kwarg

        # default values
        func_defaults = args.defaults
        func_kw_defaults = args.kw_defaults

        if node.returns is not None:
            return_type = self.evaluate_type(node.returns, types).nominal
        else:
            return_type = None

        func_type = DefiniteFunction(
            name=node.name,
            posonlyargs=func_posonlyargs,
            args=func_args,
            vararg=func_vararg,
            kwonlyargs=func_kwonlyargs,
            kwarg=func_kwarg,
            defaults=func_defaults,
            kw_defaults=func_kw_defaults,
            returns=return_type,
        )

        is_argtyped = func_type.is_argtyped
        
        # Case 1: Arguments fully typed and return is FunctionReturn -> trust annotations
        if (is_argtyped and isinstance(return_type, FunctionReturn)):
            # If return type is FunctionReturn, we trust the annotation
            func_type.is_typechecked = True
            return TypeDict({node.name: TypeRef(func_type)})

        # Case 2: Arguments fully typed
        elif is_argtyped:
            return self.handle_DefiniteFunction(node, func_type, types)

        # Case 3: Variably-typed function
        else:
            func_type = VariableFunction(
                name=node.name,
                posonlyargs=func_posonlyargs,
                args=func_args,
                vararg=func_vararg,
                kwonlyargs=func_kwonlyargs,
                kwarg=func_kwarg,
                defaults=func_defaults,
                kw_defaults=func_kw_defaults,
                returns=return_type,
                # new:
                body=node.body,
            )

            return self.handle_VariableFunction(node, func_type, types)
    

    visit_AsyncFunctionDef = visit_FunctionDef
    #       | AsyncFunctionDef(identifier name, arguments args,
    #                          stmt* body, expr* decorator_list, expr? returns,
    #                          string? type_comment)

    def handle_VariableFunction(self, node:ast.FunctionDef, func_type: VariableFunction, types: DictOfTypes) -> TypeDict:

        fname = node.name
        new_types = TypeDict({fname: TypeRef(func_type)})

        # Create a typechecking function with a copy of the function AST

        def typecheck_func(arg_types):

            with EnclosingFunction(self, node, update_assigns=False):
                body_types = self.visit_block(func_type.body, types=ChainMap(arg_types, new_types, types))
                return_type = TypeRef(Undefined)
                for ret_type in self.return_types:
                    return_type = return_type.unioned(ret_type)
            
            declared_return = func_type.returns
            
            if declared_return is None:
                return_effect = FunctionReturn(return_type=return_type, effects={})
            elif self.is_subtype(return_type, declared_return):
                return_effect = FunctionReturn(return_type=declared_return, effects={})
            else:
                raise TypeInferenceError(f"Inferred return type '{return_type}' is not a definite"
                                            f" subtype of declared return type '{declared_return}'")

            return return_effect
        
        func_type.set_typecheck_func(typecheck_func)

        # Pro-forma typecheck the function with Any

        arg_types = {}
        for arg in func_type.all_args:
            arg_name = arg.id
            arg_type = arg.type
            arg_types[arg_name] = Any if arg_type.nominal is Undefined else arg_type

        with EnclosingFunction(self, node):
            body_types = self.visit_block(node.body, types=ChainMap(new_types, arg_types, types))
        
        return new_types

    def handle_DefiniteFunction(self, node:ast.FunctionDef, func_type: DefiniteFunction, types: DictOfTypes) -> TypeDict:

        fname = node.name
        new_types = TypeDict({fname: TypeRef(func_type)})
        arg_types = func_type.get_args_typedict()

        with EnclosingFunction(self, node):
            body_types = self.visit_block(node.body, types=ChainMap(arg_types, new_types, types))

            return_type = TypeRef(Undefined)
            for ret_type in self.return_types:
                return_type = return_type.unioned(ret_type)
        
        for arg, type in arg_types.items():
            if arg in body_types and body_types[arg] != type:
                raise TypeInferenceError(f"Argument '{arg}' illegally changed type in function '{fname}'")

        declared_return = func_type.returns
        if declared_return is None:
            return_effect = FunctionReturn(return_type=return_type, effects={})
        elif self.is_subtype(return_type, declared_return):
            return_effect = FunctionReturn(return_type=declared_return, effects={})
        else:
            raise TypeInferenceError(f"Inferred return type '{return_type}' is not a definite"
                                     f" subtype of declared return type '{declared_return}'")

        func_type.returns = return_effect
        func_type.is_typechecked = True
        
        return new_types

    def visit_Return(self, node: ast.Return, types: DictOfTypes) -> DictOfTypes:
    #       | Return(expr? value)
        
        return_val = node.value

        if return_val is not None:
            return_type = self.get_type_of_expr(return_val, types)
        else:
            return_type = TypeRef(Type[None])
        
        self.return_types.append(return_type)

        return {}

    def _arg_to_typed(self, arg: ast.arg, types: DictOfTypes) -> TypedArg:

        id: str = arg.arg
        annot = arg.annotation

        type = self.evaluate_type(annot, types) if annot else TypeRef(self.function_arg_default)
        
        new_node = TypedArg(id=id, type=type, default=None)
        return new_node


    def evaluate_type(self, node: AST, types: DictOfTypes) -> TypeRef:

        if isinstance(node, ast.Constant):
            if node.value is None:
                return TypeRef(Type[None])
            else:
                return self._handle_unknown_type(f"Constant '{node.value}' is not a type!")
            
        if isinstance(node, (ast.Name, UniqueName)):
            annot = node.id
            annot_type = types.get(annot, None)
            if annot_type is None:
                return self._handle_unknown_type(f"Name '{annot}' is an unknown reference!")

            if not isinstance(annot_type, TypeRef):
                raise Exception(f"{annot_type} is not TypeRef!")

            annot_type_origin = annot_type.origin
            print("annot: ", annot)
            print("annot_type: ", annot_type)
            print("type(annot_type_origin): ", type(annot_type_origin))
            print("annot_type_origin is TypeVar: ", annot_type_origin is TypeVar)
            print("isinstance(annot_type_origin, TypeVar): ", isinstance(annot_type_origin, TypeVar))
            
            if annot_type_origin is Type:
                type_args = annot_type.args
                assert len(type_args) == 1
                return type_args[0]

            elif annot_type_origin is TypeVar:
                return annot_type

            else:
                return self._handle_unknown_type(f"Name '{annot}' is not a (known) type!")
            
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple):
            # Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='int', ctx=Load())
            value = self.evaluate_type(node.value, types)
            slice_types = self.evaluate_tuple_slice(node.slice, types)
            return TypeRef(value.nominal[slice_types])

        elif isinstance(node, ast.Subscript):
            value = self.evaluate_type(node.value, types)
            slice = self.evaluate_type(node.slice, types)

            return TypeRef(value.nominal[slice])

        return self._handle_unknown_type(f"Cannot evaluate node {dump(node)} to type!")
    
    def evaluate_tuple_slice(self, tuple_node: ast.Tuple, types: DictOfTypes) -> Tuple[TypeRef, ...]:

        result_types = []

        for elt in tuple_node.elts:

            if isinstance(elt, ast.Constant) and elt.value == ...:
                result_types.append(TypeRef(Ellipsis))
    
            elif isinstance(elt, ast.Dict):
                result = {}

                for key, value in zip(elt.keys, elt.values):
                    if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                        raise TypeInferenceError("Dict type annotation must be strings!")
                    key_name = key.value
                    value_type = self.evaluate_type(value, types)
                    result[key_name] = value_type
                
                result_types.append(result)
            
            else:
                elt_type = self.evaluate_type(elt, types)
                result_types.append(elt_type)
                
        return tuple(result_types)
                
                    
    def _handle_unknown_type(self, err_msg: str) -> Union[Type, NoReturn]:

        if self.ignore_unknown_types:
            return self.unknown_type
        else:
            raise TypeInferenceError(err_msg)
    
    #############################
    # Assignment Statements
    #############################

    def visit_SingleAssign(self, node: SingleAssign, types: DictOfTypes) -> TypeDict:
        # breakpoint()
        
        target = node.target
        value = node.value
        annotation = node.annotation

        if value is not None:
            value_type = self.get_type_of_expr(value, types)
        else:
            value_type = TypeRef(Undefined)

        if annotation:
            annot_type = self.evaluate_type(annotation, types)
            value_type = value_type.unioned(annot_type)

        if self._update_assigns:
            type_str = str(value_type)
            node.annotation = ast.Name(id=type_str, ctx=ast.Load())

        new_types = self.visit_assignment(target, value_type, types)

        return new_types

    def visit_Delete(self, node: ast.Delete, types: DictOfTypes) -> TypeDict:
    #       | Delete(expr* targets)

        new_types = TypeDict()

        for target in node.targets:
            types = self.visit_assignment(target, Undefined, types)
            new_types.update(types)

        return new_types

    def visit_assignment(self, target: ast.AST, value_type: Type, types: DictOfTypes) -> TypeDict:

        new_types = TypeDict()

        # Names, Simple Access
        if isinstance(target, ast.Name):
            var_name = target.id
            new_types[var_name] = value_type

        # Partial access
        elif isinstance(target, ast.Subscript):
            # TODO
            raise NotImplementedError('Subscript')
        elif isinstance(target, ast.Attribute):

            target_type = self.get_type_of_expr(target.value, types)
            # print("target.value: ", target.value)
            # print("target_type: ", target_type)
            attr = target.attr

            target_type.structural[attr] = value_type

        # Multiple assignments
        elif isinstance(target, (ast.Tuple, ast.List, ast.Starred)):
            raise SyntaxError('Complex assignments are not supported.'
                              ' SSA conversion removes these.')
        else:
            raise NotImplementedError(f'Node type {type(target)}')

        return new_types

    #############################
    # Unchanged Statements
    #############################

    def visit_PhiAssign(self, node: PhiAssign, types: DictOfTypes) -> DictOfTypes:

        var_name = node.target
        phi_type = TypeRef(Undefined)

        if node.active:
            for uid, _ in node.operands:
                if uid in types:
                    phi_type = phi_type.unioned(types[uid])

            type_name = str(phi_type)
            node.annotation = ast.Name(id=type_name, ctx=ast.Load())
        
        return {var_name: phi_type}

    def visit_Expr(self, node: ast.Expr, types: DictOfTypes) -> DictOfTypes:
    #       | Expr(expr value)
        self.get_type_of_expr(node.value, types)
        return {}

    def visit_Assert(self, node: ast.Assert, types: DictOfTypes) -> DictOfTypes:
    #       | Assert(expr test, expr? msg)

        self.get_type_of_expr(node.test, types)
        if node.msg:
            self.get_type_of_expr(node.msg, types)
        return {}

    def visit_Raise(self, node: ast.Raise, types: DictOfTypes) -> DictOfTypes:
    #       | Raise(expr? exc, expr? cause)

        if node.exc:
            self.get_type_of_expr(node.exc, types)

        if node.cause:
            self.get_type_of_expr(node.cause, types)

        return {}

    #############################
    # Control Flow Statements
    #############################

    def visit_If(self, node: ast.If, types: DictOfTypes) -> TypeDict:
    #       | If(expr test, stmt* body, stmt* orelse)

        test_expr = node.test
        body_block = node.body
        else_block = node.orelse

        types_copy = deepcopy(types)

        new_types = {}

        self.get_type_of_expr(test_expr, types)
        body_types = self.visit_block(body_block, types=types)
        else_types = self.visit_block(else_block, types=types_copy)

        for (var, type1), (_, type2) in zip(types.items(), types_copy.items()):
            if type1 != type2:
                new_type = type1.unioned(type2)
                type1.replace(new_type)
                type2.replace(new_type)

        new_types.update(body_types)
        new_types.update(else_types)

        return new_types

    def visit_WhileCFG(self, node: WhileCFG, types: DictOfTypes) -> TypeDict:
    #       | While(expr test, stmt* body, stmt* orelse)

        head_block = node.head
        ifelse = node.ifelse
        exit_block = node.exit

        # Round 1: infer body // phi nodes only half filled
        new_types1 = TypeDict()
        round1_types = ChainMap(new_types1, types)

        with EnclosingFunction(self, node, update_assigns=False):
            head_types = self.visit_block(head_block, types=round1_types)
            new_types1.union_update(head_types)
            self.get_type_of_expr(ifelse.test, round1_types)
            body_types = self.visit_block(ifelse.body, types=round1_types)
            new_types1.union_update(body_types)

        # Round 2: reinfer for phi nodes
        new_types2 = TypeDict()
        round2_types = ChainMap(new_types2, new_types1, types)

        with EnclosingFunction(self, node, update_assigns=False):
            head_types = self.visit_block(head_block, types=round2_types)
            new_types2.union_update(head_types)
            self.get_type_of_expr(ifelse.test, round2_types)
            body_types = self.visit_block(ifelse.body, types=round2_types)
            new_types2.union_update(body_types)

        # anything that changes after 2nd round 
        # gets widened to avoid infinite types
        
        for var, typ in new_types1.items():
            typ_now = new_types2[var]
            if typ != typ_now:
                new_types1[var] = self.widen_type(typ, typ_now)

        # Round 4: redo inference after widening
        new_types = TypeDict()
        final_types = ChainMap(new_types, new_types1, types)

        head_types = self.visit_block(head_block, types=final_types)
        new_types.union_update(head_types)

        self.get_type_of_expr(ifelse.test, final_types)

        body_types = self.visit_block(ifelse.body, types=final_types)
        new_types.union_update(body_types)
        
        # Finally, do else & exit block
        else_types = self.visit_block(ifelse.orelse, types=final_types)
        new_types.union_update(else_types)
        exit_types = self.visit_block(exit_block, types=final_types)
        new_types.union_update(exit_types)
        
        return new_types

    def widen_type(self, type1: TypeRef, type2: TypeRef) -> TypeRef:

        # Simple types:
        if type1 == type2:
            return type1

        origin1 = type1.origin
        origin2 = type2.origin

        # Match types level 1 - origin
        if origin1 == origin2:
            args1 = type1.args
            args2 = type2.args

            # Match types level 2 - args
            if args1 and len(args1) == len(args2):
                new_args = [self.widen_type(a1, a2) for a1, a2 in zip(args1, args2)]

                return TypeRef(origin1[tuple(new_args)])
                
        return TypeRef(self.widening_type)

    def visit_Try(self, node: ast.Try, types: DictOfTypes) -> TypeDict:
    #       | Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)

        new_types = TypeDict()

        body_block = node.body
        else_block = node.orelse
        final_block = node.finalbody

        types = self.visit_block(body_block, types)
        new_types.update(types)

        for handler in node.handlers:
            #  ExceptHandler(expr? type, identifier? name, stmt* body)

            handler_type = handler.type
            handler_name = handler.type
            handler_block = handler.body

            if handler_type:
                self.get_type_of_expr(handler_type, types)
            if handler_name:
                self.get_type_of_expr(handler_name, types)
            types = self.visit_block(handler_block)
            new_types.update(types)

        types = self.visit_block(else_block, types)
        new_types.update(types)

        types = self.visit_block(final_block, types)
        new_types.update(types)

        return new_types

    #############################
    # Import Statements
    #############################

    def visit_Import(self, node: ast.Import, types: DictOfTypes) -> TypeDict:
    #       | Import(alias* names)

        new_types = TypeDict()

        for alias in node.names:

            module_struct = global_module_types.get(alias.name, {})

            import_name = alias.asname if (alias.asname is not None) else alias.name
            new_types[import_name] = TypeRef(self.import_default, module_struct)
        
        return new_types

    def visit_ImportFrom(self, node: ast.ImportFrom, types: DictOfTypes) -> TypeDict:
    #       | ImportFrom(identifier? module, alias* names, int? level)

        new_types = TypeDict()

        module_struct = global_module_types.get(node.module, {})

        for alias in node.names:
            import_name = alias.asname if (alias.asname is not None) else alias.name
            new_types[import_name] = module_struct.get(alias.name, self.import_default)
        
        return new_types

    #############################
    # Ignored Statements
    #############################

    def visit_Global(self, node: ast.Global, types: DictOfTypes) -> NoReturn:
    #       | Global(identifier* names)
        raise SyntaxError("Global statements are not supported!")

    def visit_Nonlocal(self, node: ast.Nonlocal, types: DictOfTypes) -> NoReturn:
    #       | Nonlocal(identifier* names)
        raise SyntaxError("Nonlocal statements are not supported!")

    def visit_Assign(self, node: ast.Assign, types: DictOfTypes) -> NoReturn:
    #       | Assign(expr* targets, expr value, string? type_comment)
        raise NotImplementedError("Assign statements are removed by SSA")
        
    def visit_AnnAssign(self, node: ast.AnnAssign, types: DictOfTypes) -> NoReturn:
    #       | AnnAssign(expr target, expr annotation, expr? value, int simple)
        raise NotImplementedError("AnnotatedAssign statements are removed by SSA")

    def do_nothing(self, node: ast.AST, types: DictOfTypes) -> DictOfTypes:
    #       | Pass | Break | Continue
        return {}

    visit_Pass = visit_Break = visit_Continue = do_nothing
        

    #       | AugAssign(expr target, operator op, expr value)
    
    #       | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
    #       | AsyncFor(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
    
    #       | With(withitem* items, stmt* body, string? type_comment)
    #       | AsyncWith(withitem* items, stmt* body, string? type_comment)

