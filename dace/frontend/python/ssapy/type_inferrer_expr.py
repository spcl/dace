
import ast

from ast import AST, NodeVisitor, unparse
from collections import ChainMap
from copy import deepcopy

from .type_types import Any, ClassAttributes, Dict, FunctionReturn, List, TypeDict, Union, Type, Tuple, Set, Callable, TypeVar, NoReturn  # well-known Types
from .type_types import TypeRef, FunctionType, DefiniteFunction, VariableFunction, DictOfTypes, Namespace, Undefined  # new types
from .type_helpers import TypeInferenceError, get_origin_type, get_args


# Abbreviated Types
Condition = Tuple[TypeVar, Type]
Sideeffect = DictOfTypes

unop_to_dunder: Dict[AST, List[str]] = {

    # Unary Operators:  Invert | Not | UAdd | USub
    ast.Invert: ['__invert__'],  # ~
    ast.Not: ['????'],
    ast.UAdd: ['__pos__'],
    ast.USub: ['__neg__'],
    # xx: ['__abs__'],   ==> Not an operator, but a builtin method!
    
}

binop_to_dunder: Dict[AST, List[str]] = {

    # Binary Operators:   Add | Sub | Mult | MatMult | Div | Mod | Pow 
    #                   | LShift | RShift | BitOr | BitXor | BitAnd | FloorDiv
    ast.Add: ['__add__', '__radd__'],
    ast.Sub: ['__sub__', '__rsub__'],
    ast.Mult: ['__mul__', '__rmul__'],
    ast.MatMult: ['__matmul__', '__rmatmul__'],
    ast.Div: ['__truediv__', '__rtruediv__'],
    ast.FloorDiv: ['__floordiv__', '__rfloordiv__'],
    ast.Mod: ['__mod__', '__rmod__'],
    ast.Pow: ['__pow__', '__rpow__'],
    ast.LShift: ['__lshift__', '__rlshift__'],
    ast.RShift: ['__rshift__', '__rrshift__'],
    ast.BitOr: ['__or__', '__ror__'],
    ast.BitXor: ['__xor__', '__rxor__'],
    ast.BitAnd: ['__and__', '__rand__'],
    # xx: ['__divmod__', '__rdivmod__],   ==> Not an operator, but a builtin method!
}

comp_to_dunder: Dict[AST, List[str]] = {

    # Rich Comparisons:  Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
    ast.Eq: ['__eq__'],
    ast.NotEq: ['__ne__'],
    ast.Lt: ['__lt__'],
    ast.LtE: ['__le__'],
    ast.Gt: ['__gt__'],
    ast.GtE: ['__ge__'],
    # ast.Is: ['__sub__'],
    # ast.IsNot: ['__sub__'],
    # ast.In: ['__sub__'],
    # ast.NotIn: ['__sub__'],
    
}


class Subtyper:

    def is_consistent(self, subtype: Type, supertype: Type) -> Tuple[bool, List[Condition]]:
        
        # Any is compatible with everything
        if subtype is Any:
            return True, []
        else:
            return self.is_subtype(subtype, supertype)

    def is_subtype(self, subtype: Type, supertype: Type) -> Tuple[bool, List[Condition]]:
        
        # Undefined is compatible with nothing
        if subtype is Undefined:
            return False, []

        type_origin = get_origin_type(subtype)

        # handle Union specially?
        if type_origin is Union:

            union_types = get_args(subtype)
            are_consistent = []
            conditions = []

            for ty in union_types:
                is_cons, cond = self.is_consistent(ty, supertype)
                are_consistent.append(is_cons)
                conditions.extend(cond)

            #any_is = any(are_consistent)
            all_are = all(are_consistent)

            return all_are, conditions

        # handle TypeVar separately
        if type_origin is TypeVar:
            # TODO
            ...
        
        subtype_func = self.get_subtype_func(supertype)
        return subtype_func(subtype, supertype)

    def is_nonvariant(self, subtype: Type, supertype: Type) -> Tuple[bool, List[Condition]]:

        if subtype == supertype:
            return True, []

        subtype_origin = get_origin_type(subtype)
        supertype_origin = get_origin_type(supertype)

        if subtype_origin is Union and supertype_origin is Union:
            super_union_types = set(get_args(supertype))
            sub_union_types = set(get_args(supertype))
            is_nonvariant = sub_union_types.issubset(super_union_types)
        elif supertype_origin is Union:
            super_union_types = get_args(supertype)
            is_nonvariant = subtype in super_union_types
        else:
            is_nonvariant = (subtype == supertype)
        
        return is_nonvariant, []
        
    #######################
    # Subtype Functions
    #######################

    def get_subtype_func(self, supertype: Type) -> Callable[[Type], Tuple[bool, List[Condition]]]:
        "modelled after Nodevisitor.visit()"

        method = 'subtype_' + supertype.__name__
        subtype_func = getattr(self, method, None)

        if subtype_func is not None:
            return subtype_func
        else:
            raise NotImplementedError(f"No subtype function defined for supertype {supertype}")

    # Special Types
    #######################

    def subtype_Any(self, subtype: Type, supertype: Type) -> Tuple[bool, List[Condition]]:
        return True, []

    def subtype_Undefined(self, subtype: Type, supertype: Type) -> Tuple[bool, List[Condition]]:
        return False, []

    # Basic Types
    #######################

    def subtype_object(self, subtype: Type, supertype: Type[object]) -> Tuple[bool, List[Condition]]:
        return True, []

    def subtype_bool(self, subtype: Type, supertype: Type[bool]) -> Tuple[bool, List[Condition]]:
        is_subtype = subtype in (bool, )
        return is_subtype, []

    def subtype_int(self, subtype: Type, supertype: Type[int]) -> Tuple[bool, List[Condition]]:
        is_subtype = subtype in (int, bool, )
        return is_subtype, []

    def subtype_float(self, subtype: Type, supertype: Type[float]) -> Tuple[bool, List[Condition]]:
        is_subtype = subtype in (float, int, bool, )
        return is_subtype, []

    def subtype_str(self, subtype: Type, supertype: Type[str]) -> Tuple[bool, List[Condition]]:
        is_subtype = subtype in (str, )
        return is_subtype, []

    def subtype_bytes(self, subtype: Type, supertype: Type[bytes]) -> Tuple[bool, List[Condition]]:
        is_subtype = subtype in (bytes, )
        return is_subtype, []

    # Container Types
    #######################

    def subtype_Union(self, subtype: Type, supertype: Type[List]) -> Tuple[bool, List[Condition]]:
        return self.is_nonvariant(subtype, supertype)

    def subtype_List(self, subtype: Type, supertype: Type[List]) -> Tuple[bool, List[Condition]]:
        subtype_origin = get_origin_type(subtype)
        if subtype_origin is not List:
            return False, []

        subtype_args = get_args(subtype)
        supertype_args = get_args(supertype)
        return self.is_nonvariant(subtype_args, supertype_args)

    def subtype_TypeVar(self, subtype: Type, supertype: Type[List]) -> Tuple[bool, List[Condition]]:

        condition = (supertype, subtype)
        return True, [condition]

    # TODO
    # Tuple, Dict, etc


class ExpressionTyper(NodeVisitor):

    attribute_undefined = Any

    def __init__(self, subtyper=Subtyper):

        self.subtyper = subtyper()

    def is_definite_consistent(self, subtype: TypeRef, supertype: TypeRef) -> bool:

        is_subtype, conditions = self.subtyper.is_consistent(subtype.nominal, supertype.nominal)
        return is_subtype and (len(conditions) == 0)

    def is_conditional_consistent(self, subtype: TypeRef, supertype: TypeRef) -> Tuple[bool, List[Condition]]:
        return self.subtyper.is_consistent(subtype.nominal, supertype.nominal)

    def is_definite_subtype(self, subtype: TypeRef, supertype: TypeRef) -> bool:

        is_subtype, conditions = self.subtyper.is_subtype(subtype.nominal, supertype.nominal)
        return is_subtype and (len(conditions) == 0)

    def is_conditional_subtype(self, subtype: TypeRef, supertype: TypeRef) -> Tuple[bool, List[Condition]]:
        return self.subtyper.is_subtype(subtype.nominal, supertype.nominal)

    # Default

    def visit(self, node: ast.AST, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, None)

        if visitor is not None:
            return visitor(node, types, classes)
        else:
            return TypeRef(Any)

    __call__ = visit

    def update_type(self, old_type: Type, new_type: Type) -> None:

        if old_type == new_type:
            return

        old_origin = get_origin_type(old_type)
        new_origin = get_origin_type(new_type)

        if old_origin != new_origin:
            raise TypeInferenceError(f"Illegal type update from {old_type} to {new_type}."
                                     " Only generic arguments may be updated!")
        
        new_args = get_args(new_type)
        old_type.__args__ = new_args
        
    ##########################
    # Expressions (supported)
    ##########################

    # Constants / Basic Datatypes
    #############################

    def visit_Constant(self, node: ast.Constant, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | Constant(constant value, string? kind)
        const_type = type(node.value)
        return TypeRef(const_type)

    def visit_Name(self, node: ast.Name, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | Name(identifier id, expr_context ctx)
        return types.get(node.id, TypeRef(Undefined))

    def visit_List(self, node: ast.List, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | List(expr* elts, expr_context ctx)

        element_type = TypeRef(Undefined)

        for elt in node.elts:
            elt_type = self.visit(elt, types, classes)
            element_type = element_type.unioned(elt_type)

        list_type = List[element_type]
        return TypeRef(list_type)

    def visit_Tuple(self, node: ast.Tuple, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | Tuple(expr* elts, expr_context ctx)

        elt_types = tuple([self.visit(e, types, classes) for e in node.elts])
        tuple_type = Tuple[elt_types]
        
        return TypeRef(tuple_type)

    def visit_Set(self, node: ast.Set, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | Set(expr* elts)

        element_type = TypeRef(Undefined)

        for elt in node.elts:
            elt_type = self.visit(elt, types, classes)
            element_type = element_type.unioned(elt_type)

        set_type = Set[element_type]
        return TypeRef(set_type)

    def visit_Dict(self, node: ast.Dict, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | Dict(expr* keys, expr* values)

        key_type = TypeRef(Undefined)
        val_type = TypeRef(Undefined)

        for k, v in zip(node.keys, node.values):
            k_typ = self.visit(k, types, classes)
            v_typ = self.visit(v, types, classes)
            key_type = key_type.unioned(k_typ)
            val_type = val_type.unioned(v_typ)

        dict_type = Dict[key_type, val_type]

        return TypeRef(dict_type)

    def visit_JoinedStr(self, node: ast.JoinedStr, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | JoinedStr(expr* values)
        return TypeRef(str)

    def visit_FormattedValue(self, node: ast.FormattedValue, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | FormattedValue(expr value, int conversion, expr? format_spec)
        # this should really never be called anyway :)
        return TypeRef(str)

    # Indirect Function Calls
    #############################

    def visit_UnaryOp(self, node: ast.UnaryOp, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | UnaryOp(unaryop op, expr operand)

        operand_type = self.visit(node.operand, types, classes)
        unary_op = node.op

        if isinstance(unary_op, ast.Not):
            self.check_booly_args([operand_type], classes)
            return TypeRef(bool)

        method_name = unop_to_dunder[type(unary_op)]

        method = self.get_attr(operand_type, method_name, classes, is_data_descr=True)
        return self.handle_typed_call(method, [], {}, classes)

    def visit_BoolOp(self, node: ast.BoolOp, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | BoolOp(boolop op, expr* values)
        self.check_booly_args(node.values, classes)
        return TypeRef(bool)

    def check_booly_args(self, args: List[TypeRef], classes: ClassAttributes) -> Union[None, NoReturn]:

        bool_type = TypeRef(bool)

        for arg_type in args:
            if self.is_definite_subtype(arg_type, bool_type):
                continue

            method = self.get_attr(arg_type, '__bool__', classes, is_data_descr=True)
            bool_type = self.handle_typed_call(method, [], {}, classes)

            if not self.is_definite_subtype(bool_type, bool_type):
                raise TypeInferenceError("Method '__bool__' of type {arg_type} returns non-bool value")

    def visit_BinOp(self, node: ast.BinOp, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | BinOp(expr left, operator op, expr right)

        l_type = self.visit(node.left, types, classes)
        r_type = self.visit(node.right, types, classes)
        
        l_method, r_method = binop_to_dunder[type(node.op)]

        # Try 1: Left Method
        try:
            method = self.get_attr(l_type, l_method, classes, is_data_descr=True)
            return self.handle_typed_call(method, [r_type], {}, classes)
        
        except TypeInferenceError as e:

            # Try 2: Right Method
            if r_method is not None:
                try:
                    method = self.get_attr(r_type, r_method, classes, is_data_descr=True)
                    return self.handle_typed_call(method, [l_type], {}, classes)
                except TypeInferenceError:
                    pass
            
            raise e

    def visit_Subscript(self, node: ast.Subscript, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | Subscript(expr value, expr slice, expr_context ctx)

        value_type = self.visit(node.value, types, classes)

        if value_type is Any:
            return TypeRef(Any)
        
        value_type_origin = value_type.origin
        
        if value_type_origin is Tuple:
            return self.tuple_subscript(value_type, node.slice, types, classes)
        else:
            method = self.get_attr(value_type, '__getitem__', classes, is_data_descr=True)
            return self.handle_typed_call(method, [], {}, classes)
    
    def tuple_subscript(self, tuple_type: TypeRef, slice: ast.AST, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:

        tuple_args = tuple_type.args
        slice_type = self.visit(slice, types, classes)
        is_infinite = tuple_type.is_infinite_tuple

        # Slicing into tuple
        if isinstance(slice, (ast.Slice, ast.Ellipsis)):
            if is_infinite:
                return TypeRef(tuple_type)
            else:
                return TypeRef(Tuple[Union[tuple_args], ...])  # type: ignore

        # Constant Indexing
        if isinstance(slice, ast.Constant) and (slice_type.origin is int):
            slice_index = slice.value
            if is_infinite:
                return tuple_args[0]
            elif slice_index < len(tuple_args):
                return tuple_args[slice_index]
            else:
                raise IndexError(f'Tuple slice {unparse(slice)} out of bounds'
                                 f' for tuple of type {tuple_type}')

        # Element Indexing
        if self.is_definite_subtype(slice_type.nominal, TypeRef(int)):
            return TypeRef(Union[tuple_args])  # type: ignore

        # Fallback: we cannot know if slice object takes 1 or many elements out
        if is_infinite:
            element_type = tuple_args[0]
            full_type = tuple_type
        else:
            element_type = Union[tuple_args]  # type: ignore
            full_type = Tuple[element_type, ...]
        return TypeRef(Union[element_type, full_type])

    # Function Calls
    #############################

    def visit_Call(self, node: ast.Call, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | Call(expr func, expr* args, keyword* keywords)

        func_type = self.visit(node.func, types, classes)

        if func_type.nominal is Any:
            return TypeRef(Any)

        arg_types = [self.visit(arg, types, classes) for arg in node.args]
        kwarg_types = {kwarg.arg: self.visit(kwarg.value, types, classes) for kwarg in node.keywords}

        return self.handle_typed_call(func_type, arg_types, kwarg_types, classes)
    
    def handle_typed_call(self, func_type: TypeRef, arg_types: List[TypeRef], kwarg_types: Dict[str, TypeRef], classes: ClassAttributes) ->  TypeRef:

        func_type_origin = func_type.origin

        # Object instantiation
        if func_type.is_type:
            return_type = func_type.single_arg
            return_effects = {}
            method = self.get_attr(return_type, '__init__', classes, is_data_descr=True)
            self.handle_typed_call(method, arg_types, kwarg_types, classes)

        elif func_type_origin is Union:

            union_args = func_type.args
            return_types = []
            return_effects = TypeDict()

            for f_type in union_args:
                try:
                    type, struct, effect = self.handle_typed_call(f_type, arg_types, kwarg_types, classes)
                except TypeInferenceError:
                    pass
                else:
                    return_types.append(type)
                    return_effects.union_update(effect)
            
            return_type = Union[tuple(return_types)] if return_types else Union  # type: ignore

        elif isinstance(func_type.nominal, DefiniteFunction):

            call_types = self.create_call_dict(func_type.nominal, arg_types, kwarg_types)
            func_return = func_type.nominal.typecheck(call_types)
            assert isinstance(func_return, FunctionReturn)

            return_type = func_return.return_type
            return_effects = func_return.effects if func_return.effects is not None else {}

            for var, new_type in return_effects.items():
                old_type = call_types[var]
                old_type.update(new_type)

        elif isinstance(func_type.nominal, VariableFunction):
            
            call_types = self.create_call_dict(func_type.nominal, arg_types, kwarg_types, is_variable=True)
            func_return = func_type.nominal.typecheck(call_types)
            return_type = func_return.return_type
            return_effects = func_return.effects if func_return.effects is not None else {}
        
            for var, new_type in return_effects.items():
                old_type = call_types[var]
                old_type.update(new_type)

        else:
            # print("failed to infer Call")
            return_type = TypeRef(Any)

        return return_type

    def create_call_dict(self, func_type: FunctionType, call_args: List[Type], call_kwargs: Dict[str, Type], *, is_variable: bool = False) -> DictOfTypes:

        def raise_call_error(arg_name: str, is_type: Type, arg_type: Type):
            raise TypeInferenceError(f"Invalid call type '{is_type}' for argument '{arg_name}' with type '{arg_type}'")
        
        def is_variably_typed(default_type: Type) -> bool:
            return is_variable and (default_type.nominal in {Any, Undefined})

        fname = func_type.name
        num_posonly = len(func_type.posonlyargs)
        num_args = len(func_type.args)

        call_dict = {}
        conditions = []

        # 1) Check positional only
        if len(call_args) < num_posonly:
            # Calling with less than required posonly is illegal
            raise TypeInferenceError(f"Function {fname} called with {len(call_args)} but at least"
                                     f" {num_posonly} positional arguments were expected")

        for call_type, func_arg in zip(call_args[:num_posonly], func_type.posonlyargs):
            arg_name, arg_type = func_arg.id, func_arg.type
            if self.is_definite_subtype(call_type, arg_type):
                call_dict[arg_name] = deepcopy(arg_type)
            elif is_variably_typed(arg_type):
                call_dict[arg_name] = call_type
            else:
                raise_call_error(arg_name, call_type, arg_type)

        # 2) Check positional OR keyword
        call_args = call_args[num_posonly:]

        if len(call_args) > num_args and func_type.vararg is None:
            # if there are more positional arguments there must be *arg
            raise TypeInferenceError(f"Function {fname} called with {len(call_args) + num_posonly} positional"
                                     f" arguments but at most {num_args + num_posonly} were expected")
        # After this we can assume
        # EITHER there are less or equal call_args than args
        # OR there is a *arg
        
        for call_type, func_arg in zip(call_args[:num_args], func_type.args):
            arg_name, arg_type = func_arg.id, func_arg.type
            if self.is_definite_subtype(call_type, arg_type) or is_variably_typed(arg_type):
                call_dict[arg_name] = call_type
            else:
                raise_call_error(arg_name, call_type, arg_type)
        
        # 3) Check *args
        if call_args[num_args:]:
            # there must be a vararg (see reasoning above)
            for call_type in call_args:
                vararg_name, vararg_type = func_type.vararg.id, func_type.vararg.type
                if self.is_definite_subtype(call_type, vararg_type):
                    call_dict[vararg_name] = deepcopy(vararg_type)
                elif is_variably_typed(vararg_type):
                    call_dict[vararg_name] = call_type
                else:
                    raise_call_error(vararg_name, call_type, vararg_type)
                    
            func_kwargs = []
        else:
            # there must be remaining args (see reasoning above)
            func_kwargs = func_type.args[len(call_args):]

        # 4) Check keyword only
        func_kwargs.extend(func_type.kwonlyargs)

        for kwarg in func_kwargs:
            if kwarg.id not in call_kwargs:
                raise TypeInferenceError(f"Function {fname} missing keyword argument '{kwarg.id}'")
            call_type = call_kwargs.pop(kwarg.id)
            kwarg_name, kwarg_type = kwarg.id, kwarg.type
            if self.is_definite_subtype(call_type, kwarg_type):
                call_dict[kwarg_name] = deepcopy(kwarg_type)
            elif is_variably_typed(kwarg_type):
                call_dict[kwarg_name] = call_type
            else:
                raise_call_error(kwarg_name, call_type, kwarg_type)

        # 5) Check **kwargs
        if (func_type.kwarg is None) and len(call_kwargs) > 0:
            sep = "', '"
            raise TypeInferenceError(f'Function {fname} called with unexpected keyword arguments \'{sep.join(call_kwargs)}\'')
        
        for _, call_type in call_kwargs.items():
            kwarg_name, kwarg_type = func_type.kwarg.id, func_type.kwarg.type
            if self.is_definite_subtype(call_type, kwarg_type):
                call_dict[kwarg_name] = deepcopy(kwarg_type)
            elif is_variably_typed(kwarg_type):
                call_dict[kwarg_name] = call_type
            else:
                raise_call_error(kwarg_name, call_type, kwarg_type)
        
        return call_dict
       
    def visit_Attribute(self, node: ast.Attribute, types: DictOfTypes, classes: ClassAttributes) ->  TypeRef:
        #  | Attribute(expr value, identifier attr, expr_context ctx)

        attr_name: str = node.attr
        value_type = self.visit(node.value, types, classes)
        return self.get_attr(value_type, attr_name, classes)

    def get_attr(self, value_type: TypeRef, attr_name: str, classes: ClassAttributes, is_data_descr=False) -> TypeRef:
        nominal_type = value_type.nominal
        structural_type = value_type.structural

        # 1) Check instance attribute
        if not is_data_descr and (attr_name in structural_type):
            attr = structural_type[attr_name]

        # 2) Check class attributes
        else:
            if nominal_type not in classes:
                return TypeRef(self.attribute_undefined)
            
            class_namespace = classes[nominal_type]

            if attr_name not in class_namespace:
                return TypeRef(self.attribute_undefined)
            
            attr = class_namespace[attr_name]

        attr_nominal = attr.nominal

        # If attribute is a method -> bind it
        if isinstance(attr_nominal, VariableFunction):
            bound_method = attr_nominal.as_bound_method(value_type)
            return TypeRef(bound_method)
        elif isinstance(attr_nominal, DefiniteFunction):
            first_arg = attr_nominal.get_self_arg()
            matches_type = self.is_definite_subtype(value_type, first_arg.type)
            bound_method = attr_nominal.as_bound_method(well_typed=matches_type)
            return TypeRef(bound_method)
        else:
            return attr

    # ToDo Expression Types
    # expr =
    #      -- the grammar constrains where yield expressions can occur
        #  | Await(expr value)
        #  | Yield(expr? value)
        #  | YieldFrom(expr value)
    #      -- need sequences for compare to distinguish between
    #      -- x < 4 < 3 and (x < 4) < 3
    # !!!! Careful with "in" operator
        #  | Compare(expr left, cmpop* ops, expr* comparators)

    #      -- the following expression can appear in assignment context
        #  | Starred(expr value, expr_context ctx)   --> any iterable


    # Removed Expression Types
    #############################
    @staticmethod
    def __raise_unsupported_error(node: AST) -> NoReturn:
        raise NotImplementedError('Node type {type(node)} not supported! '
                                  'Use SSA to remove prior to type inference.')

    # These can't show up after SSA Transformation and are therefore disallowed
    def visit_NamedExpr(self, node: ast.NamedExpr, types: DictOfTypes, classes: ClassAttributes) -> NoReturn:
        #  | NamedExpr(expr target, expr value)
        self.__raise_unsupported_error(node)
    def visit_ListComp(self, node: ast.ListComp, types: DictOfTypes, classes: ClassAttributes) -> NoReturn:
        #  | ListComp(expr elt, comprehension* generators)
        self.__raise_unsupported_error(node)
    def visit_SetComp(self, node: ast.SetComp, types: DictOfTypes, classes: ClassAttributes) -> NoReturn:
        #  | SetComp(expr elt, comprehension* generators)
        self.__raise_unsupported_error(node)
    def visit_DictComp(self, node: ast.DictComp, types: DictOfTypes, classes: ClassAttributes) -> NoReturn:
        #  | DictComp(expr key, expr value, comprehension* generators)
        self.__raise_unsupported_error(node)
    def visit_GeneratorExp(self, node: ast.GeneratorExp, types: DictOfTypes, classes: ClassAttributes) -> NoReturn:
        #  | GeneratorExp(expr elt, comprehension* generators)
        self.__raise_unsupported_error(node)
    def visit_Lambda(self, node: ast.Lambda, types: DictOfTypes, classes: ClassAttributes) -> NoReturn:
        #  | Lambda(arguments args, expr body)
        self.__raise_unsupported_error(node)
    def visit_IfExp(self, node: ast.IfExp, types: DictOfTypes, classes: ClassAttributes) -> NoReturn:
        #  | IfExp(expr test, expr body, expr orelse)
        self.__raise_unsupported_error(node)
