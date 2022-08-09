
from copy import copy as _copy
from ast import AST as _AST, unparse as _unparse
from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod
from dataclasses import dataclass as _dataclass, field as _field

from typing import get_args as _get_args, get_origin as _get_origin

#######################
# Imported Basic Types
#######################

from typing import (

    # Redefined Types
    NewType as _NewType, 

    # Type Creation
    Type, Generic, TypeVar,

    # PEP 483 Types
    Any, Union, Tuple, Callable, Optional,

    # Builtin Generics
    Dict, List, Set,

    # Structural Types
    Iterable, 

    # Others
    NoReturn,

)


####################################
# Base Type (Nominal & Structural)
####################################

_types_to_bases: Dict[Type, Type] = {
    type: Type,
    list: List,
    set: Set,
    dict: Dict,
    tuple: Tuple,
}

@_dataclass
class TypeRef:
    nominal: Type
    structural: Dict[str, 'TypeRef'] = _field(default_factory=dict)

    DEBUG = False
    DEBUG = True

    @property
    def origin(self) -> Type:
        if _get_args(self.nominal):
            origin = _get_origin(self.nominal)
            return _types_to_bases.get(origin, origin)
        else:
            return self.nominal
    
    @property
    def args(self) -> Tuple['TypeRef', ...]:
        return _get_args(self.nominal)

    @property
    def is_type(self) -> bool:
        return self.origin is Type

    @property
    def single_arg(self) -> bool:
        type_args = self.args
        assert len(type_args) == 1
        return type_args[0]

    @property
    def is_infinite_tuple(self) -> bool:
        args = self.args
        return (len(args) == 2) and (args[1].nominal == ...)

    @property
    def is_data_descriptor(self) -> bool:
        has_set = '__set__' in self.structural
        has_del = '__delete__' in self.structural
        return has_set or has_del

    @property
    def is_non_data_descriptor(self) -> bool:
        return '__get__' in self.structural


    def unionIP(self, other: 'TypeRef'):

        if not isinstance(other, TypeRef):
            raise TypeError(f"TypeRef.union() called with {type(other)}, not TypeRef!")

        # Structural Types are dropped, currently
        self.nominal = self._nominal_union(other)
    
    def unioned(self, other: 'TypeRef'):

        if not isinstance(other, TypeRef):
            raise TypeError(f"TypeRef.union() called with {type(other)}, not TypeRef!")

        # Structural Types are dropped, currently
        union_type = self._nominal_union(other)

        return TypeRef(union_type)
    
    def _nominal_union(self, other: 'TypeRef'):

        if self.nominal is Undefined:
            union_type = other.nominal
        elif other.nominal is Undefined:
            union_type = self.nominal
        elif self.nominal == other.nominal:
            union_type = self.nominal
        else:
            union_type = Union[TypeRef(self.nominal), TypeRef(other.nominal)]

        return union_type

    def update(self, other: 'TypeRef'):

        if not isinstance(other, TypeRef):
            raise TypeError(f"TypeRef.update() called with {type(other)}, not TypeRef!")

        self_origin = self.origin
        other_origin = other.origin
        
        if self_origin != other_origin:
            raise TypeError(f"TypeRef.update() cannot update origin type! ({self_origin} to {other_origin})")

        self.nominal = other.nominal

    def replace(self, other: 'TypeRef'):

        if not isinstance(other, TypeRef):
            raise TypeError(f"TypeRef.update() called with {type(other)}, not TypeRef!")

        self.nominal = other.nominal

    def __call__(self):
        raise NotImplementedError('TypeRef instances cannot be called!')
    
    def __hash__(self):
        return 42

    def __str__(self):

        # HACK
        if self.nominal is None:
            return "(None as Type?!)"
        
        try:
            name = self.origin.__qualname__
        except AttributeError:
            name = str(self.origin)
        name = name.replace('typing.', '')

        if self.args:
             name += '['
             name += ', '.join([str(arg) for arg in self.args])
             name += ']'

        if not self.DEBUG:
            return name
        else:
            return f"TypeRef({name})"


#######################
# Custom Basic Types
#######################

# NewType = _NewType

class NewType(_NewType):

    def __repr__(self):
        return self.__name__


Undefined = NewType('Undefined', type)

DictOfTypes = Dict[str, TypeRef]
Namespace = Dict[str, DictOfTypes]
ClassAttributes = Dict[Type, Namespace]


class TypeDict(dict, DictOfTypes):

    def union_update(self, other_dict: DictOfTypes):

        for name, type in other_dict.items():
            if name not in self:
                self[name] = type
            else:
                self[name] = Union[self[name], type]
        
    def subset_union(self, other_dict):
        
        for key in self:
            if key in other_dict:
                self[key] = Union[self[key], other_dict[key]]
            else:
                self[key] = Union[self[key], Undefined]
    
    def get_merged(self, other_dict):
        raise NotImplementedError('merging TODO')


#######################
# Internal Types
#######################


@_dataclass
class FunctionReturn:
    """A type to annotate both the return type of a functions as well as any side-effects
    the function has on it's argument types.

    NOTE on side-effect dict: Missing entries are implicitly assumed to be equal to input type.
    effects=None or effects={} imply no side-effects on input types"""

    return_type: TypeRef
    effects: Dict[str, TypeRef] = _field(default_factory=dict)

    def __call__(self):
        raise NotImplementedError('FunctionReturn instances cannot be called!')

    def __repr__(self):
        effects_str = ', '.join([f"'{var}': {type}" for var, type in self.effects.items()])
        return f"FunctionReturn(return={self.return_type}, effects={{{effects_str}}})"

    def __eq__(self, other):
        if not isinstance(other, FunctionReturn):
            return False
        same_return = self.return_type == other.return_type
        same_effects = self.effects == other.effects
        return same_return and same_effects
    
    def __hash__(self):
        return hash(self.return_type)
    
    def __class_getitem__(cls, item):

        if len(item) != 2:
            raise ValueError("FunctionReturn expects 2 arguments: return type and return effects!")
        
        ret_type, ret_effect = item
        return cls(return_type=ret_type, effects=ret_effect)


@_dataclass
class TypedArg(_AST):

    id: str = "__TypedArg_default__"
    type: Type = Undefined
    default: Optional[_AST] = None


class FunctionType(metaclass=_ABCMeta):

    @property
    def all_args(self):
        all_args = (self.posonlyargs + self.args + ([self.vararg] if self.vararg else [])
                    + self.kwonlyargs + ([self.kwarg] if self.kwarg else []))
        return all_args

    @_abstractmethod
    def __init__(self, name: str, posonlyargs: List[TypedArg], args: List[TypedArg],
                 vararg: Optional[TypedArg], kwonlyargs: List[TypedArg],
                 kwarg: Optional[TypedArg], defaults: List[_AST], kw_defaults: List[_AST]):

        self.name = name

        # all the different possible arguments (in order)
        self.posonlyargs = posonlyargs
        self.args = args
        self.vararg = vararg
        self.kwonlyargs = kwonlyargs
        self.kwarg = kwarg

        for arg, default in zip(reversed(posonlyargs + args), reversed(defaults)):
            arg.default = default

        for arg, default in zip(kwonlyargs, kw_defaults):
            arg.default = default

        # check if fully typed
        all_args_typed = all([arg.type.nominal not in {Undefined, Any} for arg in self.all_args])
        self.is_argtyped = all_args_typed

        self.is_typechecked = False

    def __call__(self) -> NoReturn:
        raise NotImplementedError('FunctionType instances cannot be called!')

    def __repr__(self) -> str:

        repr_parts = ['!(' if self.is_typechecked else '*(']

        # posonly
        for arg in self.posonlyargs:
            def_val = f' = {_unparse(arg.default)}' if arg.default is not None else ''
            repr_parts.append(f"{arg.id}: {arg.type}{def_val}, ")
        
        if self.posonlyargs:
            repr_parts.append("/, ")

        # args
        for arg in self.args:
            def_val = f' = {_unparse(arg.default)}' if arg.default is not None else ''
            repr_parts.append(f"{arg.id}: {arg.type}{def_val}, ")

        # *arg
        if self.vararg:
            repr_parts.append(f"*{self.vararg.id}: {self.vararg.type}, ")
        elif self.kwonlyargs:
            repr_parts.append("*, ")
        
        # kwarg
        for arg in self.kwonlyargs:
            def_val = f' = {_unparse(arg.default)}' if arg.default is not None else ''
            repr_parts.append(f"{arg.id}: {arg.type}{def_val}, ")

        # **kwarg
        if self.kwarg:
            repr_parts.append(f"**{self.kwarg.id}: {self.kwarg.type}, ")
        
        if self.returns is None:
            return_str = ') -> ?'
        else:
            return_str = f') -> {self.returns}'
        repr_parts.append(return_str)

        return ''.join(repr_parts)

    def __eq__(self, other: 'FunctionType') -> bool:
        return False
    
    def __hash__(self) -> int:
        return 1

    def get_self_arg(self) -> TypedArg:

        if self.posonlyargs:
            first_arg = self.posonlyargs[0]
        elif self.args:
            first_arg = self.args[0]
        else:
            raise TypeError(f"Function {self.name} does not take implicit 'self' argument")

        return first_arg

    def extract_self_arg(self) -> TypedArg:

        if self.posonlyargs:
            first_arg = self.posonlyargs[0]
            self.posonlyargs = self.posonlyargs[1:]
        elif self.args:
            first_arg = self.args[0]
            self.args = self.args[1:]
        else:
            raise TypeError(f"Function {self.name} does not take implicit 'self' argument")

        return first_arg


class DefiniteFunction(FunctionType):

    def __init__(self, *, name: str, posonlyargs: List[TypedArg], args: List[TypedArg],
                 vararg: Optional[TypedArg], kwonlyargs: List[TypedArg], kwarg: Optional[TypedArg], 
                 defaults: List[_AST], kw_defaults: List[_AST], returns: FunctionReturn):

        FunctionType.__init__(self, name, posonlyargs, args, vararg, kwonlyargs, kwarg, defaults, kw_defaults)

        self.returns = returns

    def get_args_typedict(self) -> TypeDict:

        types = TypeDict()

        for arg in self.posonlyargs:
            types[arg.id] = arg.type if arg.type else Any

        for arg in self.args:
            types[arg.id] = arg.type if arg.type else Any

        if self.vararg:
            types[self.vararg.id] = Tuple[self.vararg.type]
        
        for arg in self.kwonlyargs:
            types[arg.id] = arg.type if arg.type else Any

        if self.kwarg:
            types[self.kwarg.id] = Dict[str, self.kwarg.type]
        
        return types

    def typecheck(self, argument_types: DictOfTypes):
        return self.returns

    def as_bound_method(self, well_typed: bool):

        self_copy = _copy(self)
        first_arg = self_copy.extract_self_arg()

        def typecheck_method(argument_types: DictOfTypes):

            if well_typed:
                return self.returns
            else:
                raise TypeError(f"Function {self.name} called on mismatching 'self' object")
        
        self_copy.typecheck = typecheck_method

        return self_copy


class VariableFunction(FunctionType):

    def __init__(self, *, name: str, posonlyargs: List[TypedArg], args: List[TypedArg], vararg: Optional[TypedArg],
                 kwonlyargs: List[TypedArg], kwarg: Optional[TypedArg], defaults: List[_AST], kw_defaults: List[_AST], 
                 returns: Optional[TypeRef], body: List[_AST]):

        FunctionType.__init__(self, name, posonlyargs, args, vararg, kwonlyargs, kwarg, defaults, kw_defaults)

        self.body = body
        self.returns = returns
        self.typecheck_func = None
    
    def set_typecheck_func(self, func):

        self.typecheck_func = func
    
    def typecheck(self, argument_types: DictOfTypes):

        if self.typecheck_func is None:
            raise AttributeError(f"No typechecking function set for VariableFunction '{self.name}'")

        return self.typecheck_func(argument_types)

    def as_bound_method(self, instance_type):

        self_copy = _copy(self)
        first_arg = self_copy.extract_self_arg()

        def typecheck_method(argument_types: DictOfTypes):

            first_arg_name = first_arg.id

            if first_arg_name in argument_types:
                raise TypeError(f"Method {self.name} of class {instance_type} got two inputs 'self' argument")
            
            all_arg_types = {first_arg_name: instance_type}
            all_arg_types.update(argument_types)

            return self.typecheck(all_arg_types)
        
        self_copy.typecheck = typecheck_method

        return self_copy


#######################
# Helpers for Types
#######################

def get_builtins_types():

    return {
        # 'all': Callable[[Iterable[object]], ReturnEffect[bool, None]],
        # 'any': Callable[[Iterable[object]], ReturnEffect[bool, None]],
        # 'print': Callable[_P, ReturnEffect[None, None]]
    }

def type_to_str(type: Type) -> str:

    # HACK
    if type is None:
        return "None as Type?!"

    if hasattr(type, '__args__'):
        name = repr(type)
    elif isinstance(type, (FunctionType, FunctionReturn)):
        name = repr(type)
    else:
        name = type.__qualname__

    name = name.replace('typing.', '')

    return name
