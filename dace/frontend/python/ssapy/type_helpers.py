
import typing
from collections.abc import Generator

from typing import get_args, get_origin

from .type_types import Any, Union, Dict, Type, Tuple, List, Undefined, TypeRef


class TypeInferenceError(TypeError):
    ...


types_to_bases: Dict[Type, Type] = {
    type: Type,
    list: List,
    dict: Dict,
    tuple: Tuple,
    Generator: typing.Generator
}


def tuple_is_infinite(tuple_type):
    args = get_args(tuple_type)
    return (len(args) == 2) and (args[1] == ...)


def is_definite_type(type):

    args = get_args(type)

    if type in (Any, Undefined):
        return False
    
    definite_args = [is_definite_type(arg) for arg in args]
    return all(definite_args)


def type_of_type(base_type):
    return TypeRef(Type[TypeRef(base_type)])


def get_origin_type(base_type: Type) -> Type:

    if get_args(base_type):
        origin = get_origin(base_type)
        return types_to_bases.get(origin, origin)
    else:
        return base_type


def add_type(base_type: Type, *contents):
    old_contents = get_args(base_type)
    origin = get_origin_type(base_type)
    if len(old_contents) == 0:
        return origin[old_contents]
    tup = tuple(Union[old, new] for old, new in zip(old_contents, contents))
    return origin[tup]


class EnclosingFunction:

    def __init__(self, ast_visitor, func_node, update_assigns=True):

        self.ast_visitor = ast_visitor
        self.func_node = func_node
        self.update_assigns = update_assigns
    
    def __enter__(self):

        ast_visitor = self.ast_visitor
        func_node = self.func_node

        # save old state
        self.prev_func = getattr(ast_visitor, 'current_func', None)
        self.prev_returns = getattr(ast_visitor, 'return_types', False)
        self.prev_update = getattr(ast_visitor, '_update_assigns')

        # create new state
        ast_visitor.current_func = func_node
        ast_visitor.return_types = []
        ast_visitor._update_assigns = self.update_assigns
    
    def __exit__(self, exc_type,exc_value, exc_traceback):

        # restore old state
        self.ast_visitor.current_func = self.prev_func
        self.ast_visitor.return_types = self.prev_returns
        self.ast_visitor._update_assigns = self.prev_update


# def transpose(list_of_lists, *, size=2):
#     if len(list_of_lists):
#         return list(map(list, zip(*list_of_lists)))
#     else:
#         return [[] for _ in range(size)]

