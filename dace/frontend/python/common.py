# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import collections
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from dace import data
from dace.sdfg.sdfg import SDFG


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

        if self.visitor is not None:
            return (self.message + "\n  File \"" + str(self.visitor.filename) + "\", line " + str(line) + ", column " +
                    str(col))
        else:
            return (self.message + "\n  in line " + str(line) + ":" + str(col))


def inverse_dict_lookup(dict: Dict[str, Any], value: Any):
    """ Finds the first key in a dictionary with the input value. """
    for k, v in dict.items():
        if v == value:
            return k
    return None


@dataclass(unsafe_hash=True)
class StringLiteral:
    """ A string literal found in a parsed DaCe program. """
    value: Union[str, bytes]

    def __str__(self) -> str:
        return self.value


class SDFGConvertible(object):
    """ 
    A mixin that defines the interface to annotate SDFG-convertible objects.
    """

    def __sdfg__(self, *args, **kwargs) -> SDFG:
        """
        Returns an SDFG representation of this object.

        :param args: Arguments or argument types (given as DaCe data 
                     descriptors) that can be used for compilation.
        :param kwargs: Keyword arguments or argument types (given as DaCe data 
                       descriptors) that can be used for compilation.
        :return: A parsed SDFG object representing this object.
        """
        raise NotImplementedError

    def __sdfg_closure__(self, reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """ 
        Returns the closure arrays of the SDFG represented by this object
        as a mapping between array name and the corresponding value.

        :param reevaluate: If given, re-evaluates closure elements based on the
                           input mapping (keys: array names, values: expressions
                           to evaluate). Otherwise, re-evaluates default
                           argument names.
        :return: A dictionary mapping between a name in the closure and the 
                 currently evaluated value.
        """
        raise NotImplementedError

    def __sdfg_signature__(self) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Returns the SDFG signature represented by this object, as a sequence
        of all argument names that will be found in a call to this object
        (i.e., including regular and constant arguments, but excluding "self"
        for bound methods) and a sequence of the constant argument names from
        the first sequence.
        
        :return: A 2-tuple of (all arguments, constant arguments).
        """
        raise NotImplementedError

    def closure_resolver(self,
                         constant_args: Dict[str, Any],
                         given_args: Set[str],
                         parent_closure: Optional['SDFGClosure'] = None) -> 'SDFGClosure':
        """ 
        Returns an SDFGClosure object representing the closure of the
        object to be converted to an SDFG.

        :param constant_args: Arguments whose values are already resolved to
                              compile-time values.
        :param given_args: Arguments that were given at call-time (used for
                           determining which arguments with defaults were provided).
        :param parent_closure: The parent SDFGClosure object (used for, e.g.,
                               recursion detection).
        :return: New SDFG closure object representing the convertible object.
        """
        return SDFGClosure()


@dataclass
class SDFGClosure:
    """
    Represents a reduced closure of a parsed DaCe program.
    A dace.program's closure is composed of its used constants, arrays, and
    other internal SDFG-convertible objects.
    """

    # Constants that are part of the closure (mapping from name to value)
    closure_constants: Dict[str, Any]

    # Mutable arrays that are part of the closure, mapping from data descriptor
    # names to a 4-tuple of (python name, descriptor, callable that returns
    # array, does the array belong to a nested SDFG).
    closure_arrays: Dict[str, Tuple[str, data.Data, Callable[[], Any], bool]]

    # Nested SDFGs and SDFG-convertible objects that are used in the program
    # (mapping from object id to (name, object))
    closure_sdfgs: Dict[int, Tuple[str, Union[SDFG, SDFGConvertible]]]

    # Callbacks to Python callables that are used in the program
    # Mapping from unique names to a 3-tuple of (python name, callable,
    # does the callback belong to a nested SDFG).
    callbacks: Dict[str, Tuple[str, Callable[..., Any], bool]]

    # List of nested SDFG-convertible closure objects and their names
    nested_closures: List[Tuple[str, 'SDFGClosure']]

    # Maps same array objects (checked via python id) to the same name
    array_mapping: Dict[int, str]

    # Trace of called functions (as their `id`) until this point
    callstack: List[int]

    def __init__(self):
        self.closure_constants = {}
        self.closure_arrays = {}
        self.closure_sdfgs = collections.OrderedDict()
        self.callbacks = collections.OrderedDict()
        self.nested_closures = []
        self.array_mapping = {}
        self.callstack = []

    def print_call_tree(self, name, indent=0):
        print('  ' * indent + name)
        for cname, child in self.nested_closures:
            child.print_call_tree(cname, indent + 1)

    def call_tree_length(self) -> int:
        value = 1
        for _, child in self.nested_closures:
            value += child.call_tree_length()
        return value

    def combine_nested_closures(self):
        # Remove previous nested closures if there are any
        # self.closure_arrays = {
        #     k: v
        #     for k, v in self.closure_arrays.items() if v[3] is False
        # }

        for _, child in self.nested_closures:
            for arrname, (_, desc, evaluator, _) in sorted(child.closure_arrays.items()):

                # Check if the same array is already passed as part of a
                # nested closure
                arr = evaluator()
                if id(arr) in self.array_mapping:
                    continue

                new_name = data.find_new_name(arrname, self.closure_arrays.keys())
                if not desc.transient:
                    self.closure_arrays[new_name] = (arrname, desc, evaluator, True)
                    self.array_mapping[id(arr)] = new_name

            for cbname, (_, cb, _) in sorted(child.callbacks.items()):
                new_name = data.find_new_name(cbname, self.callbacks.keys())
                self.callbacks[new_name] = (cbname, cb, True)
                self.array_mapping[id(cb)] = new_name
