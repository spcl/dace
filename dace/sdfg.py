import astunparse
import collections
import copy
import errno
import itertools
from inspect import getframeinfo, stack
import os
import pickle, json
from pydoc import locate
import random
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
import warnings
import numpy as np
import sympy as sp

import dace
import dace.serialize
from dace import (data as dt, memlet as mm, subsets as sbs, dtypes, properties,
                  symbolic)
from dace.config import Config
from dace.frontend.python import wrappers
from dace.frontend.python.astutils import ASTFindReplace
from dace.graph import edges as ed, nodes as nd, labeling
from dace.graph.labeling import propagate_memlet, propagate_labels_sdfg
from dace.data import validate_name
from dace.graph import dot
from dace.graph.graph import (OrderedDiGraph, OrderedMultiDiConnectorGraph,
                              SubgraphView, Edge, MultiConnectorEdge)
from dace.properties import (make_properties, Property, CodeProperty,
                             OrderedDictProperty)


def getcaller() -> Tuple[str, int]:
    """ Returns the file and line of the function that called the current
        function (the one that calls getcaller()).
        :return: 2-tuple of file and line.
    """
    caller = getframeinfo(stack()[2][0])
    return (caller.filename, caller.lineno)


def getdebuginfo(old_dinfo=None) -> dtypes.DebugInfo:
    """ Returns a DebugInfo object for the position that called this function.
        :param old_dinfo: Another DebugInfo object that will override the
                          return value of this function
        :return: DebugInfo containing line number and calling file.
    """
    if old_dinfo is not None:
        return old_dinfo

    caller = getframeinfo(stack()[2][0])
    return dtypes.DebugInfo(caller.lineno, 0, caller.lineno, 0,
                            caller.filename)


class Scope(object):
    """ A class defining a scope, its parent and children scopes, variables, and
        scope entry/exit nodes. """

    def __init__(self, entrynode, exitnode):
        self.parent = None
        self.children = []
        self.defined_vars = []
        self.entry = entrynode
        self.exit = exitnode


class InvalidSDFGError(Exception):
    """ A class of exceptions thrown when SDFG validation fails. """

    def __init__(self, message: str, sdfg, state_id):
        self.message = message
        self.sdfg = sdfg
        self.state_id = state_id

    def __str__(self):
        if self.state_id is not None:
            state = self.sdfg.nodes()[self.state_id]
            return "%s (at state %s)" % (self.message, str(state.label))
        else:
            return "%s" % self.message


class InvalidSDFGInterstateEdgeError(InvalidSDFGError):
    """ Exceptions of invalid inter-state edges in an SDFG. """

    def __init__(self, message: str, sdfg, edge_id):
        self.message = message
        self.sdfg = sdfg
        self.edge_id = edge_id

    def __str__(self):
        if self.edge_id is not None:
            e = self.sdfg.edges()[self.edge_id]
            edgestr = ' (at edge "%s" (%s -> %s)' % (
                e.data.label,
                str(e.src),
                str(e.dst),
            )
        else:
            edgestr = ""

        return "%s%s" % (self.message, edgestr)


class InvalidSDFGNodeError(InvalidSDFGError):
    """ Exceptions of invalid nodes in an SDFG state. """

    def __init__(self, message: str, sdfg, state_id, node_id):
        self.message = message
        self.sdfg = sdfg
        self.state_id = state_id
        self.node_id = node_id

    def __str__(self):
        state = self.sdfg.nodes()[self.state_id]

        if self.node_id is not None:
            node = state.nodes()[self.node_id]
            nodestr = ", node %s" % str(node)
        else:
            nodestr = ""

        return "%s (at state %s%s)" % (self.message, str(state.label), nodestr)


class InvalidSDFGEdgeError(InvalidSDFGError):
    """ Exceptions of invalid edges in an SDFG state. """

    def __init__(self, message: str, sdfg, state_id, edge_id):
        self.message = message
        self.sdfg = sdfg
        self.state_id = state_id
        self.edge_id = edge_id

    def __str__(self):
        state = self.sdfg.nodes()[self.state_id]

        if self.edge_id is not None:
            e = state.edges()[self.edge_id]
            edgestr = ", edge %s (%s:%s -> %s:%s)" % (
                str(e.data),
                str(e.src),
                e.src_conn,
                str(e.dst),
                e.dst_conn,
            )
        else:
            edgestr = ""

        return "%s (at state %s%s)" % (self.message, str(state.label), edgestr)


def _arrays_to_json(arrays):
    if arrays is None:
        return None
    return {k: dace.serialize.to_json(v) for k, v in arrays.items()}


def _arrays_from_json(obj, context=None):
    if obj is None:
        return {}
    return {k: dace.serialize.from_json(v, context) for k, v in obj.items()}


@make_properties
class SDFG(OrderedDiGraph):
    """ The main intermediate representation of code in DaCe.

        A Stateful DataFlow multiGraph (SDFG) is a directed graph of directed
        acyclic multigraphs (i.e., where two nodes can be connected by more
        than one edge). The top-level directed graph represents a state
        machine, where edges can contain state transition conditions and
        assignments (see the `InterstateEdge` class documentation). The nested
        acyclic multigraphs represent dataflow, where nodes may represent data
        regions in memory, tasklets, or parametric graph scopes (see
        `dace.graph.nodes` for a full list of available node types); edges in
        the multigraph represent data movement using memlets, as described in
        the `Memlet` class documentation.
    """

    #arg_types = Property(dtype=dict, default={}, desc="Formal parameter list")
    arg_types = OrderedDictProperty(default={}, desc="Formal parameter list")
    constants_prop = Property(
        dtype=dict, default={}, desc="Compile-time constants")
    _arrays = Property(
        dtype=dict,
        desc="Data descriptors for this SDFG",
        to_json=_arrays_to_json,
        from_json=_arrays_from_json)

    global_code = CodeProperty(
        desc=
        "Code generated in a global scope on the frame-code generated file.",
        default="")
    init_code = CodeProperty(
        desc="Code generated in the `__dapp_init` function.", default="")
    exit_code = CodeProperty(
        desc="Code generated in the `__dapp_exit` function.", default="")

    def __init__(self,
                 name: str,
                 arg_types: Dict[str, dt.Data] = None,
                 constants: Dict[str, Tuple[dt.Data, Any]] = None,
                 propagate: bool = True,
                 parent=None):
        """ Constructs a new SDFG.
            :param name: Name for the SDFG (also used as the filename for
                         the compiled shared library).
            :param symbols: Additional dictionary of symbol names -> types that the SDFG
                            defines, apart from symbolic data sizes.
            :param propagate: If False, disables automatic propagation of
                              memlet subsets from scopes outwards. Saves
                              processing time but disallows certain
                              transformations.
            :param parent: The parent SDFG or SDFG state (for nested SDFGs).
        """
        super(SDFG, self).__init__()
        self._name = name
        if name is not None and not validate_name(name):
            raise InvalidSDFGError('Invalid SDFG name "%s"' % name, self, None)

        self.arg_types = arg_types or collections.OrderedDict()
        self.constants_prop = {}
        if constants is not None:
            for cstname, (cst_dtype, cstval) in constants.items():
                self.add_constant(cstname, cstval, cst_dtype)

        self._propagate = propagate
        self._parent = parent
        self._symbols = {}  # type: Dict[str, dtypes.typeclass]
        self._parent_sdfg = None
        self._sdfg_list = [self]
        self._instrumented_parent = (
            False
        )  # Same as above. This flag is needed to know if the parent is instrumented (it's possible for a parent to be serial and instrumented.)
        self._start_state = None
        self._arrays = {}  # type: Dict[str, dt.Array]
        self.global_code = ''
        self.init_code = ''
        self.exit_code = ''

        # Counter to make it easy to create temp transients
        self._temp_transients = 0

        # Counter to resolve name conflicts
        self._orig_name = name
        self._num = 0

    def to_json(self):
        """ Serializes this object to JSON format.
            :return: A string representing the JSON-serialized SDFG.
        """
        tmp = super().to_json()

        # Inject the undefined symbols
        tmp['undefined_symbols'] = [
            (k, v.to_json())
            for k, v in sorted(self.undefined_symbols(True).items())
        ]
        tmp['scalar_parameters'] = [(k, v.to_json()) for k, v in sorted(
            self.scalar_parameters(True), key=lambda x: x[0])]

        tmp['attributes']['name'] = self.name

        return tmp

    @classmethod
    def from_json(cls, json_obj, context_info={'sdfg': None}):

        _type = json_obj['type']
        if _type != cls.__name__:
            raise TypeError("Class type mismatch")

        attrs = json_obj['attributes']
        nodes = json_obj['nodes']
        edges = json_obj['edges']

        ret = SDFG(
            name=attrs['name'],
            arg_types=dace.serialize.loads(
                dace.serialize.dumps(attrs['arg_types'])),
            constants=dace.serialize.loads(
                dace.serialize.dumps(attrs['constants_prop'])),
            parent=context_info['sdfg'])

        dace.serialize.set_properties_from_json(
            ret, json_obj, ignore_properties={'constants_prop'})

        for n in nodes:
            nci = copy.deepcopy(context_info)
            nci['sdfg'] = ret

            state = SDFGState.from_json(n, context=nci)
            ret.add_node(state)

        for e in edges:
            e = dace.serialize.loads(dace.serialize.dumps(e))
            ret.add_edge(ret.node(int(e.src)), ret.node(int(e.dst)), e.data)

        # Redefine symbols
        for k, v in json_obj['undefined_symbols']:
            v = dace.serialize.from_json(v)
            symbolic.symbol(k, v.dtype, override_dtype=True)

        for k, v in json_obj['scalar_parameters']:
            v = dace.serialize.from_json(v)
            ret.add_symbol(k, v.dtype, override_dtype=True)

        ret.validate()

        return ret

        # Counter to make it easy to create temp transients
        self._temp_transients = 0

    @property
    def arrays(self):
        """ Returns a dictionary of data descriptors (`Data` objects) used
            in this SDFG, with an extra `None` entry for empty memlets.
        """
        return self._arrays

    @property
    def symbols(self):
        """ Returns a dictionary of symbols (constant variables) used in this
            SDFG. """
        return self._symbols

    def data(self, dataname: str):
        """ Looks up a data descriptor from its name, which can be an array, stream, or scalar symbol. """
        if dataname in self._arrays:
            return self._arrays[dataname]
        if dataname in self._symbols:
            return self._symbols[dataname]
        raise KeyError(
            'Data descriptor with name "%s" not found in SDFG' % dataname)

    def replace(self, name: str, new_name: str):
        """ Finds and replaces all occurrences of a symbol or array name in SDFG.
            :param name: Name to find.
            :param new_name: Name to replace.
            :raise FileExistsError: If name and new_name already exist as data descriptors or symbols.
        """

        def replace_dict(d, old, new):
            if old in d:
                if new in d:
                    raise FileExistsError('"%s" already exists in SDFG' % new)
                d[new] = d[old]
                del d[old]

        if name == new_name:
            return

        # Replace in arrays and symbols
        replace_dict(self._arrays, name, new_name)
        replace_dict(self._symbols, name, new_name)

        # Replace in inter-state edges
        for edge in self.edges():
            replace_dict(edge.data.assignments, name, new_name)
            for k, v in edge.data.assignments.items():
                edge.data.assignments[k] = v.replace(name, new_name)
            condition = CodeProperty.to_string(edge.data.condition)
            edge.data.condition = condition.replace(name, new_name)
            # for k, v in edge.data.condition.items():
            #     edge.data.condition[k] = v.replace(name, new_name)

        # Replace in states
        for state in self.nodes():
            state.replace(name, new_name)

    def add_symbol(self, name, stype, override_dtype=False):
        """ Adds a symbol to the SDFG.
            :param name: Symbol name.
            :param stype: Symbol type.
            :param override_dtype: If True, overrides existing symbol type in
                                   symbol registry.
        """
        if name in self._symbols:
            raise FileExistsError('Symbol "%s" already exists in SDFG' % name)
        if not isinstance(stype, dtypes.typeclass):
            stype = dtypes.DTYPE_TO_TYPECLASS[stype]

        symbolic.symbol(name, stype, override_dtype=override_dtype)
        self._symbols[name] = stype

    @property
    def start_state(self):
        """ Returns the starting state of this SDFG. """
        source_nodes = self.source_nodes()
        if len(source_nodes) == 1:
            return source_nodes[0]
        if self._start_state is None:
            raise ValueError('Ambiguous or undefined starting state for SDFG')

        return self.node(self._start_state)

    def set_start_state(self, state_id):
        """ Manually sets the starting state of this SDFG.
            :param state_id: The node ID (use `node_id(state)`) of the
                             state to set.
        """
        if state_id < 0 or state_id >= len(self.nodes()):
            raise ValueError("Invalid state ID")
        self._start_state = state_id

    def set_global_code(self, cpp_code: str):
        """ Sets C++ code that will be generated in a global scope on the frame-code generated file. """
        self.global_code = {
            'code_or_block': cpp_code,
            'language': dace.dtypes.Language.CPP
        }

    def set_init_code(self, cpp_code: str):
        """ Sets C++ code, generated in the `__dapp_init` function. """
        self.init_code = {
            'code_or_block': cpp_code,
            'language': dace.dtypes.Language.CPP
        }

    def set_exit_code(self, cpp_code: str):
        """ Sets C++ code, generated in the `__dapp_exit` function. """
        self.exit_code = {
            'code_or_block': cpp_code,
            'language': dace.dtypes.Language.CPP
        }

    def has_instrumented_parent(self):
        return self._instrumented_parent

    def set_instrumented_parent(self):
        self._instrumented_parent = (
            True
        )  # When this is set: Under no circumstances try instrumenting this (or any transitive children)

    def remove_data(self, name, validate=True):
        """ Removes a data descriptor from the SDFG.
            :param name: The name of the data descriptor to remove.
            :param validate: If True, verifies that there are no access
                             nodes that are using this data descriptor
                             prior to removing it.
        """
        # Verify first that there are no access nodes that use this data
        if validate:
            for state in self.nodes():
                for node in state.nodes():
                    if isinstance(node, nd.AccessNode) and nd.data == name:
                        raise ValueError(
                            "Data descriptor %s is already used"
                            "in node %s, state %s" % (name, node, state))

        del self._arrays[name]

    def update_sdfg_list(self, sdfg_list):
        # TODO: Refactor
        sub_sdfg_list = self._sdfg_list
        for sdfg in sdfg_list:
            if sdfg not in sub_sdfg_list:
                sub_sdfg_list.append(sdfg)
        if self._parent_sdfg is not None:
            self._parent_sdfg.update_sdfg_list(sub_sdfg_list)
            self._sdfg_list = self._parent_sdfg.sdfg_list
            for sdfg in sub_sdfg_list:
                sdfg._sdfg_list = self._sdfg_list
        else:
            self._sdfg_list = sub_sdfg_list

    @property
    def sdfg_list(self):
        return self._sdfg_list

    def set_sourcecode(self, code, lang=None):
        """ Set the source code of this SDFG (for IDE purposes).
            :param code: A string of source code.
            :param lang: A string representing the language of the source code,
                         for syntax highlighting and completion.
        """
        self.sourcecode = {'code_or_block': code, 'language': lang}

    @property
    def name(self):
        """ The name of this SDFG. """
        if self._name != self._orig_name:
            return self._name
        newname = self._orig_name
        numbers = []
        for sdfg in self._sdfg_list:
            if sdfg is not self and sdfg._orig_name == self._orig_name:
                numbers.append(sdfg._num)
        while self._num in numbers:
            self._num += 1
        if self._num > 0:
            newname = '{}_{}'.format(self._orig_name, self._num)
            self._name = newname
        return newname

    @property
    def label(self):
        """ The name of this SDFG. """
        #return self._name
        return self.name

    @property
    def constants(self):
        """ A dictionary of compile-time constants defined in this SDFG. """
        result = {}
        # Merge with parent's constants
        if self._parent_sdfg is not None:
            result.update(self._parent_sdfg.constants)

        def cast(dtype: dt.Data, value: Any):
            """ Cast a value to the given data type. """
            if isinstance(dtype, dt.Array):
                return value
            elif isinstance(dtype, dt.Scalar):
                return dtype.dtype(value)
            raise TypeError('Unsupported data type %s' % dtype)

        result.update({k: cast(*v) for k, v in self.constants_prop.items()})
        return result

    def add_constant(self, name: str, value: Any, dtype: dt.Data = None):
        """ Adds/updates a new compile-time constant to this SDFG. A constant
            may either be a scalar or a numpy ndarray thereof.
            :param name: The name of the constant.
            :param value: The constant value.
            :param dtype: Optional data type of the symbol, or None to deduce
                          automatically.
        """

        def get_type(obj):
            if isinstance(obj, np.ndarray):
                return dt.Array(
                    dtypes.DTYPE_TO_TYPECLASS[obj.dtype.type], shape=obj.shape)
            elif isinstance(obj, dtypes.typeclass):
                return dt.Scalar(type(obj))
            elif type(obj) in dtypes.DTYPE_TO_TYPECLASS:
                return dt.Scalar(dtypes.DTYPE_TO_TYPECLASS[type(obj)])
            raise TypeError('Unrecognized constant type: %s' % type(obj))

        self.constants_prop[name] = (dtype or get_type(value), value)

    @property
    def propagate(self):
        return self._propagate

    @propagate.setter
    def propagate(self, propagate: bool):
        self._propagate = propagate

    @property
    def parent(self):
        """ Returns the parent SDFG state of this SDFG, if exists. """
        return self._parent

    @property
    def parent_sdfg(self):
        """ Returns the parent SDFG of this SDFG, if exists. """
        return self._parent_sdfg

    @parent.setter
    def parent(self, value):
        self._parent = value

    @parent_sdfg.setter
    def parent_sdfg(self, value):
        self._parent_sdfg = value

    def add_node(self, node, is_start_state=False):
        """ Adds a new node to the SDFG. Must be an SDFGState or a subclass
            thereof.
            :param node: The node to add.
            :param is_start_state: If True, sets this node as the starting
                                   state.
        """
        if not isinstance(node, SDFGState):
            raise TypeError("Expected SDFGState, got " + str(type(node)))

        # If no start state has been defined, define to be the first state
        if is_start_state == True:
            self._start_state = len(self.nodes())

        return super(SDFG, self).add_node(node)

    def add_edge(self, u, v, edge):
        """ Adds a new edge to the SDFG. Must be an InterstateEdge or a
            subclass thereof.
            :param u: Source node.
            :param v: Destination node.
            :param edge: The edge to add.
        """
        if not isinstance(u, SDFGState):
            raise TypeError("Expected SDFGState, got: {}".format(
                type(u).__name__))
        if not isinstance(v, SDFGState):
            raise TypeError("Expected SDFGState, got: {}".format(
                type(v).__name__))
        if not isinstance(edge, ed.InterstateEdge):
            raise TypeError("Expected InterstateEdge, got: {}".format(
                type(edge).__name__))
        return super(SDFG, self).add_edge(u, v, edge)

    def states(self):
        """ Alias that returns the nodes (states) in this SDFG. """
        return self.nodes()

    def all_nodes_recursive(self):
        """ Iterate over all nodes in this SDFG, including states, nodes in
            states, and recursive states and nodes within nested SDFGs,
            returning tuples on the form (node, parent), where the parent is
            either the SDFG (for states) or a DFG (nodes). """
        all_nodes = []
        for node in self.nodes():
            all_nodes.append((node, self))
            all_nodes += node.all_nodes_recursive()
        return all_nodes

    def all_edges_recursive(self):
        """ Iterate over all edges in this SDFG, including state edges,
            inter-state edges, and recursively edges within nested SDFGs,
            returning tuples on the form (edge, parent), where the parent is
            either the SDFG (for states) or a DFG (nodes). """
        all_edges = [(e, self) for e in self.edges()]
        for node in self.nodes():
            all_edges += node.all_edges_recursive()
        return all_edges

    def arrays_recursive(self):
        """ Iterate over all arrays in this SDFG, including arrays within
            nested SDFGs. Yields 3-tuples of (sdfg, array name, array)."""
        for aname, arr in self.arrays.items():
            yield self, aname, arr
        for state in self.nodes():
            for node in state.nodes():
                if isinstance(node, nd.NestedSDFG):
                    yield from node.sdfg.arrays_recursive()

    def interstate_symbols(self):
        """ Returns variables are assigned/used in the top-level and can be
            shared between states.
        """

        assigned = collections.OrderedDict()
        used = collections.OrderedDict()

        # Find symbols in inter-state edges
        for _, _, edge_data in self.edges():
            for var, expr in edge_data.assignments.items():
                assigned[var] = dt.Scalar(symbolic.symtype(expr))
                if isinstance(expr, str):
                    expr = symbolic.pystr_to_symbolic(expr, simplify=False)
                if isinstance(expr, sp.Expr):
                    for s in dace.symbolic.symbols_in_sympy_expr(expr):
                        used[s] = dt.Scalar(symbolic.symbol(s).dtype)
                elif expr is None or isinstance(expr, int):
                    pass  # Nothing to extract, or a constant
                else:
                    raise TypeError("Unexpected type: {}".format(type(expr)))
            for s in edge_data.condition_symbols():
                used[s] = dt.Scalar(symbolic.symbol(s).dtype)
        for state in self.nodes():
            a, u = state.interstate_symbols()
            assigned.update(a)
            used.update(u)

        assigned = collections.OrderedDict([(k, v)
                                            for k, v in assigned.items()
                                            if not k.startswith('__dace')])
        used = collections.OrderedDict(
            [(k, v) for k, v in used.items() if not k.startswith('__dace')])

        return assigned, used

    def scalar_parameters(self, include_constants):
        """ Returns all scalar data arguments to the SDFG (this excludes
            symbols used to define array sizes)."""
        return [
            (name, dt.Scalar(stype)) for name, stype in self._symbols.items()
            # Exclude constant variables if requested
            if (include_constants or (name not in self.constants))
        ]

    def symbols_defined_at(self, node, state=None):
        """ Returns all symbols available to a given node, including only
            scope-defined variables that encompass the node, assuming that all
            required inputs to the SDFG have been resolved. """
        if node is None:
            return collections.OrderedDict()

        # From e.g., Data or SDFG to the corresponding node
        resolved = self.resolve_node(node)
        if len(resolved) > 1:
            raise ValueError("Node {} is present multiple times in SDFG: "
                             "result is ambiguous".format(node))
        node = resolved[0]

        if state is None:
            state = self.states_for_node(node)
            if len(state) > 1:
                raise ValueError('Node "{}" is present in multiple states, '
                                 "result is ambiguous: {}".format(
                                     node, ", ".join(state)))
            state = state[0]
        else:
            if node not in state.nodes():
                raise ValueError(
                    'Node "{}" does not exist in state "{}"'.format(
                        node, state))

        # All scalar inputs, data symbols and interstate symbols are assumed to
        # have been resolved at this point
        symbols = collections.OrderedDict(
            (name, data) for name, data in self.scalar_parameters(True))
        symbols.update(self.data_symbols(True))
        assigned, used = self.interstate_symbols()
        symbols.update(assigned)
        #symbols.update(used)

        # Explore scope of node to find iteration variables
        scope_dict = state.scope_dict()
        if isinstance(node, dace.graph.nodes.EntryNode):
            scope = node
        else:
            scope = scope_dict[node]
        while scope is not None:
            if isinstance(scope, dace.graph.nodes.MapEntry):
                for param in scope.params:
                    symbols[param] = dt.Scalar(symbolic.symbol(param).dtype)
                for sym in scope.range.free_symbols:
                    symbols[sym] = dt.Scalar(symbolic.symbol(sym).dtype)
            elif isinstance(scope, dace.graph.nodes.ConsumeEntry):
                symbols[scope.consume.pe_index] = dt.Scalar(
                    symbolic.symbol(scope.consume.pe_index).dtype)
                for sym in scope.consume.num_pes.free_symbols:
                    symbols[sym] = dt.Scalar(symbolic.symbol(sym).dtype)
            else:
                raise TypeError("Unsupported entry node type: {}".format(
                    type(scope).__name__))
            scope = scope_dict[scope]

        # Call recursively on parents
        if self.parent is not None:
            # Find parent Nested SDFG node
            parent_node = next(
                n for n in self.parent.nodes()
                if isinstance(n, nd.NestedSDFG) and n.sdfg.name == self.name)
            symbols.update(
                self._parent_sdfg.symbols_defined_at(parent_node, self.parent))

        symbols.update(self.constants)

        return symbols

    def data_symbols(self, include_constants):
        """ Returns all symbols used in data nodes within the SDFG. """
        symbols = collections.OrderedDict()
        for state in self.nodes():
            symbols.update(state.data_symbols())
        if include_constants:
            return symbols
        else:
            return collections.OrderedDict((key, val)
                                           for key, val in symbols.items()
                                           if key not in self.constants)

    def scope_symbols(self):
        """ Returns all symbols used in scopes (maps) within the SDFG. """
        iteration_variables = collections.OrderedDict()
        subset_symbols = collections.OrderedDict()
        for state in self.nodes():
            iv, ss = state.scope_symbols()
            iteration_variables.update(iv)
            subset_symbols.update(ss)
        return iteration_variables, subset_symbols

    def all_symbols(self, include_constants):
        """ Returns all symbols used in this SDFG, including scalar parameters
            to the SDFG, loop iteration variables, array sizes and variables
            used in interstate edges. """
        symbols = collections.OrderedDict(
            (name, data)
            for name, data in self.scalar_parameters(include_constants))
        symbols.update(self.data_symbols(True))
        assigned, used = self.interstate_symbols()
        symbols.update(used)
        iteration_variables, subset_symbols = self.scope_symbols()
        symbols.update(subset_symbols)
        symbols.update(iteration_variables)
        if include_constants:
            return symbols
        else:
            return collections.OrderedDict((key, val)
                                           for key, val in symbols.items()
                                           if key not in self.constants)

    def undefined_symbols(self, include_scalar_data):
        """ Returns all symbols used in this SDFG that are undefined, and thus
            must be given as input parameters. """
        return undefined_symbols(self, self, include_scalar_data)

    def resolve_node(self, node):
        """ Resolves data objects and SDFG objects into their corresponding
            nodes in the SDFG. """
        if isinstance(node, dace.graph.nodes.Node):
            return [node]
        all_nodes = [(self, None)] + self.all_nodes_recursive()
        if isinstance(node, dace.data.Data):
            resolved = [
                n for n, _ in all_nodes
                if isinstance(n, dace.graph.nodes.AccessNode)
                and n.desc(self) == node
            ]
        elif isinstance(node, SDFG):
            resolved = [
                n for n, _ in all_nodes if
                isinstance(n, dace.graph.nodes.NestedSDFG) and n.sdfg == node
            ]
        else:
            raise TypeError("Unrecognized type {} passed.".format(
                type(node).__name__))
        if len(resolved) == 0:
            raise RuntimeError("Node {} of type {} not found "
                               "in SDFG {}.".format(node.data,
                                                    type(node).__name__,
                                                    self.name))
        return resolved

    def states_for_node(self, node):
        """ Finds which states a node is located in. """
        if isinstance(node, dace.data.Data):
            states = [
                s for s in self.nodes()
                if node in [n.data for n in s.data_nodes()]
            ]
        elif isinstance(node, SDFG):
            states = [
                s for s in self.nodes() if node in [
                    n.sdfg for n in s.nodes()
                    if isinstance(n, dace.graph.nodes.NestedSDFG)
                ]
            ]
        else:
            states = [s for s in self.nodes() if node in s.nodes()]
        if len(states) == 0:
            raise ValueError('Node "{}" not found'.format(node))
        return states

    def arglist(self):
        """ Returns a list of argument names required to call this SDFG.
            The return type is a dictionary of names to dtypes. """
        data_args = []
        for state in self.nodes():
            data_args += [
                (n.data, n.desc(self)) for n in state.nodes()
                if isinstance(n, nd.AccessNode) and not n.desc(self).transient
            ]
        data_args = sorted(dtypes.deduplicate(data_args))

        sym_args = sorted(self.undefined_symbols(True).items())

        # Arguments are sorted as follows:
        # 1. Program arguments, as given in the dace program definition
        # 2. Other free symbols, sorted by name
        # 3. Data arguments inferred from the SDFG, if not given in the program
        #    definition (or if not created from a dace.program)
        arg_list = collections.OrderedDict()
        for key, val in itertools.chain(data_args, sym_args):
            if key not in self.constants_prop and key not in arg_list:
                arg_list[key] = val

        return arg_list

    def signature_arglist(self, with_types=True, for_call=False):
        """ Returns a list of arguments necessary to call this SDFG,
            formatted as a list of C definitions.
            :param with_types: If True, includes argument types in the result.
            :param for_call: If True, returns arguments that can be used when
                             calling the SDFG. This means that immaterial data
                             will generate "nullptr" arguments instead of the
                             argument names.
            :return: A list of strings. For example: `['float *A', 'int b']`.
        """
        arg_list = self.arglist()

        signature_args = []
        for name, arg_type in arg_list.items():
            if isinstance(arg_type, dace.data.Data):
                signature_args.append(
                    arg_type.signature(
                        name=name, with_types=with_types, for_call=for_call))
            else:
                raise TypeError("Unsupported argument type")

        return signature_args

    def signature(self, with_types=True, for_call=False):
        """ Returns a C/C++ signature of this SDFG, used when generating code.
            :param with_types: If True, includes argument types (can be used
                               for a function prototype). If False, only
                               include argument names (can be used for function
                               calls).
            :param for_call: If True, returns arguments that can be used when
                             calling the SDFG. This means that immaterial data
                             will generate "nullptr" arguments instead of the
                             argument names.
        """
        return ", ".join(self.signature_arglist(with_types, for_call))

    def draw_to_file(self,
                     filename="sdfg.dot",
                     fill_connectors=True,
                     recursive=True):
        """ Draws the SDFG to a GraphViz (.dot) file.
            :param filename: The file to draw the SDFG to (will be written to
                             '_dotgraphs/<filename>').
            :param fill_connectors: Whether to fill missing scope (e.g., "IN_")
                                    connectors prior to drawing the graph.
            :param recursive: If True, also draws nested SDFGs.
        """
        if fill_connectors:
            self.fill_scope_connectors()

        try:
            os.makedirs("_dotgraphs")
        # Python 2.x does not have FileExistsError
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        with open(os.path.join("_dotgraphs", filename), "w") as outFile:
            outFile.write(self.draw())

        if recursive:
            for state in self.nodes():
                for node in state.nodes():
                    if isinstance(node, dace.graph.nodes.NestedSDFG):
                        node.sdfg.draw_to_file(
                            filename=node.sdfg.name + "_" + filename,
                            recursive=True)

    def draw(self):
        """ Creates a GraphViz representation of the full SDFG, including all
            states and transitions.
            :return: A string representing the SDFG in .dot format.
        """

        nodes = []

        # Redirect all edges between states to point at the boundaries
        edges = []
        for ind, edge in enumerate(self.edges()):
            srcState, dstState, data = edge
            srcDotName = "state_" + str(self.node_id(srcState))
            dstDotName = "state_" + str(self.node_id(dstState))
            srcCluster = "cluster_" + srcDotName
            dstCluster = "cluster_" + dstDotName

            if len(srcState.nodes()) > 0:
                srcNode = srcState.sink_nodes()[0]
                srcName = "s%d_%d" % (self.node_id(srcState),
                                      srcState.node_id(srcNode))
            else:
                srcName = "dummy_" + str(self.node_id(srcState))
            if len(dstState.nodes()) > 0:
                dstNode = dstState.source_nodes()[0]
                dstName = "s%d_%d" % (self.node_id(dstState),
                                      dstState.node_id(dstNode))
            else:
                dstName = "dummy_" + str(self.node_id(dstState))

            if srcState != dstState:
                edges.append(
                    dot.draw_interstate_edge_by_name(
                        srcName,
                        dstName,
                        edge,
                        self,
                        srcState,
                        ltail=srcCluster,
                        lhead=dstCluster,
                    ))
            else:
                redName = srcDotName + "_to_" + dstDotName
                nodes.append(dot.draw_invisible_node(redName))

                edges.append(
                    dot.draw_edge_explicit(
                        srcName,
                        redName,
                        Edge(srcState, srcState, ed.RedirectEdge()),
                        self,
                        srcState,
                        ltail=srcCluster,
                    ))
                edges.append(
                    dot.draw_edge_explicit(
                        redName,
                        dstName,
                        edge,
                        self,
                        srcState,
                        lhead=dstCluster))

        # Mark first and last states
        first = self.start_state

        # A state is considered a last state if it has no outgoing edges that
        # lead to another state
        last = self.sink_nodes()

        clusters = []
        for state in self.nodes():
            if state == first and state not in last:
                clusterLabel = state.label + " (BEGIN)"
                clusterColor = "#f7dede"
            elif state in last and state != first:
                clusterLabel = state.label + " (END)"
                clusterColor = "#f7dede"
            else:
                clusterLabel = state.label
                clusterColor = "#deebf7"
            cluster = """
subgraph cluster_state_{state} {{
      label = "{label}";
      labeljust = r;
      bgcolor = "{color}"; color = "{color}";""".format(
                state=self.node_id(state),
                label=clusterLabel,
                color=clusterColor)
            subNodes, subEdges = dot.draw_graph(self, state, standalone=False)
            cluster += "\n        ".join(subNodes + subEdges)
            if len(subNodes) == 0:
                cluster += "\n"
                cluster += dot.draw_invisible_node("dummy_" +
                                                   str(self.node_id(state)))
            cluster += "\n}"
            clusters.append(cluster)

        return (
            "digraph SDFG {\n    outputorder=nodesfirst;\n" +
            "    compound=true;\n" + "    newrank=true;\n" +
            "\n    ".join(nodes + edges) + "\n" + "\n".join(clusters) + "\n}")

    # TODO(later): Also implement the "_repr_svg_" method for static output
    def _repr_html_(self):
        """ HTML representation of the SDFG, used mainly for Jupyter
            notebooks. """
        from dace.jupyter import isnotebook, preamble

        result = ''
        if not isnotebook():
            result = preamble()

        # Create renderer canvas and load SDFG
        result += """
<div id="contents_{uid}" style="position: relative; resize: vertical; overflow: auto"></div>
<script>
    var sdfg_{uid} = {sdfg};
    var renderer_{uid} = new SDFGRenderer(parse_sdfg(sdfg_{uid}),
        document.getElementById('contents_{uid}'));
</script>""".format(
            # Dumping to a string so that Jupyter Javascript can parse it
            # recursively
            sdfg=dace.serialize.dumps(dace.serialize.dumps(self.to_json())),
            uid=random.randint(0, sys.maxsize - 1))

        return result

    def transients(self):
        """ Returns a dictionary mapping transient data descriptors to their
            parent scope entry node, or None if top-level (i.e., exists in
            multiple scopes). """

        result = {}
        tstate = {}

        for (i, state) in enumerate(self.nodes()):
            scope_dict = state.scope_dict()
            for node in state.nodes():
                if isinstance(node,
                              nd.AccessNode) and node.desc(self).transient:
                    arrname = node.data
                    # If transient is accessed in more than one state, it is a
                    # top-level transient
                    if arrname in tstate and tstate[arrname] != i:
                        tstate[arrname] = None
                        result[arrname] = None
                    else:
                        tstate[arrname] = i
                        result[arrname] = scope_dict[node]

        return result

    def shared_transients(self):
        """ Returns a list of transient data that appears in more than one
            state. """
        seen = {}
        shared = []

        # If a transient is present in an inter-state edge, it is shared
        for interstate_edge in self.edges():
            for sym in interstate_edge.data.condition_symbols():
                if sym in self.arrays and self.arrays[sym].transient:
                    seen[sym] = interstate_edge
                    shared.append(sym)

        # If transient is accessed in more than one state, it is shared
        for state in self.nodes():
            for node in state.nodes():
                if isinstance(node,
                              nd.AccessNode) and node.desc(self).transient:
                    if node.desc(self).toplevel or (node.data in seen and
                                                    seen[node.data] != state):
                        shared.append(node.data)
                    seen[node.data] = state

        return dtypes.deduplicate(shared)

    def input_arrays(self):
        """ Returns a list of input arrays that need to be fed into the SDFG.
        """
        result = []
        for state in self.nodes():
            for node in state.source_nodes():
                if isinstance(node, nd.AccessNode):
                    if node not in result:
                        result.append(node)
        return result

    def output_arrays(self):
        """ Returns a list of output arrays that need to be returned from the
            SDFG. """
        result = []
        for state in self.nodes():
            for node in state.sink_nodes():
                if isinstance(node, nd.AccessNode):
                    if node not in result:
                        result.append(node)
        return result

    def save(self, filename: str, use_pickle=False, with_metadata=False):
        """ Save this SDFG to a file.
            :param filename: File name to save to.
            :param use_pickle: Use Python pickle as the SDFG format (default:
                               JSON).
            :param with_metadata: Save property metadata (e.g. name,
                                  description). False or True override current
                                  option, whereas None keeps default
        """
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        except (FileNotFoundError, FileExistsError):
            pass

        if use_pickle:
            with open(filename, "wb") as fp:
                symbolic.SympyAwarePickler(fp).dump(self)
        else:
            if with_metadata is not None:
                old_meta = dace.serialize.JSON_STORE_METADATA
                dace.serialize.JSON_STORE_METADATA = with_metadata
            with open(filename, "w") as fp:
                fp.write(dace.serialize.dumps(self.to_json()))
            if with_metadata is not None:
                dace.serialize.JSON_STORE_METADATA = old_meta

    @staticmethod
    def from_file(filename: str):
        """ Constructs an SDFG from a file.
            :param filename: File name to load SDFG from.
            :return: An SDFG.
        """
        with open(filename, "rb") as fp:
            firstbyte = fp.read(1)
            fp.seek(0)
            if firstbyte == b'{':  # JSON file
                sdfg_json = json.load(fp)
                sdfg = SDFG.from_json(sdfg_json)
            else:  # Pickle
                sdfg = symbolic.SympyAwareUnpickler(fp).load()

            if not isinstance(sdfg, SDFG):
                raise TypeError("Loaded file is not an SDFG (loaded "
                                "type: %s)" % type(sdfg).__name__)
            return sdfg

    # Dynamic SDFG creation API
    ##############################
    def add_state(self, label=None, is_start_state=False):
        """ Adds a new SDFG state to this graph and returns it.
            :param label: State label.
            :param is_start_state: If True, resets SDFG starting state to this
                                   state.
            :return: A new SDFGState object.
        """
        if label is None or any([s.label == label for s in self.nodes()]):
            i = len(self)
            base = "state" if label is None else label
            while True:
                # Append a number. If the state already exists, increment the
                # number until it doesn't
                label = "{}_{}".format(base, i)
                if any([s.label == label for s in self.nodes()]):
                    i += 1
                else:
                    break
        state = SDFGState(label, self)

        self.add_node(state, is_start_state=is_start_state)
        return state

    def _find_new_name(self, name: str):
        """ Tries to find a new name by adding an underscore and a number. """
        index = 0
        while (name + ('_%d' % index)) in self._arrays:
            index += 1

        return name + ('_%d' % index)

    def add_array(self,
                  name: str,
                  shape,
                  dtype,
                  storage=dtypes.StorageType.Default,
                  materialize_func=None,
                  transient=False,
                  strides=None,
                  offset=None,
                  toplevel=False,
                  debuginfo=None,
                  allow_conflicts=False,
                  total_size=None,
                  find_new_name=False):
        """ Adds an array to the SDFG data descriptor store. """

        if not isinstance(name, str):
            raise TypeError(
                "Array name must be a string. Got %s" % type(name).__name__)

        # If exists, fail
        if name in self._arrays:
            if not find_new_name:
                raise NameError(
                    'Array or Stream with name "%s" already exists '
                    "in SDFG" % name)
            else:
                name = self._find_new_name(name)

        # convert strings to int if possible
        newshape = []
        for s in shape:
            try:
                newshape.append(int(s))
            except:
                newshape.append(dace.symbolic.pystr_to_symbolic(s))
        shape = newshape

        if isinstance(dtype, type) and dtype in dtypes._CONSTANT_TYPES[:-1]:
            dtype = dtypes.typeclass(dtype)

        desc = dt.Array(
            dtype,
            shape,
            storage=storage,
            materialize_func=materialize_func,
            allow_conflicts=allow_conflicts,
            transient=transient,
            strides=strides,
            offset=offset,
            toplevel=toplevel,
            debuginfo=debuginfo,
            total_size=total_size)

        self._arrays[name] = desc
        return name, desc

    def add_stream(self,
                   name: str,
                   dtype,
                   veclen=1,
                   buffer_size=1,
                   shape=(1, ),
                   storage=dtypes.StorageType.Default,
                   transient=False,
                   offset=None,
                   toplevel=False,
                   debuginfo=None,
                   find_new_name=False):
        """ Adds a stream to the SDFG data descriptor store. """
        if not isinstance(name, str):
            raise TypeError(
                "Stream name must be a string. Got %s" % type(name).__name__)

        # If exists, fail
        if name in self._arrays:
            if not find_new_name:
                raise NameError(
                    'Array or Stream with name "%s" already exists '
                    "in SDFG" % name)
            else:
                name = self._find_new_name(name)

        if isinstance(dtype, type) and dtype in dtypes._CONSTANT_TYPES[:-1]:
            dtype = dtypes.typeclass(dtype)

        desc = dt.Stream(
            dtype,
            veclen,
            buffer_size,
            shape=shape,
            storage=storage,
            transient=transient,
            offset=offset,
            toplevel=toplevel,
            debuginfo=debuginfo,
        )

        self._arrays[name] = desc
        return name, desc

    def add_scalar(self,
                   name: str,
                   dtype,
                   storage=dtypes.StorageType.Default,
                   transient=False,
                   toplevel=False,
                   debuginfo=None,
                   find_new_name=False):
        """ Adds a scalar to the SDFG data descriptor store. """
        if not isinstance(name, str):
            raise TypeError(
                "Scalar name must be a string. Got %s" % type(name).__name__)
        # If exists, fail
        if name in self._arrays:
            if not find_new_name:
                raise NameError(
                    'Array or Stream with name "%s" already exists '
                    "in SDFG" % name)
            else:
                name = self._find_new_name(name)

        if isinstance(dtype, type) and dtype in dtypes._CONSTANT_TYPES[:-1]:
            dtype = dtypes.typeclass(dtype)

        desc = dt.Scalar(
            dtype,
            storage=storage,
            transient=transient,
            toplevel=toplevel,
            debuginfo=debuginfo,
        )

        self._arrays[name] = desc
        return name, desc

    def add_transient(self,
                      name,
                      shape,
                      dtype,
                      storage=dtypes.StorageType.Default,
                      materialize_func=None,
                      strides=None,
                      offset=None,
                      toplevel=False,
                      debuginfo=None,
                      allow_conflicts=False,
                      total_size=None,
                      find_new_name=False):
        """ Convenience function to add a transient array to the data
            descriptor store. """
        return self.add_array(
            name,
            shape,
            dtype,
            storage,
            materialize_func,
            True,
            strides,
            offset,
            toplevel=toplevel,
            debuginfo=debuginfo,
            allow_conflicts=allow_conflicts,
            total_size=total_size,
            find_new_name=find_new_name)

    def temp_data_name(self):
        """ Returns a temporary data descriptor name that can be used in this SDFG. """

        name = '__tmp%d' % self._temp_transients
        while name in self._arrays:
            self._temp_transients += 1
            name = '__tmp%d' % self._temp_transients
        self._temp_transients += 1

        return name

    def add_temp_transient(self,
                           shape,
                           dtype,
                           storage=dtypes.StorageType.Default,
                           materialize_func=None,
                           strides=None,
                           offset=None,
                           toplevel=False,
                           debuginfo=None,
                           allow_conflicts=False,
                           total_size=None):
        """ Convenience function to add a transient array with a temporary name to the data
            descriptor store. """
        return self.add_array(
            self.temp_data_name(),
            shape,
            dtype,
            storage,
            materialize_func,
            True,
            strides,
            offset,
            toplevel=toplevel,
            debuginfo=debuginfo,
            allow_conflicts=allow_conflicts,
            total_size=total_size)

    def add_datadesc(self, name: str, datadesc: dt.Data, find_new_name=False):
        """ Adds an existing data descriptor to the SDFG array store.
            :param name: Name to use.
            :param datadesc: Data descriptor to add.
            :param find_new_name: If True and data descriptor with this name
                                  exists, finds a new name to add.
            :return: Name of the new data descriptor
        """
        if not isinstance(name, str):
            raise TypeError("Data descriptor name must be a string. Got %s" %
                            type(name).__name__)
        # If exists, fail
        if name in self._arrays:
            if find_new_name:
                name = self._find_new_name(name)
            else:
                raise NameError(
                    'Array or Stream with name "%s" already exists '
                    "in SDFG" % name)
        self._arrays[name] = datadesc
        return name

    def add_loop(
            self,
            before_state,
            loop_state,
            after_state,
            loop_var: str,
            initialize_expr: str,
            condition_expr: str,
            increment_expr: str,
            loop_end_state=None,
    ):
        """ Helper function that adds a looping state machine around a
            given state (or sequence of states).
            :param before_state: The state after which the loop should
                                 begin, or None if the loop is the first
                                 state (creates an empty state).
            :param loop_state: The state that begins the loop. See also
                               `loop_end_state` if the loop is multi-state.
            :param after_state: The state that should be invoked after
                                the loop ends, or None if the program
                                should terminate (creates an empty state).
            :param loop_var: A name of an inter-state variable to use
                             for the loop. If None, `initialize_expr`
                             and `increment_expr` must be None.
            :param initialize_expr: A string expression that is assigned
                                    to `loop_var` before the loop begins.
                                    If None, does not define an expression.
            :param condition_expr: A string condition that occurs every
                                   loop iteration. If None, loops forever
                                   (undefined behavior).
            :param increment_expr: A string expression that is assigned to
                                   `loop_var` after every loop iteration.
                                    If None, does not define an expression.
            :param loop_end_state: If the loop wraps multiple states, the
                                   state where the loop iteration ends.
                                   If None, sets the end state to
                                   `loop_state` as well.
            :return: A 3-tuple of (`before_state`, generated loop guard state,
                                   `after_state`).
        """
        from dace.frontend.python.astutils import negate_expr  # Avoid import loops

        # Argument checks
        if loop_var is None and (initialize_expr or increment_expr):
            raise ValueError("Cannot initalize or increment an empty loop"
                             " variable")

        # Handling empty states
        if loop_end_state is None:
            loop_end_state = loop_state
        if before_state is None:
            before_state = self.add_state()
        if after_state is None:
            after_state = self.add_state()

        # Create guard state
        guard = self.add_state("guard")

        # Loop initialization
        init = None if initialize_expr is None else {loop_var: initialize_expr}
        self.add_edge(before_state, guard, ed.InterstateEdge(assignments=init))

        # Loop condition
        if condition_expr:
            cond_ast = CodeProperty.from_string(condition_expr,
                                                dtypes.Language.Python)
        else:
            cond_ast = CodeProperty.from_string('True', dtypes.Language.Python)
        self.add_edge(guard, loop_state, ed.InterstateEdge(cond_ast))
        self.add_edge(guard, after_state,
                      ed.InterstateEdge(negate_expr(cond_ast)))

        # Loop incrementation
        incr = None if increment_expr is None else {loop_var: increment_expr}
        self.add_edge(
            loop_end_state, guard, ed.InterstateEdge(assignments=incr))

        return before_state, guard, after_state

    # SDFG queries
    ##############################

    def find_state(self, state_id_or_label):
        """ Finds a state according to its ID (if integer is provided) or
            label (if string is provided).

            :param state_id_or_label: State ID (if int) or label (if str).
            :return: An SDFGState object.
        """

        if isinstance(state_id_or_label, str):
            for s in self.nodes():
                if s.label == state_id_or_label:
                    return s
            raise LookupError("State %s not found" % state_id_or_label)
        elif isinstance(state_id_or_label, int):
            return self.nodes()[state_id_or_label]
        else:
            raise TypeError(
                "state_id_or_label is not an int nor string: {}".format(
                    state_id_or_label))

    def find_node(self, state_id_or_label, node_id_or_label):
        """ Finds a node within a state according to its ID (if integer is
            provided) or label (if string is provided).

            :param state_id_or_label: State ID (if int) or label (if str).
            :param node_id_or_label:  Node ID (if int) or label (if str)
                                      within the given state.
            :return: A nodes.Node object.
        """
        state = self.find_state(state_id_or_label)
        return state.find_node(node_id_or_label)

    def specialize(self, additional_symbols=None, specialize_all_symbols=True):
        """ Sets symbolic values in this SDFG to constants.
            :param additional_symbols: Additional values to specialize.
            :param specialize_all_symbols: If True, raises an
                   UnboundLocalError if at least one of the symbols in the
                   SDFG is unset.
        """
        syms = {}
        additional_symbols = additional_symbols or {}
        undefined_symbols = self.undefined_symbols(False)
        # scalar_arguments = self.scalar_parameters(False)
        for (
                symname
        ) in undefined_symbols:  # itertools.chain(undefined_symbols, scalar_arguments):
            try:
                syms[symname] = symbolic.symbol(symname).get()
            except UnboundLocalError:
                # Allow scalar arguments to remain undefined, but fail on
                # symbols
                if specialize_all_symbols and symname not in additional_symbols:
                    pass

        # Augment symbol values from additional symbols
        syms.update({
            # If symbols are passed, extract the value. If constants are
            # passed, use them directly.
            name: val.get() if isinstance(val, dace.symbolic.symbol) else val
            for name, val in additional_symbols.items()
        })

        # Update constants
        for k, v in syms.items():
            self.add_constant(k, v)

    def compile(self, specialize=None, optimizer=None, output_file=None):
        """ Compiles a runnable binary from this SDFG.

            :param specialize: If True, specializes all symbols to their
                               defined values as constants. If None, uses
                               configuration setting.
            :param optimizer: If defines a valid class name, it will be called
                              during compilation to transform the SDFG as
                              necessary. If None, uses configuration setting.
            :param output_file: If not None, copies the output library file to
                                the specified path.
            :return: A callable CompiledSDFG object.
        """

        # Importing these outside creates an import loop
        from dace.codegen import codegen, compiler

        if Config.get_bool("compiler", "use_cache"):
            # Try to see if a cached version of the binary exists
            # print("looking for cached binary: " + compiler.get_binary_name(self.name))
            binary_filename = compiler.get_binary_name(self.name)
            if os.path.isfile(binary_filename):
                # print("A cached binary was found!")
                return compiler.load_from_file(self, binary_filename)

        ############################
        # DaCe Compilation Process #

        # Clone SDFG as the other modules may modify its contents
        sdfg = copy.deepcopy(self)

        # Fill in scope entry/exit connectors
        sdfg.fill_scope_connectors()

        # Propagate memlets in the graph
        if self._propagate:
            propagate_labels_sdfg(sdfg)

        # Specialize SDFG to its symbol values
        if (specialize is None and Config.get_bool(
                "optimizer", "autospecialize")) or specialize == True:
            sdfg.specialize()

        # Optimize SDFG using the CLI or external hooks
        optclass = _get_optimizer_class(optimizer)
        if optclass is not None:
            opt = optclass(sdfg)
            sdfg = opt.optimize()

        sdfg.save(os.path.join('_dotgraphs', 'program.sdfg'))

        # Generate code for the program by traversing the SDFG state by state
        program_objects = codegen.generate_code(sdfg)

        # Generate the program folder and write the source files
        program_folder = compiler.generate_program_folder(
            self, program_objects, os.path.join(".dacecache", sdfg.name))

        # Compile the code and get the shared library path
        shared_library = compiler.configure_and_compile(program_folder)

        # If provided, save output to path or filename
        if output_file is not None:
            if os.path.isdir(output_file):
                output_file = os.path.join(output_file,
                                           os.path.basename(shared_library))
            shutil.copyfile(shared_library, output_file)

        # Get the function handle
        return compiler.get_program_handle(shared_library, sdfg)

    def argument_typecheck(self, args, kwargs, types_only=False):
        """ Checks if arguments and keyword arguments match the SDFG
            types. Raises RuntimeError otherwise.

            :raise RuntimeError: Argument count mismatch.
            :raise TypeError: Argument type mismatch.
            :raise NotImplementedError: Unsupported argument type.
        """
        expected_args = self.arglist()
        num_args_passed = len(args) + len(kwargs)
        num_args_expected = len(expected_args)
        if num_args_passed < num_args_expected:
            expected_kwargs = list(expected_args.keys())[len(args):]
            missing_args = [k for k in expected_kwargs if k not in kwargs]
            raise RuntimeError(
                "Missing arguments to SDFG: '%s'" % (', '.join(missing_args)))
        elif num_args_passed > num_args_expected:
            unnecessary_args = []
            extra_args = len(args) - len(expected_args)
            if extra_args > 0:
                unnecessary_args.extend(
                    'Argument #%d' % (i + len(expected_args) + 1)
                    for i in range(extra_args))
                unnecessary_args.extend(kwargs.keys())
            else:
                unnecessary_args = [
                    k for k in kwargs.keys() if k not in expected_args
                ]
            raise RuntimeError("Too many arguments to SDFG. Unnecessary "
                               "arguments: %s" % ', '.join(unnecessary_args))
        positional_args = list(args)
        for i, arg in enumerate(expected_args):
            expected = expected_args[arg]
            if i < len(positional_args):
                passed = positional_args[i]
            else:
                if arg not in kwargs:
                    raise RuntimeError(
                        "Missing argument to DaCe program: {}".format(arg))
                passed = kwargs[arg]
            if types_only:
                desc = dt.create_datadescriptor(passed)
                if not expected.is_equivalent(desc):
                    raise TypeError("Type mismatch for argument: "
                                    "expected %s, got %s" % (expected, desc))
                else:
                    continue
            if isinstance(expected, dace.data.Array):
                if not isinstance(passed, np.ndarray):
                    raise TypeError("Type mismatch for argument {}: "
                                    "expected array type, got {}".format(
                                        arg, type(passed)))
            elif (isinstance(expected, dace.data.Scalar)
                  or isinstance(expected, dace.dtypes.typeclass)):
                if (not dace.dtypes.isconstant(passed)
                        and not isinstance(passed, dace.symbolic.symbol)):
                    raise TypeError("Type mismatch for argument {}: "
                                    "expected scalar type, got {}".format(
                                        arg, type(passed)))
            elif isinstance(expected, dace.data.Stream):
                if not isinstance(passed, dace.dtypes.stream):
                    raise TypeError("Type mismatch for argument {}: "
                                    "expected stream type, got {}".format(
                                        arg, type(passed)))
            else:
                raise NotImplementedError(
                    "Type checking not implemented for type {} (argument "
                    "{})".format(type(expected).__name__, arg))

    def __call__(self, *args, **kwargs):
        """ Invokes an SDFG, generating and compiling code if necessary. """

        binaryobj = self.compile()

        # Verify passed arguments (unless disabled by the user)
        if dace.config.Config.get_bool("execution", "general", "check_args"):
            self.argument_typecheck(args, kwargs)
        return binaryobj(*args, **kwargs)

    def fill_scope_connectors(self):
        """ Fills missing scope connectors (i.e., "IN_#"/"OUT_#" on entry/exit
            nodes) according to data on the memlets. """
        for state in self.nodes():
            state.fill_scope_connectors()

    def predecessor_state_transitions(self, state):
        """ Yields paths (lists of edges) that the SDFG can pass through
            before computing the given state. """
        from networkx import all_simple_paths

        for path in all_simple_paths(self, self._start_state, state):
            yield [
                next(e for e in self.out_edges(s) if e.dst == d)
                for s, d in zip(path[:-1], path[1:])
            ]

    def predecessor_states(self, state):
        """ Returns a list of unique states that the SDFG can pass through
            before computing the given state. """
        from networkx import all_simple_paths

        start_state = self._start_state or self.source_nodes()[0]
        return set([
            n for path in all_simple_paths(self, start_state, state)
            for n in path
        ])

    def validate(self) -> None:
        """ Verifies the correctness of an SDFG by applying multiple tests.

            Raises an InvalidSDFGError with the erroneous node/edge
            on failure.
        """
        try:
            # SDFG-level checks
            if not validate_name(self.name):
                raise InvalidSDFGError("Invalid name", self, None)

            if len(self.source_nodes()) > 1 and self._start_state is None:
                raise InvalidSDFGError("Starting state undefined", self, None)

            if len(set([s.label for s in self.nodes()])) != len(self.nodes()):
                raise InvalidSDFGError(
                    "Found multiple states with the same name", self, None)

            # Validate array names
            for name in self._arrays.keys():
                if name is not None and not validate_name(name):
                    raise InvalidSDFGError("Invalid array name %s" % name,
                                           self, None)

            # Check every state separately
            for sid, state in enumerate(self.nodes()):
                state.validate(self, sid)

            # Interstate edge checks
            for eid, edge in enumerate(self.edges()):

                # Name validation
                if len(edge.data.assignments) > 0:
                    for assign in edge.data.assignments.keys():
                        if not validate_name(assign):
                            raise InvalidSDFGInterstateEdgeError(
                                "Invalid interstate symbol name %s" % assign,
                                self, eid)

            # TODO: Check interstate edges with undefined symbols

        except InvalidSDFGError:
            # If the SDFG is invalid, save it
            self.save(os.path.join('_dotgraphs', 'invalid.sdfg'))
            raise

    def is_valid(self) -> bool:
        """ Returns True if the SDFG is verified correctly (using `validate`).
        """
        try:
            self.validate()
        except InvalidSDFGError:
            return False
        return True

    def apply_strict_transformations(self, validate=True):
        """ Applies safe transformations (that will surely increase the
            performance) on the SDFG. For example, this fuses redundant states
            (safely) and removes redundant arrays.

            B{Note:} This is an in-place operation on the SDFG.
        """
        from dace.transformation.dataflow import RedundantArray, MergeArrays
        from dace.transformation.interstate import StateFusion, InlineSDFG

        strict_transformations = [
            StateFusion, RedundantArray, MergeArrays, InlineSDFG
        ]

        self.apply_transformations(
            strict_transformations, validate=validate, strict=True)

    def apply_transformations(self,
                              patterns: Union[Type, List[Type]],
                              validate: bool = True,
                              strict: bool = False,
                              states: Optional[List[Any]] = None,
                              apply_once: bool = False,
                              properties: Dict[str, Any] = None):
        """ This function applies transformations as given in the argument
            patterns. Operates in-place.
            :param patterns: A Transformation class or a list thereof to apply.
            :param validate: If True, validates after every transformation.
            :param strict: If True, operates in strict transformation mode.
            :param states: If not None, specifies a subset of states to
                           apply transformations on.
            :param apply_once: If True, applies the first found transformation
                               and returns. Otherwise, applies until no further
                               transformations are found.
            :param properties: Properties to set when applying transformations.
        """
        # Avoiding import loops
        from dace.transformation import optimizer
        from dace.transformation.pattern_matching import Transformation

        if isinstance(patterns, type) and issubclass(patterns, Transformation):
            patterns = [patterns]

        # Apply strict state fusions greedily.
        opt = optimizer.SDFGOptimizer(self, inplace=True)
        applied = True
        applied_transformations = collections.defaultdict(int)
        while applied:
            applied = False
            # Find and apply immediately
            for match in opt.get_pattern_matches(
                    strict=strict, patterns=patterns, states=states):
                sdfg = self.sdfg_list[match.sdfg_id]
                if properties is not None:
                    for prop_name, prop_val in properties.items():
                        setattr(match, prop_name, prop_val)
                match.apply(sdfg)
                applied_transformations[type(match).__name__] += 1
                if validate:
                    self.fill_scope_connectors()
                    self.validate()
                applied = True
                break
            if apply_once and applied:
                break

        if Config.get_bool('debugprint') and len(applied_transformations) > 0:
            print('Applied {}.'.format(', '.join([
                '%d %s' % (v, k) for k, v in applied_transformations.items()
            ])))

    def apply_gpu_transformations(self,
                                  states=None,
                                  validate=True,
                                  strict=True):
        """ Applies a series of transformations on the SDFG for it to
            generate GPU code.
            @note: It is recommended to apply redundant array removal
            transformation after this transformation. Alternatively,
            you can apply_strict_transformations() after this transformation.
            @note: This is an in-place operation on the SDFG.
        """
        # Avoiding import loops
        from dace.transformation.dataflow import GPUTransformLocalStorage

        patterns = [GPUTransformLocalStorage]
        self.apply_transformations(
            patterns, validate=validate, strict=strict, states=states)

    def generate_code(self, specialize=None):
        """ Generates code from this SDFG and returns it.
            :param specialize: If True, specializes all set symbols to their
                               values in the generated code. If None,
                               uses default configuration value.
            :return: A list of `CodeObject` objects containing the generated
                      code of different files and languages.
        """

        # Import loop "fix"
        from dace.codegen import codegen

        ################################
        # DaCe Code Generation Process #
        sdfg = copy.deepcopy(self)

        # Fill in scope entry/exit connectors
        sdfg.fill_scope_connectors()

        # Propagate memlets in the graph
        if sdfg.propagate:
            labeling.propagate_labels_sdfg(sdfg)

        # Specialize SDFG to its symbol values
        if (specialize is None and Config.get_bool(
                "optimizer", "autospecialize")) or specialize == True:
            sdfg.specialize()

        sdfg.draw_to_file()
        sdfg.save(os.path.join('_dotgraphs', 'program.sdfg'))

        # Generate code for the program by traversing the SDFG state by state
        program_code = codegen.generate_code(sdfg)

        return program_code


class MemletTrackingView(object):
    """ A mixin class that enables tracking memlets in directed acyclic multigraphs. """

    def memlet_path(self,
                    edge: MultiConnectorEdge) -> List[MultiConnectorEdge]:
        """ Given one edge, returns a list of edges representing a path
            between its source and sink nodes. Used for memlet tracking.

            @note: Behavior is undefined when there is more than one path
                   involving this edge.
            :param edge: An edge within this state.
            :return: A list of edges from a source node to a destination node.
            """
        result = [edge]

        # Obtain the full state (to work with paths that trace beyond a scope)
        state = self._graph

        # If empty memlet, return itself as the path
        if edge.src_conn is None and edge.dst_conn is None and edge.data.data is None:
            return result

        # Prepend incoming edges until reaching the source node
        curedge = edge
        while not isinstance(curedge.src,
                             (nd.CodeNode, nd.AccessNode, nd.Reduce)):
            # Trace through scopes using OUT_# -> IN_#
            if isinstance(curedge.src, (nd.EntryNode, nd.ExitNode)):
                if curedge.src_conn is None:
                    raise ValueError(
                        "Source connector cannot be None for {}".format(
                            curedge.src))
                assert curedge.src_conn.startswith("OUT_")
                next_edge = next(
                    e for e in state.in_edges(curedge.src)
                    if e.dst_conn == "IN_" + curedge.src_conn[4:])
                result.insert(0, next_edge)
                curedge = next_edge

        # Prepend outgoing edges until reaching the sink node
        curedge = edge
        while not isinstance(curedge.dst,
                             (nd.CodeNode, nd.AccessNode, nd.Reduce)):
            # Trace through scope entry using IN_# -> OUT_#
            if isinstance(curedge.dst, (nd.EntryNode, nd.ExitNode)):
                if curedge.dst_conn is None:
                    raise ValueError(
                        "Destination connector cannot be None for {}".format(
                            curedge.dst))
                if not curedge.dst_conn.startswith("IN_"):  # Map variable
                    break
                next_edge = next(
                    e for e in state.out_edges(curedge.dst)
                    if e.src_conn == "OUT_" + curedge.dst_conn[3:])
                result.append(next_edge)
                curedge = next_edge

        return result

    def memlet_tree(self, edge: MultiConnectorEdge) -> mm.MemletTree:
        """ Given one edge, returns a tree of edges between its node source(s)
            and sink(s). Used for memlet tracking.

            :param edge: An edge within this state.
            :return: A tree of edges whose root is the source/sink node
                     (depending on direction) and associated children edges.
            """
        propagate_forward = False
        propagate_backward = False
        if ((isinstance(edge.src, nd.EntryNode) and edge.src_conn is not None)
                or
            (isinstance(edge.dst, nd.EntryNode) and edge.dst_conn is not None
             and edge.dst_conn.startswith('IN_'))):
            propagate_forward = True
        if ((isinstance(edge.src, nd.ExitNode) and edge.src_conn is not None)
                or
            (isinstance(edge.dst, nd.ExitNode) and edge.dst_conn is not None)):
            propagate_backward = True

        # If either both are False (no scopes involved) or both are True
        # (invalid SDFG), we return only the current edge as a degenerate tree
        if propagate_forward == propagate_backward:
            return mm.MemletTree(edge)

        # Obtain the full state (to work with paths that trace beyond a scope)
        state = self._graph

        # Find tree root
        curedge = edge
        if propagate_forward:
            while (isinstance(curedge.src, nd.EntryNode)
                   and curedge.src_conn is not None):
                assert curedge.src_conn.startswith('OUT_')
                cname = curedge.src_conn[4:]
                curedge = next(
                    e for e in state.in_edges(curedge.src)
                    if e.dst_conn == 'IN_%s' % cname)
        elif propagate_backward:
            while (isinstance(curedge.dst, nd.ExitNode)
                   and curedge.dst_conn is not None):
                assert curedge.dst_conn.startswith('IN_')
                cname = curedge.dst_conn[3:]
                curedge = next(
                    e for e in state.out_edges(curedge.dst)
                    if e.src_conn == 'OUT_%s' % cname)
        tree_root = mm.MemletTree(curedge)

        # Collect children (recursively)
        def add_children(treenode):
            if propagate_forward:
                if not (isinstance(treenode.edge.dst, nd.EntryNode)
                        and treenode.edge.dst_conn
                        and treenode.edge.dst_conn.startswith('IN_')):
                    return
                conn = treenode.edge.dst_conn[3:]
                treenode.children = [
                    mm.MemletTree(e, parent=treenode)
                    for e in state.out_edges(treenode.edge.dst)
                    if e.src_conn == 'OUT_%s' % conn
                ]
            elif propagate_backward:
                if (not isinstance(treenode.edge.src, nd.ExitNode)
                        or treenode.edge.src_conn is None):
                    return
                conn = treenode.edge.src_conn[4:]
                treenode.children = [
                    mm.MemletTree(e, parent=treenode)
                    for e in state.in_edges(treenode.edge.src)
                    if e.dst_conn == 'IN_%s' % conn
                ]

            for child in treenode.children:
                add_children(child)

        # Start from root node (obtained from above parent traversal)
        add_children(tree_root)

        # Find edge in tree
        def traverse(node):
            if node.edge == edge:
                return node
            for child in node.children:
                res = traverse(child)
                if res is not None:
                    return res
            return None

        # Return node that corresponds to current edge
        return traverse(tree_root)


class ScopeSubgraphView(SubgraphView, MemletTrackingView):
    """ An extension to SubgraphView that enables the creation of scope
        dictionaries in subgraphs and free symbols. """

    def __init__(self, graph, subgraph_nodes):
        super(ScopeSubgraphView, self).__init__(graph, subgraph_nodes)
        self._clear_scopedict_cache()

    @property
    def parent(self):
        return self._graph.parent

    def _clear_scopedict_cache(self):
        """ Clears the cached results for the scope_dict function.

            For use when the graph mutates (e.g., new edges/nodes, deletions).
        """
        self._scope_dict_toparent_cached = None
        self._scope_dict_tochildren_cached = None

    def scope_dict(self,
                   node_to_children=False,
                   return_ids=False,
                   validate=True):
        """ Returns a dictionary that segments an SDFG state into
            entry-node/exit-node scopes.

            :param node_to_children: If False (default), returns a mapping
                                     of each node to its parent scope
                                     (ScopeEntry) node. If True, returns a
                                     mapping of each parent node to a list of
                                     children nodes.
            :type node_to_children: bool
            :param return_ids: Return node ID numbers instead of node objects.
            :type return_ids: bool
            :param validate: Ensure that the graph is not malformed when
                 computing dictionary.
            :return: The mapping from a node to its parent scope node, or the
                     mapping from a node to a list of children nodes.
            :rtype: dict(Node, Node) or dict(Node, list(Node))
        """
        result = None
        if not node_to_children and self._scope_dict_toparent_cached is not None:
            result = copy.copy(self._scope_dict_toparent_cached)
        elif node_to_children and self._scope_dict_tochildren_cached is not None:
            result = copy.copy(self._scope_dict_tochildren_cached)

        if result is None:
            result = {}
            node_queue = collections.deque(self.source_nodes())
            eq = _scope_dict_inner(self, node_queue, None, node_to_children,
                                   result)

            # Sanity check
            if validate:
                assert len(eq) == 0

            # Cache result
            if node_to_children:
                self._scope_dict_tochildren_cached = result
            else:
                self._scope_dict_toparent_cached = result

            result = copy.copy(result)

        if return_ids:
            return _scope_dict_to_ids(self, result)
        return result

    def scope_subgraph(self, entry_node, include_entry=True,
                       include_exit=True):
        """ Returns a subgraph that only contains the scope, defined by the
            given entry node.
        """
        return _scope_subgraph(self, entry_node, include_entry, include_exit)

    def top_level_transients(self):
        return top_level_transients(self)

    def all_transients(self):
        return all_transients(self)

    def entry_node(self, exit_node):
        """ Returns the entry node corresponding to the passed exit node. """
        return self.scope_dict()[exit_node]

    def exit_nodes(self, entry_node):
        """ Returns the exit node leaving the context opened by
            the given entry node. """

        if not isinstance(entry_node, nd.EntryNode):
            raise TypeError(
                "Received {}: should be dace.nodes.EntryNode".format(
                    type(entry_node).__name__))

        node_to_children = self.scope_dict(True)
        return [
            v for v in node_to_children[entry_node]
            if isinstance(v, nd.ExitNode)
        ]

    def data_symbols(self):
        """Returns all symbols used in data nodes."""
        return data_symbols(self)

    def scope_symbols(self):
        """Returns all symbols defined by scopes within this state."""
        return scope_symbols(self)

    def interstate_symbols(self):
        """Returns all symbols (assigned, used) in interstate edges in nested
           SDFGs within this subgraph."""
        return interstate_symbols(self)

    def undefined_symbols(self, sdfg, include_scalar_data):
        return undefined_symbols(sdfg, self, include_scalar_data)

    def all_nodes_recursive(self):
        all_nodes = []
        for node in self.nodes():
            all_nodes.append((node, self))
            if isinstance(node, dace.graph.nodes.NestedSDFG):
                all_nodes += node.sdfg.all_nodes_recursive()
        return all_nodes

    def all_edges_recursive(self):
        all_edges = [(e, self) for e in self.edges()]
        for node in self.nodes():
            if isinstance(node, dace.graph.nodes.NestedSDFG):
                all_edges += node.sdfg.all_edges_recursive()
        return all_edges


# TODO: Use mixin for SDFGState and ScopeSubgraphView for scope dict
@make_properties
class SDFGState(OrderedMultiDiConnectorGraph, MemletTrackingView):
    """ An acyclic dataflow multigraph in an SDFG, corresponding to a
        single state in the SDFG state machine. """

    is_collapsed = Property(
        dtype=bool,
        desc="Show this node/scope/state as collapsed",
        default=False)

    nosync = Property(
        dtype=bool,
        default=False,
        desc="Do not synchronize at the end of the state")

    instrument = Property(
        choices=dtypes.InstrumentationType,
        desc="Measure execution statistics with given method",
        default=dtypes.InstrumentationType.No_Instrumentation)

    def __init__(self, label=None, sdfg=None, debuginfo=None):
        """ Constructs an SDFG state.
            :param label: Name for the state (optional).
            :param sdfg: A reference to the parent SDFG.
            :param debuginfo: Source code locator for debugging.
        """
        super(SDFGState, self).__init__()
        self._label = label
        self._parent = sdfg
        self._graph = self  # Allowing MemletTrackingView mixin to work
        self._clear_scopedict_cache()
        self._debuginfo = debuginfo
        self.is_collapsed = False
        self.nosync = False
        self._parallel_parent = (
            None
        )  # This (and is_parallel and set_parallel_parent) are duplicated...
        self._instrumented_parent = (
            False
        )  # Same as above. This flag is needed to know if the parent is instrumented (it's possible for a parent to be serial and instrumented.)

    @property
    def parent(self):
        """ Returns the parent SDFG of this state. """
        return self._parent

    def has_instrumented_parent(self):
        return self._instrumented_parent

    def set_instrumented_parent(self):
        self._instrumented_parent = (
            True
        )  # When this is set: Under no circumstances try instrumenting this (or any transitive children)

    def is_parallel(self):
        return self._parallel_parent is not None

    def set_parallel_parent(self, parallel_parent):
        self._parallel_parent = parallel_parent

    def get_parallel_parent(self):
        return self._parallel_parent

    def __str__(self):
        return self._label

    # Clears the cached results for the scope_dict function.
    # For use when the graph mutates (e.g., new edges/nodes, deletions)
    def _clear_scopedict_cache(self):
        self._scope_dict_toparent_cached = None
        self._scope_dict_tochildren_cached = None
        self._scope_tree_cached = None
        self._scope_leaves_cached = None

    @property
    def label(self):
        return self._label

    @property
    def name(self):
        return self._label

    def set_label(self, label):
        self._label = label

    def replace(self, name: str, new_name: str):
        """ Finds and replaces all occurrences of a symbol or array in this
            state.
            :param name: Name to find.
            :param new_name: Name to replace.
        """
        replace(self, name, new_name)

    def add_node(self, node):
        if not isinstance(node, nd.Node):
            raise TypeError("Expected Node, got " + str(type(node)) + " (" +
                            str(node) + ")")
        self._clear_scopedict_cache()
        return super(SDFGState, self).add_node(node)

    def remove_node(self, node):
        self._clear_scopedict_cache()
        super(SDFGState, self).remove_node(node)

    def add_edge(self, u, u_connector, v, v_connector, memlet):
        if not isinstance(u, nd.Node):
            raise TypeError(
                "Source node is not of type nd.Node (type: %s)" % str(type(u)))
        if u_connector is not None and not isinstance(u_connector, str):
            raise TypeError("Source connector is not string (type: %s)" % str(
                type(u_connector)))
        if not isinstance(v, nd.Node):
            raise TypeError("Destination node is not of type nd.Node (type: " +
                            "%s)" % str(type(v)))
        if v_connector is not None and not isinstance(v_connector, str):
            raise TypeError("Destination connector is not string (type: %s)" %
                            str(type(v_connector)))
        if not isinstance(memlet, mm.Memlet):
            raise TypeError(
                "Memlet is not of type Memlet (type: %s)" % str(type(memlet)))

        self._clear_scopedict_cache()
        return super(SDFGState, self).add_edge(u, u_connector, v, v_connector,
                                               memlet)

    def remove_edge(self, edge):
        self._clear_scopedict_cache()
        super(SDFGState, self).remove_edge(edge)

    def remove_edge_and_connectors(self, edge):
        self._clear_scopedict_cache()
        super(SDFGState, self).remove_edge(edge)
        if edge.src_conn in edge.src.out_connectors:
            edge.src._out_connectors.remove(edge.src_conn)
        if edge.dst_conn in edge.dst.in_connectors:
            edge.dst._in_connectors.remove(edge.dst_conn)

    def all_nodes_recursive(self):
        all_nodes = []
        for node in self.nodes():
            all_nodes.append((node, self))
            if isinstance(node, dace.graph.nodes.NestedSDFG):
                all_nodes += node.sdfg.all_nodes_recursive()
        return all_nodes

    def all_edges_recursive(self):
        all_edges = [(e, self) for e in self.edges()]
        for node in self.nodes():
            if isinstance(node, dace.graph.nodes.NestedSDFG):
                all_edges += node.sdfg.all_edges_recursive()
        return all_edges

    def data_symbols(self):
        """ Returns all symbols used in data nodes. """
        return data_symbols(self)

    def scope_symbols(self):
        """ Returns all symbols defined by scopes within this state. """
        return scope_symbols(self)

    def interstate_symbols(self):
        """ Returns all symbols assigned/used in interstate edges in nested
           SDFGs within this state. """
        return interstate_symbols(self)

    def undefined_symbols(self, sdfg, include_scalar_data):
        return undefined_symbols(sdfg, self, include_scalar_data)

    def data_nodes(self):
        """ Returns all data_nodes (arrays) present in this state. """
        return [n for n in self.nodes() if isinstance(n, nd.AccessNode)]

    def memlets_for_array(self, arrayname):
        return [e for e in self.edges() if e[3].data == arrayname]

    def draw_node(self, graph):
        return dot.draw_node(graph, self, shape="Msquare")

    def to_json(self, parent=None):
        ret = {
            'type':
            type(self).__name__,
            'label':
            self.name,
            'id':
            parent.node_id(self) if parent is not None else None,
            'collapsed':
            self.is_collapsed,
            'scope_dict': {
                k: sorted(v)
                for k, v in sorted(
                    self.scope_dict(node_to_children=True, return_ids=True)
                    .items())
            },
            'nodes': [n.to_json(self) for n in self.nodes()],
            'edges': [
                e.to_json(self) for e in sorted(
                    self.edges(),
                    key=lambda e: (e.src_conn or '', e.dst_conn or ''))
            ],
            'attributes':
            dace.serialize.all_properties_to_json(self),
        }

        return ret

    @classmethod
    def from_json(cls, json_obj, context={'sdfg': None}):
        """ Loads the node properties, label and type into a dict.
            :param json_obj: The object containing information about this node.
                             NOTE: This may not be a string!
            :return: An SDFGState instance constructed from the passed data
        """

        _type = json_obj['type']
        if _type != cls.__name__:
            raise Exception("Class type mismatch")

        attrs = json_obj['attributes']
        nodes = json_obj['nodes']
        edges = json_obj['edges']

        ret = SDFGState(
            label=json_obj['label'], sdfg=context['sdfg'], debuginfo=None)

        rec_ci = {
            'sdfg': context['sdfg'],
            'sdfg_state': ret,
            'callback': context['callback'] if 'callback' in context else None
        }
        dace.serialize.set_properties_from_json(ret, json_obj, rec_ci)

        for n in nodes:
            nret = dace.serialize.loads(
                dace.serialize.dumps(n), context=rec_ci)
            ret.add_node(nret)

        # Connect using the edges
        for e in edges:
            eret = dace.serialize.loads(
                dace.serialize.dumps(e), context=rec_ci)

            ret.add_edge(eret.src, eret.src_conn, eret.dst, eret.dst_conn,
                         eret.data)

        # Fix potentially broken scopes
        for n in nodes:
            if isinstance(n, dace.graph.nodes.MapExit):
                n.map = ret.entry_node(n).map
            elif isinstance(n, dace.graph.nodes.ConsumeExit):
                n.consume = ret.entry_node(n).consume

        return ret

    def _repr_html_(self):
        """ HTML representation of a state, used mainly for Jupyter
            notebooks. """
        # Create dummy SDFG with this state as the only one
        arrays = set(n.data for n in self.data_nodes())
        sdfg = SDFG(self.label)
        sdfg._arrays = {k: self._parent.arrays[k] for k in arrays}
        sdfg.add_node(self)

        return sdfg._repr_html_()

    def scope_tree(self):
        if (hasattr(self, '_scope_tree_cached')
                and self._scope_tree_cached is not None):
            return copy.copy(self._scope_tree_cached)

        sdp = self.scope_dict(node_to_children=False)
        sdc = self.scope_dict(node_to_children=True)

        result = {}

        sdfg_symbols = self.parent.undefined_symbols(True).keys()

        # Get scopes
        for node, scopenodes in sdc.items():
            if node is None:
                exit_node = None
            else:
                exit_node = next(
                    v for v in scopenodes if isinstance(v, nd.ExitNode))
            scope = Scope(node, exit_node)
            scope.defined_vars = set(
                symbolic.pystr_to_symbolic(s)
                for s in (self.parent.symbols_defined_at(node, self).keys()
                          | sdfg_symbols))
            result[node] = scope

        # Scope parents and children
        for node, scope in result.items():
            if node is not None:
                scope.parent = result[sdp[node]]
            scope.children = [
                result[n] for n in sdc[node] if isinstance(n, nd.EntryNode)
            ]

        self._scope_tree_cached = result

        return copy.copy(self._scope_tree_cached)

    def scope_leaves(self):
        if (hasattr(self, '_scope_leaves_cached')
                and self._scope_leaves_cached is not None):
            return copy.copy(self._scope_leaves_cached)
        st = self.scope_tree()
        self._scope_leaves_cached = [
            scope for scope in st.values() if len(scope.children) == 0
        ]
        return copy.copy(self._scope_leaves_cached)

    def scope_dict(self,
                   node_to_children=False,
                   return_ids=False,
                   validate=True):
        """ Returns a dictionary that segments an SDFG state into
            entry-node/exit-node scopes.

            :param node_to_children: If False (default), returns a mapping
                                     of each node to its parent scope
                                     (ScopeEntry) node. If True, returns a
                                     mapping of each parent node to a list of
                                     children nodes.
            :type node_to_children: bool
            :param return_ids: Return node ID numbers instead of node objects.
            :type return_ids: bool
            :param validate: Ensure that the graph is not malformed when
                             computing dictionary.
            :return: The mapping from a node to its parent scope node, or the
                     mapping from a node to a list of children nodes.
            :rtype: dict(Node, Node) or dict(Node, list(Node))
        """
        result = None
        if not node_to_children and self._scope_dict_toparent_cached is not None:
            result = copy.copy(self._scope_dict_toparent_cached)
        elif node_to_children and self._scope_dict_tochildren_cached is not None:
            result = copy.copy(self._scope_dict_tochildren_cached)

        if result is None:
            result = {}
            node_queue = collections.deque(self.source_nodes())
            eq = _scope_dict_inner(self, node_queue, None, node_to_children,
                                   result)

            # Sanity check
            if validate and len(eq) != 0:
                raise RuntimeError("Leftover nodes in queue: {}".format(eq))

            # Cache result
            if node_to_children:
                self._scope_dict_tochildren_cached = result
            else:
                self._scope_dict_toparent_cached = result

            result = copy.copy(result)

        if return_ids:
            return _scope_dict_to_ids(self, result)
        return result

    def scope_subgraph(self, entry_node, include_entry=True,
                       include_exit=True):
        return _scope_subgraph(self, entry_node, include_entry, include_exit)

    def top_level_transients(self):
        """Iterate over top-level transients of this state."""
        return top_level_transients(self)  # Free function

    def all_transients(self):
        """Iterate over all transients in this state."""
        return all_transients(self)

    def entry_node(self, node):
        """ Returns the scope entry node of the given node, or None if
            top-level. """
        return self.scope_dict(False)[node]

    def exit_nodes(self, entry_node):
        """ Returns the exit node leaving the context opened by
            the given entry node. """

        if not isinstance(entry_node, nd.EntryNode):
            raise TypeError(
                "Received {}: should be dace.nodes.EntryNode".format(
                    type(entry_node).__name__))

        node_to_children = self.scope_dict(True)
        return [
            v for v in node_to_children[entry_node]
            if isinstance(v, nd.ExitNode)
        ]

    # Dynamic SDFG creation API
    ##############################
    def add_read(self, array_or_stream_name: str,
                 debuginfo=None) -> nd.AccessNode:
        """ Adds a read-only access node to this SDFG state.
            :param array_or_stream_name: The name of the array/stream.
            :return: An array access node.
        """
        debuginfo = getdebuginfo(debuginfo)
        node = nd.AccessNode(
            array_or_stream_name,
            dtypes.AccessType.ReadOnly,
            debuginfo=debuginfo)
        self.add_node(node)
        return node

    def add_write(self, array_or_stream_name: str,
                  debuginfo=None) -> nd.AccessNode:
        """ Adds a write-only access node to this SDFG state.
            :param array_or_stream_name: The name of the array/stream.
            :return: An array access node.
        """
        debuginfo = getdebuginfo(debuginfo)
        node = nd.AccessNode(
            array_or_stream_name,
            dtypes.AccessType.WriteOnly,
            debuginfo=debuginfo)
        self.add_node(node)
        return node

    def add_access(self, array_or_stream_name: str,
                   debuginfo=None) -> nd.AccessNode:
        """ Adds a general (read/write) access node to this SDFG state.
            :param array_or_stream_name: The name of the array/stream.
            :return: An array access node.
        """
        debuginfo = getdebuginfo(debuginfo)
        node = nd.AccessNode(
            array_or_stream_name,
            dtypes.AccessType.ReadWrite,
            debuginfo=debuginfo)
        self.add_node(node)
        return node

    def add_tasklet(
            self,
            name: str,
            inputs: Set[str],
            outputs: Set[str],
            code: str,
            language: dtypes.Language = dtypes.Language.Python,
            code_global: str = "",
            code_init: str = "",
            code_exit: str = "",
            location: str = "-1",
            debuginfo=None,
    ):
        """ Adds a tasklet to the SDFG state. """
        debuginfo = getdebuginfo(debuginfo)
        tasklet = nd.Tasklet(
            name,
            inputs,
            outputs,
            code,
            language,
            code_global=code_global,
            code_init=code_init,
            code_exit=code_exit,
            location=location,
            debuginfo=debuginfo,
        )
        self.add_node(tasklet)
        return tasklet

    def add_nested_sdfg(
            self,
            sdfg: SDFG,
            parent,
            inputs: Set[str],
            outputs: Set[str],
            name=None,
            schedule=dtypes.ScheduleType.Default,
            location="-1",
            debuginfo=None,
    ):
        """ Adds a nested SDFG to the SDFG state. """
        if name is None:
            name = sdfg.label
        debuginfo = getdebuginfo(debuginfo)

        sdfg.parent = self
        sdfg._parent_sdfg = self.parent

        sdfg.update_sdfg_list([])

        s = nd.NestedSDFG(
            name,
            sdfg,
            inputs,
            outputs,
            schedule=schedule,
            location=location,
            debuginfo=debuginfo,
        )
        self.add_node(s)
        return s

    def _map_from_ndrange(self,
                          name,
                          schedule,
                          unroll,
                          ndrange,
                          debuginfo=None):
        # Input can either be a dictionary or a list of pairs
        if isinstance(ndrange, list):
            params = [k for k, v in ndrange]
            ndrange = {k: v for k, v in ndrange}
        else:
            params = list(ndrange.keys())

        map_range = properties.SubsetProperty.from_string(", ".join(
            [ndrange[p] for p in params]))
        map = nd.Map(
            name, params, map_range, schedule, unroll, debuginfo=debuginfo)
        return map

    def add_map(
            self,
            name,
            ndrange: Dict[str, str],
            schedule=dtypes.ScheduleType.Default,
            unroll=False,
            debuginfo=None,
    ) -> Tuple[nd.MapEntry, nd.MapExit]:
        """ Adds a map entry and map exit.
            :param name:      Map label
            :param ndrange:   Mapping between range variable names and their
                              subsets (parsed from strings)
            :param schedule:  Map schedule type
            :param unroll:    True if should unroll the map in code generation

            :return: (map_entry, map_exit) node 2-tuple
        """
        debuginfo = getdebuginfo(debuginfo)
        map = self._map_from_ndrange(
            name, schedule, unroll, ndrange, debuginfo=debuginfo)
        map_entry = nd.MapEntry(map)
        map_exit = nd.MapExit(map)
        self.add_nodes_from([map_entry, map_exit])
        return map_entry, map_exit

    def add_consume(
            self,
            name,
            elements: Tuple[str, str],
            condition: str = None,
            schedule=dtypes.ScheduleType.Default,
            chunksize=1,
            debuginfo=None,
    ) -> Tuple[nd.ConsumeEntry, nd.ConsumeExit]:
        """ Adds consume entry and consume exit nodes.
            :param name:      Label
            :param elements:  A 2-tuple signifying the processing element
                              index and number of total processing elements
            :param condition: Quiescence condition to finish consuming, or
                              None (by default) to consume until the stream
                              is empty for the first time. If false, will
                              consume forever.
            :param schedule:  Consume schedule type
            :param chunksize: Maximal number of elements to consume at a time

            :return: (consume_entry, consume_exit) node 2-tuple
        """
        if len(elements) != 2:
            raise TypeError("Elements must be a 2-tuple of "
                            "(PE_index, num_PEs)")
        pe_tuple = (elements[0],
                    properties.SymbolicProperty.from_string(elements[1]))

        debuginfo = getdebuginfo(debuginfo)
        consume = nd.Consume(
            name,
            pe_tuple,
            condition,
            schedule,
            chunksize,
            debuginfo=debuginfo)
        entry = nd.ConsumeEntry(consume)
        exit = nd.ConsumeExit(consume)

        self.add_nodes_from([entry, exit])
        return entry, exit

    def add_mapped_tasklet(
            self,
            name: str,
            map_ranges: Dict[str, sbs.Subset],
            inputs: Dict[str, mm.Memlet],
            code: str,
            outputs: Dict[str, mm.Memlet],
            schedule=dtypes.ScheduleType.Default,
            unroll_map=False,
            code_global="",
            code_init="",
            code_exit="",
            location="-1",
            language=dtypes.Language.Python,
            debuginfo=None,
            external_edges=False,
    ) -> Tuple[nd.Tasklet, nd.MapEntry, nd.MapExit]:
        """ Convenience function that adds a map entry, tasklet, map exit,
            and the respective edges to external arrays.
            :param name:       Tasklet (and wrapping map) name
            :param map_ranges: Mapping between variable names and their
                               subsets
            :param inputs:     Mapping between input local variable names and
                               their memlets
            :param code:       Code (written in `language`)
            :param outputs:    Mapping between output local variable names and
                               their memlets
            :param schedule:   Map schedule
            :param unroll_map: True if map should be unrolled in code
                               generation
            :param code_global: (optional) Global code (outside functions)
            :param language:   Programming language in which the code is
                               written
            :param debuginfo:  Debugging information (mostly for DIODE)
            :param external_edges: Create external access nodes and connect
                                   them with memlets automatically

            :return: tuple of (tasklet, map_entry, map_exit)
        """
        map_name = name + "_map"
        debuginfo = getdebuginfo(debuginfo)
        tasklet = nd.Tasklet(
            name,
            set(inputs.keys()),
            set(outputs.keys()),
            code,
            language=language,
            code_global=code_global,
            code_init=code_init,
            code_exit=code_exit,
            location=location,
            debuginfo=debuginfo,
        )
        map = self._map_from_ndrange(
            map_name, schedule, unroll_map, map_ranges, debuginfo=debuginfo)
        map_entry = nd.MapEntry(map)
        map_exit = nd.MapExit(map)
        self.add_nodes_from([map_entry, tasklet, map_exit])

        # Create access nodes
        if external_edges:
            input_data = set(memlet.data for memlet in inputs.values())
            output_data = set(memlet.data for memlet in outputs.values())
            inpdict = {}
            outdict = {}
            for inp in input_data:
                inpdict[inp] = self.add_read(inp)
            for out in output_data:
                outdict[out] = self.add_write(out)

        # Connect inputs from map to tasklet
        tomemlet = {}
        for name, memlet in inputs.items():
            # Set memlet local name
            memlet.name = name
            # Add internal memlet edge
            self.add_edge(map_entry, None, tasklet, name, memlet)
            tomemlet[memlet.data] = memlet

        # If there are no inputs, add empty memlet
        if len(inputs) == 0:
            self.add_edge(map_entry, None, tasklet, None, mm.EmptyMemlet())

        if external_edges:
            for inp, inpnode in inpdict.items():
                # Add external edge
                outer_memlet = propagate_memlet(self, tomemlet[inp], map_entry,
                                                True)
                self.add_edge(inpnode, None, map_entry, "IN_" + inp,
                              outer_memlet)

                # Add connectors to internal edges
                for e in self.out_edges(map_entry):
                    if e.data.data == inp:
                        e._src_conn = "OUT_" + inp

                # Add connectors to map entry
                map_entry.add_in_connector("IN_" + inp)
                map_entry.add_out_connector("OUT_" + inp)

        # Connect outputs from tasklet to map
        tomemlet = {}
        for name, memlet in outputs.items():
            # Set memlet local name
            memlet.name = name
            # Add internal memlet edge
            self.add_edge(tasklet, name, map_exit, None, memlet)
            tomemlet[memlet.data] = memlet

        # If there are no outputs, add empty memlet
        if len(outputs) == 0:
            self.add_edge(tasklet, None, map_exit, None, mm.EmptyMemlet())

        if external_edges:
            for out, outnode in outdict.items():
                # Add external edge
                outer_memlet = propagate_memlet(self, tomemlet[out], map_exit,
                                                True)
                self.add_edge(map_exit, "OUT_" + out, outnode, None,
                              outer_memlet)

                # Add connectors to internal edges
                for e in self.in_edges(map_exit):
                    if e.data.data == out:
                        e._dst_conn = "IN_" + out

                # Add connectors to map entry
                map_exit.add_in_connector("IN_" + out)
                map_exit.add_out_connector("OUT_" + out)

        return tasklet, map_entry, map_exit

    def add_reduce(
            self,
            wcr,
            axes,
            wcr_identity=None,
            schedule=dtypes.ScheduleType.Default,
            debuginfo=None,
    ):
        """ Adds a reduction node.
            :param wcr: A lambda function representing the reduction operation
            :param axes: A tuple of axes to reduce the input memlet from, or
                         None for all axes
            :param wcr_identity: If not None, initializes output memlet values
                                 with this value
            :param schedule: Reduction schedule type

            :return: A Reduce node
        """
        debuginfo = getdebuginfo(debuginfo)
        result = nd.Reduce(
            wcr, axes, wcr_identity, schedule, debuginfo=debuginfo)
        self.add_node(result)
        return result

    def add_edge_pair(
            self,
            scope_node,
            internal_node,
            external_node,
            internal_memlet,
            external_memlet=None,
            scope_connector=None,
            internal_connector=None,
            external_connector=None,
    ):
        """ Adds two edges around a scope node (e.g., map entry, consume
            exit).

            The internal memlet (connecting to the internal node) has to be
            specified. If external_memlet (i.e., connecting to the node out
            of the scope) is not specified, it is propagated automatically
            using internal_memlet and the scope.

            :param scope_node: A scope node (for example, map exit) to add
                               edges around.
            :param internal_node: The node within the scope to connect to. If
                                  `scope_node` is an entry node, this means
                                  the node connected to the outgoing edge,
                                  else incoming.
            :param external_node: The node out of the scope to connect to.
            :param internal_memlet: The memlet on the edge to/from
                                    internal_node.
            :param external_memlet: The memlet on the edge to/from
                                    external_node (optional, will propagate
                                    internal_memlet if not specified).
            :param scope_connector: A scope connector name (or a unique
                                    number if not specified).
            :param internal_connector: The connector on internal_node to
                                       connect to.
            :param external_connector: The connector on external_node to
                                       connect to.
            :return: A 2-tuple representing the (internal, external) edges.
        """
        if not isinstance(scope_node, (nd.EntryNode, nd.ExitNode)):
            raise ValueError("scope_node is not a scope entry/exit")

        # Autodetermine scope connector ID
        if scope_connector is None:
            # Pick out numbered connectors that do not lead into the scope range
            conn_id = 1
            for conn in scope_node.in_connectors | scope_node.out_connectors:
                if conn.startswith("IN_") or conn.startswith("OUT_"):
                    conn_name = conn[conn.find("_") + 1:]
                    try:
                        cid = int(conn_name)
                        if cid >= conn_id:
                            conn_id = cid + 1
                    except (TypeError, ValueError):
                        pass
            scope_connector = str(conn_id)

        # Add connectors
        scope_node.add_in_connector("IN_" + scope_connector)
        scope_node.add_out_connector("OUT_" + scope_connector)
        ##########################

        # Add internal edge
        if isinstance(scope_node, nd.EntryNode):
            iedge = self.add_edge(
                scope_node,
                "OUT_" + scope_connector,
                internal_node,
                internal_connector,
                internal_memlet,
            )
        else:
            iedge = self.add_edge(
                internal_node,
                internal_connector,
                scope_node,
                "IN_" + scope_connector,
                internal_memlet,
            )

        # Add external edge
        if external_memlet is None:
            # If undefined, propagate
            external_memlet = propagate_memlet(self, internal_memlet,
                                               scope_node, True)

        if isinstance(scope_node, nd.EntryNode):
            eedge = self.add_edge(
                external_node,
                external_connector,
                scope_node,
                "IN_" + scope_connector,
                external_memlet,
            )
        else:
            eedge = self.add_edge(
                scope_node,
                "OUT_" + scope_connector,
                external_node,
                external_connector,
                external_memlet,
            )

        return (iedge, eedge)

    def add_memlet_path(self,
                        *path_nodes,
                        memlet=None,
                        src_conn=None,
                        dst_conn=None):
        """ Adds a path of memlet edges between the given nodes, propagating
            from the given innermost memlet.

            :param *path_nodes: Nodes participating in the path (in the given
                                order).
            :keyword memlet: (mandatory) The memlet at the innermost scope
                             (e.g., the incoming memlet to a tasklet (last
                             node), or an outgoing memlet from an array
                             (first node), followed by scope exits).
            :keyword src_conn: Connector at the beginning of the path.
            :keyword dst_conn: Connector at the end of the path.
        """
        if memlet is None:
            raise TypeError("Innermost memlet cannot be None")
        if len(path_nodes) < 2:
            raise ValueError("Memlet path must consist of at least 2 nodes")

        src_node = path_nodes[0]
        dst_node = path_nodes[-1]

        # Add edges first so that scopes can be understood
        edges = [
            self.add_edge(path_nodes[i], None, path_nodes[i + 1], None,
                          mm.EmptyMemlet())
            for i in range(len(path_nodes) - 1)
        ]

        if not isinstance(memlet, dace.memlet.Memlet):
            raise TypeError("Expected Memlet, got: {}".format(
                type(memlet).__name__))

        sdict = self.scope_dict(validate=False)
        if scope_contains_scope(sdict, src_node, dst_node):
            propagate_forward = False
        else:  # dst node's scope is higher than src node, propagate out
            propagate_forward = True

        # Innermost edge memlet
        cur_memlet = memlet

        # Verify that connectors exist
        if (not isinstance(memlet, dace.memlet.EmptyMemlet)
                and hasattr(edges[0].src, "out_connectors")
                and isinstance(edges[0].src, nd.CodeNode) and
            (src_conn is None or src_conn not in edges[0].src.out_connectors)):
            raise ValueError("Output connector {} does not exist in {}".format(
                src_conn, edges[0].src.label))
        if (not isinstance(memlet, dace.memlet.EmptyMemlet)
                and hasattr(edges[-1].dst, "in_connectors")
                and isinstance(edges[-1].dst, nd.CodeNode) and
            (dst_conn is None or dst_conn not in edges[-1].dst.in_connectors)):
            raise ValueError("Input connector {} does not exist in {}".format(
                dst_conn, edges[-1].dst.label))

        path = edges if propagate_forward else reversed(edges)
        # Propagate and add edges
        for i, edge in enumerate(path):
            # Figure out source and destination connectors
            if propagate_forward:
                sconn = src_conn if i == 0 else (
                    "OUT_" + edge.src.last_connector())
                dconn = (dst_conn if i == len(edges) - 1 else
                         ("IN_" + edge.dst.next_connector()))
            else:
                sconn = (src_conn if i == len(edges) - 1 else
                         ("OUT_" + edge.src.next_connector()))
                dconn = dst_conn if i == 0 else (
                    "IN_" + edge.dst.last_connector())

            if isinstance(cur_memlet, dace.memlet.EmptyMemlet):
                if propagate_forward:
                    sconn = src_conn if i == 0 else None
                    dconn = dst_conn if i == len(edges) - 1 else None
                else:
                    sconn = src_conn if i == len(edges) - 1 else None
                    dconn = dst_conn if i == 0 else None

            # Modify edge to match memlet path
            edge._src_conn = sconn
            edge._dst_conn = dconn
            edge._data = cur_memlet

            # Add connectors to edges
            if propagate_forward:
                if dconn is not None:
                    edge.dst.add_in_connector(dconn)
                if sconn is not None:
                    edge.src.add_out_connector(sconn)
            else:
                if dconn is not None:
                    edge.dst.add_in_connector(dconn)
                if sconn is not None:
                    edge.src.add_out_connector(sconn)

            # Propagate current memlet to produce the next one
            if i < len(edges) - 1:
                snode = edge.dst if propagate_forward else edge.src
                if not isinstance(cur_memlet, dace.memlet.EmptyMemlet):
                    cur_memlet = propagate_memlet(self, cur_memlet, snode,
                                                  True)

    # DEPRECATED FUNCTIONS
    ######################################
    def add_array(self,
                  name,
                  shape,
                  dtype,
                  storage=dtypes.StorageType.Default,
                  materialize_func=None,
                  transient=False,
                  strides=None,
                  offset=None,
                  toplevel=False,
                  debuginfo=None,
                  total_size=None,
                  find_new_name=False):
        """ @attention: This function is deprecated. """
        warnings.warn(
            'The "SDFGState.add_array" API is deprecated, please '
            'use "SDFG.add_array" and "SDFGState.add_access"',
            DeprecationWarning)
        # Workaround to allow this legacy API
        if name in self.parent._arrays:
            del self.parent._arrays[name]
        self.parent.add_array(
            name,
            shape,
            dtype,
            storage,
            materialize_func,
            transient,
            strides,
            offset,
            toplevel,
            debuginfo,
            find_new_name=find_new_name,
            total_size=total_size)
        return self.add_access(name, debuginfo)

    def add_stream(
            self,
            name,
            dtype,
            veclen=1,
            buffer_size=1,
            shape=(1, ),
            storage=dtypes.StorageType.Default,
            transient=False,
            offset=None,
            toplevel=False,
            debuginfo=None,
    ):
        """ @attention: This function is deprecated. """
        warnings.warn(
            'The "SDFGState.add_stream" API is deprecated, please '
            'use "SDFG.add_stream" and "SDFGState.add_access"',
            DeprecationWarning)
        # Workaround to allow this legacy API
        if name in self.parent._arrays:
            del self.parent._arrays[name]
        self.parent.add_stream(
            name,
            dtype,
            veclen,
            buffer_size,
            shape,
            storage,
            transient,
            offset,
            toplevel,
            debuginfo,
        )
        return self.add_access(name, debuginfo)

    def add_scalar(
            self,
            name,
            dtype,
            storage=dtypes.StorageType.Default,
            transient=False,
            toplevel=False,
            debuginfo=None,
    ):
        """ @attention: This function is deprecated. """
        warnings.warn(
            'The "SDFGState.add_scalar" API is deprecated, please '
            'use "SDFG.add_scalar" and "SDFGState.add_access"',
            DeprecationWarning)
        # Workaround to allow this legacy API
        if name in self.parent._arrays:
            del self.parent._arrays[name]
        self.parent.add_scalar(name, dtype, storage, transient, toplevel,
                               debuginfo)
        return self.add_access(name, debuginfo)

    def add_transient(self,
                      name,
                      shape,
                      dtype,
                      storage=dtypes.StorageType.Default,
                      materialize_func=None,
                      strides=None,
                      offset=None,
                      toplevel=False,
                      debuginfo=None,
                      total_size=None):
        """ @attention: This function is deprecated. """
        return self.add_array(
            name,
            shape,
            dtype,
            storage,
            materialize_func,
            True,
            strides,
            offset,
            toplevel,
            debuginfo,
            total_size=total_size)

    # SDFG queries
    ######################################
    def find_node(self, node_id_or_label):
        """ Finds a node according to its ID (if integer is
            provided) or label (if string is provided).

            :param node_id_or_label  Node ID (if int) or label (if str)
            :return A nodes.Node object
        """

        if isinstance(node_id_or_label, str):
            for n in self.nodes():
                if n.label == node_id_or_label:
                    return n
            raise LookupError("Node %s not found" % node_id_or_label)
        elif isinstance(node_id_or_label, int):
            return self.nodes()[node_id_or_label]
        else:
            raise TypeError("node_id_or_label is not an int nor string")

    def is_empty(self):
        return len([
            n for n in self.nodes() if not isinstance(n, nd.EmptyTasklet)
        ]) == 0

    def fill_scope_connectors(self):
        """ Creates new "IN_%d" and "OUT_%d" connectors on each scope entry
            and exit, depending on array names. """
        for nid, node in enumerate(self.nodes()):
            ####################################################
            # Add connectors to scope entries
            if isinstance(node, nd.EntryNode):
                # Find current number of input connectors
                num_inputs = len([
                    e for e in self.in_edges(node)
                    if e.dst_conn is not None and e.dst_conn.startswith("IN_")
                ])

                conn_to_data = {}

                # Append input connectors and get mapping of connectors to data
                for edge in self.in_edges(node):
                    if edge.dst_conn is not None and edge.dst_conn.startswith(
                            "IN_"):
                        conn_to_data[edge.data.data] = edge.dst_conn[3:]

                    # We're only interested in edges without connectors
                    if edge.dst_conn is not None or edge.data.data is None:
                        continue
                    edge._dst_conn = "IN_" + str(num_inputs + 1)
                    node._in_connectors.add(edge.dst_conn)
                    conn_to_data[edge.data.data] = num_inputs + 1

                    num_inputs += 1

                # Set the corresponding output connectors
                for edge in self.out_edges(node):
                    if edge.src_conn is not None:
                        continue
                    if edge.data.data is None:
                        continue
                    edge._src_conn = "OUT_" + str(conn_to_data[edge.data.data])
                    node._out_connectors.add(edge.src_conn)
            ####################################################
            # Same treatment for scope exits
            if isinstance(node, nd.ExitNode):
                # Find current number of output connectors
                num_outputs = len([
                    e for e in self.out_edges(node)
                    if e.src_conn is not None and e.src_conn.startswith("OUT_")
                ])

                conn_to_data = {}

                # Append output connectors and get mapping of connectors to data
                for edge in self.out_edges(node):
                    if edge.src_conn is not None and edge.src_conn.startswith(
                            "OUT_"):
                        conn_to_data[edge.data.data] = edge.src_conn[4:]

                    # We're only interested in edges without connectors
                    if edge.src_conn is not None or edge.data.data is None:
                        continue
                    edge._src_conn = "OUT_" + str(num_outputs + 1)
                    node._out_connectors.add(edge.src_conn)
                    conn_to_data[edge.data.data] = num_outputs + 1

                    num_outputs += 1

                # Set the corresponding input connectors
                for edge in self.in_edges(node):
                    if edge.dst_conn is not None:
                        continue
                    if edge.data.data is None:
                        continue
                    edge._dst_conn = "IN_" + str(conn_to_data[edge.data.data])
                    node._in_connectors.add(edge.dst_conn)

    def validate(self, sdfg, state_id) -> None:
        """ Verifies the correctness of an SDFG state by applying multiple
            tests. Raises an InvalidSDFGError with the erroneous node on
            failure.
        """
        if not validate_name(self._label):
            raise InvalidSDFGError("Invalid state name", sdfg, state_id)

        if self._parent != sdfg:
            raise InvalidSDFGError(
                "State does not point to the correct "
                "parent", sdfg, state_id)

        # Used in memlet validation
        if dace.Config.get_bool('experimental', 'validate_undefs'):
            scope_tree = self.scope_tree()

        # Unreachable
        ########################################
        if (sdfg.number_of_nodes() > 1 and sdfg.in_degree(self) == 0
                and sdfg.out_degree(self) == 0):
            raise InvalidSDFGError("Unreachable state", sdfg, state_id)

        for nid, node in enumerate(self.nodes()):
            # Node validation
            try:
                node.validate(sdfg, self)
            except InvalidSDFGError:
                raise
            except Exception as ex:
                raise InvalidSDFGNodeError(
                    "Node validation failed: " + str(ex), sdfg, state_id, nid)

            # Isolated nodes
            ########################################
            if self.in_degree(node) + self.out_degree(node) == 0:
                # One corner case: OK if this is a code node
                if isinstance(node, nd.CodeNode):
                    pass
                else:
                    raise InvalidSDFGNodeError("Isolated node", sdfg, state_id,
                                               nid)

            # Scope tests
            ########################################
            if isinstance(node, nd.EntryNode):
                if len(self.exit_nodes(node)) == 0:
                    raise InvalidSDFGNodeError(
                        "Entry node does not have matching "
                        "exit node",
                        sdfg,
                        state_id,
                        nid,
                    )

            if isinstance(node, (nd.EntryNode, nd.ExitNode)):
                for iconn in node.in_connectors:
                    if (iconn is not None and iconn.startswith("IN_") and
                        ("OUT_" + iconn[3:]) not in node.out_connectors):
                        raise InvalidSDFGNodeError(
                            "No match for input connector %s in output "
                            "connectors" % iconn,
                            sdfg,
                            state_id,
                            nid,
                        )
                for oconn in node.out_connectors:
                    if (oconn is not None and oconn.startswith("OUT_")
                            and ("IN_" + oconn[4:]) not in node.in_connectors):
                        raise InvalidSDFGNodeError(
                            "No match for output connector %s in input "
                            "connectors" % oconn,
                            sdfg,
                            state_id,
                            nid,
                        )

            # Node-specific tests
            ########################################
            if isinstance(node, nd.AccessNode):
                if node.data not in sdfg.arrays:
                    raise InvalidSDFGNodeError(
                        "Access node must point to a valid array name in the SDFG",
                        sdfg,
                        state_id,
                        nid,
                    )

                # Find uninitialized transients
                arr = sdfg.arrays[node.data]
                if (arr.transient and self.in_degree(node) == 0
                        and self.out_degree(node) > 0):
                    # Find other instances of node in predecessor states
                    states = sdfg.predecessor_states(self)
                    input_found = False
                    for state in states:
                        for onode in state.nodes():
                            if (isinstance(onode, nd.AccessNode)
                                    and onode.data == node.data):
                                if state.in_degree(onode) > 0:
                                    input_found = True
                                    break
                        if input_found:
                            break
                    if not input_found and node.setzero == False:
                        warnings.warn(
                            'WARNING: Use of uninitialized transient "%s" in state %s'
                            % (node.data, self.label))

            if isinstance(node,
                          nd.Reduce) and (len(self.in_edges(node)) != 1
                                          or len(self.out_edges(node)) != 1):
                raise InvalidSDFGNodeError(
                    "Reduce node must have exactly one input and output edges",
                    sdfg,
                    state_id,
                    nid,
                )

            if (isinstance(node, nd.ConsumeEntry)
                    and "IN_stream" not in node.in_connectors):
                raise InvalidSDFGNodeError(
                    "Consume entry node must have an input stream", sdfg,
                    state_id, nid)
            if (isinstance(node, nd.ConsumeEntry)
                    and "OUT_stream" not in node.out_connectors):
                raise InvalidSDFGNodeError(
                    "Consume entry node must have an internal stream",
                    sdfg,
                    state_id,
                    nid,
                )

            # Connector tests
            ########################################
            # Check for duplicate connector names (unless it's a nested SDFG)
            if (len(node.in_connectors & node.out_connectors) > 0
                    and not isinstance(node, nd.NestedSDFG)):
                dups = node.in_connectors & node.out_connectors
                raise InvalidSDFGNodeError(
                    "Duplicate connectors: " + str(dups), sdfg, state_id, nid)

            # Check for dangling connectors (incoming)
            for conn in node.in_connectors:
                incoming_edges = 0
                for e in self.in_edges(node):
                    # Connector found
                    if e.dst_conn == conn:
                        incoming_edges += 1

                if incoming_edges == 0:
                    raise InvalidSDFGNodeError(
                        "Dangling in-connector %s" % conn, sdfg, state_id, nid)
                # Connectors may have only one incoming edge
                # Due to input connectors of scope exit, this is only correct
                # in some cases:
                if incoming_edges > 1 and not isinstance(node, nd.ExitNode):
                    raise InvalidSDFGNodeError(
                        "Connector %s cannot have more "
                        "than one incoming edge, found %d" % (conn,
                                                              incoming_edges),
                        sdfg,
                        state_id,
                        nid,
                    )

            # Check for dangling connectors (outgoing)
            for conn in node.out_connectors:
                outgoing_edges = 0
                for e in self.out_edges(node):
                    # Connector found
                    if e.src_conn == conn:
                        outgoing_edges += 1

                if outgoing_edges == 0:
                    raise InvalidSDFGNodeError(
                        "Dangling out-connector %s" % conn, sdfg, state_id,
                        nid)

                # In case of scope exit, only one outgoing edge per connector
                # is allowed.
                if outgoing_edges > 1 and isinstance(node, nd.ExitNode):
                    raise InvalidSDFGNodeError(
                        "Connector %s cannot have more "
                        "than one outgoing edge, found %d" % (conn,
                                                              outgoing_edges),
                        sdfg,
                        state_id,
                        nid,
                    )

            # Check for edges to nonexistent connectors
            for e in self.in_edges(node):
                if e.dst_conn is not None and e.dst_conn not in node.in_connectors:
                    raise InvalidSDFGNodeError(
                        ("Memlet %s leading to " + "nonexistent connector %s")
                        % (str(e.data), e.dst_conn),
                        sdfg,
                        state_id,
                        nid,
                    )
            for e in self.out_edges(node):
                if e.src_conn is not None and e.src_conn not in node.out_connectors:
                    raise InvalidSDFGNodeError(
                        ("Memlet %s coming from " + "nonexistent connector %s")
                        % (str(e.data), e.src_conn),
                        sdfg,
                        state_id,
                        nid,
                    )
            ########################################

        # Memlet checks
        scope = self.scope_dict()
        for eid, e in enumerate(self.edges()):
            # Edge validation
            try:
                e.data.validate(sdfg, self)
            except InvalidSDFGError:
                raise
            except Exception as ex:
                raise InvalidSDFGEdgeError(
                    "Edge validation failed: " + str(ex), sdfg, state_id, eid)

            # For every memlet, obtain its full path in the DFG
            path = self.memlet_path(e)
            src_node = path[0].src
            dst_node = path[-1].dst

            # Check if memlet data matches src or dst nodes
            if (e.data.data is not None
                    and (isinstance(src_node, nd.AccessNode)
                         or isinstance(dst_node, nd.AccessNode))
                    and (not isinstance(src_node, nd.AccessNode)
                         or e.data.data != src_node.data)
                    and (not isinstance(dst_node, nd.AccessNode)
                         or e.data.data != dst_node.data)):
                raise InvalidSDFGEdgeError(
                    "Memlet data does not match source or destination "
                    "data nodes)",
                    sdfg,
                    state_id,
                    eid,
                )

            # Check memlet subset validity with respect to source/destination nodes
            if e.data.data is not None and e.data.allow_oob == False:
                subset_node = (dst_node if isinstance(dst_node, nd.AccessNode)
                               and e.data.data == dst_node.data else src_node)
                other_subset_node = (dst_node
                                     if isinstance(dst_node, nd.AccessNode)
                                     and e.data.data != dst_node.data else
                                     src_node)

                if isinstance(subset_node, nd.AccessNode):
                    arr = sdfg.arrays[subset_node.data]
                    # Dimensionality
                    if e.data.subset.dims() != len(arr.shape):
                        raise InvalidSDFGEdgeError(
                            "Memlet subset does not match node dimension "
                            "(expected %d, got %d)" % (len(arr.shape),
                                                       e.data.subset.dims()),
                            sdfg,
                            state_id,
                            eid,
                        )

                    # Bounds
                    if any(((minel + off) < 0) == True for minel, off in zip(
                            e.data.subset.min_element(), arr.offset)):
                        raise InvalidSDFGEdgeError(
                            "Memlet subset negative out-of-bounds", sdfg,
                            state_id, eid)
                    if any(((maxel + off) >= s) == True
                           for maxel, s, off in zip(
                               e.data.subset.max_element(), arr.shape,
                               arr.offset)):
                        raise InvalidSDFGEdgeError(
                            "Memlet subset out-of-bounds", sdfg, state_id, eid)
                # Test other_subset as well
                if e.data.other_subset is not None and isinstance(
                        other_subset_node, nd.AccessNode):
                    arr = sdfg.arrays[other_subset_node.data]
                    # Dimensionality
                    if e.data.other_subset.dims() != len(arr.shape):
                        raise InvalidSDFGEdgeError(
                            "Memlet other_subset does not match node dimension "
                            "(expected %d, got %d)" % (len(
                                arr.shape), e.data.other_subset.dims()),
                            sdfg,
                            state_id,
                            eid,
                        )

                    # Bounds
                    if any(((minel + off) < 0) == True for minel, off in zip(
                            e.data.other_subset.min_element(), arr.offset)):
                        raise InvalidSDFGEdgeError(
                            "Memlet other_subset negative out-of-bounds",
                            sdfg,
                            state_id,
                            eid,
                        )
                    if any(((maxel + off) >= s) == True
                           for maxel, s, off in zip(
                               e.data.other_subset.max_element(), arr.shape,
                               arr.offset)):
                        raise InvalidSDFGEdgeError(
                            "Memlet other_subset out-of-bounds", sdfg,
                            state_id, eid)

                # Test subset and other_subset for undefined symbols
                if dace.Config.get_bool('experimental', 'validate_undefs'):
                    defined_symbols = set(
                        map(str, scope_tree[scope[e.dst]].defined_vars))
                    undefs = (e.data.subset.free_symbols - defined_symbols)
                    if len(undefs) > 0:
                        raise InvalidSDFGEdgeError(
                            'Undefined symbols %s found in memlet subset' %
                            undefs, sdfg, state_id, eid)
                    if e.data.other_subset is not None:
                        undefs = (
                            e.data.other_subset.free_symbols - defined_symbols)
                        if len(undefs) > 0:
                            raise InvalidSDFGEdgeError(
                                'Undefined symbols %s found in memlet '
                                'other_subset' % undefs, sdfg, state_id, eid)
            #######################################

            # Memlet path scope lifetime checks
            # If scope(src) == scope(dst): OK
            if scope[src_node] == scope[dst_node] or src_node == scope[dst_node]:
                pass
            # If scope(src) contains scope(dst), then src must be a data node
            elif scope_contains_scope(scope, src_node, dst_node):
                if not isinstance(src_node, nd.AccessNode):
                    raise InvalidSDFGEdgeError(
                        "Memlet creates an "
                        "invalid path (source node %s should "
                        "be a data node)" % str(src_node),
                        sdfg,
                        state_id,
                        eid,
                    )
            # If scope(dst) contains scope(src), then dst must be a data node
            elif scope_contains_scope(scope, dst_node, src_node):
                if not isinstance(dst_node, nd.AccessNode):
                    raise InvalidSDFGEdgeError(
                        "Memlet creates an "
                        "invalid path (sink node %s should "
                        "be a data node)" % str(dst_node),
                        sdfg,
                        state_id,
                        eid,
                    )
            # If scope(dst) is disjoint from scope(src), it's an illegal memlet
            else:
                raise InvalidSDFGEdgeError(
                    "Illegal memlet between disjoint scopes", sdfg, state_id,
                    eid)

            # Check dimensionality of memory access
            if isinstance(e.data.subset, (sbs.Range, sbs.Indices)):
                if e.data.subset.dims() != len(sdfg.arrays[e.data.data].shape):
                    raise InvalidSDFGEdgeError(
                        "Memlet subset uses the wrong dimensions"
                        " (%dD for a %dD data node)" %
                        (e.data.subset.dims(),
                         len(sdfg.arrays[e.data.data].shape)),
                        sdfg,
                        state_id,
                        eid,
                    )

            # Verify that source and destination subsets contain the same
            # number of elements
            if e.data.other_subset is not None and not (
                (isinstance(src_node, nd.AccessNode)
                 and isinstance(sdfg.arrays[src_node.data], dt.Stream)) or
                (isinstance(dst_node, nd.AccessNode)
                 and isinstance(sdfg.arrays[dst_node.data], dt.Stream))):
                if (e.data.subset.num_elements() !=
                        e.data.other_subset.num_elements()):
                    raise InvalidSDFGEdgeError(
                        'Dimensionality mismatch between src/dst subsets',
                        sdfg, state_id, eid)
        ########################################


def scope_contains_scope(sdict, node, other_node):
    """ Returns true iff scope of `node` contains the scope of  `other_node`.
    """
    curnode = other_node
    nodescope = sdict[node]
    while curnode is not None:
        curnode = sdict[curnode]
        if curnode == nodescope:
            return True
    return False


def find_input_arraynode(graph, edge):
    result = graph.memlet_path(edge)[0]
    if not isinstance(result.src, nd.AccessNode):
        raise RuntimeError("Input array node not found for memlet " +
                           str(edge.data))
    return result.src


def find_output_arraynode(graph, edge):
    result = graph.memlet_path(edge)[-1]
    if not isinstance(result.dst, nd.AccessNode):
        raise RuntimeError("Output array node not found for memlet " +
                           str(edge.data))
    return result.dst


def _scope_subgraph(graph, entry_node, include_entry, include_exit):
    if not isinstance(entry_node, nd.EntryNode):
        raise TypeError("Received {}: should be dace.nodes.EntryNode".format(
            type(entry_node).__name__))
    node_to_children = graph.scope_dict(True)
    if include_exit:
        children_nodes = set(node_to_children[entry_node])
    else:
        # Assume the last node in the scope list is the exit node
        children_nodes = set(node_to_children[entry_node][:-1])
    map_nodes = [
        node for node in children_nodes if isinstance(node, nd.EntryNode)
    ]
    while len(map_nodes) > 0:
        next_map_nodes = []
        # Traverse children map nodes
        for map_node in map_nodes:
            # Get child map subgraph (1 level)
            more_nodes = set(node_to_children[map_node])
            # Unionize children_nodes with new nodes
            children_nodes |= more_nodes
            # Add nodes of the next level to next_map_nodes
            next_map_nodes.extend([
                node for node in more_nodes if isinstance(node, nd.EntryNode)
            ])
        map_nodes = next_map_nodes

    if include_entry:
        children_nodes.add(entry_node)

    # Preserve order of nodes
    return ScopeSubgraphView(graph,
                             [n for n in graph.nodes() if n in children_nodes])


def _scope_dict_inner(graph, node_queue, current_scope, node_to_children,
                      result):
    """ Returns a queue of nodes that are external to the current scope. """
    # Initialize an empty list, if necessary
    if node_to_children and current_scope not in result:
        result[current_scope] = []

    external_queue = collections.deque()

    visited = set()
    while len(node_queue) > 0:
        node = node_queue.popleft()

        # If this node has been visited already, skip it
        if node in visited:
            continue
        visited.add(node)

        # Set the node parent (or its parent's children)
        if not node_to_children:
            result[node] = current_scope
        else:
            result[current_scope].append(node)

        successors = [n for n in graph.successors(node) if n not in visited]

        # If this is an Entry Node, we need to recurse further
        if isinstance(node, nd.EntryNode):
            node_queue.extend(
                _scope_dict_inner(graph, collections.deque(successors), node,
                                  node_to_children, result))
        # If this is an Exit Node, we push the successors to the external
        # queue
        elif isinstance(node, nd.ExitNode):
            external_queue.extend(successors)
        # Otherwise, it is a plain node, and we push its successors to the
        # same queue
        else:
            node_queue.extend(successors)

    return external_queue


def _scope_dict_to_ids(state: SDFGState, scope_dict: Dict[Any, List[Any]]):
    """ Return a JSON-serializable dictionary of a scope dictionary,
        using integral node IDs instead of object references. """

    def node_id_or_none(node):
        if node is None: return -1
        return state.node_id(node)

    return {
        node_id_or_none(k): [node_id_or_none(vi) for vi in v]
        for k, v in scope_dict.items()
    }


def concurrent_subgraphs(graph):
    """ Finds subgraphs of an SDFGState or ScopeSubgraphView that can
        run concurrently. """
    if not (isinstance(graph, SDFGState)
            or isinstance(graph, ScopeSubgraphView)):
        raise TypeError(
            "Expected SDFGState or ScopeSubgraphView, got: {}".format(
                type(graph).__name__))
    candidates = graph.source_nodes()
    components = collections.OrderedDict()  # {start node: nodes in component}
    for cand in candidates:
        if isinstance(cand, dace.graph.nodes.AccessNode):
            # AccessNodes can be read from multiple concurrent components, so
            # check all out edges
            start_nodes = [e.dst for e in graph.out_edges(cand)]
            for n in start_nodes:
                if n not in components:
                    components[n] = {cand, n}
                else:
                    # Components can read from multiple start arrays
                    components[n].add(cand)
        else:
            # The source node == the first control or compute node
            components[cand] = {cand}
    subgraphs = []  # [{nodes in subgraph}]
    for i, start_node in enumerate(components):
        # Do BFS and find all nodes reachable from this start node
        seen = set()
        to_search = [start_node]
        while len(to_search) > 0:
            node = to_search.pop()
            if node in seen:
                continue
            seen.add(node)
            for e in graph.out_edges(node):
                if e.dst not in seen:
                    to_search.append(e.dst)
        # If this component overlaps with any previously determined components,
        # fuse them
        for other in subgraphs:
            if len(other & seen) > 0:
                # Add both traversed node and potential data source nodes
                other |= seen | components[start_node]
                break
        else:
            # If there was no overlap, this is a concurrent subgraph
            subgraphs.append(seen | components[start_node])
    # Now stick each of the found components in a ScopeSubgraphView and return
    # them. Sort according to original order of nodes
    all_nodes = graph.nodes()
    return [
        ScopeSubgraphView(graph, [n for n in all_nodes if n in sg])
        for sg in subgraphs
    ]


def scope_symbols(dfg):
    """ Returns all symbols used in scopes within the given DFG, separated
        into (iteration variables, symbols used in subsets). """
    iteration_variables = collections.OrderedDict()
    subset_symbols = collections.OrderedDict()
    sdict = dfg.scope_dict()
    for n in dfg.nodes():
        # TODO(later): Refactor to method on Node objects
        if isinstance(n, dace.graph.nodes.NestedSDFG):
            iv, ss = n.sdfg.scope_symbols()
            iteration_variables.update(iv)
            subset_symbols.update(ss)
            continue
        if not isinstance(n, dace.graph.nodes.EntryNode):
            continue
        if isinstance(n, dace.graph.nodes.MapEntry):
            # Collect dynamic map range symbols from parent scopes
            dynamic_symbols = set()
            parent = n
            while parent is not None:
                dynamic_symbols |= parent.in_connectors
                parent = sdict[parent]

            for param in n.params:
                iteration_variables[param] = dt.Scalar(
                    symbolic.symbol(param).dtype)
            for dim in n.map.range:
                try:
                    for i in dim:
                        if isinstance(i, sp.Expr):
                            subset_symbols.update(
                                (k.name, dt.Scalar(k.dtype))
                                for k in i.free_symbols
                                if k.name not in dynamic_symbols)
                except TypeError:  # X object is not iterable
                    if isinstance(dim, sp.Expr):
                        subset_symbols.update((k.name, dt.Scalar(k.dtype))
                                              for k in dim.free_symbols
                                              if k.name not in dynamic_symbols)
                    else:
                        raise TypeError(
                            "Unexpected map range type for {}: {}".format(
                                n.map,
                                type(n.map.range).__name__))
        elif isinstance(n, dace.graph.nodes.ConsumeEntry):
            # Collect dynamic map range symbols from parent scopes
            dynamic_symbols = set()
            parent = n
            while parent is not None:
                dynamic_symbols |= parent.in_connectors
                parent = sdict[parent]

            # Add PE index as iteration variable
            iteration_variables[n.consume.pe_index] = dt.Scalar(
                symbolic.symbol(n.consume.pe_index).dtype)
            if isinstance(n.consume.num_pes, sp.Expr):
                subset_symbols.update((k.name, dt.Scalar(k.dtype))
                                      for k in n.consume.num_pes.free_symbols
                                      if k.name not in dynamic_symbols)
        else:
            raise TypeError("Unsupported entry node type: {}".format(
                type(n).__name__))
    return iteration_variables, subset_symbols


def data_symbols(dfg):
    """ Returns all symbols used in data nodes within the specified DFG. """
    sdfg = dfg.parent
    result = collections.OrderedDict()
    # Scalars determining the size of arrays
    for d in dfg.nodes():
        # Update symbols with symbols in nested SDFGs
        if isinstance(d, nd.NestedSDFG):
            result.update(d.sdfg.data_symbols(True))
            continue
        if not isinstance(d, nd.AccessNode):
            continue
        ddesc = d.desc(sdfg)
        for s in itertools.chain(ddesc.shape, ddesc.strides, ddesc.offset):
            if isinstance(s, sp.Expr):
                result.update((k.name, dt.Scalar(k.dtype))
                              for k in s.free_symbols
                              if not k.name.startswith("__dace"))
    return result


def undefined_symbols(sdfg, obj, include_scalar_data):
    """ Returns all symbols used in this object that are undefined, and thus
        must be given as input parameters. """
    scalar_arguments = sdfg.scalar_parameters(False)
    if include_scalar_data:
        symbols = collections.OrderedDict(
            (name, data) for name, data in scalar_arguments)
    else:
        symbols = collections.OrderedDict()
    defined = set(sdfg.constants.keys())
    symbols.update(
        obj.data_symbols(True)
        if isinstance(obj, SDFG) else obj.data_symbols())
    assigned, used = obj.interstate_symbols()
    defined |= assigned.keys()
    symbols.update(used)
    iteration_variables, subset_symbols = obj.scope_symbols()
    symbols.update(subset_symbols)
    if sdfg.parent is not None:
        # Find parent Nested SDFG node
        parent_node = next(
            n for n in sdfg.parent.nodes()
            if isinstance(n, nd.NestedSDFG) and n.sdfg.name == sdfg.name)
        defined |= sdfg._parent_sdfg.symbols_defined_at(
            parent_node, sdfg.parent).keys()
    # Don't include iteration variables
    # (TODO: this is too lenient; take scope into account)
    defined |= iteration_variables.keys()
    defined |= {
        n.data
        for n, scope in obj.all_nodes_recursive()
        if (isinstance(n, dace.graph.nodes.AccessNode) and (
            scope.parent is None and n.desc(scope).transient or scope.parent))
    }
    symbols = collections.OrderedDict(
        (key, value) for key, value in symbols.items()
        if key not in defined and not key.startswith('__dace'))
    return symbols


def interstate_symbols(dfg):
    """ Returns all symbols used in interstate edges in nested SDFGs within
        this state. """
    assigned = collections.OrderedDict()
    used = collections.OrderedDict()
    for node in dfg.nodes():
        if isinstance(node, dace.graph.nodes.NestedSDFG):
            a, u = node.sdfg.interstate_symbols()
            assigned.update(a)

            # Filter used symbols if they belong to SDFG input/output connectors
            u = {
                k: v
                for k, v in u.items()
                if k not in (node.in_connectors | node.out_connectors)
            }
            used.update(u)

    return assigned, used


def top_level_transients(dfg):
    """ Iterate over top-level transients (i.e., ones that exist in multiple
        states or scopes) of the passed dataflow graph. """
    sdfg = dfg.parent
    visited_transients = set()
    scope_dict = dfg.scope_dict(node_to_children=True)
    for node in scope_dict[None]:  # Top-level nodes
        if not isinstance(node, nd.AccessNode):
            continue
        if node.data in visited_transients:
            continue
        if not node.desc(sdfg).transient:
            continue
        visited_transients.add(node.data)
        yield node.data


def all_transients(dfg):
    """ Iterate over all transient data in the specified dataflow graph. """
    visited = set()
    for node in dfg.nodes():
        if not isinstance(node, dace.graph.nodes.AccessNode):
            continue
        if not node.desc(dfg.parent).transient:
            continue
        if node.data in visited:
            continue
        visited.add(node.data)
        yield node.data


def _transients_in_scope(sdfg, scope, scope_dict):
    return set(node.data
               for node in scope_dict[scope.entry if scope else scope]
               if isinstance(node, nd.AccessNode)
               and sdfg.arrays[node.data].transient)


def local_transients(sdfg, dfg, entry_node):
    """ Returns transients local to the scope defined by the specified entry
        node in the dataflow graph. """
    state: SDFGState = dfg._graph
    scope_dict = state.scope_dict(node_to_children=True)
    scope_tree = state.scope_tree()
    current_scope = scope_tree[entry_node]

    # Start by setting shared transients as defined
    defined_transients = set(sdfg.shared_transients())

    # Get access nodes in current scope
    transients = _transients_in_scope(sdfg, current_scope, scope_dict)

    # Add transients defined in parent scopes
    while current_scope is not None:
        current_scope = current_scope.parent
        defined_transients.update(
            _transients_in_scope(sdfg, current_scope, scope_dict))

    return sorted(list(transients - defined_transients))


def compile(function_or_sdfg, *args, **kwargs):
    """ Obtain a runnable binary from a Python (@dace.program) function. """
    if isinstance(function_or_sdfg, dace.frontend.python.parser.DaceProgram):
        sdfg = dace.frontend.python.parser.parse_from_function(
            function_or_sdfg, *args, **kwargs)
    elif isinstance(function_or_sdfg, SDFG):
        sdfg = function_or_sdfg
    else:
        raise TypeError("Unsupported function type")
    return sdfg.compile(**kwargs)


def is_devicelevel(sdfg: SDFG, state: SDFGState, node: dace.graph.nodes.Node):
    """ Tests whether a node in an SDFG is contained within GPU device-level
        code.
        :param sdfg: The SDFG in which the node resides.
        :param state: The SDFG state in which the node resides.
        :param node: The node in question
        :return: True if node is in device-level code, False otherwise.
    """
    while sdfg is not None:
        sdict = state.scope_dict()
        scope = sdict[node]
        while scope is not None:
            if scope.schedule in dtypes.GPU_SCHEDULES:
                return True
            scope = sdict[scope]
        # Traverse up nested SDFGs
        if sdfg.parent is not None:
            if isinstance(sdfg.parent, SDFGState):
                parent = sdfg.parent.parent
            else:
                parent = sdfg.parent
            state, node = next(
                (s, n) for s in parent.nodes() for n in s.nodes()
                if isinstance(n, nd.NestedSDFG) and n.sdfg.name == sdfg.name)
        else:
            parent = sdfg.parent
        sdfg = parent
    return False


def replace(subgraph: Union[SDFGState, ScopeSubgraphView, SubgraphView],
            name: str, new_name: str):
    """ Finds and replaces all occurrences of a symbol or array in the given
        subgraph.
        :param subgraph: The given graph or subgraph to replace in.
        :param name: Name to find.
        :param new_name: Name to replace.
    """
    symrepl = {
        symbolic.symbol(name):
        symbolic.symbol(new_name) if isinstance(new_name, str) else new_name
    }

    def replsym(symlist):
        if symlist is None:
            return None
        if isinstance(symlist, (symbolic.SymExpr, symbolic.symbol, sp.Basic)):
            return symlist.subs(symrepl)
        for i, dim in enumerate(symlist):
            try:
                symlist[i] = tuple(
                    d.subs(symrepl) if symbolic.issymbolic(d) else d
                    for d in dim)
            except TypeError:
                symlist[i] = (dim.subs(symrepl)
                              if symbolic.issymbolic(dim) else dim)
        return symlist

    # Replace in node properties
    for node in subgraph.nodes():
        for propclass, propval in node.properties():
            pname = propclass.attr_name
            if isinstance(propclass, properties.SymbolicProperty):
                setattr(node, pname, propval.subs({name: new_name}))
            if isinstance(propclass, properties.DataProperty):
                if propval == name:
                    setattr(node, pname, new_name)
            if isinstance(propclass, properties.RangeProperty):
                setattr(node, pname, replsym(propval))
            if isinstance(propclass, properties.CodeProperty):
                for stmt in propval['code_or_block']:
                    ASTFindReplace({name: new_name}).visit(stmt)

    # Replace in memlets
    for edge in subgraph.edges():
        if edge.data.data == name:
            edge.data.data = new_name
        edge.data.subset = replsym(edge.data.subset)
        edge.data.other_subset = replsym(edge.data.other_subset)


def is_array_stream_view(sdfg: SDFG, dfg: SDFGState, node: nd.AccessNode):
    """ Test whether a stream is directly connected to an array. """

    # Test all memlet paths from the array. If the path goes directly
    # to/from a stream, construct a stream array view
    all_source_paths = []
    source_paths = []
    all_sink_paths = []
    sink_paths = []
    for e in dfg.in_edges(node):
        src_node = dfg.memlet_path(e)[0].src
        # Append empty path to differentiate between a copy and an array-view
        if isinstance(src_node, nd.CodeNode):
            all_source_paths.append(None)
        # Append path from source node
        if isinstance(src_node, nd.AccessNode) and isinstance(
                src_node.desc(sdfg), dt.Array):
            source_paths.append(src_node)
    for e in dfg.out_edges(node):
        sink_node = dfg.memlet_path(e)[-1].dst

        # Append empty path to differentiate between a copy and an array-view
        if isinstance(sink_node, nd.CodeNode):
            all_sink_paths.append(None)
        # Append path to sink node
        if isinstance(sink_node, nd.AccessNode) and isinstance(
                sink_node.desc(sdfg), dt.Array):
            sink_paths.append(sink_node)

    all_sink_paths.extend(sink_paths)
    all_source_paths.extend(source_paths)

    # Special case: stream can be represented as a view of an array
    if ((len(all_source_paths) > 0 and len(sink_paths) == 1)
            or (len(all_sink_paths) > 0 and len(source_paths) == 1)):
        # TODO: What about a source path?
        arrnode = sink_paths[0]
        # Only works if the stream itself is not an array of streams
        if list(node.desc(sdfg).shape) == [1]:
            node.desc(sdfg).sink = arrnode.data  # For memlet generation
            arrnode.desc(
                sdfg).src = node.data  # TODO: Move src/sink to node, not array
            return True
    return False


def dynamic_map_inputs(state: SDFGState,
                       map_entry: nd.MapEntry) -> List[MultiConnectorEdge]:
    return [
        e for e in state.in_edges(map_entry)
        if e.dst_conn and not e.dst_conn.startswith('IN_')
    ]


def has_dynamic_map_inputs(state: SDFGState, map_entry: nd.MapEntry) -> bool:
    return len(dynamic_map_inputs(state, map_entry)) > 0


def _get_optimizer_class(class_override):
    """ Imports and returns a class string defined in the configuration
        (under "optimizer.interface") or overridden in the input
        class_override argument. Empty string, False, or failure to find the
        class skips the process.

        @note: This method uses pydoc to locate the class.
    """
    clazz = class_override
    if class_override is None:
        clazz = Config.get("optimizer", "interface")

    if clazz == "" or clazz == False:
        return None

    result = locate(clazz)
    if result is None:
        warnings.warn('Optimizer interface class "%s" not found' % clazz)

    return result
