# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import collections
import copy
import ctypes
import itertools
import gzip
from numbers import Integral
import os
import pickle, json
from hashlib import md5, sha256
from pydoc import locate
import random
import re
import shutil
import sys
import time
from typing import Any, AnyStr, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type, Union
import warnings
import numpy as np
import sympy as sp

import dace
import dace.serialize
from dace import (data as dt, memlet as mm, subsets as sbs, dtypes, properties, symbolic)
from dace.sdfg.scope import ScopeTree
from dace.sdfg.replace import replace, replace_properties, replace_properties_dict
from dace.sdfg.validation import (InvalidSDFGError, validate_sdfg)
from dace.config import Config
from dace.frontend.python import astutils, wrappers
from dace.sdfg import nodes as nd
from dace.sdfg.graph import OrderedDiGraph, Edge, SubgraphView
from dace.sdfg.state import SDFGState
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.distr_types import ProcessGrid, SubArray, RedistrArray
from dace.dtypes import validate_name
from dace.properties import (DebugInfoProperty, EnumProperty, ListProperty, make_properties, Property, CodeProperty,
                             TransformationHistProperty, OptionalSDFGReferenceProperty, DictProperty, CodeBlock)
from typing import BinaryIO

# NOTE: In shapes, we try to convert strings to integers. In ranks, a string should be interpreted as data (scalar).
ShapeType = Sequence[Union[Integral, str, symbolic.symbol, symbolic.SymExpr, symbolic.sympy.Basic]]
RankType = Union[Integral, str, symbolic.symbol, symbolic.SymExpr, symbolic.sympy.Basic]


def _arrays_to_json(arrays):
    if arrays is None:
        return None
    return {k: dace.serialize.to_json(v) for k, v in arrays.items()}


def _arrays_from_json(obj, context=None):
    if obj is None:
        return {}
    return {k: dace.serialize.from_json(v, context) for k, v in obj.items()}


def _replace_dict_keys(d, old, new):
    if old in d:
        if new in d:
            warnings.warn('"%s" already exists in SDFG' % new)
        d[new] = d[old]
        del d[old]


def _replace_dict_values(d, old, new):
    for k, v in d.items():
        if v == old:
            d[k] = new


def _assignments_from_string(astr):
    """ Returns a dictionary of assignments from a semicolon-delimited
        string of expressions. """

    result = {}
    for aitem in astr.split(';'):
        aitem = aitem.strip()
        m = re.search(r'([^=\s]+)\s*=\s*([^=]+)', aitem)
        result[m.group(1)] = m.group(2)

    return result


def _assignments_to_string(assdict):
    """ Returns a semicolon-delimited string from a dictionary of assignment
        expressions. """
    return '; '.join(['%s=%s' % (k, v) for k, v in assdict.items()])


@make_properties
class LogicalGroup(object):
    """ Logical element groupings on a per-SDFG level.
    """

    nodes = ListProperty(element_type=tuple, desc='Nodes in this group given by [State, Node] id tuples')
    states = ListProperty(element_type=int, desc='States in this group given by their ids')
    name = Property(dtype=str, desc='Logical group name')
    color = Property(dtype=str, desc='Color for the group, given as a hexadecimal string')

    def __init__(self, name, color, nodes=[], states=[]):
        self.nodes = nodes
        self.states = states
        self.color = color
        self.name = name

    def to_json(self):
        retdict = dace.serialize.all_properties_to_json(self)
        retdict['type'] = type(self).__name__
        return retdict

    @staticmethod
    def from_json(json_obj, context=None):
        ret = LogicalGroup('', '')
        dace.serialize.set_properties_from_json(ret, json_obj, context=context, ignore_properties={'type'})
        return ret


@make_properties
class InterstateEdge(object):
    """ An SDFG state machine edge. These edges can contain a condition
        (which may include data accesses for data-dependent decisions) and
        zero or more assignments of values to inter-state variables (e.g.,
        loop iterates).
    """

    assignments = DictProperty(key_type=str,
                               value_type=CodeBlock,
                               desc="Assignments to perform upon transition (e.g., 'x=x+1; y = 0')",
                               from_string=_assignments_from_string,
                               to_string=_assignments_to_string)
    condition = CodeProperty(desc="Transition condition", default=CodeBlock("1"))

    def __init__(self, condition: CodeBlock = None, assignments=None):
        if condition is None:
            condition = CodeBlock("1")

        if assignments is None:
            assignments = {}

        if isinstance(condition, str):
            self.condition = CodeBlock(condition)
        elif isinstance(condition, ast.AST):
            self.condition = CodeBlock([condition])
        elif isinstance(condition, list):
            self.condition = CodeBlock(condition)
        else:
            self.condition = condition
        self.assignments = {k: InterstateEdge._convert_assignment(v) for k, v in assignments.items()}
        self._cond_sympy = None
        self._uncond = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'condition' or name == '_condition':
            super().__setattr__('_cond_sympy', None)
            super().__setattr__('_uncond', None)
        return super().__setattr__(name, value)

    @staticmethod
    def _convert_assignment(assignment: Union[CodeBlock, ast.AST, str]) -> CodeBlock:
        if isinstance(assignment, CodeBlock):
            return assignment
        elif isinstance(assignment, ast.AST):
            return CodeBlock([assignment])
        return CodeBlock(assignment)

    def is_unconditional(self):
        """ Returns True if the state transition is unconditional. """
        if self._uncond is not None:
            return self._uncond
        self._uncond = (self.condition is None or InterstateEdge.condition.to_string(self.condition).strip() == "1"
                        or self.condition.as_string == "")
        return self._uncond

    def condition_sympy(self):
        if self._cond_sympy is not None:
            return self._cond_sympy
        self._cond_sympy = symbolic.pystr_to_symbolic(self.condition.as_string)
        return self._cond_sympy

    @property
    def free_symbols(self) -> Set[str]:
        """ Returns a set of symbols used in this edge's properties. """
        # Symbols in conditions and assignments
        result = set(map(str, dace.symbolic.symbols_in_ast(self.condition.code[0])))
        for assign in self.assignments.values():
            result |= set(map(str, dace.symbolic.symbols_in_ast(assign.code[0])))
        return result - set(self.assignments.keys())

    def replace_dict(self, repl: Dict[str, str], replace_keys=True, replace_condition=True) -> None:
        """
        Replaces all given keys with their corresponding values.
        :param repl: Replacement dictionary.
        :param replace_keys: If False, skips replacing assignment keys.
        :param replace_condition: If False, replaces the value only in the assignment.
        """
        if not repl:
            return

        if replace_keys:
            for name, new_name in repl.items():
                _replace_dict_keys(self.assignments, name, new_name)
        replacer = astutils.ASTFindReplace(repl)

        for k, v in self.assignments.items():
            replacer.replace_count = 0
            vast = replacer.visit(v.code[0])
            if replacer.replace_count > 0:
                self.assignments[k] = CodeBlock(vast)

        if replace_condition:
            replacer.replace_count = 0
            condition = replacer.visit(self.condition.code[0])
            if replacer.replace_count > 0:
                self.condition = CodeBlock(condition)
                self._uncond = None
                self._cond_sympy = None

    def replace(self, name: str, new_name: str, replace_keys=True) -> None:
        """
        Replaces all occurrences of ``name`` with ``new_name``.
        :param name: The source name.
        :param new_name: The replacement name.
        :param replace_keys: If False, skips replacing assignment keys.
        """
        self.replace_dict({name: new_name}, replace_keys)

    def new_symbols(self, sdfg, symbols) -> Dict[str, dtypes.typeclass]:
        """
        Returns a mapping between symbols defined by this edge (i.e.,
        assignments) to their type.
        """
        from dace.codegen.tools.type_inference import infer_expr_type

        if sdfg is not None:
            alltypes = copy.copy(symbols)
            alltypes.update({k: v.dtype for k, v in sdfg.arrays.items()})
        else:
            alltypes = symbols

        return {k: infer_expr_type(v, alltypes) for k, v in self.assignments.items()}

    def to_json(self, parent=None):
        return {
            'type': type(self).__name__,
            'attributes': dace.serialize.all_properties_to_json(self),
            'label': self.label
        }

    @staticmethod
    def from_json(json_obj, context=None):
        # Create dummy object
        ret = InterstateEdge()
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)

        return ret

    @property
    def label(self):
        assignments = ','.join(['%s=%s' % (k, v.as_string) for k, v in self.assignments.items()])

        # Edge with assigment only (no condition)
        if self.condition.as_string == '1':
            # Edge without conditions or assignments
            if len(self.assignments) == 0:
                return ''
            return assignments

        # Edge with condition only (no assignment)
        if len(self.assignments) == 0:
            return self.condition.as_string

        # Edges with assigments and conditions
        return self.condition.as_string + '; ' + assignments


@make_properties
class SDFG(OrderedDiGraph[SDFGState, InterstateEdge]):
    """ The main intermediate representation of code in DaCe.

        A Stateful DataFlow multiGraph (SDFG) is a directed graph of directed
        acyclic multigraphs (i.e., where two nodes can be connected by more
        than one edge). The top-level directed graph represents a state
        machine, where edges can contain state transition conditions and
        assignments (see the `InterstateEdge` class documentation). The nested
        acyclic multigraphs represent dataflow, where nodes may represent data
        regions in memory, tasklets, or parametric graph scopes (see
        `dace.sdfg.nodes` for a full list of available node types); edges in
        the multigraph represent data movement using memlets, as described in
        the `Memlet` class documentation.
    """

    arg_names = ListProperty(element_type=str, desc='Ordered argument names (used for calling conventions).')
    constants_prop = Property(dtype=dict, default={}, desc="Compile-time constants")
    _arrays = Property(dtype=dict,
                       desc="Data descriptors for this SDFG",
                       to_json=_arrays_to_json,
                       from_json=_arrays_from_json)
    symbols = DictProperty(str, dtypes.typeclass, desc="Global symbols for this SDFG")

    instrument = EnumProperty(dtype=dtypes.InstrumentationType,
                              desc="Measure execution statistics with given method",
                              default=dtypes.InstrumentationType.No_Instrumentation)

    global_code = DictProperty(str, CodeBlock, desc="Code generated in a global scope on the output files.")
    init_code = DictProperty(str, CodeBlock, desc="Code generated in the `__dace_init` function.")
    exit_code = DictProperty(str, CodeBlock, desc="Code generated in the `__dace_exit` function.")

    orig_sdfg = OptionalSDFGReferenceProperty(allow_none=True)
    transformation_hist = TransformationHistProperty()

    logical_groups = ListProperty(element_type=LogicalGroup, desc='Logical groupings of nodes and edges')

    openmp_sections = Property(dtype=bool,
                               default=Config.get_bool('compiler', 'cpu', 'openmp_sections'),
                               desc='Whether to generate OpenMP sections in code')

    debuginfo = DebugInfoProperty(allow_none=True)

    _pgrids = DictProperty(str,
                           ProcessGrid,
                           desc="Process-grid descriptors for this SDFG",
                           to_json=_arrays_to_json,
                           from_json=_arrays_from_json)
    _subarrays = DictProperty(str,
                              SubArray,
                              desc="Sub-array descriptors for this SDFG",
                              to_json=_arrays_to_json,
                              from_json=_arrays_from_json)
    _rdistrarrays = DictProperty(str,
                                 RedistrArray,
                                 desc="Sub-array redistribution descriptors for this SDFG",
                                 to_json=_arrays_to_json,
                                 from_json=_arrays_from_json)

    callback_mapping = DictProperty(str,
                                    str,
                                    desc='Mapping between callback name and its original callback '
                                    '(for when the same callback is used with a different signature)')

    def __init__(self,
                 name: str,
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

        self.constants_prop = {}
        if constants is not None:
            for cstname, (cst_dtype, cstval) in constants.items():
                self.add_constant(cstname, cstval, cst_dtype)

        self._propagate = propagate
        self._parent = parent
        self.symbols = {}
        self._parent_sdfg = None
        self._parent_nsdfg_node = None
        self._sdfg_list = [self]
        self._start_state: Optional[int] = None
        self._cached_start_state: Optional[SDFGState] = None
        self._arrays = {}  # type: Dict[str, dt.Array]
        self._labels: Set[str] = set()
        self.global_code = {'frame': CodeBlock("", dtypes.Language.CPP)}
        self.init_code = {'frame': CodeBlock("", dtypes.Language.CPP)}
        self.exit_code = {'frame': CodeBlock("", dtypes.Language.CPP)}
        self.orig_sdfg = None
        self.transformation_hist = []
        self.callback_mapping = {}
        # Counter to make it easy to create temp transients
        self._temp_transients = 0

        # Grid-distribution-related fields
        self._pgrids = {}
        self._subarrays = {}
        self._rdistrarrays = {}

        # Counter to resolve name conflicts
        self._orig_name = name
        self._num = 0

    @property
    def sdfg_id(self):
        """
        Returns the unique index of the current SDFG within the current
        tree of SDFGs (top-level SDFG is 0, nested SDFGs are greater).
        """
        return self.sdfg_list.index(self)

    def to_json(self, hash=False):
        """ Serializes this object to JSON format.
            :return: A string representing the JSON-serialized SDFG.
        """
        # Location in the SDFG list (only for root SDFG)
        if self.parent_sdfg is None:
            self.reset_sdfg_list()

        tmp = super().to_json()

        # Ensure properties are serialized correctly
        tmp['attributes']['constants_prop'] = json.loads(dace.serialize.dumps(tmp['attributes']['constants_prop']))

        tmp['sdfg_list_id'] = int(self.sdfg_id)
        tmp['start_state'] = self._start_state

        tmp['attributes']['name'] = self.name
        if hash:
            tmp['attributes']['hash'] = self.hash_sdfg(tmp)

        if int(self.sdfg_id) == 0:
            tmp['dace_version'] = dace.__version__

        return tmp

    @classmethod
    def from_json(cls, json_obj, context_info=None):
        context_info = context_info or {'sdfg': None}
        _type = json_obj['type']
        if _type != cls.__name__:
            raise TypeError("Class type mismatch")

        attrs = json_obj['attributes']
        nodes = json_obj['nodes']
        edges = json_obj['edges']

        ret = SDFG(name=attrs['name'],
                   constants=dace.serialize.loads(dace.serialize.dumps(attrs['constants_prop'])),
                   parent=context_info['sdfg'])

        dace.serialize.set_properties_from_json(ret,
                                                json_obj,
                                                ignore_properties={'constants_prop', 'name', 'hash', 'start_state'})

        nodelist = []
        for n in nodes:
            nci = copy.copy(context_info)
            nci['sdfg'] = ret

            state = SDFGState.from_json(n, context=nci)
            ret.add_node(state)
            nodelist.append(state)

        for e in edges:
            e = dace.serialize.from_json(e)
            ret.add_edge(nodelist[int(e.src)], nodelist[int(e.dst)], e.data)

        if 'start_state' in json_obj:
            ret._start_state = json_obj['start_state']

        return ret

    def hash_sdfg(self, jsondict: Optional[Dict[str, Any]] = None) -> str:
        '''
        Returns a hash of the current SDFG, without considering IDs and attribute names.
        :param jsondict: If not None, uses given JSON dictionary as input.
        :return: The hash (in SHA-256 format).
        '''
        def keyword_remover(json_obj: Any, last_keyword=""):
            # Makes non-unique in SDFG hierarchy v2
            # Recursively remove attributes from the SDFG which are not used in
            # uniquely representing the SDFG. This, among other things, includes
            # the hash, name, transformation history, and meta attributes.
            if isinstance(json_obj, dict):
                if 'sdfg_list_id' in json_obj:
                    del json_obj['sdfg_list_id']

                keys_to_delete = []
                kv_to_recurse = []
                for key, value in json_obj.items():
                    if (isinstance(key, str)
                            and (key.startswith('_meta_')
                                 or key in ['name', 'hash', 'orig_sdfg', 'transformation_hist', 'instrument'])):
                        keys_to_delete.append(key)
                    else:
                        kv_to_recurse.append((key, value))

                for key in keys_to_delete:
                    del json_obj[key]

                for key, value in kv_to_recurse:
                    keyword_remover(value, last_keyword=key)
            elif isinstance(json_obj, (list, tuple)):
                for value in json_obj:
                    keyword_remover(value)

        # Clean SDFG of nonstandard objects
        jsondict = (json.loads(json.dumps(jsondict)) if jsondict is not None else self.to_json())

        keyword_remover(jsondict)  # Make non-unique in SDFG hierarchy

        string_representation = json.dumps(jsondict)  # dict->str
        hsh = sha256(string_representation.encode('utf-8'))
        return hsh.hexdigest()

    @property
    def arrays(self):
        """ Returns a dictionary of data descriptors (`Data` objects) used
            in this SDFG, with an extra `None` entry for empty memlets.
        """
        return self._arrays

    @property
    def process_grids(self):
        """ Returns a dictionary of process-grid descriptors (`ProcessGrid` objects) used in this SDFG. """
        return self._pgrids

    @property
    def subarrays(self):
        """ Returns a dictionary of sub-array descriptors (`SubArray` objects) used in this SDFG. """
        return self._subarrays

    @property
    def rdistrarrays(self):
        """ Returns a dictionary of sub-array redistribution descriptors (`RedistrArray` objects) used in this SDFG. """
        return self._rdistrarrays

    def data(self, dataname: str):
        """ Looks up a data descriptor from its name, which can be an array, stream, or scalar symbol. """
        if dataname in self._arrays:
            return self._arrays[dataname]
        if str(dataname) in self.symbols:
            return self.symbols[str(dataname)]
        raise KeyError('Data descriptor with name "%s" not found in SDFG' % dataname)

    def replace(self, name: str, new_name: str):
        """ Finds and replaces all occurrences of a symbol or array name in SDFG.
            :param name: Name to find.
            :param new_name: Name to replace.
            :raise FileExistsError: If name and new_name already exist as data descriptors or symbols.
        """
        if name == new_name:
            return
        self.replace_dict({name: new_name})

    def replace_dict(self,
                     repldict: Dict[str, str],
                     symrepl: Optional[Dict[symbolic.SymbolicType, symbolic.SymbolicType]] = None,
                     replace_in_graph: bool = True,
                     replace_keys: bool = True) -> None:
        """
        Replaces all occurrences of keys in the given dictionary with the mapped
        values.
        :param repldict: The replacement dictionary.
        :param replace_keys: If False, skips replacing assignment keys.
        :param symrepl: A symbolic expression replacement dictionary (for performance reasons).
        :param replace_in_graph: Whether to replace in SDFG nodes / edges.
        :param replace_keys: If True, replaces in SDFG property names (e.g., array, symbol, and constant names).
        """
        symrepl = symrepl or {
            symbolic.symbol(k): symbolic.pystr_to_symbolic(v) if isinstance(v, (str, properties.CodeBlock)) else v
            for k, v in repldict.items()
        }

        # Replace in arrays and symbols (if a variable name)
        if replace_keys:
            for name, new_name in repldict.items():
                if validate_name(new_name):
                    _replace_dict_keys(self._arrays, name, new_name)
                    _replace_dict_keys(self.symbols, name, new_name)
                    _replace_dict_keys(self.constants_prop, name, new_name)
                    _replace_dict_keys(self.callback_mapping, name, new_name)
                    _replace_dict_values(self.callback_mapping, name, new_name)

        # Replace inside data descriptors
        for array in self.arrays.values():
            replace_properties_dict(array, repldict, symrepl)

        if replace_in_graph:
            # Replace in inter-state edges
            for edge in self.edges():
                edge.data.replace_dict(repldict)

            # Replace in states
            for state in self.nodes():
                state.replace_dict(repldict, symrepl)

    def add_symbol(self, name, stype):
        """ Adds a symbol to the SDFG.
            :param name: Symbol name.
            :param stype: Symbol type.
        """
        if name in self.symbols:
            raise FileExistsError('Symbol "%s" already exists in SDFG' % name)
        if not isinstance(stype, dtypes.typeclass):
            stype = dtypes.DTYPE_TO_TYPECLASS[stype]
        self.symbols[name] = stype

    def remove_symbol(self, name):
        """ Removes a symbol from the SDFG.
            :param name: Symbol name.
        """
        del self.symbols[name]
        # Clean up from symbol mapping if this SDFG is nested
        nsdfg = self.parent_nsdfg_node
        if nsdfg is not None and name in nsdfg.symbol_mapping:
            del nsdfg.symbol_mapping[name]

    @property
    def start_state(self):
        """ Returns the starting state of this SDFG. """
        if self._cached_start_state is not None:
            return self._cached_start_state

        source_nodes = self.source_nodes()
        if len(source_nodes) == 1:
            self._cached_start_state = source_nodes[0]
            return source_nodes[0]
        # If starting state is ambiguous (i.e., loop to initial state or more
        # than one possible start state), allow manually overriding start state
        if self._start_state is not None:
            self._cached_start_state = self.node(self._start_state)
            return self._cached_start_state
        raise ValueError('Ambiguous or undefined starting state for SDFG, '
                         'please use "is_start_state=True" when adding the '
                         'starting state with "add_state"')

    @start_state.setter
    def start_state(self, state_id):
        """ Manually sets the starting state of this SDFG.
            :param state_id: The node ID (use `node_id(state)`) of the
                             state to set.
        """
        if state_id < 0 or state_id >= self.number_of_nodes():
            raise ValueError("Invalid state ID")
        self._start_state = state_id
        self._cached_start_state = self.node(state_id)

    def set_global_code(self, cpp_code: str, location: str = 'frame'):
        """
        Sets C++ code that will be generated in a global scope on
        one of the generated code files.
        :param cpp_code: The code to set.
        :param location: The file/backend in which to generate the code.
                         Options are None (all files), "frame", "openmp",
                         "cuda", "xilinx", "intel_fpga", or any code generator
                         name.
        """
        self.global_code[location] = CodeBlock(cpp_code, dace.dtypes.Language.CPP)

    def set_init_code(self, cpp_code: str, location: str = 'frame'):
        """
        Sets C++ code that will be generated in the __dace_init_* functions on
        one of the generated code files.
        :param cpp_code: The code to set.
        :param location: The file/backend in which to generate the code.
                         Options are None (all files), "frame", "openmp",
                         "cuda", "xilinx", "intel_fpga", or any code generator
                         name.
        """
        self.init_code[location] = CodeBlock(cpp_code, dtypes.Language.CPP)

    def set_exit_code(self, cpp_code: str, location: str = 'frame'):
        """
        Sets C++ code that will be generated in the __dace_exit_* functions on
        one of the generated code files.
        :param cpp_code: The code to set.
        :param location: The file/backend in which to generate the code.
                         Options are None (all files), "frame", "openmp",
                         "cuda", "xilinx", "intel_fpga", or any code generator
                         name.
        """
        self.exit_code[location] = CodeBlock(cpp_code, dtypes.Language.CPP)

    def append_global_code(self, cpp_code: str, location: str = 'frame'):
        """
        Appends C++ code that will be generated in a global scope on
        one of the generated code files.
        :param cpp_code: The code to set.
        :param location: The file/backend in which to generate the code.
                         Options are None (all files), "frame", "openmp",
                         "cuda", "xilinx", "intel_fpga", or any code generator
                         name.
        """
        if location not in self.global_code:
            self.global_code[location] = CodeBlock('', dtypes.Language.CPP)
        self.global_code[location].code += cpp_code

    def append_init_code(self, cpp_code: str, location: str = 'frame'):
        """
        Appends C++ code that will be generated in the __dace_init_* functions on
        one of the generated code files.
        :param cpp_code: The code to append.
        :param location: The file/backend in which to generate the code.
                         Options are None (all files), "frame", "openmp",
                         "cuda", "xilinx", "intel_fpga", or any code generator
                         name.
        """
        if location not in self.init_code:
            self.init_code[location] = CodeBlock('', dtypes.Language.CPP)
        self.init_code[location].code += cpp_code

    def append_exit_code(self, cpp_code: str, location: str = 'frame'):
        """
        Appends C++ code that will be generated in the __dace_exit_* functions on
        one of the generated code files.
        :param cpp_code: The code to append.
        :param location: The file/backend in which to generate the code.
                         Options are None (all files), "frame", "openmp",
                         "cuda", "xilinx", "intel_fpga", or any code generator
                         name.
        """
        if location not in self.exit_code:
            self.exit_code[location] = CodeBlock('', dtypes.Language.CPP)
        self.exit_code[location].code += cpp_code

    def prepend_exit_code(self, cpp_code: str, location: str = 'frame'):
        """
        Prepends C++ code that will be generated in the __dace_exit_* functions on
        one of the generated code files.
        :param cpp_code: The code to prepend.
        :param location: The file/backend in which to generate the code.
                         Options are None (all files), "frame", "openmp",
                         "cuda", "xilinx", "intel_fpga", or any code generator
                         name.
        """
        if location not in self.exit_code:
            self.exit_code[location] = CodeBlock('', dtypes.Language.CPP)
        self.exit_code[location].code = cpp_code + self.exit_code[location].code

    def append_transformation(self, transformation):
        """
        Appends a transformation to the treansformation history of this SDFG.
        If this is the first transformation being applied, it also saves the
        initial state of the SDFG to return to and play back the history.
        :param transformation: The transformation to append.
        """
        if Config.get_bool('store_history') is False:
            return
        # Make sure the transformation is appended to the root SDFG.
        if self.sdfg_id != 0:
            self.sdfg_list[0].append_transformation(transformation)
            return

        if not self.orig_sdfg:
            clone = copy.deepcopy(self)
            clone.transformation_hist = []
            clone.orig_sdfg = None
            self.orig_sdfg = clone
        self.transformation_hist.append(transformation)

    ##########################################
    # Instrumentation-related methods

    def is_instrumented(self) -> bool:
        """ Returns True if the SDFG has performance instrumentation enabled on
            it or any of its elements. """
        if self.instrument != dtypes.InstrumentationType.No_Instrumentation:
            return True
        try:
            next(n for n, _ in self.all_nodes_recursive()
                 if hasattr(n, 'instrument') and n.instrument != dtypes.InstrumentationType.No_Instrumentation)
            return True
        except StopIteration:
            return False

    def get_instrumentation_reports(self) -> \
            List['dace.codegen.instrumentation.InstrumentationReport']:
        """
        Returns a list of instrumentation reports from previous runs of
        this SDFG.
        :return: A List of timestamped InstrumentationReport objects.
        """
        # Avoid import loops
        from dace.codegen.instrumentation import InstrumentationReport

        path = os.path.join(self.build_folder, 'perf')
        return [
            InstrumentationReport(os.path.join(path, fname)) for fname in os.listdir(path)
            if fname.startswith('report-')
        ]

    def clear_instrumentation_reports(self):
        """
        Clears the instrumentation report folder of this SDFG.
        """
        path = os.path.join(self.build_folder, 'perf')
        for fname in os.listdir(path):
            if not fname.startswith('report-'):
                continue
            os.unlink(os.path.join(path, fname))

    def get_latest_report(self) -> Optional['dace.codegen.instrumentation.InstrumentationReport']:
        """
        Returns an instrumentation report from the latest run of this SDFG, or
        None if the file does not exist.
        :return: A timestamped InstrumentationReport object, or None if does
                 not exist.
        """
        path = os.path.join(self.build_folder, 'perf')
        files = [f for f in os.listdir(path) if f.startswith('report-')]
        if len(files) == 0:
            return None

        # Avoid import loops
        from dace.codegen.instrumentation import InstrumentationReport

        return InstrumentationReport(os.path.join(path, sorted(files, reverse=True)[0]))

    def get_instrumented_data(
        self,
        timestamp: Optional[int] = None
    ) -> Optional['dace.codegen.instrumentation.data.data_report.InstrumentedDataReport']:
        """
        Returns an instrumented data report from the latest run of this SDFG, with a given timestamp, or
        None if no reports exist.
        :param timestamp: An optional timestamp to use for the report.
        :return: An InstrumentedDataReport object, or None if one does not exist.
        """
        # Avoid import loops
        from dace.codegen.instrumentation.data.data_report import InstrumentedDataReport

        if timestamp is None:
            reports = self.available_data_reports()
            if not reports:
                return None
            timestamp = sorted(reports)[-1]

        folder = os.path.join(self.build_folder, 'data', str(timestamp))
        if not os.path.exists(folder):
            return None

        return InstrumentedDataReport(self, folder)

    def available_data_reports(self) -> List[str]:
        """
        Returns a list of available instrumented data reports for this SDFG.
        """
        path = os.path.join(self.build_folder, 'data')
        if os.path.exists(path):
            return os.listdir(path)
        else:
            return []

    def clear_data_reports(self):
        """
        Clears the instrumented data report folders of this SDFG.
        """
        reports = self.available_data_reports()
        path = os.path.join(self.build_folder, 'data')
        for report in reports:
            shutil.rmtree(os.path.join(path, report))

    def call_with_instrumented_data(self,
                                    dreport: 'dace.codegen.instrumentation.data.data_report.InstrumentedDataReport',
                                    *args, **kwargs):
        """
        Invokes an SDFG with an instrumented data report, generating and compiling code if necessary. 
        Arguments given as ``args`` and ``kwargs`` will be overriden by the data containers defined in the report.
        :param dreport: The instrumented data report to use upon calling.
        :param args: Arguments to call SDFG with.
        :param kwargs: Keyword arguments to call SDFG with.
        :return: The return value(s) of this SDFG.
        """
        from dace.codegen.compiled_sdfg import CompiledSDFG  # Avoid import loop

        binaryobj: CompiledSDFG = self.compile()
        set_report = binaryobj.get_exported_function('__dace_set_instrumented_data_report')
        if set_report is None:
            raise ValueError(
                'Data instrumentation report function not found. This is likely because the SDFG is not instrumented '
                'with `dace.DataInstrumentationType.Restore`')

        # Initialize the compiled SDFG to get the handle, then set the report folder
        handle = binaryobj.initialize(*args, **kwargs)
        set_report(handle, ctypes.c_char_p(os.path.abspath(dreport.folder).encode('utf-8')))

        # Verify passed arguments (unless disabled by the user)
        if dace.config.Config.get_bool("execution", "general", "check_args"):
            self.argument_typecheck(args, kwargs)
        return binaryobj(*args, **kwargs)

    ##########################################

    @property
    def build_folder(self) -> str:
        """ Returns a relative path to the build cache folder for this SDFG. """
        if hasattr(self, '_build_folder'):
            return self._build_folder
        cache_config = Config.get('cache')
        base_folder = Config.get('default_build_folder')
        if cache_config == 'single':
            # Always use the same directory, overwriting any other program,
            # preventing parallelism and caching of multiple programs, but
            # saving space and potentially build time
            return os.path.join(base_folder, 'single_cache')
        elif cache_config == 'hash':
            # Any change to the SDFG will result in a new cache folder
            md5_hash = md5(str(self.to_json()).encode('utf-8')).hexdigest()
            return os.path.join(base_folder, f'{self.name}_{md5_hash}')
        elif cache_config == 'unique':
            # Base name on location in memory, so no caching is possible between
            # processes or subsequent invocations
            md5_hash = md5(str(os.getpid()).encode('utf-8')).hexdigest()
            return os.path.join(base_folder, f'{self.name}_{md5_hash}')
        elif cache_config == 'name':
            # Overwrites previous invocations, and can clash with other programs
            # if executed in parallel in the same working directory
            return os.path.join(base_folder, self.name)
        else:
            raise ValueError(f'Unknown cache configuration: {cache_config}')

    @build_folder.setter
    def build_folder(self, newfolder: str):
        self._build_folder = newfolder

    def remove_data(self, name, validate=True):
        """ Removes a data descriptor from the SDFG.
            :param name: The name of the data descriptor to remove.
            :param validate: If True, verifies that there are no access
                             nodes that are using this data descriptor
                             prior to removing it.
        """

        # Verify that the data descriptor exists
        if name not in self._arrays:
            return

        # Verify that there are no access nodes that use this data
        if validate:
            for state in self.nodes():
                for node in state.nodes():
                    if isinstance(node, nd.AccessNode) and node.data == name:
                        raise ValueError(f"Cannot remove data descriptor "
                                         f"{name}: it is accessed by node "
                                         f"{node} in state {state}.")

        del self._arrays[name]

    def reset_sdfg_list(self):
        if self.parent_sdfg is not None:
            return self.parent_sdfg.reset_sdfg_list()
        else:
            # Propagate new SDFG list to all children
            all_sdfgs = list(self.all_sdfgs_recursive())
            for sd in all_sdfgs:
                sd._sdfg_list = all_sdfgs
        return self._sdfg_list

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
    def sdfg_list(self) -> List['SDFG']:
        return self._sdfg_list

    def set_sourcecode(self, code: str, lang=None):
        """ Set the source code of this SDFG (for IDE purposes).
            :param code: A string of source code.
            :param lang: A string representing the language of the source code,
                         for syntax highlighting and completion.
        """
        self.sourcecode = {'code': code, 'language': lang}

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

    @name.setter
    def name(self, newname: str):
        self._name = newname

    @property
    def label(self):
        """ The name of this SDFG. """
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
        self.constants_prop[name] = (dtype or dt.create_datadescriptor(value), value)

    @property
    def propagate(self):
        return self._propagate

    @propagate.setter
    def propagate(self, propagate: bool):
        self._propagate = propagate

    @property
    def parent(self) -> SDFGState:
        """ Returns the parent SDFG state of this SDFG, if exists. """
        return self._parent

    @property
    def parent_sdfg(self) -> 'SDFG':
        """ Returns the parent SDFG of this SDFG, if exists. """
        return self._parent_sdfg

    @property
    def parent_nsdfg_node(self) -> nd.NestedSDFG:
        """ Returns the parent NestedSDFG node of this SDFG, if exists. """
        return self._parent_nsdfg_node

    @parent.setter
    def parent(self, value):
        self._parent = value

    @parent_sdfg.setter
    def parent_sdfg(self, value):
        self._parent_sdfg = value

    @parent_nsdfg_node.setter
    def parent_nsdfg_node(self, value):
        self._parent_nsdfg_node = value

    def add_node(self, node, is_start_state=False):
        """ Adds a new node to the SDFG. Must be an SDFGState or a subclass
            thereof.
            :param node: The node to add.
            :param is_start_state: If True, sets this node as the starting
                                   state.
        """
        if not isinstance(node, SDFGState):
            raise TypeError("Expected SDFGState, got " + str(type(node)))
        super(SDFG, self).add_node(node)
        self._cached_start_state = None
        if is_start_state is True:
            self.start_state = len(self.nodes()) - 1
            self._cached_start_state = node

    def remove_node(self, node: SDFGState):
        if node is self._cached_start_state:
            self._cached_start_state = None
        return super().remove_node(node)

    def add_edge(self, u, v, edge):
        """ Adds a new edge to the SDFG. Must be an InterstateEdge or a
            subclass thereof.
            :param u: Source node.
            :param v: Destination node.
            :param edge: The edge to add.
        """
        if not isinstance(u, SDFGState):
            raise TypeError("Expected SDFGState, got: {}".format(type(u).__name__))
        if not isinstance(v, SDFGState):
            raise TypeError("Expected SDFGState, got: {}".format(type(v).__name__))
        if not isinstance(edge, InterstateEdge):
            raise TypeError("Expected InterstateEdge, got: {}".format(type(edge).__name__))
        if v is self._cached_start_state:
            self._cached_start_state = None
        return super(SDFG, self).add_edge(u, v, edge)

    def states(self):
        """ Alias that returns the nodes (states) in this SDFG. """
        return self.nodes()

    def all_nodes_recursive(self) -> Iterator[Tuple[nd.Node, Union['SDFG', 'SDFGState']]]:
        """ Iterate over all nodes in this SDFG, including states, nodes in
            states, and recursive states and nodes within nested SDFGs,
            returning tuples on the form (node, parent), where the parent is
            either the SDFG (for states) or a DFG (nodes). """
        for node in self.nodes():
            yield node, self
            yield from node.all_nodes_recursive()

    def all_sdfgs_recursive(self):
        """ Iterate over this and all nested SDFGs. """
        yield self
        for state in self.nodes():
            for node in state.nodes():
                if isinstance(node, nd.NestedSDFG):
                    yield from node.sdfg.all_sdfgs_recursive()

    def all_edges_recursive(self):
        """ Iterate over all edges in this SDFG, including state edges,
            inter-state edges, and recursively edges within nested SDFGs,
            returning tuples on the form (edge, parent), where the parent is
            either the SDFG (for states) or a DFG (nodes). """
        for e in self.edges():
            yield e, self
        for node in self.nodes():
            yield from node.all_edges_recursive()

    def arrays_recursive(self):
        """ Iterate over all arrays in this SDFG, including arrays within
            nested SDFGs. Yields 3-tuples of (sdfg, array name, array)."""
        for aname, arr in self.arrays.items():
            yield self, aname, arr
        for state in self.nodes():
            for node in state.nodes():
                if isinstance(node, nd.NestedSDFG):
                    yield from node.sdfg.arrays_recursive()

    @property
    def free_symbols(self) -> Set[str]:
        """
        Returns a set of symbol names that are used by the SDFG, but not
        defined within it. This property is used to determine the symbolic
        parameters of the SDFG and verify that ``SDFG.symbols`` is complete.
        :note: Assumes that the graph is valid (i.e., without undefined or
               overlapping symbols).
        """
        defined_syms = set()
        free_syms = set()

        # Start with the set of SDFG free symbols
        free_syms |= set(self.symbols.keys())

        # Exclude data descriptor names
        for name, desc in self.arrays.items():
            defined_syms.add(name)

        # Add free state symbols
        for state in self.nodes():
            free_syms |= state.free_symbols

        # Add free inter-state symbols
        for e in self.edges():
            defined_syms |= set(e.data.assignments.keys())
            free_syms |= e.data.free_symbols

        defined_syms |= set(self.constants.keys())

        # Subtract symbols defined in inter-state edges and constants
        return free_syms - defined_syms

    def read_and_write_sets(self) -> Tuple[Set[AnyStr], Set[AnyStr]]:
        """
        Determines what data containers are read and written in this SDFG. Does
        not include reads to subsets of containers that have previously been
        written within the same state.
        :return: A two-tuple of sets of things denoting
                 ({data read}, {data written}).
        """
        read_set = set()
        write_set = set()
        for state in self.states():
            for edge in self.in_edges(state):
                read_set |= edge.data.free_symbols & self.arrays.keys()
            # Get dictionaries of subsets read and written from each state
            rs, ws = state._read_and_write_sets()
            read_set |= rs.keys()
            write_set |= ws.keys()
        return read_set, write_set

    def arglist(self, scalars_only=False, free_symbols=None) -> Dict[str, dt.Data]:
        """
        Returns an ordered dictionary of arguments (names and types) required
        to invoke this SDFG.

        The arguments follow the following order:
        <sorted data arguments>, <sorted scalar arguments>.
        Data arguments are all the non-transient data containers in the
        SDFG; and scalar arguments are all the non-transient scalar data
        containers and free symbols (see ``SDFG.free_symbols``). This structure
        will create a sorted list of pointers followed by a sorted list of PoDs
        and structs.

        :return: An ordered dictionary of (name, data descriptor type) of all
                 the arguments, sorted as defined here.
        """
        # Start with data descriptors
        if scalars_only:
            data_args = {}
        else:
            data_args = {k: v for k, v in self.arrays.items() if not v.transient and not isinstance(v, dt.Scalar)}

        scalar_args = {
            k: v
            for k, v in self.arrays.items()
            if not v.transient and isinstance(v, dt.Scalar) and not k.startswith('__dace')
        }

        # Add global free symbols to scalar arguments
        free_symbols = free_symbols if free_symbols is not None else self.free_symbols
        scalar_args.update({k: dt.Scalar(self.symbols[k]) for k in free_symbols if not k.startswith('__dace')})

        # Fill up ordered dictionary
        result = collections.OrderedDict()
        result.update(sorted(data_args.items()))
        result.update(sorted(scalar_args.items()))

        return result

    def init_signature(self, for_call=False, free_symbols=None) -> str:
        """ Returns a C/C++ signature of this SDFG, used when generating the initalization code.
            It only contains symbols.

            :param for_call: If True, returns arguments that can be used when calling the SDFG.
        """
        # Get global free symbols scalar arguments
        free_symbols = free_symbols or self.free_symbols
        return ", ".join(
            dt.Scalar(self.symbols[k]).as_arg(name=k, with_types=not for_call, for_call=for_call)
            for k in sorted(free_symbols) if not k.startswith('__dace'))

    def signature_arglist(self, with_types=True, for_call=False, with_arrays=True, arglist=None) -> List[str]:
        """ Returns a list of arguments necessary to call this SDFG,
            formatted as a list of C definitions.
            :param with_types: If True, includes argument types in the result.
            :param for_call: If True, returns arguments that can be used when
                             calling the SDFG.
            :param with_arrays: If True, includes arrays, otherwise,
                                only symbols and scalars are included.
            :param arglist: An optional cached argument list.
            :return: A list of strings. For example: `['float *A', 'int b']`.
        """
        arglist = arglist or self.arglist(scalars_only=not with_arrays)
        return [v.as_arg(name=k, with_types=with_types, for_call=for_call) for k, v in arglist.items()]

    def signature(self, with_types=True, for_call=False, with_arrays=True, arglist=None) -> str:
        """ Returns a C/C++ signature of this SDFG, used when generating code.
            :param with_types: If True, includes argument types (can be used
                               for a function prototype). If False, only
                               include argument names (can be used for function
                               calls).
            :param for_call: If True, returns arguments that can be used when
                             calling the SDFG.
            :param with_arrays: If True, includes arrays, otherwise,
                                only symbols and scalars are included.
            :param arglist: An optional cached argument list.
        """
        return ", ".join(self.signature_arglist(with_types, for_call, with_arrays, arglist))

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
    var sdfv_{uid} = new SDFV();
    var renderer_{uid} = new SDFGRenderer(sdfv_{uid}, parse_sdfg(sdfg_{uid}),
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
                if isinstance(node, nd.AccessNode) and node.desc(self).transient:
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

    def shared_transients(self, check_toplevel=True) -> List[str]:
        """ Returns a list of transient data that appears in more than one
            state. """
        seen = {}
        shared = []

        # If a transient is present in an inter-state edge, it is shared
        for interstate_edge in self.edges():
            for sym in interstate_edge.data.free_symbols:
                if sym in self.arrays and self.arrays[sym].transient:
                    seen[sym] = interstate_edge
                    shared.append(sym)

        # If transient is accessed in more than one state, it is shared
        for state in self.nodes():
            for node in state.nodes():
                if isinstance(node, nd.AccessNode) and node.desc(self).transient:
                    if (check_toplevel and node.desc(self).toplevel) or (node.data in seen
                                                                         and seen[node.data] != state):
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

    def save(self, filename: str, use_pickle=False, hash=None, exception=None, compress=False) -> Optional[str]:
        """ Save this SDFG to a file.
            :param filename: File name to save to.
            :param use_pickle: Use Python pickle as the SDFG format (default:
                               JSON).
            :param hash: By default, saves the hash if SDFG is JSON-serialized.
                         Otherwise, if True, saves the hash along with the SDFG.
            :param exception: If not None, stores error information along with
                              SDFG.
            :param compress: If True, uses gzip to compress the file upon saving.
            :return: The hash of the SDFG, or None if failed/not requested.
        """
        if compress:
            fileopen = lambda file, mode: gzip.open(file, mode + 't')
        else:
            fileopen = open

        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        except (FileNotFoundError, FileExistsError):
            pass

        if use_pickle:
            with fileopen(filename, "wb") as fp:
                symbolic.SympyAwarePickler(fp).dump(self)
            if hash is True:
                return self.hash_sdfg()
        else:
            hash = True if hash is None else hash
            with fileopen(filename, "w") as fp:
                json_output = self.to_json(hash=hash)
                if exception:
                    json_output['error'] = exception.to_json()
                dace.serialize.dump(json_output, fp)
            if hash and 'hash' in json_output['attributes']:
                return json_output['attributes']['hash']

        return None

    def view(self, filename=None):
        """View this sdfg in the system's HTML viewer
           :param filename: the filename to write the HTML to. If `None`, a temporary file will be created.
        """
        from dace.cli.sdfv import view
        view(self, filename=filename)

    @staticmethod
    def _from_file(fp: BinaryIO) -> 'SDFG':
        firstbyte = fp.read(1)
        fp.seek(0)
        if firstbyte == b'{':  # JSON file
            sdfg_json = json.load(fp)
            sdfg = SDFG.from_json(sdfg_json)
        else:  # Pickle
            sdfg = symbolic.SympyAwareUnpickler(fp).load()

        if not isinstance(sdfg, SDFG):
            raise TypeError("Loaded file is not an SDFG (loaded type: %s)" % type(sdfg).__name__)
        return sdfg

    @staticmethod
    def from_file(filename: str) -> 'SDFG':
        """ Constructs an SDFG from a file.
            :param filename: File name to load SDFG from.
            :return: An SDFG.
        """
        # Try compressed first. If fails, try uncompressed
        try:
            with gzip.open(filename, 'rb') as fp:
                return SDFG._from_file(fp)
        except OSError:
            pass
        with open(filename, "rb") as fp:
            return SDFG._from_file(fp)

    # Dynamic SDFG creation API
    ##############################
    def add_state(self, label=None, is_start_state=False) -> 'SDFGState':
        """ Adds a new SDFG state to this graph and returns it.
            :param label: State label.
            :param is_start_state: If True, resets SDFG starting state to this
                                   state.
            :return: A new SDFGState object.
        """
        if self._labels is None or len(self._labels) != self.number_of_nodes():
            self._labels = set(s.label for s in self.nodes())
        label = label or 'state'
        existing_labels = self._labels
        label = dt.find_new_name(label, existing_labels)
        state = SDFGState(label, self)
        self._labels.add(label)

        self.add_node(state, is_start_state=is_start_state)
        return state

    def add_state_before(self, state: 'SDFGState', label=None, is_start_state=False) -> 'SDFGState':
        """ Adds a new SDFG state before an existing state, reconnecting
            predecessors to it instead.
            :param state: The state to prepend the new state before.
            :param label: State label.
            :param is_start_state: If True, resets SDFG starting state to this
                                   state.
            :return: A new SDFGState object.
        """
        new_state = self.add_state(label, is_start_state)
        # Reconnect
        for e in self.in_edges(state):
            self.remove_edge(e)
            self.add_edge(e.src, new_state, e.data)
        # Add unconditional connection between the new state and the current
        self.add_edge(new_state, state, InterstateEdge())
        return new_state

    def add_state_after(self, state: 'SDFGState', label=None, is_start_state=False) -> 'SDFGState':
        """ Adds a new SDFG state after an existing state, reconnecting
            it to the successors instead.
            :param state: The state to append the new state after.
            :param label: State label.
            :param is_start_state: If True, resets SDFG starting state to this
                                   state.
            :return: A new SDFGState object.
        """
        new_state = self.add_state(label, is_start_state)
        # Reconnect
        for e in self.out_edges(state):
            self.remove_edge(e)
            self.add_edge(new_state, e.dst, e.data)
        # Add unconditional connection between the current and the new state
        self.add_edge(state, new_state, InterstateEdge())
        return new_state

    def _find_new_name(self, name: str):
        """ Tries to find a new name by adding an underscore and a number. """
        index = 0
        names = (self._arrays.keys() | self.constants_prop.keys() | self._pgrids.keys() | self._subarrays.keys()
                 | self._rdistrarrays.keys())
        while (name + ('_%d' % index)) in names:
            index += 1

        return name + ('_%d' % index)

    def find_new_constant(self, name: str):
        """
        Tries to find a new constant name by adding an underscore and a number.
        """
        constants = self.constants
        if name not in constants:
            return name

        index = 0
        while (name + ('_%d' % index)) in constants:
            index += 1

        return name + ('_%d' % index)

    def find_new_symbol(self, name: str):
        """
        Tries to find a new symbol name by adding an underscore and a number.
        """
        symbols = self.symbols
        if name not in symbols:
            return name

        index = 0
        while (name + ('_%d' % index)) in symbols:
            index += 1

        return name + ('_%d' % index)

    def add_array(self,
                  name: str,
                  shape,
                  dtype,
                  storage=dtypes.StorageType.Default,
                  location=None,
                  transient=False,
                  strides=None,
                  offset=None,
                  lifetime=dace.dtypes.AllocationLifetime.Scope,
                  debuginfo=None,
                  allow_conflicts=False,
                  total_size=None,
                  find_new_name=False,
                  alignment=0,
                  may_alias=False) -> Tuple[str, dt.Array]:
        """ Adds an array to the SDFG data descriptor store. """

        # convert strings to int if possible
        newshape = []
        for s in shape:
            try:
                newshape.append(int(s))
            except:
                newshape.append(dace.symbolic.pystr_to_symbolic(s))
        shape = newshape
        strides = strides or None

        if isinstance(dtype, type) and dtype in dtypes._CONSTANT_TYPES[:-1]:
            dtype = dtypes.typeclass(dtype)

        desc = dt.Array(dtype,
                        shape,
                        storage=storage,
                        location=location,
                        allow_conflicts=allow_conflicts,
                        transient=transient,
                        strides=strides,
                        offset=offset,
                        lifetime=lifetime,
                        alignment=alignment,
                        debuginfo=debuginfo,
                        total_size=total_size,
                        may_alias=may_alias)

        return self.add_datadesc(name, desc, find_new_name=find_new_name), desc

    def add_view(self,
                 name: str,
                 shape,
                 dtype,
                 storage=dtypes.StorageType.Default,
                 strides=None,
                 offset=None,
                 debuginfo=None,
                 allow_conflicts=False,
                 total_size=None,
                 find_new_name=False,
                 alignment=0,
                 may_alias=False) -> Tuple[str, dt.View]:
        """ Adds a view to the SDFG data descriptor store. """

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

        desc = dt.View(dtype,
                       shape,
                       storage=storage,
                       allow_conflicts=allow_conflicts,
                       transient=True,
                       strides=strides,
                       offset=offset,
                       lifetime=dtypes.AllocationLifetime.Scope,
                       alignment=alignment,
                       debuginfo=debuginfo,
                       total_size=total_size,
                       may_alias=may_alias)

        return self.add_datadesc(name, desc, find_new_name=find_new_name), desc

    def add_reference(self,
                      name: str,
                      shape,
                      dtype,
                      storage=dtypes.StorageType.Default,
                      strides=None,
                      offset=None,
                      debuginfo=None,
                      allow_conflicts=False,
                      total_size=None,
                      find_new_name=False,
                      alignment=0,
                      may_alias=False) -> Tuple[str, dt.Reference]:
        """ Adds a reference to the SDFG data descriptor store. """

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

        desc = dt.Reference(dtype,
                            shape,
                            storage=storage,
                            allow_conflicts=allow_conflicts,
                            transient=True,
                            strides=strides,
                            offset=offset,
                            lifetime=dtypes.AllocationLifetime.Scope,
                            alignment=alignment,
                            debuginfo=debuginfo,
                            total_size=total_size,
                            may_alias=may_alias)

        return self.add_datadesc(name, desc, find_new_name=find_new_name), desc

    def add_stream(self,
                   name: str,
                   dtype,
                   buffer_size=1,
                   shape=(1, ),
                   storage=dtypes.StorageType.Default,
                   transient=False,
                   offset=None,
                   lifetime=dace.dtypes.AllocationLifetime.Scope,
                   debuginfo=None,
                   find_new_name=False) -> Tuple[str, dt.Stream]:
        """ Adds a stream to the SDFG data descriptor store. """

        # Convert to int if possible, otherwise to symbolic
        _shape = []
        for s in shape:
            try:
                _shape.append(int(s))
            except:
                _shape.append(dace.symbolic.pystr_to_symbolic(s))
        shape = _shape

        if isinstance(dtype, type) and dtype in dtypes._CONSTANT_TYPES[:-1]:
            dtype = dtypes.typeclass(dtype)

        desc = dt.Stream(
            dtype=dtype,
            buffer_size=buffer_size,
            shape=shape,
            storage=storage,
            transient=transient,
            offset=offset,
            lifetime=lifetime,
            debuginfo=debuginfo,
        )

        return self.add_datadesc(name, desc, find_new_name=find_new_name), desc

    def add_scalar(self,
                   name: str,
                   dtype,
                   storage=dtypes.StorageType.Default,
                   transient=False,
                   lifetime=dace.dtypes.AllocationLifetime.Scope,
                   debuginfo=None,
                   find_new_name=False) -> Tuple[str, dt.Scalar]:
        """ Adds a scalar to the SDFG data descriptor store. """

        if isinstance(dtype, type) and dtype in dtypes._CONSTANT_TYPES[:-1]:
            dtype = dtypes.typeclass(dtype)

        desc = dt.Scalar(
            dtype,
            storage=storage,
            transient=transient,
            lifetime=lifetime,
            debuginfo=debuginfo,
        )

        return self.add_datadesc(name, desc, find_new_name=find_new_name), desc

    def add_transient(self,
                      name,
                      shape,
                      dtype,
                      storage=dtypes.StorageType.Default,
                      location=None,
                      strides=None,
                      offset=None,
                      lifetime=dace.dtypes.AllocationLifetime.Scope,
                      debuginfo=None,
                      allow_conflicts=False,
                      total_size=None,
                      find_new_name=False,
                      alignment=0,
                      may_alias=False) -> Tuple[str, dt.Array]:
        """ Convenience function to add a transient array to the data
            descriptor store. """
        return self.add_array(name,
                              shape,
                              dtype,
                              storage=storage,
                              location=location,
                              transient=True,
                              strides=strides,
                              offset=offset,
                              lifetime=lifetime,
                              debuginfo=debuginfo,
                              allow_conflicts=allow_conflicts,
                              total_size=total_size,
                              alignment=alignment,
                              may_alias=may_alias,
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
                           location=None,
                           strides=None,
                           offset=None,
                           lifetime=dace.dtypes.AllocationLifetime.Scope,
                           debuginfo=None,
                           allow_conflicts=False,
                           total_size=None,
                           alignment=0,
                           may_alias=False):
        """ Convenience function to add a transient array with a temporary name to the data
            descriptor store. """
        return self.add_array(self.temp_data_name(),
                              shape,
                              dtype,
                              storage=storage,
                              location=location,
                              transient=True,
                              strides=strides,
                              offset=offset,
                              lifetime=lifetime,
                              alignment=alignment,
                              debuginfo=debuginfo,
                              allow_conflicts=allow_conflicts,
                              total_size=total_size,
                              may_alias=may_alias)

    def add_temp_transient_like(self, desc: Union[dt.Array, dt.Scalar], dtype=None, debuginfo=None):
        """ Convenience function to add a transient array with a temporary name to the data
            descriptor store. """
        debuginfo = debuginfo or desc.debuginfo
        dtype = dtype or desc.dtype
        newdesc = desc.clone()
        newdesc.dtype = dtype
        newdesc.transient = True
        newdesc.debuginfo = debuginfo
        return self.add_datadesc(self.temp_data_name(), newdesc), newdesc

    def add_datadesc(self, name: str, datadesc: dt.Data, find_new_name=False) -> str:
        """ Adds an existing data descriptor to the SDFG array store.
            :param name: Name to use.
            :param datadesc: Data descriptor to add.
            :param find_new_name: If True and data descriptor with this name
                                  exists, finds a new name to add.
            :return: Name of the new data descriptor
        """
        if not isinstance(name, str):
            raise TypeError("Data descriptor name must be a string. Got %s" % type(name).__name__)
        # If exists, fail
        if name in self._arrays:
            if find_new_name:
                name = self._find_new_name(name)
            else:
                raise NameError('Array or Stream with name "%s" already exists ' "in SDFG" % name)
        self._arrays[name] = datadesc

        # Add free symbols to the SDFG global symbol storage
        for sym in datadesc.free_symbols:
            if sym.name not in self.symbols:
                self.add_symbol(sym.name, sym.dtype)

        return name

    def add_pgrid(self,
                  shape: ShapeType = None,
                  parent_grid: str = None,
                  color: Sequence[Union[Integral, bool]] = None,
                  exact_grid: RankType = None,
                  root: RankType = 0):
        """ Adds a process-grid to the process-grid descriptor store.
            For more details on process-grids, please read the documentation of the ProcessGrid class.
            :param shape: Shape of the process-grid (see `dims` parameter of [MPI_Cart_create](https://www.mpich.org/static/docs/latest/www3/MPI_Cart_create.html)), e.g., [2, 3, 3].
            :param parent_grid: Parent process-grid (similar to the `comm` parameter of [MPI_Cart_sub](https://www.mpich.org/static/docs/v3.2/www3/MPI_Cart_sub.html)).
            :param color: The i-th entry specifies whether the i-th dimension is kept in the sub-grid or is dropped (see `remain_dims` input of [MPI_Cart_sub](https://www.mpich.org/static/docs/v3.2/www3/MPI_Cart_sub.html)).
            :param exact_grid: If set then, out of all the sub-grids created, only the one that contains the rank with id `exact_grid` will be utilized for collective communication.
            :param root: Root rank (used for collective communication).
            :return: Name of the new process-grid descriptor.
        """

        if not (shape or parent_grid):
            raise ValueError("Process-grid must either have its shape defined or be linked to a parent-grid.")

        # convert strings to int if possible
        shape = shape or []
        newshape = []
        for s in shape:
            try:
                newshape.append(int(s))
            except:
                newshape.append(dace.symbolic.pystr_to_symbolic(s))
        shape = newshape

        grid_name = self._find_new_name('__pgrid')
        is_subgrid = (parent_grid is not None)
        if parent_grid and isinstance(parent_grid, str):
            parent_grid = self._pgrids[parent_grid]

        self._pgrids[grid_name] = ProcessGrid(grid_name, is_subgrid, shape, parent_grid, color, exact_grid, root)

        self.append_init_code(self._pgrids[grid_name].init_code())
        self.append_exit_code(self._pgrids[grid_name].exit_code())

        return grid_name

    def add_subarray(self,
                     dtype: dtypes.typeclass,
                     shape: ShapeType,
                     subshape: ShapeType,
                     pgrid: str = None,
                     correspondence: Sequence[Integral] = None):
        """ Adds a sub-array to the sub-array descriptor store.
            For more details on sub-arrays, please read the documentation of the SubArray class.
            :param dtype: Datatype of the array (see `oldtype` parameter of [MPI_Type_create_subarray](https://www.mpich.org/static/docs/v3.2/www3/MPI_Type_create_subarray.html)).
            :param shape: Shape of the sub-array (see `array_of_sizes` parameter of [MPI_Type_create_subarray](https://www.mpich.org/static/docs/v3.2/www3/MPI_Type_create_subarray.html)).
            :param subshape: Sub-shape of the sub-array (see `array_of_subsizes` parameter of [MPI_Type_create_subarray](https://www.mpich.org/static/docs/v3.2/www3/MPI_Type_create_subarray.html)).
            :param pgrid: Process-grid used for collective scatter/gather operations.
            :param correspondence: Matching among array dimensions and process-grid dimensions.
            :return: Name of the new sub-array descriptor.
        """

        # convert strings to int if possible
        shape = shape or []
        newshape = []
        for s in shape:
            try:
                newshape.append(int(s))
            except:
                newshape.append(dace.symbolic.pystr_to_symbolic(s))
        shape = newshape
        subshape = subshape or []
        newshape = []
        for s in subshape:
            try:
                newshape.append(int(s))
            except:
                newshape.append(dace.symbolic.pystr_to_symbolic(s))
        subshape = newshape

        subarray_name = self._find_new_name('__subarray')
        self._subarrays[subarray_name] = SubArray(subarray_name, dtype, shape, subshape, pgrid, correspondence)

        self.append_init_code(self._subarrays[subarray_name].init_code())
        self.append_exit_code(self._subarrays[subarray_name].exit_code())

        return subarray_name

    def add_rdistrarray(self, array_a: str, array_b: str):
        """ Adds a sub-array redistribution to the sub-array redistribution descriptor store.
            For more details on redistributions, please read the documentation of the RedistrArray class.
            :param array_a: Input sub-array descriptor.
            :param array_b: Output sub-array descriptor.
            :return: Name of the new redistribution descriptor.
        """

        rdistrarray_name = self._find_new_name('__rdistrarray')
        self._rdistrarrays[rdistrarray_name] = RedistrArray(rdistrarray_name, array_a, array_b)
        self.append_init_code(self._rdistrarrays[rdistrarray_name].init_code(self))
        self.append_exit_code(self._rdistrarrays[rdistrarray_name].exit_code(self))
        return rdistrarray_name

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
            raise ValueError("Cannot initalize or increment an empty loop variable")

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
        self.add_edge(before_state, guard, InterstateEdge(assignments=init))

        # Loop condition
        if condition_expr:
            cond_ast = CodeBlock(condition_expr).code
        else:
            cond_ast = CodeBlock('True').code
        self.add_edge(guard, loop_state, InterstateEdge(cond_ast))
        self.add_edge(guard, after_state, InterstateEdge(negate_expr(cond_ast)))

        # Loop incrementation
        incr = None if increment_expr is None else {loop_var: increment_expr}
        self.add_edge(loop_end_state, guard, InterstateEdge(assignments=incr))

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
            raise TypeError("state_id_or_label is not an int nor string: {}".format(state_id_or_label))

    def specialize(self, symbols: Dict[str, Any]):
        """ Sets symbolic values in this SDFG to constants.
            :param symbols: Values to specialize.
        """
        # Set symbol values to add
        syms = {
            # If symbols are passed, extract the value. If constants are
            # passed, use them directly.
            name: val.get() if isinstance(val, dace.symbolic.symbol) else val
            for name, val in symbols.items()
        }

        # Update constants
        for k, v in syms.items():
            self.add_constant(str(k), v)

    def optimize(self, optimizer=None) -> 'SDFG':
        """
        Optimize an SDFG using the CLI or external hooks.
        :param optimizer: If defines a valid class name, it will be called
                          during compilation to transform the SDFG as
                          necessary. If None, uses configuration setting.
        :return: An SDFG (returns self if optimizer is in place)
        """
        # Fill in scope entry/exit connectors
        self.fill_scope_connectors()

        optclass = _get_optimizer_class(optimizer)
        if optclass is not None:
            # Propagate memlets in the graph
            if self._propagate:
                propagate_memlets_sdfg(self)

            opt = optclass(self)
            sdfg = opt.optimize()
        else:
            sdfg = self

        sdfg.save(os.path.join('_dacegraphs', 'program.sdfg'))
        return sdfg

    def is_loaded(self) -> bool:
        """
        Returns True if the SDFG binary is already loaded in the current
        process.
        """
        # Avoid import loops
        from dace.codegen import compiled_sdfg as cs, compiler

        binary_filename = compiler.get_binary_name(self.build_folder, self.name)
        dll = cs.ReloadableDLL(binary_filename, self.name)
        return dll.is_loaded()

    def compile(self, output_file=None, validate=True) -> \
            'dace.codegen.compiler.CompiledSDFG':
        """ Compiles a runnable binary from this SDFG.
            :param output_file: If not None, copies the output library file to
                                the specified path.
            :param validate: If True, validates the SDFG prior to generating
                             code.
            :return: A callable CompiledSDFG object.
        """

        # Importing these outside creates an import loop
        from dace.codegen import codegen, compiler

        # Compute build folder path before running codegen
        build_folder = self.build_folder

        if Config.get_bool('compiler', 'use_cache'):
            # Try to see if a cached version of the binary exists
            binary_filename = compiler.get_binary_name(build_folder, self.name)
            if os.path.isfile(binary_filename):
                return compiler.load_from_file(self, binary_filename)

        ############################
        # DaCe Compilation Process #

        # Clone SDFG as the other modules may modify its contents
        sdfg = copy.deepcopy(self)
        # Fix the build folder name on the copied SDFG to avoid it changing
        # if the codegen modifies the SDFG (thereby changing its hash)
        sdfg.build_folder = build_folder

        # Rename SDFG to avoid runtime issues with clashing names
        index = 0
        while sdfg.is_loaded():
            sdfg._name = f'{self._name}_{index}'
            index += 1
        if self.name != sdfg.name:
            warnings.warn('SDFG "%s" is already loaded by another object, '
                          'recompiling under a different name.' % self.name)

        try:
            # Fill in scope entry/exit connectors
            sdfg.fill_scope_connectors()

            # Generate code for the program by traversing the SDFG state by state
            program_objects = codegen.generate_code(sdfg, validate=validate)
        except Exception:
            self.save(os.path.join('_dacegraphs', 'failing.sdfg'))
            raise

        # Generate the program folder and write the source files
        program_folder = compiler.generate_program_folder(sdfg, program_objects, build_folder)

        # Compile the code and get the shared library path
        shared_library = compiler.configure_and_compile(program_folder, sdfg.name)

        # If provided, save output to path or filename
        if output_file is not None:
            if os.path.isdir(output_file):
                output_file = os.path.join(output_file, os.path.basename(shared_library))
            shutil.copyfile(shared_library, output_file)

        # Ensure that an SDFG link file is created along with the SDFG, linking
        # it to the generating code and storing command line arguments that
        # were provided.
        if sys.argv is not None and len(sys.argv) > 0:
            os.makedirs(build_folder, exist_ok=True)
            with open(os.path.join(build_folder, 'program.sdfgl'), 'w') as launchfiles_file:
                launchfiles_file.write('name,SDFG_intermediate,SDFG,source,' +
                                       ','.join(['argv_' + str(i) for i in range(len(sys.argv))]) + '\n')
                launchfiles_file.write(sdfg.name + ',' + os.path.abspath(os.path.join(build_folder, 'program.sdfg')) +
                                       ',' + os.path.abspath(os.path.join('_dacegraphs', 'program.sdfg')) + ',' +
                                       os.path.abspath(sys.argv[0]) + ',' + ','.join([str(el) for el in sys.argv]))

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

        # Omit return values from arguments
        expected_args = collections.OrderedDict([(k, v) for k, v in expected_args.items()
                                                 if not k.startswith('__return')])
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('__return')}

        num_args_passed = len(args) + len(kwargs)
        num_args_expected = len(expected_args)
        if num_args_passed < num_args_expected:
            expected_kwargs = list(expected_args.keys())[len(args):]
            missing_args = [k for k in expected_kwargs if k not in kwargs]
            raise RuntimeError("Missing arguments to SDFG: '%s'" % (', '.join(missing_args)))
        elif num_args_passed > num_args_expected:
            unnecessary_args = []
            extra_args = len(args) - len(expected_args)
            if extra_args > 0:
                unnecessary_args.extend('Argument #%d' % (i + len(expected_args) + 1) for i in range(extra_args))
                unnecessary_args.extend(kwargs.keys())
            else:
                unnecessary_args = [k for k in kwargs.keys() if k not in expected_args]
            raise RuntimeError("Too many arguments to SDFG. Unnecessary arguments: %s" % ', '.join(unnecessary_args))
        positional_args = list(args)
        for i, arg in enumerate(expected_args):
            expected = expected_args[arg]
            if i < len(positional_args):
                passed = positional_args[i]
            else:
                if arg not in kwargs:
                    raise RuntimeError("Missing argument to DaCe program: {}".format(arg))
                passed = kwargs[arg]
            if types_only:
                desc = dt.create_datadescriptor(passed)
                if not expected.is_equivalent(desc):
                    raise TypeError("Type mismatch for argument: expected %s, got %s" % (expected, desc))
                else:
                    continue
            if isinstance(expected, dace.data.Array):
                if not dtypes.is_array(passed):
                    raise TypeError("Type mismatch for argument {}: "
                                    "expected array type, got {}".format(arg, type(passed)))
            elif (isinstance(expected, dace.data.Scalar) or isinstance(expected, dace.dtypes.typeclass)):
                if (not dtypes.isconstant(passed) and not isinstance(passed, dace.symbolic.symbol)):
                    raise TypeError("Type mismatch for argument {}: "
                                    "expected scalar type, got {}".format(arg, type(passed)))
            elif isinstance(expected, dace.data.Stream):
                if not isinstance(passed, dace.dtypes.stream):
                    raise TypeError("Type mismatch for argument {}: "
                                    "expected stream type, got {}".format(arg, type(passed)))
            else:
                raise NotImplementedError("Type checking not implemented for type {} (argument "
                                          "{})".format(type(expected).__name__, arg))

    def __call__(self, *args, **kwargs):
        """ Invokes an SDFG, generating and compiling code if necessary. """
        if Config.get_bool('optimizer', 'transform_on_call'):
            sdfg = self.optimize()
        else:
            sdfg = self

        binaryobj = sdfg.compile()

        # Verify passed arguments (unless disabled by the user)
        if dace.config.Config.get_bool("execution", "general", "check_args"):
            sdfg.argument_typecheck(args, kwargs)
        return binaryobj(*args, **kwargs)

    def fill_scope_connectors(self):
        """ Fills missing scope connectors (i.e., "IN_#"/"OUT_#" on entry/exit
            nodes) according to data on the memlets. """
        for state in self.nodes():
            state.fill_scope_connectors()

    def predecessor_state_transitions(self, state):
        """ Yields paths (lists of edges) that the SDFG can pass through
            before computing the given state. """
        return self.bfs_edges(state, reverse=True)

    def predecessor_states(self, state):
        """ Returns a list of unique states that the SDFG can pass through
            before computing the given state. """
        return (e.src for e in self.bfs_edges(state, reverse=True))

    def validate(self, references: Optional[Set[int]] = None) -> None:
        validate_sdfg(self, references)

    def is_valid(self) -> bool:
        """ Returns True if the SDFG is verified correctly (using `validate`).
        """
        try:
            self.validate()
        except InvalidSDFGError:
            return False
        return True

    def apply_strict_transformations(self, validate=True, validate_all=False):
        """
        This method is DEPRECATED in favor of ``simplify``.
        Applies safe transformations (that will surely increase the
        performance) on the SDFG. For example, this fuses redundant states
        (safely) and removes redundant arrays.

        B{Note:} This is an in-place operation on the SDFG.
        """
        warnings.warn('SDFG.apply_strict_transformations is deprecated, use SDFG.simplify instead.', DeprecationWarning)
        return self.simplify(validate, validate_all)

    def simplify(self, validate=True, validate_all=False):
        """ Applies safe transformations (that will surely increase the
            performance) on the SDFG. For example, this fuses redundant states
            (safely) and removes redundant arrays.

            B{Note:} This is an in-place operation on the SDFG.
        """
        # These are imported in order to update the transformation registry
        from dace.transformation import dataflow, interstate
        from dace.transformation.dataflow import RedundantReadSlice, RedundantWriteSlice
        from dace.sdfg import utils as sdutil
        # This is imported here to avoid an import loop
        from dace.transformation.transformation import simplification_transformations

        # First step is to apply multi-state inline, before any state fusion can
        # occur
        sdutil.inline_sdfgs(self, multistate=True)
        if validate_all:
            self.validate()
        sdutil.fuse_states(self)

        self.apply_transformations_repeated([RedundantReadSlice, RedundantWriteSlice],
                                            validate=False,
                                            permissive=False,
                                            validate_all=validate_all)
        self.apply_transformations_repeated(simplification_transformations(),
                                            validate=validate,
                                            permissive=False,
                                            validate_all=validate_all)

    def _initialize_transformations_from_type(
        self,
        xforms: Union[Type, List[Type], 'dace.transformation.PatternTransformation'],
        options: Union[Dict[str, Any], List[Dict[str, Any]], None] = None
    ) -> List['dace.transformation.PatternTransformation']:
        """
        Initializes given pattern-matching transformations with the options given.
        This method receives different formats and makes one kind of output.

        :param xforms: One or more PatternTransformation objects or classes.
        :param options: Zero or more transformation initialization option dictionaries.
        :return: List of PatternTransformation objects inititalized with their properties.
        """
        from dace.transformation import PatternTransformation  # Avoid import loops

        if isinstance(xforms, (PatternTransformation, type)):
            xforms = [xforms]
        if isinstance(options, dict):
            options = [options]
        options = options or [dict() for _ in xforms]

        if len(options) != len(xforms):
            raise ValueError('Length of options and transformations mismatch')

        result: List[PatternTransformation] = []
        for xftype, opts in zip(xforms, options):
            if isinstance(xftype, PatternTransformation):
                # Object was given, use as-is
                result.append(xftype)
            else:
                # Class was given, initialize
                opts = opts or {}
                try:
                    result.append(xftype(**opts))
                except TypeError:
                    # Backwards compatibility, transformation does not support ctor arguments
                    t = xftype()
                    # Set manually
                    for oname, oval in opts.items():
                        setattr(t, oname, oval)
                    result.append(t)

        return result

    def apply_transformations(self,
                              xforms: Union[Type, List[Type]],
                              options: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
                              validate: bool = True,
                              validate_all: bool = False,
                              permissive: bool = False,
                              states: Optional[List[Any]] = None,
                              print_report: Optional[bool] = None) -> int:
        """ This function applies a transformation or a sequence thereof
            consecutively. Operates in-place.

            :param xforms: A PatternTransformation class or a sequence.
            :param options: An optional dictionary (or sequence of dictionaries)
                            to modify transformation parameters.
            :param validate: If True, validates after all transformations.
            :param validate_all: If True, validates after every transformation.
            :param permissive: If True, operates in permissive mode.
            :param states: If not None, specifies a subset of states to
                           apply transformations on.
            :param print_report: Whether to show debug prints or not (None if
                                 the DaCe config option 'debugprint' should
                                 apply)
            :return: Number of transformations applied.

            Examples::

                      # Applies MapTiling, then MapFusion, followed by
                      # GPUTransformSDFG, specifying parameters only for the
                      # first transformation.
                      sdfg.apply_transformations(
                        [MapTiling, MapFusion, GPUTransformSDFG],
                        options=[{'tile_size': 16}, {}, {}])
        """
        from dace.transformation.passes.pattern_matching import PatternMatchAndApply  # Avoid import loops

        xforms = self._initialize_transformations_from_type(xforms, options)

        pazz = PatternMatchAndApply(xforms,
                                    permissive=permissive,
                                    validate=validate,
                                    validate_all=validate_all,
                                    states=states,
                                    print_report=print_report)
        results = pazz.apply_pass(self, {})

        # Return number of transformations applied
        if results is None:
            return 0
        return sum(len(v) for v in results.values())

    def apply_transformations_repeated(self,
                                       xforms: Union[Type, List[Type]],
                                       options: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
                                       validate: bool = True,
                                       validate_all: bool = False,
                                       permissive: bool = False,
                                       states: Optional[List[Any]] = None,
                                       print_report: Optional[bool] = None,
                                       order_by_transformation: bool = True,
                                       progress: Optional[bool] = None) -> int:
        """ This function repeatedly applies a transformation or a set of
            (unique) transformations until none can be found. Operates in-place.

            :param xforms: A PatternTransformation class or a set thereof.
            :param options: An optional dictionary (or sequence of dictionaries)
                            to modify transformation parameters.
            :param validate: If True, validates after all transformations.
            :param validate_all: If True, validates after every transformation.
            :param permissive: If True, operates in permissive mode.
            :param states: If not None, specifies a subset of states to
                           apply transformations on.
            :param print_report: Whether to show debug prints or not (None if
                                 the DaCe config option 'debugprint' should
                                 apply).
            :param order_by_transformation: Try to apply transformations ordered
                                            by class rather than SDFG.
            :param progress: If True, prints every intermediate transformation
                             applied. If False, never prints anything. If None
                             (default), prints only after 5 seconds of
                             transformations.
            :return: Number of transformations applied.

            Examples::

                    # Applies InlineSDFG until no more subgraphs can be inlined
                    sdfg.apply_transformations_repeated(InlineSDFG)
        """
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

        xforms = self._initialize_transformations_from_type(xforms, options)

        pazz = PatternMatchAndApplyRepeated(xforms, permissive, validate, validate_all, states, print_report, progress,
                                            order_by_transformation)
        results = pazz.apply_pass(self, {})

        # Return number of transformations applied
        if results is None:
            return 0
        return sum(len(v) for v in results.values())

    def apply_transformations_once_everywhere(self,
                                              xforms: Union[Type, List[Type]],
                                              options: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
                                              validate: bool = True,
                                              validate_all: bool = False,
                                              permissive: bool = False,
                                              states: Optional[List[Any]] = None,
                                              print_report: Optional[bool] = None,
                                              order_by_transformation: bool = True,
                                              progress: Optional[bool] = None) -> int:
        """ 
        This function applies a transformation or a set of (unique) transformations
        until throughout the entire SDFG once. Operates in-place.

            :param xforms: A PatternTransformation class or a set thereof.
            :param options: An optional dictionary (or sequence of dictionaries)
                            to modify transformation parameters.
            :param validate: If True, validates after all transformations.
            :param validate_all: If True, validates after every transformation.
            :param permissive: If True, operates in permissive mode.
            :param states: If not None, specifies a subset of states to
                           apply transformations on.
            :param print_report: Whether to show debug prints or not (None if
                                 the DaCe config option 'debugprint' should
                                 apply).
            :param order_by_transformation: Try to apply transformations ordered
                                            by class rather than SDFG.
            :param progress: If True, prints every intermediate transformation
                             applied. If False, never prints anything. If None
                             (default), prints only after 5 seconds of
                             transformations.
            :return: Number of transformations applied.

            Examples::

                    # Tiles all maps once
                    sdfg.apply_transformations_once_everywhere(MapTiling, options=dict(tile_size=16))
        """
        from dace.transformation.passes.pattern_matching import PatternApplyOnceEverywhere

        xforms = self._initialize_transformations_from_type(xforms, options)

        pazz = PatternApplyOnceEverywhere(xforms, permissive, validate, validate_all, states, print_report, progress,
                                          order_by_transformation)
        results = pazz.apply_pass(self, {})

        # Return number of transformations applied
        if results is None:
            return 0
        return sum(len(v) for v in results.values())

    def apply_gpu_transformations(self,
                                  states=None,
                                  validate=True,
                                  validate_all=False,
                                  permissive=False,
                                  sequential_innermaps=True,
                                  register_transients=True):
        """ Applies a series of transformations on the SDFG for it to
            generate GPU code.
            :param sequential_innermaps: Make all internal maps Sequential.
            :param register_transients: Make all transients inside GPU maps registers.
            :note: It is recommended to apply redundant array removal
            transformation after this transformation. Alternatively,
            you can simplify() after this transformation.
            :note: This is an in-place operation on the SDFG.
        """
        # Avoiding import loops
        from dace.transformation.interstate import GPUTransformSDFG

        self.apply_transformations(GPUTransformSDFG,
                                   options=dict(sequential_innermaps=sequential_innermaps,
                                                register_trans=register_transients),
                                   validate=validate,
                                   validate_all=validate_all,
                                   permissive=permissive,
                                   states=states)

    def apply_fpga_transformations(self, states=None, validate=True, validate_all=False, permissive=False):
        """ Applies a series of transformations on the SDFG for it to
            generate FPGA code.

            :note: This is an in-place operation on the SDFG.
        """
        # Avoiding import loops
        from dace.transformation.interstate import FPGATransformSDFG

        self.apply_transformations(FPGATransformSDFG,
                                   validate=validate,
                                   validate_all=validate_all,
                                   permissive=permissive,
                                   states=states)

    def expand_library_nodes(self, recursive=True):
        """
        Recursively expand all unexpanded library nodes in the SDFG,
        resulting in a "pure" SDFG that the code generator can handle.
        :param recursive: If True, expands all library nodes recursively,
                          including library nodes that expand to library nodes.
        """

        states = list(self.states())
        while len(states) > 0:
            state = states.pop()
            expanded_something = False
            for node in list(state.nodes()):  # Make sure we have a copy
                if isinstance(node, nd.NestedSDFG):
                    node.sdfg.expand_library_nodes(recursive=recursive)  # Call recursively
                elif isinstance(node, nd.LibraryNode):
                    impl_name = node.expand(self, state)
                    if Config.get_bool('debugprint'):
                        print('Automatically expanded library node \"{}\" with '
                              'implementation \"{}\".'.format(str(node), impl_name))
                    # We made a copy of the original list of nodes, so we keep
                    # iterating even though this list has now changed
                    if recursive:
                        expanded_something = True
            if expanded_something:
                states.append(state)  # Nodes have changed. Check state again

    def generate_code(self):
        """ Generates code from this SDFG and returns it.
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

        # Generate code for the program by traversing the SDFG state by state
        program_code = codegen.generate_code(sdfg)

        return program_code

    def make_array_memlet(self, array: str):
        """Convenience method to generate a Memlet that transfers a full array.

           :param array: the name of the array
           :return: a Memlet that fully transfers array
        """
        return dace.Memlet.from_array(array, self.arrays[array])


def _get_optimizer_class(class_override):
    """ Imports and returns a class string defined in the configuration
        (under "optimizer.interface") or overridden in the input
        class_override argument. Empty string, False, or failure to find the
        class skips the process.

        :note: This method uses pydoc to locate the class.
    """
    clazz = class_override
    if class_override is None:
        clazz = Config.get("optimizer", "interface")

    if clazz == "" or clazz is False or str(clazz).strip() == "":
        return None

    result = locate(clazz)
    if result is None:
        warnings.warn('Optimizer interface class "%s" not found' % clazz)

    return result
