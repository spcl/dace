# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import collections
import copy
import ctypes
import gzip
from numbers import Integral
import os
import json
from hashlib import md5, sha256
import random
import shutil
import sys
from typing import Any, AnyStr, Dict, List, Optional, Sequence, Set, Tuple, Type, TYPE_CHECKING, Union
import warnings

import dace
from dace.sdfg.graph import generate_element_id
import dace.serialize
from dace import (data as dt, hooks, memlet as mm, subsets as sbs, dtypes, symbolic)
from dace.sdfg.replace import replace_properties_dict
from dace.sdfg.validation import (InvalidSDFGError, validate_sdfg)
from dace.config import Config
from dace.frontend.python import astutils
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, SDFGState, ControlFlowRegion
from dace.distr_types import ProcessGrid, SubArray, RedistrArray
from dace.dtypes import validate_name
from dace.properties import (DebugInfoProperty, EnumProperty, ListProperty, make_properties, Property, CodeProperty,
                             TransformationHistProperty, OptionalSDFGReferenceProperty, DictProperty, CodeBlock)
from typing import BinaryIO

# NOTE: In shapes, we try to convert strings to integers. In ranks, a string should be interpreted as data (scalar).
ShapeType = Sequence[Union[Integral, str, symbolic.symbol, symbolic.SymExpr, symbolic.sympy.Basic]]
RankType = Union[Integral, str, symbolic.symbol, symbolic.SymExpr, symbolic.sympy.Basic]

if TYPE_CHECKING:
    from dace.codegen.instrumentation.report import InstrumentationReport
    from dace.codegen.instrumentation.data.data_report import InstrumentedDataReport
    from dace.codegen.compiled_sdfg import CompiledSDFG
    from dace.sdfg.analysis.schedule_tree.treenodes import ScheduleTreeScope


class NestedDict(dict):

    def __init__(self, mapping=None):
        mapping = mapping or {}
        super(NestedDict, self).__init__(mapping)

    def __getitem__(self, key):
        tokens = key.split('.') if isinstance(key, str) else [key]
        token = tokens.pop(0)
        result = super(NestedDict, self).__getitem__(token)
        while tokens:
            token = tokens.pop(0)
            result = result.members[token]
        return result

    def __setitem__(self, key, val):
        if isinstance(key, str) and '.' in key:
            raise KeyError('NestedDict does not support setting nested keys')
        super(NestedDict, self).__setitem__(key, val)

    def __contains__(self, key):
        tokens = key.split('.') if isinstance(key, str) else [key]
        token = tokens.pop(0)
        result = super(NestedDict, self).__contains__(token)
        desc = None
        while tokens and result:
            if desc is None:
                desc = super(NestedDict, self).__getitem__(token)
            else:
                desc = desc.members[token]
            token = tokens.pop(0)
            result = hasattr(desc, 'members') and token in desc.members
        return result

    def keys(self):
        result = super(NestedDict, self).keys()
        for k, v in self.items():
            if isinstance(v, dt.Structure):
                result |= set(map(lambda x: k + '.' + x, v.keys()))
        return result


def _arrays_to_json(arrays):
    if arrays is None:
        return None
    return {k: dace.serialize.to_json(v) for k, v in arrays.items()}


def _arrays_from_json(obj, context=None):
    if obj is None:
        return {}
    return {k: dace.serialize.from_json(v, context) for k, v in obj.items()}


def _nested_arrays_from_json(obj, context=None):
    if obj is None:
        return NestedDict({})
    return NestedDict({k: dace.serialize.from_json(v, context) for k, v in obj.items()})


def _replace_dict_keys(d, old, new):
    if old == new:
        warnings.warn(f"Trying to replace key with the same name {old} ... skipping.")
        return
    if old in d:
        if new in d:
            warnings.warn('"%s" already exists in SDFG' % new)
        d[new] = d[old]
        del d[old]


def _replace_dict_values(d, old, new):
    for k, v in d.items():
        if v == old:
            d[k] = new


def memlets_in_ast(node: ast.AST, arrays: Dict[str, dt.Data]) -> List[mm.Memlet]:
    """
    Generates a list of memlets from each of the subscripts that appear in the Python AST.
    Assumes the subscript slice can be coerced to a symbolic expression (e.g., no indirect access).

    :param node: The AST node to find memlets in.
    :param arrays: A dictionary mapping array names to their data descriptors (a-la ``sdfg.arrays``)
    :return: A list of Memlet objects in the order they appear in the AST.
    """
    result: List[mm.Memlet] = []

    for subnode in ast.walk(node):
        if isinstance(subnode, ast.Subscript):
            data = astutils.rname(subnode.value)
            data, slc = astutils.subscript_to_slice(subnode, arrays)
            subset = sbs.Range(slc)
            result.append(mm.Memlet(data=data, subset=subset))

    return result


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

    assignments = Property(dtype=dict, desc="Assignments to perform upon transition (e.g., 'x=x+1; y = 0')")
    condition = CodeProperty(desc="Transition condition", default=CodeBlock("1"))
    guid = Property(dtype=str, allow_none=False)

    def __init__(self,
                 condition: Optional[Union[CodeBlock, str, ast.AST, list]] = None,
                 assignments: Optional[Dict] = None):
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

        self.guid = generate_element_id(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'condition' or name == '_condition':
            super().__setattr__('_cond_sympy', None)
            super().__setattr__('_uncond', None)
        return super().__setattr__(name, value)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'guid':  # Skip ID
                continue
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    @staticmethod
    def _convert_assignment(assignment) -> str:
        if isinstance(assignment, ast.AST):
            return CodeBlock(assignment).as_string
        return str(assignment)

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

    def read_symbols(self) -> Set[str]:
        """
        Returns a set of symbols read in this edge (including symbols in the condition and assignment values).
        """
        # Symbols in conditions and assignments
        result = set(map(str, dace.symbolic.symbols_in_ast(self.condition.code[0])))
        for assign in self.assignments.values():
            result |= symbolic.free_symbols_and_functions(assign)

        return result

    def used_symbols(self, all_symbols: bool = False, union_lhs_symbols: bool = False) -> Set[str]:
        """ Returns a set of symbols used in this edge's properties. """
        # NOTE: The former algorithm for computing an edge's free symbols was:
        #       `self.read_symbols() - set(self.assignments.keys())`
        #       The issue with the above algorithm is that any symbols that are first read and then assigned will not
        #       be considered free symbols. For example, the former algorithm will fail for the following edges:
        #       - assignments = {'i': 'i + 1'}
        #       - condition = 'i < 10', assignments = {'i': '3'}
        #       - assignments = {'j': 'i + 1', 'i': '3'}
        #       The new algorithm below addresses the issue by iterating over the edge's condition and assignments and
        #       exlcuding keys from being considered "defined" if they have been already read.

        # Symbols in conditions are always free, because the condition is executed before the assignments
        cond_symbols = set(map(str, dace.symbolic.symbols_in_ast(self.condition.code[0])))
        # Symbols in assignment keys are candidate defined symbols
        lhs_symbols = set()
        # Symbols in assignment values are candidate free symbols
        rhs_symbols = set()
        for lhs, rhs in self.assignments.items():
            # Always add LHS symbols to the set of candidate free symbols
            rhs_symbols |= set(map(str, dace.symbolic.symbols_in_ast(ast.parse(rhs))))
            # Add the RHS to the set of candidate defined symbols ONLY if it has not been read yet
            # This also solves the ordering issue that may arise in cases like the 3rd example above
            if lhs not in cond_symbols and lhs not in rhs_symbols:
                lhs_symbols.add(lhs)
        # Return the set of candidate free symbols minus the set of candidate defined symbols
        if union_lhs_symbols:
            return (cond_symbols | rhs_symbols | lhs_symbols)
        else:
            return (cond_symbols | rhs_symbols) - lhs_symbols

    def used_sdfg_symbols(self, arrays: Dict[str, dt.Data], union_lhs_symbols: bool = False) -> Set[str]:
        """
        Returns a set of symbols used in this edge's properties (i.e., condition and assignments) that are not
        registered as data descriptors to the SDFG.
        :param arrays: A dictionary mapping names to their corresponding data descriptors (`sdfg.arrays`)
        :param union_lhs_symbols: If True, returns all symbols used in the edge, including those on the LHS.
        :return: A set of symbols names.
        """
        # all_symbols does not matter but need to provide something
        symbol_names = self.used_symbols(all_symbols=True, union_lhs_symbols=union_lhs_symbols)
        assert all([isinstance(s, str) for s in symbol_names])
        real_symbol_names = {s for s in symbol_names if s not in arrays}
        assert all([isinstance(s, str) for s in real_symbol_names])
        return real_symbol_names

    def used_arrays(self, arrays: Dict[str, dt.Data], union_lhs_symbols: bool = False) -> Set[str]:
        """
        Returns a set of arrays used in this edge's properties (i.e., condition and assignments).
        :param arrays: A dictionary mapping names to their corresponding data descriptors (`sdfg.arrays`)
        :param union_lhs_symbols: If True, returns all symbols used in the edge, including those on the LHS.
        :return: A set of array names.
        """
        # all_symbols does not matter but need to provide something
        symbol_names = self.used_symbols(all_symbols=True, union_lhs_symbols=union_lhs_symbols)
        assert all([isinstance(s, str) for s in symbol_names])
        used_array_names = {s for s in symbol_names if s in arrays}
        assert all([isinstance(s, str) for s in used_array_names])
        return used_array_names

    @property
    def free_symbols(self) -> Set[str]:
        """ Returns a set of symbols used in this edge's properties. """
        return self.used_symbols(all_symbols=True)

    def replace_dict(self, repl: Dict[str, str], replace_keys=True) -> None:
        """
        Replaces all given keys with their corresponding values.

        :param repl: Replacement dictionary.
        :param replace_keys: If False, skips replacing assignment keys.
        """
        if not repl:
            return

        if replace_keys:
            for name, new_name in repl.items():
                _replace_dict_keys(self.assignments, name, new_name)

        for k, v in self.assignments.items():
            vast = ast.parse(v)
            vast = astutils.ASTFindReplace(repl).visit(vast)
            newv = astutils.unparse(vast)
            if newv != v:
                self.assignments[k] = newv
        condition = ast.parse(self.condition.as_string)
        condition = astutils.ASTFindReplace(repl).visit(condition)
        newc = astutils.unparse(condition)
        if newc != condition:
            self.condition.as_string = newc
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

        inferred_lhs_symbols = {k: infer_expr_type(v, alltypes) for k, v in self.assignments.items()}

        # Symbols in assignment keys are candidate newly defined symbols
        lhs_symbols = set()
        # Symbols already defined
        rhs_symbols = set()
        for lhs, rhs in self.assignments.items():
            rhs_symbols |= symbolic.free_symbols_and_functions(rhs)
            # Only add LHS to the set of candidate newly defined symbols if it has not been defined yet
            if lhs not in rhs_symbols:
                lhs_symbols.add(lhs)

        return {k: v for k, v in inferred_lhs_symbols.items() if k in lhs_symbols}

    def get_read_memlets(self, arrays: Dict[str, dt.Data]) -> List[mm.Memlet]:
        """
        Returns a list of memlets (with data descriptors and subsets) used in this edge. This includes
        both reads in the condition and in every assignment.

        :param arrays: A dictionary mapping names to their corresponding data descriptors (a-la ``sdfg.arrays``)
        :return: A list of Memlet objects for each read.
        """
        result: List[mm.Memlet] = []
        result.extend(memlets_in_ast(self.condition.code[0], arrays))
        for assign in self.assignments.values():
            vast = ast.parse(assign)
            result.extend(memlets_in_ast(vast, arrays))

        return result

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
        assignments = ','.join(['%s=%s' % (k, v) for k, v in self.assignments.items()])

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
class SDFG(ControlFlowRegion):
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

    name = Property(dtype=str, desc="Name of the SDFG")
    arg_names = ListProperty(element_type=str, desc='Ordered argument names (used for calling conventions).')
    constants_prop: Dict[str, Tuple[dt.Data, Any]] = Property(
        dtype=dict,
        default={},
        desc='Compile-time constants. The dictionary maps between a constant name to '
        'a tuple of its type and the actual constant data.')
    _arrays = Property(dtype=NestedDict,
                       desc="Data descriptors for this SDFG",
                       to_json=_arrays_to_json,
                       from_json=_nested_arrays_from_json)
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

    using_explicit_control_flow = Property(dtype=bool,
                                           default=False,
                                           desc="Whether the SDFG contains explicit control flow constructs")

    def __init__(self,
                 name: str,
                 constants: Dict[str, Tuple[dt.Data, Any]] = None,
                 propagate: bool = True,
                 parent=None):
        """ Constructs a new SDFG.

            :param name: Name for the SDFG (also used as the filename for
                         the compiled shared library).
            :param constants: Additional dictionary of compile-time constants
                              {name (str): tuple(type (dace.data.Data), value (Any))}.
            :param propagate: If False, disables automatic propagation of
                              memlet subsets from scopes outwards. Saves
                              processing time but disallows certain
                              transformations.
            :param parent: The parent SDFG or SDFG state (for nested SDFGs).
        """
        super(SDFG, self).__init__()
        self.name = name
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
        self._arrays = NestedDict()  # type: Dict[str, dt.Array]
        self.arg_names = []
        self._labels: Set[str] = set()
        self.global_code = {'frame': CodeBlock("", dtypes.Language.CPP)}
        self.init_code = {'frame': CodeBlock("", dtypes.Language.CPP)}
        self.exit_code = {'frame': CodeBlock("", dtypes.Language.CPP)}
        self.orig_sdfg = None
        self.transformation_hist = []
        self.callback_mapping = {}
        # Counter to make it easy to create temp transients
        self._temp_transients = 0

        # Helper fields to avoid code generation and compilation
        self._regenerate_code = True
        self._recompile = True

        # Grid-distribution-related fields
        self._pgrids = {}
        self._subarrays = {}
        self._rdistrarrays = {}

        # Counter to resolve name conflicts
        self._orig_name = name
        self._num = 0

        self._sdfg = self

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # Skip derivative attributes and GUID
            if k in ('_cached_start_block', '_edges', '_nodes', '_parent', '_parent_sdfg', '_parent_nsdfg_node',
                     '_cfg_list', '_transformation_hist', 'guid'):
                continue
            setattr(result, k, copy.deepcopy(v, memo))
        # Copy edges and nodes
        result._edges = copy.deepcopy(self._edges, memo)
        result._nodes = copy.deepcopy(self._nodes, memo)
        result._cached_start_block = copy.deepcopy(self._cached_start_block, memo)
        # Copy parent attributes
        for k in ('_parent', '_parent_sdfg', '_parent_nsdfg_node'):
            if id(getattr(self, k)) in memo:
                setattr(result, k, memo[id(getattr(self, k))])
            else:
                setattr(result, k, None)
        # Copy SDFG list and transformation history
        if hasattr(self, '_transformation_hist'):
            setattr(result, '_transformation_hist', copy.deepcopy(self._transformation_hist, memo))
        result._cfg_list = []
        if self._parent_sdfg is None:
            # Avoid import loops
            from dace.transformation.passes.fusion_inline import FixNestedSDFGReferences

            result._cfg_list = result.reset_cfg_list()
            fixed = FixNestedSDFGReferences().apply_pass(result, {})
            if fixed:
                warnings.warn(f'Fixed {fixed} nested SDFG parent references during deep copy.')

        return result

    @property
    def sdfg_id(self):
        """
        Returns the unique index of the current CFG within the current tree of CFGs (Top-level CFG/SDFG is 0, nested
        CFGs/SDFGs are greater).
        :note: ``sdfg_id`` is deprecated, please use ``cfg_id`` instead.
        """
        return self.cfg_id

    def to_json(self, hash=False):
        """ Serializes this object to JSON format.

            :return: A string representing the JSON-serialized SDFG.
        """
        # Location in the SDFG list (only for root SDFG)
        is_root = self.parent_sdfg is None
        if is_root:
            self.reset_cfg_list()

        tmp = super().to_json()

        # Ensure properties are serialized correctly
        if 'constants_prop' in tmp['attributes']:
            tmp['attributes']['constants_prop'] = json.loads(dace.serialize.dumps(tmp['attributes']['constants_prop']))

        tmp['attributes']['name'] = self.name
        if hash:
            tmp['attributes']['hash'] = self.hash_sdfg(tmp)

        if is_root:
            tmp['dace_version'] = dace.__version__

        return tmp

    @classmethod
    def from_json(cls, json_obj, context=None):
        context = context or {'sdfg': None}
        _type = json_obj['type']
        if _type != cls.__name__:
            raise TypeError("Class type mismatch")

        attrs = json_obj['attributes']
        nodes = json_obj['nodes']
        edges = json_obj['edges']

        if 'constants_prop' in attrs:
            constants_prop = dace.serialize.loads(dace.serialize.dumps(attrs['constants_prop']))
        else:
            constants_prop = None

        ret = SDFG(name=attrs['name'], constants=constants_prop, parent=context['sdfg'])

        dace.serialize.set_properties_from_json(ret, json_obj, ignore_properties={'constants_prop', 'name', 'hash'})

        nodelist = []
        for n in nodes:
            nci = copy.copy(context)
            nci['sdfg'] = ret

            block = dace.serialize.from_json(n, context=nci)
            ret.add_node(block)
            nodelist.append(block)

        for e in edges:
            e = dace.serialize.from_json(e)
            ret.add_edge(nodelist[int(e.src)], nodelist[int(e.dst)], e.data)

        if 'start_block' in json_obj:
            ret._start_block = json_obj['start_block']

        return ret

    def hash_sdfg(self, jsondict: Optional[Dict[str, Any]] = None) -> str:
        """
        Returns a hash of the current SDFG, without considering IDs and attribute names.

        :param jsondict: If not None, uses given JSON dictionary as input.
        :return: The hash (in SHA-256 format).
        """

        def keyword_remover(json_obj: Any, last_keyword=""):
            # Makes non-unique in SDFG hierarchy v2
            # Recursively remove attributes from the SDFG which are not used in
            # uniquely representing the SDFG. This, among other things, includes
            # the hash, name, transformation history, and meta attributes.
            if isinstance(json_obj, dict):
                if 'cfg_list_id' in json_obj:
                    del json_obj['cfg_list_id']

                keys_to_delete = []
                kv_to_recurse = []
                for key, value in json_obj.items():
                    if (isinstance(key, str)
                            and (key.startswith('_meta_')
                                 or key in ['name', 'hash', 'orig_sdfg', 'transformation_hist', 'instrument', 'guid'])):
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
        if dataname in self.constants_prop:
            return self.constants_prop[dataname][0]
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

        repldict = {k: v for k, v in repldict.items() if k != v}
        if symrepl:
            symrepl = {k: v for k, v in symrepl.items() if str(k) != str(v)}

        symrepl = symrepl or {
            symbolic.pystr_to_symbolic(k): symbolic.pystr_to_symbolic(v) if isinstance(k, str) else v
            for k, v in repldict.items()
        }

        # Replace in arrays and symbols (if a variable name)
        if replace_keys:
            # Filter out nested data names, as we cannot and do not want to replace names in nested data descriptors
            repldict_filtered = {k: v for k, v in repldict.items() if '.' not in k}
            for name, new_name in repldict_filtered.items():
                if validate_name(new_name):
                    _replace_dict_keys(self._arrays, name, new_name)
                    _replace_dict_keys(self.symbols, name, new_name)
                    _replace_dict_keys(self.constants_prop, name, new_name)
                    _replace_dict_keys(self.callback_mapping, name, new_name)
                    _replace_dict_values(self.callback_mapping, name, new_name)

        # Replace inside data descriptors
        for array in self.arrays.values():
            replace_properties_dict(array, repldict, symrepl)

        super().replace_dict(repldict, symrepl, replace_in_graph, replace_keys)

    def add_symbol(self, name, stype, find_new_name: bool = False):
        """ Adds a symbol to the SDFG.

            :param name: Symbol name.
            :param stype: Symbol type.
            :param find_new_name: Find a new name.
        """
        if find_new_name:
            name = self._find_new_name(name)
        else:
            # We do not check for data constant, because there is a link between the constants and
            #  the data descriptors.
            if name in self.symbols:
                raise FileExistsError(f'Symbol "{name}" already exists in SDFG')
            if name in self.arrays:
                raise FileExistsError(f'Cannot create symbol "{name}", the name is used by a data descriptor.')
            if name in self._subarrays:
                raise FileExistsError(f'Cannot create symbol "{name}", the name is used by a subarray.')
            if name in self._rdistrarrays:
                raise FileExistsError(f'Cannot create symbol "{name}", the name is used by a RedistrArray.')
            if name in self._pgrids:
                raise FileExistsError(f'Cannot create symbol "{name}", the name is used by a ProcessGrid.')
        if not isinstance(stype, dtypes.typeclass):
            stype = dtypes.dtype_to_typeclass(stype)
        self.symbols[name] = stype
        return name

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
        return self.start_block

    @start_state.setter
    def start_state(self, state_id):
        self.start_block = state_id

    @property
    def regenerate_code(self):
        return self._regenerate_code

    @regenerate_code.setter
    def regenerate_code(self, value):
        self._regenerate_code = value

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
        Appends a transformation to the transformation history of this SDFG.
        If this is the first transformation being applied, it also saves the
        initial state of the SDFG to return to and play back the history.

        :param transformation: The transformation to append.
        """
        if Config.get_bool('store_history') is False:
            return
        # Make sure the transformation is appended to the root SDFG.
        if self.cfg_id != 0:
            self.cfg_list[0].append_transformation(transformation)
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

    def get_instrumentation_reports(self) -> List['InstrumentationReport']:
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
        try:
            files = os.listdir(path)
        except FileNotFoundError:
            return
        for fname in files:
            if not fname.startswith('report-'):
                continue
            os.unlink(os.path.join(path, fname))

    def get_latest_report_path(self) -> Optional[str]:
        """
        Returns an instrumentation report file path from the latest run of this SDFG, or
        None if the file does not exist.

        :return: A path to the latest instrumentation report, or None if one does not exist.
        """
        path = os.path.join(self.build_folder, 'perf')
        files = [f for f in os.listdir(path) if f.startswith('report-')]
        if len(files) == 0:
            return None

        return os.path.join(path, sorted(files, reverse=True)[0])

    def get_latest_report(self) -> Optional['InstrumentationReport']:
        """
        Returns an instrumentation report from the latest run of this SDFG, or
        None if the file does not exist.

        :return: A timestamped InstrumentationReport object, or None if does not exist.
        """
        # Avoid import loops
        from dace.codegen.instrumentation import InstrumentationReport

        path = self.get_latest_report_path()
        if path is None:
            return None

        return InstrumentationReport(path)

    def get_instrumented_data(self, timestamp: Optional[int] = None) -> Optional['InstrumentedDataReport']:
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

    def call_with_instrumented_data(self, dreport: 'InstrumentedDataReport', *args, **kwargs):
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

        # Verify passed arguments (if enabled)
        if Config.get_bool('frontend', 'check_args'):
            self.argument_typecheck(args, kwargs)
        return binaryobj(*args, **kwargs)

    ##########################################

    def as_schedule_tree(self, in_place: bool = False) -> 'ScheduleTreeScope':
        """
        Creates a schedule tree from this SDFG and all nested SDFGs. The schedule tree is a tree of nodes that represent
        the execution order of the SDFG.
        Each node in the tree can either represent a single statement (symbol assignment, tasklet, copy, library node,
        etc.) or a ``ScheduleTreeScope`` block (map, for-loop, pipeline, etc.) that contains other nodes.

        It can be used to generate code from an SDFG, or to perform schedule transformations on the SDFG. For example,
        erasing an empty if branch, or merging two consecutive for-loops.

        :param in_place: If True, the SDFG is modified in-place. Otherwise, a copy is made. Note that the SDFG might
                         not be usable after the conversion if ``in_place`` is True!
        :return: A schedule tree representing the given SDFG.
        """
        # Avoid import loop
        from dace.sdfg.analysis.schedule_tree import sdfg_to_tree as s2t
        return s2t.as_schedule_tree(self, in_place=in_place)

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
            for state in self.states():
                for node in state.nodes():
                    if isinstance(node, nd.AccessNode) and node.data == name:
                        raise ValueError(f"Cannot remove data descriptor "
                                         f"{name}: it is accessed by node "
                                         f"{node} in state {state}.")

        del self._arrays[name]

    def reset_sdfg_list(self):
        """
        Reset the CFG list when changes have been made to the SDFG's CFG tree.
        This collects all control flow graphs recursively and propagates the collection to all CFGs as the new CFG list.
        :note: ``reset_sdfg_list`` is deprecated, please use ``reset_cfg_list`` instead.

        :return: The newly updated CFG list.
        """
        warnings.warn('reset_sdfg_list is deprecated, use reset_cfg_list instead', DeprecationWarning)
        return self.reset_cfg_list()

    def update_sdfg_list(self, sdfg_list):
        """
        Given a collection of CFGs, add them all to the current SDFG's CFG list.
        Any CFGs already in the list are skipped, and the newly updated list is propagated across all CFGs in the CFG
        tree.
        :note: ``update_sdfg_list`` is deprecated, please use ``update_cfg_list`` instead.

        :param sdfg_list: The collection of CFGs to add to the CFG list.
        """
        warnings.warn('update_sdfg_list is deprecated, use update_cfg_list instead', DeprecationWarning)
        self.update_cfg_list(sdfg_list)

    @property
    def sdfg_list(self) -> List['ControlFlowRegion']:
        warnings.warn('sdfg_list is deprecated, use cfg_list instead', DeprecationWarning)
        return self.cfg_list

    def set_sourcecode(self, code: str, lang=None):
        """ Set the source code of this SDFG (for IDE purposes).

            :param code: A string of source code.
            :param lang: A string representing the language of the source code,
                         for syntax highlighting and completion.
        """
        self.sourcecode = {'code': code, 'language': lang}

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
                return dtype.dtype.type(value)
            raise TypeError('Unsupported data type %s' % dtype)

        result.update({k: cast(*v) for k, v in self.constants_prop.items()})
        return result

    def add_constant(self, name: str, value: Any, dtype: dt.Data = None):
        """
        Adds/updates a new compile-time constant to this SDFG.

        A constant may either be a scalar or a numpy ndarray thereof. It is not an
        error if there is already a symbol or an array with the same name inside
        the SDFG. However, the data descriptors must refer to the same type.

        :param name: The name of the constant.
        :param value: The constant value.
        :param dtype: Optional data type of the symbol, or None to deduce automatically.
        """
        if name in self._subarrays:
            raise FileExistsError(f'Can not create constant "{name}", the name is used by a subarray.')
        if name in self._rdistrarrays:
            raise FileExistsError(f'Can not create constant "{name}", the name is used by a RedistrArray.')
        if name in self._pgrids:
            raise FileExistsError(f'Can not create constant "{name}", the name is used by a ProcessGrid.')
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

    def remove_node(self, node: SDFGState):
        if node is self._cached_start_block:
            self._cached_start_block = None
        return super().remove_node(node)

    def states(self):
        """ Returns the states in this SDFG, recursing into state scope blocks. """
        return list(self.all_states())

    def arrays_recursive(self, include_nested_data: bool = False):
        """ Iterate over all arrays in this SDFG, including arrays within
            nested SDFGs. Yields 3-tuples of (sdfg, array name, array).

            :param include_nested_data: If True, also yields nested data.
            :return: A generator of (sdfg, array name, array) tuples.
        """

        def _yield_nested_data(name, arr):
            for nname, narr in arr.members.items():
                if isinstance(narr, dt.Structure):
                    yield from _yield_nested_data(name + '.' + nname, narr)
                yield self, name + '.' + nname, narr

        for aname, arr in self.arrays.items():
            if isinstance(arr, dt.Structure) and include_nested_data:
                yield from _yield_nested_data(aname, arr)
            yield self, aname, arr
        for state in self.states():
            for node in state.nodes():
                if isinstance(node, nd.NestedSDFG):
                    yield from node.sdfg.arrays_recursive(include_nested_data=include_nested_data)

    def _used_symbols_internal(self,
                               all_symbols: bool,
                               defined_syms: Optional[Set] = None,
                               free_syms: Optional[Set] = None,
                               used_before_assignment: Optional[Set] = None,
                               keep_defined_in_mapping: bool = False,
                               with_contents: bool = True) -> Tuple[Set[str], Set[str], Set[str]]:
        defined_syms = set() if defined_syms is None else defined_syms
        free_syms = set() if free_syms is None else free_syms
        used_before_assignment = set() if used_before_assignment is None else used_before_assignment

        # Exclude data descriptor names and constants
        for name in self.arrays.keys():
            defined_syms.add(name)

        defined_syms |= set(self.constants_prop.keys())

        # Add used symbols from init and exit code
        for code in self.init_code.values():
            free_syms |= symbolic.symbols_in_code(code.as_string, self.symbols.keys())
        for code in self.exit_code.values():
            free_syms |= symbolic.symbols_in_code(code.as_string, self.symbols.keys())

        return super()._used_symbols_internal(all_symbols=all_symbols,
                                              keep_defined_in_mapping=keep_defined_in_mapping,
                                              defined_syms=defined_syms,
                                              free_syms=free_syms,
                                              used_before_assignment=used_before_assignment,
                                              with_contents=with_contents)

    def get_all_toplevel_symbols(self) -> Set[str]:
        """
        Returns a set of all symbol names that are used by the SDFG's state machine.
        This includes all symbols in the descriptor repository and interstate edges,
        whether free or defined. Used to identify duplicates when, e.g., inlining or
        dealiasing a set of nested SDFGs.
        """
        # Exclude constants and data descriptor names
        exclude = set(self.arrays.keys()) | set(self.constants_prop.keys())

        syms = set()

        # Start with the set of SDFG free symbols
        syms |= set(self.symbols.keys())

        # Add inter-state symbols
        for e in self.edges():
            syms |= set(e.data.assignments.keys())
            syms |= e.data.free_symbols

        # Subtract exluded symbols
        return syms - exclude

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
            # Get dictionaries of subsets read and written from each state
            rs, ws = state._read_and_write_sets()
            read_set |= rs.keys()
            write_set |= ws.keys()

        array_names = self.arrays.keys()
        for edge in self.all_interstate_edges():
            read_set |= edge.data.free_symbols & array_names

        # By definition, data that is referenced by symbolic condition expressions
        # (branching condition, loop condition, ...) is also part of the read set.
        for cfr in self.all_control_flow_regions():
            read_set |= cfr.used_symbols(all_symbols=True, with_contents=False) & array_names

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

        # Add global free symbols used in the generated code to scalar arguments
        free_symbols = free_symbols if free_symbols is not None else self.used_symbols(all_symbols=False)
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
        free_symbols = free_symbols if free_symbols is not None else self.used_symbols(all_symbols=False)
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

    def python_signature_arglist(self, with_types=True, for_call=False, with_arrays=True, arglist=None) -> List[str]:
        """ Returns a list of arguments necessary to call this SDFG,
            formatted as a list of Data-Centric Python definitions.

            :param with_types: If True, includes argument types in the result.
            :param for_call: If True, returns arguments that can be used when
                             calling the SDFG.
            :param with_arrays: If True, includes arrays, otherwise,
                                only symbols and scalars are included.
            :param arglist: An optional cached argument list.
            :return: A list of strings. For example: `['A: dace.float32[M]', 'b: dace.int32']`.
        """
        arglist = arglist or self.arglist(scalars_only=not with_arrays, free_symbols=[])
        return [v.as_python_arg(name=k, with_types=with_types, for_call=for_call) for k, v in arglist.items()]

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

    def python_signature(self, with_types=True, for_call=False, with_arrays=True, arglist=None) -> str:
        """ Returns a Data-Centric Python signature of this SDFG, used when generating code.

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
        return ", ".join(self.python_signature_arglist(with_types, for_call, with_arrays, arglist))

    def _repr_html_(self):
        """ HTML representation of the SDFG, used mainly for Jupyter
            notebooks. """
        from dace.jupyter import isnotebook, preamble

        result = ''
        if not isnotebook():
            result = preamble()

        # Create renderer canvas and load SDFG
        result += """
<div class="sdfv">
<div id="contents_{uid}" style="position: relative; resize: vertical; overflow: auto"></div>
</div>
<script>
    var sdfg_{uid} = {sdfg};
</script>
<script>
    new SDFGRenderer(
        checkCompatLoad(parse_sdfg(sdfg_{uid})),
        document.getElementById("contents_{uid}"),
        undefined, null, null, false, null, null
    );
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

        for (i, state) in enumerate(self.states()):
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

    def shared_transients(self, check_toplevel: bool = True, include_nested_data: bool = False) -> List[str]:
        """ Returns a list of transient data that appears in more than one state.

            :param check_toplevel: If True, consider the descriptors' toplevel attribute.
            :param include_nested_data: If True, also include nested data.
            :return: A list of transient data names.
        """
        seen = {}
        shared = []

        # If a transient is present in an inter-state edge, it is shared
        for interstate_edge in self.all_interstate_edges():
            for sym in interstate_edge.data.free_symbols:
                if sym in self.arrays and self.arrays[sym].transient:
                    seen[sym] = interstate_edge
                    shared.append(sym)

        # The same goes for the conditions of conditional blocks.
        for block in self.all_control_flow_blocks():
            if isinstance(block, ConditionalBlock):
                for cond, _ in block.branches:
                    if cond is not None:
                        cond_symbols = set(map(str, dace.symbolic.symbols_in_ast(cond.code[0])))
                        for sym in cond_symbols:
                            if sym in self.arrays and self.arrays[sym].transient:
                                seen[sym] = block
                                shared.append(sym)

        # If transient is accessed in more than one state, it is shared
        for state in self.states():
            for node in state.data_nodes():
                tokens = node.data.split('.')
                # NOTE: The following three lines ensure that nested data share transient and toplevel attributes.
                desc = self.arrays[tokens[0]]
                is_transient = desc.transient
                is_toplevel = desc.toplevel
                if include_nested_data:
                    datanames = set(['.'.join(tokens[:i + 1]) for i in range(len(tokens))])
                else:
                    datanames = set([tokens[0]])
                for dataname in datanames:
                    desc = self.arrays[dataname]
                    if is_transient:
                        if (check_toplevel and is_toplevel) or (dataname in seen and seen[dataname] != state):
                            shared.append(dataname)
                        seen[dataname] = state

        return dtypes.deduplicate(shared)

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
        filename = os.path.expanduser(filename)

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

    def view(self, filename=None, verbose=False):
        """
        View this sdfg in the system's HTML viewer

        :param filename: the filename to write the HTML to. If `None`, a temporary file will be created.
        :param verbose: Be verbose, `False` by default.
        """
        from dace.cli.sdfv import view
        view(self, filename=filename, verbose=verbose)

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

    def _find_new_name(self, name: str):
        """ Tries to find a new name by adding an underscore and a number. """

        names = (self._arrays.keys() | self.constants_prop.keys() | self._pgrids.keys() | self._subarrays.keys()
                 | self._rdistrarrays.keys() | self.symbols.keys())
        return dt.find_new_name(name, names)

    def is_name_used(self, name: str) -> bool:
        """ Checks if `name` is already used inside the SDFG."""
        if name in self._arrays:
            return True
        if name in self.symbols:
            return True
        if name in self.constants_prop:
            return True
        if name in self._pgrids:
            return True
        if name in self._subarrays:
            return True
        if name in self._rdistrarrays:
            return True
        return False

    def is_name_free(self, name: str) -> bool:
        """ Test if `name` is free, i.e. is not used by anything else."""
        return not self.is_name_used(name)

    def find_new_constant(self, name: str):
        """
        Tries to find a new name for a constant.
        """
        if self.is_name_free(name):
            return name
        return self._find_new_name(name)

    def find_new_symbol(self, name: str):
        """
        Tries to find a new symbol name by adding an underscore and a number.
        """
        if self.is_name_free(name):
            return name
        return self._find_new_name(name)

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
                 may_alias=False) -> Tuple[str, dt.ArrayView]:
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

        desc = dt.ArrayView(dtype,
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

        desc = dt.ArrayReference(dtype,
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

        # NOTE: Consider switching to `_find_new_name`
        #  The frontend seems to access this variable directly.
        while self.is_name_used(name):
            self._temp_transients += 1
            name = '__tmp%d' % self._temp_transients
        self._temp_transients += 1
        return name

    def refresh_temp_transients(self):
        """
        Updates the temporary transient counter of this SDFG by querying the maximum number among the
        ``__tmp###`` data descriptors.
        """
        temp_transients = [k[5:] for k in self.arrays.keys() if k.startswith('__tmp')]
        max_temp_transient = 0
        for arr_suffix in temp_transients:
            try:
                max_temp_transient = max(max_temp_transient, int(arr_suffix))
            except ValueError:  # Not of the form __tmp###
                continue
        self._temp_transients = max_temp_transient + 1

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

        if find_new_name:
            # These characters might be introduced through the creation of views to members
            #  of strictures.
            # NOTES: If `find_new_name` is `True` and the name (understood as a sequence of
            #   any characters) is not used, i.e. `assert self.is_name_free(name)`, then it
            #   is still "cleaned", i.e. dots are replaced with underscores. However, if
            #   `find_new_name` is `False` then this cleaning is not applied and it is possible
            #   to create names that are formally invalid. The above code reproduces the exact
            #   same behaviour and is maintained for  compatibility. This behaviour is
            #   triggered by tests/python_frontend/structures/structure_python_test.py::test_rgf`.
            name = self._find_new_name(name)
            name = name.replace('.', '_')
            if self.is_name_used(name):
                name = self._find_new_name(name)
        else:
            # We do not check for data constant, because there is a link between the constants and
            #  the data descriptors.
            if name in self.arrays:
                raise FileExistsError(f'Data descriptor "{name}" already exists in SDFG')
            if name in self.symbols:
                raise FileExistsError(f'Can not create data descriptor "{name}", the name is used by a symbol.')
            if name in self._subarrays:
                raise FileExistsError(f'Can not create data descriptor "{name}", the name is used by a subarray.')
            if name in self._rdistrarrays:
                raise FileExistsError(f'Can not create data descriptor "{name}", the name is used by a RedistrArray.')
            if name in self._pgrids:
                raise FileExistsError(f'Can not create data descriptor "{name}", the name is used by a ProcessGrid.')

        def _add_symbols(sdfg: SDFG, desc: dt.Data):
            if isinstance(desc, dt.Structure):
                for v in desc.members.values():
                    if isinstance(v, dt.Data):
                        _add_symbols(sdfg, v)
            for sym in desc.free_symbols:
                if sym.name not in sdfg.symbols:
                    sdfg.add_symbol(sym.name, sym.dtype)

        # Add the data descriptor to the SDFG and all symbols that are not yet known.
        self._arrays[name] = datadesc
        _add_symbols(self, datadesc)

        return name

    def add_datadesc_view(self, name: str, datadesc: dt.Data, find_new_name=False) -> str:
        """ Adds a view of a given data descriptor to the SDFG array store.

            :param name: Name to use.
            :param datadesc: Data descriptor to view.
            :param find_new_name: If True and data descriptor with this name
                                  exists, finds a new name to add.
            :return: Name of the new data descriptor
        """
        vdesc = dt.View.view(datadesc)
        return self.add_datadesc(name, vdesc, find_new_name)

    def add_datadesc_reference(self, name: str, datadesc: dt.Data, find_new_name=False) -> str:
        """ Adds a reference of a given data descriptor to the SDFG array store.

            :param name: Name to use.
            :param datadesc: Data descriptor to view.
            :param find_new_name: If True and data descriptor with this name
                                  exists, finds a new name to add.
            :return: Name of the new data descriptor
        """
        vdesc = dt.Reference.view(datadesc)
        return self.add_datadesc(name, vdesc, find_new_name)

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

        # No need to ensure unique test.
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
        # No need to ensure unique test.
        name = self._find_new_name('__rdistrarray')

        self._rdistrarrays[name] = RedistrArray(name, array_a, array_b)
        self.append_init_code(self._rdistrarrays[name].init_code(self))
        self.append_exit_code(self._rdistrarrays[name].exit_code(self))
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
        """
        Helper function that adds a looping state machine around a
        given state (or sequence of states).

        :param before_state: The state after which the loop should
                             begin, or None if the loop is the first
                             state (creates an empty state).
        :param loop_state: The state that begins the loop. See also
                           ``loop_end_state`` if the loop is multi-state.
        :param after_state: The state that should be invoked after
                            the loop ends, or None if the program
                            should terminate (creates an empty state).
        :param loop_var: A name of an inter-state variable to use
                         for the loop. If None, ``initialize_expr``
                         and ``increment_expr`` must be None.
        :param initialize_expr: A string expression that is assigned
                                to ``loop_var`` before the loop begins.
                                If None, does not define an expression.
        :param condition_expr: A string condition that occurs every
                               loop iteration. If None, loops forever
                               (undefined behavior).
        :param increment_expr: A string expression that is assigned to
                               ``loop_var`` after every loop iteration.
                               If None, does not define an expression.
        :param loop_end_state: If the loop wraps multiple states, the
                               state where the loop iteration ends.
                               If None, sets the end state to
                               ``loop_state`` as well.
        :return: A 3-tuple of (``before_state``, generated loop guard state,
                 ``after_state``).
        """
        from dace.frontend.python.astutils import negate_expr  # Avoid import loops

        warnings.warn("SDFG.add_loop is deprecated and will be removed in a future release. Use LoopRegions instead.",
                      DeprecationWarning)

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
        # Update constants
        for k, v in symbols.items():
            self.add_constant(str(k), v)

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

    def compile(self, output_file=None, validate=True, return_program_handle=True) -> 'CompiledSDFG':
        """ Compiles a runnable binary from this SDFG.

            :param output_file: If not None, copies the output library file to
                                the specified path.
            :param validate: If True, validates the SDFG prior to generating
                             code.
            :param return_program_handle: If False, does not load the generated library.
            :return: A callable CompiledSDFG object, or None if ``return_program_handle=False``.
        """

        # Importing these outside creates an import loop
        from dace.codegen import codegen, compiler

        # Compute build folder path before running codegen
        build_folder = self.build_folder

        if not self._recompile or Config.get_bool('compiler', 'use_cache'):
            # Try to see if a cached version of the binary exists
            binary_filename = compiler.get_binary_name(build_folder, self.name)
            if os.path.isfile(binary_filename):
                return compiler.load_from_file(self, binary_filename)

        ############################
        # DaCe Compilation Process #

        if self.regenerate_code or not os.path.isdir(build_folder):
            # Clone SDFG as the other modules may modify its contents
            sdfg = copy.deepcopy(self)
            # Fix the build folder name on the copied SDFG to avoid it changing
            # if the codegen modifies the SDFG (thereby changing its hash)
            sdfg.build_folder = build_folder

            # Ensure external nested SDFGs are loaded.
            for _ in sdfg.all_sdfgs_recursive(load_ext=True):
                pass

            # Rename SDFG to avoid runtime issues with clashing names
            index = 0
            while sdfg.is_loaded():
                sdfg.name = f'{self.name}_{index}'
                index += 1
            if self.name != sdfg.name:
                warnings.warn(f"SDFG '{self.name}' is already loaded by another object, recompiling under a different "
                              f"name '{sdfg.name}'.")

            try:
                # Fill in scope entry/exit connectors
                sdfg.fill_scope_connectors()

                # Generate code for the program by traversing the SDFG state by state
                program_objects = codegen.generate_code(sdfg, validate=validate)
            except Exception:
                fpath = os.path.join('_dacegraphs', 'failing.sdfgz')
                self.save(fpath, compress=True)
                print(f'Failing SDFG saved for inspection in {os.path.abspath(fpath)}')
                raise

            # Generate the program folder and write the source files
            program_folder = compiler.generate_program_folder(sdfg, program_objects, build_folder)
        else:
            # The code was already generated, just load the program folder
            program_folder = build_folder
            sdfg = self

        # Compile the code and get the shared library path
        shared_library = compiler.configure_and_compile(program_folder, sdfg.name)

        # If provided, save output to path or filename
        if output_file is not None:
            if os.path.isdir(output_file):
                output_file = os.path.join(output_file, os.path.basename(shared_library))
            shutil.copyfile(shared_library, output_file)

        # Get the function handle
        if return_program_handle:
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
        with hooks.invoke_sdfg_call_hooks(self) as sdfg:
            binaryobj = sdfg.compile()

            # Verify passed arguments (if enabled)
            if Config.get_bool('frontend', 'check_args'):
                sdfg.argument_typecheck(args, kwargs)

            return binaryobj(*args, **kwargs)

    def fill_scope_connectors(self):
        """ Fills missing scope connectors (i.e., "IN_#"/"OUT_#" on entry/exit
            nodes) according to data on the memlets. """
        for state in self.states():
            state.fill_scope_connectors()

    def predecessor_state_transitions(self, state):
        """ Yields paths (lists of edges) that the SDFG can pass through
            before computing the given state. """
        return self.edge_bfs(state, reverse=True)

    def predecessor_states(self, state):
        """ Returns a list of unique states that the SDFG can pass through
            before computing the given state. """
        return (e.src for e in self.edge_bfs(state, reverse=True))

    def validate(self, references: Optional[Set[int]] = None, **context: bool) -> None:
        validate_sdfg(self, references, **context)

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

    def simplify(self, validate=True, validate_all=False, verbose=False, skip: Optional[Set[str]] = None, options=None):
        """ Applies safe transformations (that will surely increase the
            performance) on the SDFG. For example, this fuses redundant states
            (safely) and removes redundant arrays.

            :note: This is an in-place operation on the SDFG.
        """
        from dace.transformation.passes.simplify import SimplifyPass
        return SimplifyPass(validate=validate,
                            validate_all=validate_all,
                            verbose=verbose,
                            skip=skip,
                            pass_options=options).apply_pass(self, {})

    def auto_optimize(self,
                      device: dtypes.DeviceType,
                      validate: bool = True,
                      validate_all: bool = False,
                      symbols: Dict[str, int] = None,
                      use_gpu_storage: bool = False):
        """
        Runs a basic sequence of transformations to optimize a given SDFG to decent
        performance. In particular, performs the following:

            * Simplify
            * Auto-parallelization (loop-to-map)
            * Greedy application of SubgraphFusion
            * Tiled write-conflict resolution (MapTiling -> AccumulateTransient)
            * Tiled stream accumulation (MapTiling -> AccumulateTransient)
            * Collapse all maps to parallelize across all dimensions
            * Set all library nodes to expand to ``fast`` expansion, which calls
              the fastest library on the target device

        :param device: the device to optimize for.
        :param validate: If True, validates the SDFG after all transformations
                         have been applied.
        :param validate_all: If True, validates the SDFG after every step.
        :param symbols: Optional dict that maps symbols (str/symbolic) to int/float
        :param use_gpu_storage: If True, changes the storage of non-transient data to GPU global memory.
        :note: Operates in-place on this SDFG.
        :note: This function is still experimental and may harm correctness in
               certain cases. Please report an issue if it does.
        """
        from dace.transformation.auto.auto_optimize import auto_optimize
        auto_optimize(self, device, validate, validate_all, symbols, use_gpu_storage)

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

                      # Applies MapTiling, then MapFusionVertical, followed by
                      # GPUTransformSDFG, specifying parameters only for the
                      # first transformation.
                      sdfg.apply_transformations(
                        [MapTiling, MapFusionVertical, GPUTransformSDFG],
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
                                  register_transients=True,
                                  simplify=True,
                                  host_maps=None,
                                  host_data=None):
        """ Applies a series of transformations on the SDFG for it to
            generate GPU code.

            :param sequential_innermaps: Make all internal maps Sequential.
            :param register_transients: Make all transients inside GPU maps registers.
            :note: It is recommended to apply redundant array removal
                   transformation after this transformation. Alternatively,
                   you can ``simplify()`` after this transformation.
            :note: This is an in-place operation on the SDFG.
        """
        # Avoiding import loops
        from dace.transformation.interstate import GPUTransformSDFG

        self.apply_transformations(GPUTransformSDFG,
                                   options=dict(sequential_innermaps=sequential_innermaps,
                                                register_trans=register_transients,
                                                simplify=simplify,
                                                host_maps=host_maps,
                                                host_data=host_data),
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
        return dace.Memlet.from_array(array, self.data(array))

    def recheck_using_explicit_control_flow(self) -> bool:
        found_explicit_cf_block = False
        for node, graph in self.root_sdfg.all_nodes_recursive():
            if isinstance(graph, ControlFlowRegion) and not isinstance(graph, SDFG):
                found_explicit_cf_block = True
                break
            if isinstance(node, ControlFlowBlock) and not isinstance(node, SDFGState):
                found_explicit_cf_block = True
                break
        self.root_sdfg.using_explicit_control_flow = found_explicit_cf_block
        return found_explicit_cf_block
