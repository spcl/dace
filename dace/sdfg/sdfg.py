# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import astunparse
import collections
import copy
import errno
import itertools
import os
import pickle, json
from pydoc import locate
import random
import re
import shutil
import sys
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union
import warnings
import numpy as np
import sympy as sp

import dace
import dace.serialize
from dace import (data as dt, memlet as mm, subsets as sbs, dtypes, properties,
                  symbolic)
from dace.sdfg.scope import ScopeTree
from dace.sdfg.replace import replace, replace_properties
from dace.sdfg.validation import (InvalidSDFGError, validate_sdfg)
from dace.config import Config
from dace.frontend.python import wrappers
from dace.sdfg import nodes as nd
from dace.sdfg.graph import OrderedDiGraph, Edge, SubgraphView
from dace.sdfg.state import SDFGState
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.dtypes import validate_name
from dace.properties import (make_properties, Property, CodeProperty,
                             TransformationHistProperty, SDFGReferenceProperty,
                             DictProperty, OrderedDictProperty, CodeBlock)


def _arrays_to_json(arrays):
    if arrays is None:
        return None
    return {k: dace.serialize.to_json(v) for k, v in arrays.items()}


def _arrays_from_json(obj, context=None):
    if obj is None:
        return {}
    return {k: dace.serialize.from_json(v, context) for k, v in obj.items()}


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
class InterstateEdge(object):
    """ An SDFG state machine edge. These edges can contain a condition
        (which may include data accesses for data-dependent decisions) and
        zero or more assignments of values to inter-state variables (e.g.,
        loop iterates).
    """

    assignments = Property(
        dtype=dict,
        desc="Assignments to perform upon transition (e.g., 'x=x+1; y = 0')",
        from_string=_assignments_from_string,
        to_string=_assignments_to_string)
    condition = CodeProperty(desc="Transition condition",
                             default=CodeBlock("1"))

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
        self.assignments = assignments

    def is_unconditional(self):
        """ Returns True if the state transition is unconditional. """
        return (self.condition is None or InterstateEdge.condition.to_string(
            self.condition).strip() == "1" or self.condition.as_string == "")

    def condition_sympy(self):
        return symbolic.pystr_to_symbolic(self.condition.as_string)

    @property
    def free_symbols(self) -> Set[str]:
        """ Returns a set of symbols used in this edge's properties. """
        # Symbols in conditions and assignments
        result = set(
            map(str, dace.symbolic.symbols_in_ast(self.condition.code[0])))
        for assign in self.assignments.values():
            result |= set(
                map(str,
                    symbolic.pystr_to_symbolic(assign).free_symbols))

        return result - set(self.assignments.keys())

    def new_symbols(self, symbols) -> Dict[str, dtypes.typeclass]:
        """
        Returns a mapping between symbols defined by this edge (i.e.,
        assignments) to their type.
        """
        from dace.codegen.tools.type_inference import infer_expr_type
        return {
            k: infer_expr_type(v, symbols)
            for k, v in self.assignments.items()
        }

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
        assignments = ','.join(
            ['%s=%s' % (k, v) for k, v in self.assignments.items()])

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
class SDFG(OrderedDiGraph):
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

    arg_types = OrderedDictProperty(default={}, desc="Formal parameter list")
    constants_prop = Property(dtype=dict,
                              default={},
                              desc="Compile-time constants")
    _arrays = Property(dtype=dict,
                       desc="Data descriptors for this SDFG",
                       to_json=_arrays_to_json,
                       from_json=_arrays_from_json)
    symbols = DictProperty(str,
                           dtypes.typeclass,
                           desc="Global symbols for this SDFG")

    global_code = DictProperty(
        str,
        CodeBlock,
        desc="Code generated in a global scope on the output files.")
    init_code = DictProperty(
        str, CodeBlock, desc="Code generated in the `__dace_init` function.")
    exit_code = DictProperty(
        str, CodeBlock, desc="Code generated in the `__dace_exit` function.")

    orig_sdfg = SDFGReferenceProperty(allow_none=True)
    transformation_hist = TransformationHistProperty()

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
        self.symbols = {}
        self._parent_sdfg = None
        self._parent_nsdfg_node = None
        self._sdfg_list = [self]
        self._start_state = None
        self._arrays = {}  # type: Dict[str, dt.Array]
        self.global_code = {'frame': CodeBlock("", dtypes.Language.CPP)}
        self.init_code = {'frame': CodeBlock("", dtypes.Language.CPP)}
        self.exit_code = {'frame': CodeBlock("", dtypes.Language.CPP)}
        self.orig_sdfg = None
        self.transformation_hist = []

        # Counter to make it easy to create temp transients
        self._temp_transients = 0

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

    def to_json(self):
        """ Serializes this object to JSON format.
            :return: A string representing the JSON-serialized SDFG.
        """
        tmp = super().to_json()

        # Ensure properties are serialized correctly
        tmp['attributes']['constants_prop'] = json.loads(
            dace.serialize.dumps(tmp['attributes']['constants_prop']))

        # Location in the SDFG list
        self.reset_sdfg_list()
        tmp['sdfg_list_id'] = int(self.sdfg_id)

        tmp['attributes']['name'] = self.name

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
                   arg_types=dace.serialize.loads(
                       dace.serialize.dumps(attrs['arg_types'])),
                   constants=dace.serialize.loads(
                       dace.serialize.dumps(attrs['constants_prop'])),
                   parent=context_info['sdfg'])

        dace.serialize.set_properties_from_json(
            ret, json_obj, ignore_properties={'constants_prop', 'name'})

        for n in nodes:
            nci = copy.copy(context_info)
            nci['sdfg'] = ret

            state = SDFGState.from_json(n, context=nci)
            ret.add_node(state)

        for e in edges:
            e = dace.serialize.loads(dace.serialize.dumps(e))
            ret.add_edge(ret.node(int(e.src)), ret.node(int(e.dst)), e.data)

        return ret

    @property
    def arrays(self):
        """ Returns a dictionary of data descriptors (`Data` objects) used
            in this SDFG, with an extra `None` entry for empty memlets.
        """
        return self._arrays

    def data(self, dataname: str):
        """ Looks up a data descriptor from its name, which can be an array, stream, or scalar symbol. """
        if dataname in self._arrays:
            return self._arrays[dataname]
        if str(dataname) in self.symbols:
            return self.symbols[str(dataname)]
        raise KeyError('Data descriptor with name "%s" not found in SDFG' %
                       dataname)

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

        # Replace in arrays and symbols (if a variable name)
        if validate_name(new_name):
            replace_dict(self._arrays, name, new_name)
            replace_dict(self.symbols, name, new_name)

        # Replace inside data descriptors
        for array in self.arrays.values():
            replace_properties(array, name, new_name)

        # Replace in inter-state edges
        for edge in self.edges():
            replace_dict(edge.data.assignments, name, new_name)
            for k, v in edge.data.assignments.items():
                edge.data.assignments[k] = v.replace(name, new_name)
            condition = edge.data.condition
            edge.data.condition.as_string = condition.as_string.replace(
                name, new_name)

        # Replace in states
        for state in self.nodes():
            state.replace(name, new_name)

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

    @property
    def start_state(self):
        """ Returns the starting state of this SDFG. """
        source_nodes = self.source_nodes()
        if len(source_nodes) == 1:
            return source_nodes[0]
        # If starting state is ambiguous (i.e., loop to initial state or more
        # than one possible start state), allow manually overriding start state
        if self._start_state is not None:
            return self._start_state
        raise ValueError('Ambiguous or undefined starting state for SDFG, '
                         'please use "is_start_state=True" when adding the '
                         'starting state with "add_state"')

    @start_state.setter
    def start_state(self, state_id):
        """ Manually sets the starting state of this SDFG.
            :param state_id: The node ID (use `node_id(state)`) of the
                             state to set.
        """
        states = self.nodes()
        if state_id < 0 or state_id >= len(states):
            raise ValueError("Invalid state ID")
        self._start_state = states[state_id]

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
        self.global_code[location] = CodeBlock(cpp_code,
                                               dace.dtypes.Language.CPP)

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
        try:
            next(n for n, _ in self.all_nodes_recursive()
                 if hasattr(n, 'instrument') and
                 n.instrument != dtypes.InstrumentationType.No_Instrumentation)
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
            InstrumentationReport(os.path.join(path, fname))
            for fname in os.listdir(path) if fname.startswith('report-')
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

    def get_latest_report(self) -> \
            Optional['dace.codegen.instrumentation.InstrumentationReport']:
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

        return InstrumentationReport(
            os.path.join(path,
                         sorted(files, reverse=True)[0]))

    ##########################################

    @property
    def build_folder(self) -> str:
        """ Returns a relative path to the build cache folder for this SDFG. """
        if hasattr(self, '_build_folder'):
            return self._build_folder
        elif Config.get_bool('testing', 'single_cache'):
            return os.path.join('.dacecache', 'test')
        else:
            return os.path.join('.dacecache', self.name)

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
        # Verify first that there are no access nodes that use this data
        if validate:
            for state in self.nodes():
                for node in state.nodes():
                    if isinstance(node, nd.AccessNode) and nd.data == name:
                        raise ValueError("Data descriptor %s is already used"
                                         "in node %s, state %s" %
                                         (name, node, state))

        del self._arrays[name]

    def reset_sdfg_list(self):
        if self.parent_sdfg is not None:
            self._sdfg_list = self.parent_sdfg.reset_sdfg_list()
        else:
            self._sdfg_list = list(self.all_sdfgs_recursive())
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
    def sdfg_list(self):
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
                return dt.Array(dtypes.DTYPE_TO_TYPECLASS[obj.dtype.type],
                                shape=obj.shape)
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

    @property
    def parent_nsdfg_node(self):
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

    def nodes(self) -> List[SDFGState]:
        return super().nodes()

    def node(self, id) -> SDFGState:
        return super().node(id)

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
        if is_start_state == True:
            self.start_state = len(self.nodes()) - 1

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
        if not isinstance(edge, InterstateEdge):
            raise TypeError("Expected InterstateEdge, got: {}".format(
                type(edge).__name__))
        return super(SDFG, self).add_edge(u, v, edge)

    def states(self):
        """ Alias that returns the nodes (states) in this SDFG. """
        return self.nodes()

    def all_nodes_recursive(
            self) -> Iterator[Tuple[nd.Node, Union['SDFG', 'SDFGState']]]:
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

        # Add free data symbols and exclude data descriptor names
        for name, desc in self.arrays.items():
            free_syms |= set(map(str, desc.free_symbols))
            defined_syms.add(name)

        # Add free state symbols
        for state in self.nodes():
            free_syms |= state.free_symbols

        # Add free inter-state symbols
        for e in self.edges():
            defined_syms |= set(e.data.new_symbols({}).keys())
            free_syms |= e.data.free_symbols

        defined_syms |= set(self.constants.keys())

        # Subtract symbols defined in inter-state edges and constants
        return free_syms - defined_syms

    def arglist(self) -> Dict[str, dt.Data]:
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
        data_args = {
            k: v
            for k, v in self.arrays.items()
            if not v.transient and not isinstance(v, dt.Scalar)
        }
        scalar_args = {
            k: v
            for k, v in self.arrays.items() if not v.transient
            and isinstance(v, dt.Scalar) and not k.startswith('__dace')
        }

        # Add global free symbols to scalar arguments
        scalar_args.update({
            k: dt.Scalar(self.symbols[k])
            for k in self.free_symbols if not k.startswith('__dace')
        })

        # Fill up ordered dictionary
        result = collections.OrderedDict()
        for k, v in itertools.chain(sorted(data_args.items()),
                                    sorted(scalar_args.items())):
            result[k] = v

        return result

    def signature_arglist(self, with_types=True, for_call=False):
        """ Returns a list of arguments necessary to call this SDFG,
            formatted as a list of C definitions.
            :param with_types: If True, includes argument types in the result.
            :param for_call: If True, returns arguments that can be used when
                             calling the SDFG.
            :return: A list of strings. For example: `['float *A', 'int b']`.
        """
        return [
            v.as_arg(name=k, with_types=with_types, for_call=for_call)
            for k, v in self.arglist().items()
        ]

    def signature(self, with_types=True, for_call=False):
        """ Returns a C/C++ signature of this SDFG, used when generating code.
            :param with_types: If True, includes argument types (can be used
                               for a function prototype). If False, only
                               include argument names (can be used for function
                               calls).
            :param for_call: If True, returns arguments that can be used when
                             calling the SDFG.
        """
        return ", ".join(self.signature_arglist(with_types, for_call))

    def _repr_html_(self):
        """ HTML representation of the SDFG, used mainly for Jupyter
            notebooks. """
        from dace.jupyter import isnotebook, preamble

        result = ''
        if not isnotebook():
            result = preamble()

        # Make sure to not store metadata (saves space)
        old_meta = dace.serialize.JSON_STORE_METADATA
        dace.serialize.JSON_STORE_METADATA = False

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

        # Reset metadata state
        dace.serialize.JSON_STORE_METADATA = old_meta

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
            for sym in interstate_edge.data.free_symbols:
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

    def save(self,
             filename: str,
             use_pickle=False,
             with_metadata=False,
             exception=None):
        """ Save this SDFG to a file.
            :param filename: File name to save to.
            :param use_pickle: Use Python pickle as the SDFG format (default:
                               JSON).
            :param with_metadata: Save property metadata (e.g. name,
                                  description). False or True override current
                                  option, whereas None keeps default.
            :param exception: If not None, stores error information along with
                              SDFG.
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
                json_output = self.to_json()
                if exception:
                    json_output['error'] = exception.to_json()
                fp.write(dace.serialize.dumps(json_output))
            if with_metadata is not None:
                dace.serialize.JSON_STORE_METADATA = old_meta

    def view(self, filename=None):
        """View this sdfg in the system's HTML viewer
           :param filename: the filename to write the HTML to. If `None`, a temporary file will be created.
        """
        from diode.sdfv import view
        view(self, filename=filename)

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
    def add_state(self, label=None, is_start_state=False) -> 'SDFGState':
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

    def add_state_before(self,
                         state: 'SDFGState',
                         label=None,
                         is_start_state=False) -> 'SDFGState':
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

    def add_state_after(self,
                        state: 'SDFGState',
                        label=None,
                        is_start_state=False) -> 'SDFGState':
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
        while (name + ('_%d' % index)) in self._arrays:
            index += 1

        return name + ('_%d' % index)

    def add_array(self,
                  name: str,
                  shape,
                  dtype,
                  storage=dtypes.StorageType.Default,
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

        if isinstance(dtype, type) and dtype in dtypes._CONSTANT_TYPES[:-1]:
            dtype = dtypes.typeclass(dtype)

        desc = dt.Array(dtype,
                        shape,
                        storage=storage,
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
                              storage,
                              True,
                              strides,
                              offset,
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
                              storage,
                              True,
                              strides,
                              offset,
                              lifetime=lifetime,
                              alignment=alignment,
                              debuginfo=debuginfo,
                              allow_conflicts=allow_conflicts,
                              total_size=total_size,
                              may_alias=may_alias)

    def add_datadesc(self,
                     name: str,
                     datadesc: dt.Data,
                     find_new_name=False) -> str:
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
                raise NameError('Array or Stream with name "%s" already exists '
                                "in SDFG" % name)
        self._arrays[name] = datadesc

        # Add free symbols to the SDFG global symbol storage
        for sym in datadesc.free_symbols:
            if sym.name not in self.symbols:
                self.add_symbol(sym.name, sym.dtype)

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
            raise TypeError(
                "state_id_or_label is not an int nor string: {}".format(
                    state_id_or_label))

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
            self.add_constant(k, v)

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

    def compile(self, output_file=None) -> \
            'dace.codegen.compiler.CompiledSDFG':
        """ Compiles a runnable binary from this SDFG.
            :param output_file: If not None, copies the output library file to
                                the specified path.
            :return: A callable CompiledSDFG object.
        """

        # Importing these outside creates an import loop
        from dace.codegen import codegen, compiler

        if Config.get_bool("compiler", "use_cache"):
            # Try to see if a cached version of the binary exists
            binary_filename = compiler.get_binary_name(self.build_folder,
                                                       self.name)
            if os.path.isfile(binary_filename):
                # print("A cached binary was found!")
                return compiler.load_from_file(self, binary_filename)

        ############################
        # DaCe Compilation Process #

        # Clone SDFG as the other modules may modify its contents
        sdfg = copy.deepcopy(self)

        # Fill in scope entry/exit connectors
        sdfg.fill_scope_connectors()

        # Recursively expand library nodes that haven't been expanded yet
        sdfg.expand_library_nodes()

        # Generate code for the program by traversing the SDFG state by state
        program_objects = codegen.generate_code(sdfg)

        # Generate the program folder and write the source files
        program_folder = compiler.generate_program_folder(
            sdfg, program_objects, sdfg.build_folder)

        # Compile the code and get the shared library path
        shared_library = compiler.configure_and_compile(program_folder,
                                                        sdfg.name)

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

        # Omit return values from arguments
        expected_args = collections.OrderedDict([
            (k, v) for k, v in expected_args.items()
            if not k.startswith('__return')
        ])
        kwargs = {
            k: v
            for k, v in kwargs.items() if not k.startswith('__return')
        }

        num_args_passed = len(args) + len(kwargs)
        num_args_expected = len(expected_args)
        if num_args_passed < num_args_expected:
            expected_kwargs = list(expected_args.keys())[len(args):]
            missing_args = [k for k in expected_kwargs if k not in kwargs]
            raise RuntimeError("Missing arguments to SDFG: '%s'" %
                               (', '.join(missing_args)))
        elif num_args_passed > num_args_expected:
            unnecessary_args = []
            extra_args = len(args) - len(expected_args)
            if extra_args > 0:
                unnecessary_args.extend('Argument #%d' %
                                        (i + len(expected_args) + 1)
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
                if not dtypes.is_array(passed):
                    raise TypeError("Type mismatch for argument {}: "
                                    "expected array type, got {}".format(
                                        arg, type(passed)))
            elif (isinstance(expected, dace.data.Scalar)
                  or isinstance(expected, dace.dtypes.typeclass)):
                if (not dtypes.isconstant(passed)
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
        from networkx import all_simple_paths

        for path in all_simple_paths(self, self.start_state, state):
            yield [
                next(e for e in self.out_edges(s) if e.dst == d)
                for s, d in zip(path[:-1], path[1:])
            ]

    def predecessor_states(self, state):
        """ Returns a list of unique states that the SDFG can pass through
            before computing the given state. """
        from networkx import all_simple_paths

        return set([
            n for path in all_simple_paths(self, self.start_state, state)
            for n in path
        ])

    def validate(self) -> None:
        validate_sdfg(self)

    def is_valid(self) -> bool:
        """ Returns True if the SDFG is verified correctly (using `validate`).
        """
        try:
            self.validate()
        except InvalidSDFGError:
            return False
        return True

    def apply_strict_transformations(self, validate=True, validate_all=False):
        """ Applies safe transformations (that will surely increase the
            performance) on the SDFG. For example, this fuses redundant states
            (safely) and removes redundant arrays.

            B{Note:} This is an in-place operation on the SDFG.
        """
        # These are imported in order to update the transformation registry
        from dace.transformation import dataflow, interstate
        # This is imported here to avoid an import loop
        from dace.transformation.transformation import Transformation

        strict_transformations = [
            k for k, v in Transformation.extensions().items()
            if v.get('strict', False)
        ]
        self.apply_transformations_repeated(strict_transformations,
                                            validate=validate,
                                            strict=True,
                                            validate_all=validate_all)

    def apply_transformations(self,
                              xforms: Union[Type, List[Type]],
                              options: Optional[Union[Dict[str, Any],
                                                      List[Dict[str,
                                                                Any]]]] = None,
                              validate: bool = True,
                              validate_all: bool = False,
                              strict: bool = False,
                              states: Optional[List[Any]] = None) -> int:
        """ This function applies a transformation or a sequence thereof
            consecutively. Operates in-place.
            :param xforms: A Transformation class or a sequence.
            :param options: An optional dictionary (or sequence of dictionaries)
                            to modify transformation parameters.
            :param validate: If True, validates after all transformations.
            :param validate_all: If True, validates after every transformation.
            :param strict: If True, operates in strict transformation mode.
            :param states: If not None, specifies a subset of states to
                           apply transformations on.
            :return: Number of transformations applied.

            Examples::

                      # Applies MapTiling, then MapFusion, followed by
                      # GPUTransformSDFG, specifying parameters only for the
                      # first transformation.
                      sdfg.apply_transformations(
                        [MapTiling, MapFusion, GPUTransformSDFG],
                        options=[{'tile_size': 16}, {}, {}])
        """
        # Avoiding import loops
        from dace.transformation import optimizer
        from dace.transformation.transformation import Transformation

        applied_transformations = collections.defaultdict(int)

        if isinstance(xforms, type) and issubclass(xforms, Transformation):
            xforms = [xforms]

        if isinstance(options, dict):
            options = [options]
        options = options or [dict() for _ in xforms]
        if len(options) != len(xforms):
            raise ValueError('Length of options and transformations mismatch')

        opt = optimizer.SDFGOptimizer(self, inplace=True)

        for xform, opts in zip(xforms, options):
            # Find only the first match
            try:
                match = next(m for m in opt.get_pattern_matches(
                    strict=strict, patterns=[xform], states=states))
            except StopIteration:
                continue
            sdfg = self.sdfg_list[match.sdfg_id]

            # Set transformation properties
            for prop_name, prop_val in opts.items():
                setattr(match, prop_name, prop_val)
            match.apply(sdfg)
            applied_transformations[type(match).__name__] += 1
            if validate_all:
                self.validate()

        if validate:
            self.validate()

        if Config.get_bool('debugprint') and len(applied_transformations) > 0:
            print('Applied {}.'.format(', '.join([
                '%d %s' % (v, k) for k, v in applied_transformations.items()
            ])))

        return sum(applied_transformations.values())

    def apply_transformations_repeated(
            self,
            xforms: Union[Type, List[Type]],
            options: Optional[Union[Dict[str, Any], List[Dict[str,
                                                              Any]]]] = None,
            validate: bool = True,
            validate_all: bool = False,
            strict: bool = False,
            states: Optional[List[Any]] = None) -> int:
        """ This function repeatedly applies a transformation or a set of
            (unique) transformations until none can be found. Operates in-place.
            :param xforms: A Transformation class or a set thereof.
            :param options: An optional dictionary (or sequence of dictionaries)
                            to modify transformation parameters.
            :param validate: If True, validates after all transformations.
            :param validate_all: If True, validates after every transformation.
            :param strict: If True, operates in strict transformation mode.
            :param states: If not None, specifies a subset of states to
                           apply transformations on.
            :return: Number of transformations applied.

            Examples::

                    # Applies InlineSDFG until no more subgraphs can be inlined
                    sdfg.apply_transformations_repeated(InlineSDFG)
        """
        # Avoiding import loops
        from dace.transformation import optimizer
        from dace.transformation.transformation import Transformation

        applied_transformations = collections.defaultdict(int)

        if isinstance(xforms, type) and issubclass(xforms, Transformation):
            xforms = [xforms]

        # Ensure transformations are unique
        if len(xforms) != len(set(xforms)):
            raise ValueError('Transformation set must be unique')

        if isinstance(options, dict):
            options = [options]
        options = options or [dict() for _ in xforms]
        if len(options) != len(xforms):
            raise ValueError('Length of options and transformations mismatch')

        opt = optimizer.SDFGOptimizer(self, inplace=True)

        applied = True
        while applied:
            applied = False
            # Find and apply one of
            for match in opt.get_pattern_matches(strict=strict,
                                                 patterns=xforms,
                                                 states=states):
                sdfg = self.sdfg_list[match.sdfg_id]

                # Set transformation properties
                opts = next(o for x, o in zip(xforms, options)
                            if type(match) is x)
                for prop_name, prop_val in opts.items():
                    setattr(match, prop_name, prop_val)

                match.apply(sdfg)
                applied_transformations[type(match).__name__] += 1
                if validate_all:
                    self.validate()
                applied = True
                break

        if validate:
            self.validate()

        if Config.get_bool('debugprint') and len(applied_transformations) > 0:
            print('Applied {}.'.format(', '.join([
                '%d %s' % (v, k) for k, v in applied_transformations.items()
            ])))

        return sum(applied_transformations.values())

    def apply_gpu_transformations(self,
                                  states=None,
                                  validate=True,
                                  validate_all=False,
                                  strict=True):
        """ Applies a series of transformations on the SDFG for it to
            generate GPU code.
            :note: It is recommended to apply redundant array removal
            transformation after this transformation. Alternatively,
            you can apply_strict_transformations() after this transformation.
            :note: This is an in-place operation on the SDFG.
        """
        # Avoiding import loops
        from dace.transformation.interstate import GPUTransformSDFG

        self.apply_transformations(GPUTransformSDFG,
                                   validate=validate,
                                   validate_all=validate_all,
                                   strict=strict,
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
                    node.sdfg.expand_library_nodes()  # Call recursively
                elif isinstance(node, nd.LibraryNode):
                    node.expand(self, state)
                    print("Automatically expanded library node \"" + str(node) +
                          "\".")
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

        # Propagate memlets in the graph
        if sdfg.propagate:
            propagate_memlets_sdfg(sdfg)

        sdfg.save(os.path.join('_dacegraphs', 'program.sdfg'))

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

        @note: This method uses pydoc to locate the class.
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
