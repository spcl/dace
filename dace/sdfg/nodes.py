# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes implementing the different types of nodes of the stateful
    dataflow multigraph representation. """

import ast
from copy import deepcopy as dcpy
from collections.abc import KeysView
import dace
import itertools
import dace.serialize
from typing import Any, Dict, Optional, Set, Union
from dace.config import Config
from dace.sdfg import graph
from dace.frontend.python.astutils import unparse, rname
from dace.properties import (EnumProperty, Property, CodeProperty, LambdaProperty, RangeProperty, DebugInfoProperty,
                             SetProperty, make_properties, indirect_properties, DataProperty, SymbolicProperty,
                             ListProperty, SDFGReferenceProperty, DictProperty, LibraryImplementationProperty,
                             CodeBlock)
from dace.frontend.operations import detect_reduction_type
from dace.symbolic import issymbolic, pystr_to_symbolic
from dace import data, subsets as sbs, dtypes
from dace.sdfg import tasklet_validation as tval
from dace.sdfg.type_inference import infer_types, infer_expr_type
import pydoc
import warnings

# -----------------------------------------------------------------------------


@make_properties
class Node(object):
    """ Base node class. """

    in_connectors = DictProperty(key_type=str,
                                 value_type=dtypes.typeclass,
                                 desc="A set of input connectors for this node.")
    out_connectors = DictProperty(key_type=str,
                                  value_type=dtypes.typeclass,
                                  desc="A set of output connectors for this node.")
    guid = Property(dtype=str, allow_none=False)

    def __init__(self, in_connectors=None, out_connectors=None):
        # Convert connectors to typed connectors with autodetect type
        if isinstance(in_connectors, (set, list, KeysView)):
            in_connectors = {k: None for k in in_connectors}
        if isinstance(out_connectors, (set, list, KeysView)):
            out_connectors = {k: None for k in out_connectors}

        self.in_connectors = in_connectors or {}
        self.out_connectors = out_connectors or {}

        self.guid = graph.generate_element_id(self)

    def __str__(self):
        if hasattr(self, 'label'):
            return self.label
        else:
            return type(self).__name__

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'guid':  # Skip ID
                continue
            setattr(result, k, dcpy(v, memo))
        return result

    def validate(self, sdfg, state):
        pass

    def to_json(self, parent):
        labelstr = str(self)
        typestr = getattr(self, '__jsontype__', str(type(self).__name__))

        try:
            scope_entry_node = parent.entry_node(self)
        except (RuntimeError, ValueError, StopIteration):
            scope_entry_node = None

        if scope_entry_node is not None:
            try:
                ens = parent.exit_node(parent.entry_node(self))
                scope_exit_node = str(parent.node_id(ens))
                scope_entry_node = str(parent.node_id(scope_entry_node))
            except (RuntimeError, ValueError, StopIteration):
                scope_entry_node = scope_exit_node = None
        else:
            scope_entry_node = None
            scope_exit_node = None

        # The scope exit of an entry node is the matching exit node
        if isinstance(self, EntryNode):
            try:
                scope_exit_node = str(parent.node_id(parent.exit_node(self)))
            except (RuntimeError, ValueError, StopIteration):
                scope_exit_node = None

        retdict = {
            "type": typestr,
            "label": labelstr,
            "attributes": dace.serialize.all_properties_to_json(self),
            "id": parent.node_id(self),
            "scope_entry": scope_entry_node,
            "scope_exit": scope_exit_node
        }
        return retdict

    def __repr__(self):
        return type(self).__name__ + ' (' + self.__str__() + ')'

    def add_in_connector(self, connector_name: str, dtype: Any = None, force: bool = False):
        """ Adds a new input connector to the node. The operation will fail if
            a connector (either input or output) with the same name already
            exists in the node.

            :param connector_name: The name of the new connector.
            :param dtype: The type of the connector, or None for auto-detect.
            :param force: Add connector even if input or output connector of that name already exists.
            :return: True if the operation is successful, otherwise False.
        """

        if (not force and (connector_name in self.in_connectors or connector_name in self.out_connectors)):
            return False
        if not isinstance(dtype, dace.typeclass):
            dtype = dace.typeclass(dtype)
        self.in_connectors[connector_name] = dtype
        return True

    def add_out_connector(
        self,
        connector_name: str,
        dtype: Any = None,
        force: bool = False,
    ) -> bool:
        """ Adds a new output connector to the node. The operation will fail if
            a connector (either input or output) with the same name already
            exists in the node.

            :param connector_name: The name of the new connector.
            :param dtype: The type of the connector, or None for auto-detect.
            :param force: Add connector even if input or output connector of that name already exists.
            :return: True if the operation is successful, otherwise False.
        """

        if (not force and (connector_name in self.in_connectors or connector_name in self.out_connectors)):
            return False
        if not isinstance(dtype, dace.typeclass):
            dtype = dace.typeclass(dtype)
        self.out_connectors[connector_name] = dtype
        return True

    def _add_scope_connectors(
        self,
        connector_name: str,
        dtype: Optional[dtypes.typeclass] = None,
        force: bool = False,
    ) -> None:
        """ Adds input and output connector names to `self` in one step.

            The function will add an input connector with name `'IN_' + connector_name`
            and an output connector with name `'OUT_' + connector_name`.
            The function is a shorthand for calling `add_in_connector()` and `add_out_connector()`.

            :param connector_name: The base name of the new connectors.
            :param dtype: The type of the connectors, or `None` for auto-detect.
            :param force: Add connector even if input or output connector of that name already exists.
            :return: True if the operation is successful, otherwise False.
        """
        in_connector_name = "IN_" + connector_name
        out_connector_name = "OUT_" + connector_name
        if not force:
            if in_connector_name in self.in_connectors or in_connector_name in self.out_connectors:
                return False
            if out_connector_name in self.in_connectors or out_connector_name in self.out_connectors:
                return False
        # We force unconditionally because we have performed the tests above.
        self.add_in_connector(
            connector_name=in_connector_name,
            dtype=dtype,
            force=True,
        )
        self.add_out_connector(
            connector_name=out_connector_name,
            dtype=dtype,
            force=True,
        )
        return True

    def remove_in_connector(self, connector_name: str):
        """ Removes an input connector from the node.

            :param connector_name: The name of the connector to remove.
            :return: True if the operation was successful.
        """

        if connector_name in self.in_connectors:
            connectors = self.in_connectors
            del connectors[connector_name]
            self.in_connectors = connectors
        return True

    def remove_out_connector(self, connector_name: str):
        """ Removes an output connector from the node.

            :param connector_name: The name of the connector to remove.
            :return: True if the operation was successful.
        """

        if connector_name in self.out_connectors:
            connectors = self.out_connectors
            del connectors[connector_name]
            self.out_connectors = connectors
        return True

    def _next_connector_int(self) -> int:
        """ Returns the next unused connector ID (as an integer). Used for
            filling connectors when adding edges to scopes. """
        next_number = 1
        for conn in itertools.chain(self.in_connectors, self.out_connectors):
            if conn.startswith('IN_'):
                cconn = conn[3:]
            elif conn.startswith('OUT_'):
                cconn = conn[4:]
            else:
                continue
            try:
                curconn = int(cconn)
                if curconn >= next_number:
                    next_number = curconn + 1
            except (TypeError, ValueError):  # not integral
                continue
        return next_number

    def next_connector(self, try_name: str = None) -> str:
        """
        Returns the next unused connector ID (as a string). Used for
        filling connectors when adding edges to scopes.

        :param try_name: First try the connector with this name. If already
                         exists, use the next integer connector.
        """
        if (try_name and 'IN_' + try_name not in self.in_connectors and 'OUT_' + try_name not in self.out_connectors):
            return try_name

        return str(self._next_connector_int())

    def last_connector(self) -> str:
        """ Returns the last used connector ID (as a string). Used for
            filling connectors when adding edges to scopes. """
        return str(self._next_connector_int() - 1)

    @property
    def free_symbols(self) -> Set[str]:
        """ Returns a set of symbols used in this node's properties. """
        return set()

    def new_symbols(self, sdfg, state, symbols) -> Dict[str, dtypes.typeclass]:
        """ Returns a mapping between symbols defined by this node (e.g., for
            scope entries) to their type. """
        return {}

    def infer_connector_types(self, sdfg, state):
        """
        Infers and fills remaining connectors (i.e., set to None) with their
        types.
        """
        pass


# ------------------------------------------------------------------------------


@make_properties
class AccessNode(Node):
    """ A node that accesses data in the SDFG. Denoted by a circular shape. """

    setzero = Property(dtype=bool, desc="Initialize to zero", default=False)
    debuginfo = DebugInfoProperty()
    data = DataProperty(desc="Data (array, stream, scalar) to access")

    instrument = EnumProperty(dtype=dtypes.DataInstrumentationType,
                              desc="Instrument data contents at this access",
                              default=dtypes.DataInstrumentationType.No_Instrumentation)
    instrument_condition = CodeProperty(desc="Condition under which to trigger the instrumentation",
                                        default=CodeBlock("1", language=dtypes.Language.CPP))

    def __init__(self, data, debuginfo=None):
        super(AccessNode, self).__init__()

        # Properties
        self.debuginfo = debuginfo
        if not isinstance(data, str):
            raise TypeError('Data for AccessNode must be a string')
        self.data = data

    @staticmethod
    def from_json(json_obj, context=None):
        ret = AccessNode("Nodata")
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def __deepcopy__(self, memo):
        node = object.__new__(AccessNode)
        node._data = self._data
        node._setzero = self._setzero
        node._instrument = self._instrument
        node._instrument_condition = dcpy(self._instrument_condition, memo=memo)
        node._in_connectors = dcpy(self._in_connectors, memo=memo)
        node._out_connectors = dcpy(self._out_connectors, memo=memo)
        node._debuginfo = dcpy(self._debuginfo, memo=memo)

        node._guid = graph.generate_element_id(node)

        return node

    @property
    def label(self):
        return self.data

    @property
    def root_data(self):
        return self.data.split('.')[0]

    def __label__(self, sdfg, state):
        return self.data

    def desc(self, sdfg: Union['dace.sdfg.SDFG', 'dace.sdfg.SDFGState', 'dace.sdfg.ScopeSubgraphView']):
        if isinstance(sdfg, (dace.sdfg.SDFGState, dace.sdfg.ScopeSubgraphView)):
            sdfg = sdfg.parent
        return sdfg.arrays[self.data]

    def root_desc(self, sdfg):
        from dace.sdfg import SDFGState, ScopeSubgraphView
        if isinstance(sdfg, (SDFGState, ScopeSubgraphView)):
            sdfg = sdfg.parent
        return sdfg.arrays[self.data.split('.')[0]]

    def validate(self, sdfg, state):
        if self.data not in sdfg.arrays:
            raise KeyError('Array "%s" not found in SDFG' % self.data)

    def has_writes(self, state):
        for e in state.in_edges(self):
            if not e.data.is_empty():
                return True
        return False

    def has_reads(self, state):
        for e in state.out_edges(self):
            if not e.data.is_empty():
                return True
        return False


# ------------------------------------------------------------------------------


@make_properties
class CodeNode(Node):
    """ A node that contains runnable code with acyclic external data
        dependencies. May either be a tasklet or a nested SDFG, and
        denoted by an octagonal shape. """

    label = Property(dtype=str, desc="Name of the CodeNode")
    location = DictProperty(key_type=str,
                            value_type=dace.symbolic.pystr_to_symbolic,
                            desc='Full storage location identifier (e.g., rank, GPU ID)')
    environments = SetProperty(str,
                               desc="Environments required by CMake to build and run this code node.",
                               default=set())

    def __init__(self, label="", location=None, inputs=None, outputs=None):
        super(CodeNode, self).__init__(inputs or set(), outputs or set())
        # Properties
        self.label = label
        self.location = location if location is not None else {}

    @property
    def free_symbols(self) -> Set[str]:
        return set().union(*[v.free_symbols for v in self.location.values()])


@make_properties
class Tasklet(CodeNode):
    """ A node that contains a tasklet: a functional computation procedure
        that can only access external data specified using connectors.

        Tasklets may be implemented in Python, C++, or any supported
        language by the code generator.
    """

    code = CodeProperty(desc="Tasklet code", default=CodeBlock(""))
    state_fields = ListProperty(element_type=str, desc="Fields that are added to the global state")
    code_global = CodeProperty(desc="Global scope code needed for tasklet execution",
                               default=CodeBlock("", dtypes.Language.CPP))
    code_init = CodeProperty(desc="Extra code that is called on DaCe runtime initialization",
                             default=CodeBlock("", dtypes.Language.CPP))
    code_exit = CodeProperty(desc="Extra code that is called on DaCe runtime cleanup",
                             default=CodeBlock("", dtypes.Language.CPP))
    debuginfo = DebugInfoProperty()

    instrument = EnumProperty(dtype=dtypes.InstrumentationType,
                              desc="Measure execution statistics with given method",
                              default=dtypes.InstrumentationType.No_Instrumentation)
    side_effects = Property(dtype=bool,
                            allow_none=True,
                            default=None,
                            desc='If True, this tasklet calls a function that may have '
                            'additional side effects on the system state (e.g., callback). '
                            'Defaults to None, which lets the framework make assumptions based on '
                            'the tasklet contents')
    ignored_symbols = SetProperty(element_type=str,
                                  desc='A set of symbols to ignore when computing '
                                  'the symbols used by this tasklet. Used to skip certain symbols in non-Python '
                                  'tasklets, where only string analysis is possible; and to skip globals in Python '
                                  'tasklets that should not be given as parameters to the SDFG.')

    def __init__(self,
                 label,
                 inputs=None,
                 outputs=None,
                 code="",
                 language=dtypes.Language.Python,
                 state_fields=None,
                 code_global="",
                 code_init="",
                 code_exit="",
                 location=None,
                 side_effects=None,
                 ignored_symbols=None,
                 debuginfo=None):
        super(Tasklet, self).__init__(label, location, inputs, outputs)

        self.code = CodeBlock(code, language)

        self.state_fields = state_fields or []
        self.code_global = CodeBlock(code_global, dtypes.Language.CPP)
        self.code_init = CodeBlock(code_init, dtypes.Language.CPP)
        self.code_exit = CodeBlock(code_exit, dtypes.Language.CPP)
        self.side_effects = side_effects
        self.ignored_symbols = ignored_symbols or set()
        self.debuginfo = debuginfo

    @property
    def language(self):
        return self.code.language

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Tasklet("dummylabel")
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    @property
    def name(self):
        return self._label

    def validate(self, sdfg: 'dace.sdfg.SDFG', state: 'dace.sdfg.SDFGState'):
        if not dtypes.validate_name(self.label):
            raise NameError('Invalid tasklet name "%s"' % self.label)
        for in_conn in self.in_connectors:
            if not dtypes.validate_name(in_conn):
                raise NameError('Invalid input connector "%s"' % in_conn)
        for out_conn in self.out_connectors:
            if not dtypes.validate_name(out_conn):
                raise NameError('Invalid output connector "%s"' % out_conn)
        if self.language == dtypes.Language.Python and self.code.code:
            validator = tval.ConnectorDimensionalityValidator({e.dst_conn: e.data
                                                               for e in state.in_edges(self)},
                                                              {e.src_conn: e.data
                                                               for e in state.out_edges(self)}, sdfg)
            for stmt in self.code.code:
                validator.visit(stmt)

    @property
    def free_symbols(self) -> Set[str]:
        symbols_to_ignore = self.in_connectors.keys() | self.out_connectors.keys()
        symbols_to_ignore |= self.ignored_symbols

        return self.code.get_free_symbols(symbols_to_ignore)

    def has_side_effects(self, sdfg) -> bool:
        """
        Returns True if this tasklet may have other side effects (e.g., calling stateful libraries, communicating).
        """
        # If side effects property is set, takes precedence over node analysis
        if self.side_effects is not None:
            return self.side_effects

        # If side effect property is not defined, find calls within tasklet
        if self.code.language == dace.dtypes.Language.Python and self.code.code:
            for stmt in self.code.code:
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Call):
                        cname = rname(n.func)
                        # If the function name is a symbol or a Scalar data descriptor, it may be a dace.callback,
                        # which means side effects are possible unless otherwise mentioned
                        if cname in sdfg.symbols or cname in sdfg.arrays:
                            return True
        return False

    def infer_connector_types(self, sdfg, state):
        # If a MLIR tasklet, simply read out the types (it's explicit)
        if self.code.language == dtypes.Language.MLIR:
            # Inline import because mlir.utils depends on pyMLIR which may not be installed
            # Doesn't cause crashes due to missing pyMLIR if a MLIR tasklet is not present
            from dace.codegen.targets.mlir import utils

            mlir_ast = utils.get_ast(self.code.code)
            mlir_is_generic = utils.is_generic(mlir_ast)
            mlir_entry_func = utils.get_entry_func(mlir_ast, mlir_is_generic)

            mlir_result_type = utils.get_entry_result_type(mlir_entry_func, mlir_is_generic)
            mlir_out_name = next(iter(self.out_connectors.keys()))

            if self.out_connectors[mlir_out_name] is None or self.out_connectors[mlir_out_name].ctype == "void":
                self.out_connectors[mlir_out_name] = utils.get_dace_type(mlir_result_type)
            elif self.out_connectors[mlir_out_name] != utils.get_dace_type(mlir_result_type):
                warnings.warn("Type mismatch between MLIR tasklet out connector and MLIR code")

            for mlir_arg in utils.get_entry_args(mlir_entry_func, mlir_is_generic):
                if self.in_connectors[mlir_arg[0]] is None or self.in_connectors[mlir_arg[0]].ctype == "void":
                    self.in_connectors[mlir_arg[0]] = utils.get_dace_type(mlir_arg[1])
                elif self.in_connectors[mlir_arg[0]] != utils.get_dace_type(mlir_arg[1]):
                    warnings.warn("Type mismatch between MLIR tasklet in connector and MLIR code")

            return

        # If a Python tasklet, use type inference to figure out all None output
        # connectors
        if all(cval.type is not None for cval in self.out_connectors.values()):
            return
        if self.code.language != dtypes.Language.Python:
            return

        if any(cval.type is None for cval in self.in_connectors.values()):
            raise TypeError('Cannot infer output connectors of tasklet "%s", '
                            'not all input connectors have types' % str(self))

        # Get symbols defined at beginning of node, and infer all types in
        # tasklet
        syms = state.symbols_defined_at(self)
        syms.update(self.in_connectors)
        new_syms = infer_types(self.code.code, syms)
        for cname, oconn in self.out_connectors.items():
            if oconn.type is None:
                if cname not in new_syms:
                    raise TypeError('Cannot infer type of tasklet %s output '
                                    '"%s", please specify manually.' % (self.label, cname))
                self.out_connectors[cname] = new_syms[cname]

    def __str__(self):
        if not self.label:
            return "--Empty--"
        else:
            return self.label


@make_properties
class RTLTasklet(Tasklet):
    """ A specialized tasklet, which is a functional computation procedure
        that can only access external data specified using connectors.

        This tasklet is specialized for tasklets implemented in System Verilog
        in that it adds support for adding metadata about the IP cores in use.
    """
    # TODO to be replaced when enums have embedded properties
    ip_cores = DictProperty(key_type=str, value_type=dict, desc="A set of IP cores used by the tasklet.")

    @property
    def __jsontype__(self):
        return 'Tasklet'

    def add_ip_core(self, module_name, name, vendor, version, params):
        self.ip_cores[module_name] = {'name': name, 'vendor': vendor, 'version': version, 'params': params}


# ------------------------------------------------------------------------------


@make_properties
class NestedSDFG(CodeNode):
    """ An SDFG state node that contains an SDFG of its own, runnable using
        the data dependencies specified using its connectors.

        It is encouraged to use nested SDFGs instead of coarse-grained tasklets
        since they are analyzable with respect to transformations.

        :note: A nested SDFG cannot create recursion (one of its parent SDFGs).
    """

    # NOTE: We cannot use SDFG as the type because of an import loop
    sdfg = SDFGReferenceProperty(desc="The SDFG", allow_none=True)
    ext_sdfg_path = Property(dtype=str,
                             default=None,
                             allow_none=True,
                             desc='Path to a file containing the SDFG for this nested SDFG')
    symbol_mapping = DictProperty(key_type=str,
                                  value_type=dace.symbolic.pystr_to_symbolic,
                                  desc="Mapping between internal symbols and their values, expressed as "
                                  "symbolic expressions")
    debuginfo = DebugInfoProperty()
    is_collapsed = Property(dtype=bool, desc="Show this node/scope/state as collapsed", default=False)

    instrument = EnumProperty(dtype=dtypes.InstrumentationType,
                              desc="Measure execution statistics with given method",
                              default=dtypes.InstrumentationType.No_Instrumentation)

    no_inline = Property(dtype=bool,
                         desc="If True, this nested SDFG will not be inlined during "
                         "simplification",
                         default=False)

    unique_name = Property(dtype=str, desc="Unique name of the SDFG", default="")

    def __init__(self,
                 label,
                 sdfg: Optional['dace.SDFG'],
                 inputs: Set[str],
                 outputs: Set[str],
                 symbol_mapping: Dict[str, Any] = None,
                 location=None,
                 debuginfo=None,
                 path: Optional[str] = None):
        super(NestedSDFG, self).__init__(label, location, inputs, outputs)

        # Properties
        self.sdfg: 'dace.SDFG' = sdfg
        self.ext_sdfg_path = path
        self.symbol_mapping = symbol_mapping or {}
        self.debuginfo = debuginfo

    def load_external(self, context: Optional['dace.SDFGState']) -> None:
        if self.sdfg is None and self.ext_sdfg_path is not None:
            self.sdfg = dace.SDFG.from_file(self.ext_sdfg_path)
            self.sdfg.parent_nsdfg_node = self
            self.sdfg.parent = context
            self.sdfg.parent_sdfg = context.sdfg if context else None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # Skip GUID.
            if k in ('guid', ):
                continue
            setattr(result, k, dcpy(v, memo))
        if result._sdfg is not None:
            result._sdfg.parent_nsdfg_node = result
        return result

    @staticmethod
    def from_json(json_obj, context=None):
        from dace import SDFG  # Avoid import loop

        # We have to load the SDFG first.
        ret = NestedSDFG("nolabel", SDFG('nosdfg'), {}, {})

        dace.serialize.set_properties_from_json(ret, json_obj, context)

        if ret.sdfg is not None:
            if context and 'sdfg_state' in context:
                ret.sdfg.parent = context['sdfg_state']
            if context and 'sdfg' in context:
                ret.sdfg.parent_sdfg = context['sdfg']
            ret.sdfg.parent_nsdfg_node = ret

            ret.sdfg.update_cfg_list([])

        return ret

    def used_symbols(self, all_symbols: bool) -> Set[str]:
        free_syms = set().union(*(map(str, pystr_to_symbolic(v).free_symbols) for v in self.location.values()))

        keys_to_use = set(self.symbol_mapping.keys())

        # Filter out unused internal symbols from symbol mapping
        if not all_symbols:
            internally_used_symbols = self.sdfg.used_symbols(all_symbols=False)
            keys_to_use &= internally_used_symbols

        # Translate the internal symbols back to their external counterparts.
        free_syms |= set().union(*(map(str,
                                       pystr_to_symbolic(v).free_symbols) for k, v in self.symbol_mapping.items()
                                   if k in keys_to_use))

        return free_syms

    @property
    def free_symbols(self) -> Set[str]:
        return self.used_symbols(all_symbols=True)

    def infer_connector_types(self, sdfg, state):
        # Avoid import loop
        from dace.sdfg.infer_types import infer_connector_types, infer_aliasing

        # Propagate aliasing information into SDFG
        infer_aliasing(self, sdfg, state)

        # Infer internal connector types
        infer_connector_types(self.sdfg)

    def __str__(self):
        if not self.label:
            return "SDFG"
        else:
            return self.label

    def validate(self, sdfg, state, references: Optional[Set[int]] = None, **context: bool):
        if not dtypes.validate_name(self.label):
            raise NameError('Invalid nested SDFG name "%s"' % self.label)
        for in_conn in self.in_connectors:
            if not dtypes.validate_name(in_conn):
                raise NameError('Invalid input connector "%s"' % in_conn)
        for out_conn in self.out_connectors:
            if not dtypes.validate_name(out_conn):
                raise NameError('Invalid output connector "%s"' % out_conn)
        if self.sdfg:
            if self.sdfg.parent_nsdfg_node is not self:
                raise ValueError('Parent nested SDFG node not properly set')
            if self.sdfg.parent is not state:
                raise ValueError('Parent state not properly set for nested SDFG node')
            if self.sdfg.parent_sdfg is not sdfg:
                raise ValueError('Parent SDFG not properly set for nested SDFG node')

            connectors = self.in_connectors.keys() | self.out_connectors.keys()
            for conn in connectors:
                if conn in self.sdfg.symbols:
                    raise ValueError(f'Connector "{conn}" was given, but it refers to a symbol, which is not allowed. '
                                     'To pass symbols use "symbol_mapping".')
                if conn not in self.sdfg.arrays:
                    raise NameError(
                        f'Connector "{conn}" was given but is not a registered data descriptor in the nested SDFG. '
                        'Example: parameter passed to a function without a matching array within it.')
            for dname, desc in self.sdfg.arrays.items():
                if not desc.transient and dname not in connectors:
                    raise NameError('Data descriptor "%s" not found in nested SDFG connectors' % dname)
                if dname in connectors and desc.transient:
                    raise NameError('"%s" is a connector but its corresponding array is transient' % dname)

        # Validate inout connectors
        from dace.sdfg import utils  # Avoids circular import
        inout_connectors = self.in_connectors.keys() & self.out_connectors.keys()
        for conn in inout_connectors:
            inputs = set()
            outputs = set()
            for edge in state.in_edges_by_connector(self, conn):
                src = utils.get_global_memlet_path_src(sdfg, state, edge)
                if isinstance(src, AccessNode):
                    inputs.add(src.data)
            for edge in state.out_edges_by_connector(self, conn):
                dst = utils.get_global_memlet_path_dst(sdfg, state, edge)
                if isinstance(dst, AccessNode):
                    outputs.add(dst.data)
            if len(inputs - outputs) > 0:
                raise ValueError(f"Inout connector {conn} is connected to different input ({inputs}) and "
                                 f"output ({outputs}) arrays")

        # Validate undefined symbols
        if self.sdfg:
            symbols = set(k for k in self.sdfg.free_symbols if k not in connectors)
            missing_symbols = [s for s in symbols if s not in self.symbol_mapping]
            if missing_symbols:
                raise ValueError('Missing symbols on nested SDFG: %s' % (missing_symbols))
            extra_symbols = self.symbol_mapping.keys() - symbols
            if len(extra_symbols) > 0:
                # TODO: Elevate to an error?
                warnings.warn(f"{self.label} maps to unused symbol(s): {extra_symbols}")

            # Recursively validate nested SDFG
            self.sdfg.validate(references, **context)


# ------------------------------------------------------------------------------


# Scope entry class
class EntryNode(Node):
    """ A type of node that opens a scope (e.g., Map or Consume). """

    def validate(self, sdfg, state):
        self.map.validate(sdfg, state, self)

    add_scope_connectors = Node._add_scope_connectors


# ------------------------------------------------------------------------------


# Scope exit class
class ExitNode(Node):
    """ A type of node that closes a scope (e.g., Map or Consume). """

    def validate(self, sdfg, state):
        self.map.validate(sdfg, state, self)

    add_scope_connectors = Node._add_scope_connectors


# ------------------------------------------------------------------------------


@dace.serialize.serializable
class MapEntry(EntryNode):
    """ Node that opens a Map scope.

        :see: Map
    """

    def __init__(self, map: 'Map', dynamic_inputs=None):
        super(MapEntry, self).__init__(dynamic_inputs or set())
        if map is None:
            raise ValueError("Map for MapEntry can not be None.")
        self._map = map

    @staticmethod
    def map_type():
        return Map

    @classmethod
    def from_json(cls, json_obj, context=None):
        m = cls.map_type()("", [], [])
        ret = cls(map=m)

        try:
            # Connection of the scope nodes
            try:
                nid = int(json_obj['scope_exit'])
            except KeyError:
                # Backwards compatibility
                nid = int(json_obj['scope_exits'][0])
            except TypeError:
                nid = None

            if nid is not None:
                exit_node = context['sdfg_state'].node(nid)
                exit_node.map = m
        except graph.NodeNotFoundError:  # Exit node has a higher node ID
            # Connection of the scope nodes handled in MapExit
            pass

        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, val):
        self._map = val

    def __str__(self):
        return str(self.map)

    @property
    def free_symbols(self) -> Set[str]:
        dyn_inputs = set(c for c in self.in_connectors if not c.startswith('IN_'))
        return set(k for k in self._map.range.free_symbols if k not in dyn_inputs)

    def new_symbols(self, sdfg, state, symbols) -> Dict[str, dtypes.typeclass]:
        result = {}
        # Add map params
        for p, rng in zip(self._map.params, self._map.range):
            result[p] = dtypes.result_type_of(infer_expr_type(rng[0], symbols), infer_expr_type(rng[1], symbols))

        # Handle the dynamic map ranges.
        dyn_inputs = set(c for c in self.in_connectors if not c.startswith('IN_'))
        for e in state.in_edges(self):
            if e.dst_conn in dyn_inputs:
                result[e.dst_conn] = (self.in_connectors[e.dst_conn] or sdfg.arrays[e.data.data].dtype)

        return result

    def used_symbols_within_scope(self, parent_state: 'dace.SDFGState', all_symbols: bool = False) -> Set[str]:
        """
        Returns a set of symbol names that are used within the Map scope created by this MapEntry

        :param all_symbols: If False, only returns symbols that are needed as arguments (only used in generated code).
        """
        parent_sdfg: dace.SDFG = parent_state.sdfg

        new_symbols = set()
        free_symbols = set()

        # Free symbols from nodes
        for n in parent_state.all_nodes_between(self, parent_state.exit_node(self)):
            if isinstance(n, EntryNode):
                new_symbols |= set(n.new_symbols(parent_sdfg, parent_state, {}).keys())
            elif isinstance(n, AccessNode):
                # Add data descriptor symbols
                free_symbols |= set(map(str, n.desc(parent_sdfg).used_symbols(all_symbols)))
            elif isinstance(n, Tasklet):
                if n.language == dtypes.Language.Python:
                    # Consider callbacks defined as symbols as free
                    for stmt in n.code.code:
                        for astnode in ast.walk(stmt):
                            if (isinstance(astnode, ast.Call) and isinstance(astnode.func, ast.Name)
                                    and astnode.func.id in parent_sdfg.symbols):
                                free_symbols.add(astnode.func.id)
                else:
                    # Find all string tokens and filter them to sdfg.symbols, while ignoring connectors
                    code_symbols = dace.symbolic.symbols_in_code(
                        n.code.as_string,
                        potential_symbols=parent_sdfg.symbols.keys(),
                        symbols_to_ignore=(n.in_connectors.keys() | n.out_connectors.keys() | n.ignored_symbols),
                    )
                    free_symbols |= code_symbols
                    continue
            elif isinstance(n, NestedSDFG):
                free_symbols |= n.used_symbols(all_symbols)

            free_symbols |= n.free_symbols

        # Free symbols from memlets
        for e in parent_state.all_edges(*parent_state.all_nodes_between(self, parent_state.exit_node(self))):
            # If used for code generation, only consider memlet tree leaves
            if not all_symbols and not parent_state.is_leaf_memlet(e):
                continue

            free_symbols |= e.data.used_symbols(all_symbols, e)

        # Do not consider SDFG constants as symbols
        new_symbols.update(set(parent_sdfg.constants.keys()))
        return free_symbols - new_symbols


@dace.serialize.serializable
class MapExit(ExitNode):
    """ Node that closes a Map scope.

        :see: Map
    """

    def __init__(self, map: 'Map'):
        super(MapExit, self).__init__()
        if map is None:
            raise ValueError("Map for MapExit can not be None.")
        self._map = map

    @staticmethod
    def map_type():
        return Map

    @classmethod
    def from_json(cls, json_obj, context=None):
        try:
            # Set map reference to map entry
            entry_node = context['sdfg_state'].node(int(json_obj['scope_entry']))

            ret = cls(map=entry_node.map)
        except (IndexError, TypeError, graph.NodeNotFoundError):
            # Entry node has a higher ID than exit node
            # Connection of the scope nodes handled in MapEntry
            ret = cls(cls.map_type()('_', [], []))

        dace.serialize.set_properties_from_json(ret, json_obj, context=context)

        return ret

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, val):
        self._map = val

    @property
    def schedule(self):
        return self._map.schedule

    @schedule.setter
    def schedule(self, val):
        self._map.schedule = val

    @property
    def label(self):
        return self._map.label

    def __str__(self):
        return str(self.map)


@make_properties
class Map(object):
    """ A Map is a two-node representation of parametric graphs, containing
        an integer set by which the contents (nodes dominated by an entry
        node and post-dominated by an exit node) are replicated.

        Maps contain a `schedule` property, which specifies how the scope
        should be scheduled (execution order). Code generators can use the
        schedule property to generate appropriate code, e.g., GPU kernels.
    """

    # List of (editable) properties
    label = Property(dtype=str, desc="Label of the map")
    params = ListProperty(element_type=str, desc="Mapped parameters")
    range = RangeProperty(desc="Ranges of map parameters", default=sbs.Range([]))
    schedule = EnumProperty(dtype=dtypes.ScheduleType, desc="Map schedule", default=dtypes.ScheduleType.Default)
    unroll = Property(dtype=bool, desc="Map unrolling")
    unroll_factor = Property(dtype=int,
                             allow_none=True,
                             default=0,
                             desc="How much iterations should be unrolled."
                             " To prevent unrolling, set this value to 1.")
    collapse = Property(dtype=int, default=1, desc="How many dimensions to collapse into the parallel range")
    debuginfo = DebugInfoProperty()
    is_collapsed = Property(dtype=bool, desc="Show this node/scope/state as collapsed", default=False)

    instrument = EnumProperty(dtype=dtypes.InstrumentationType,
                              desc="Measure execution statistics with given method",
                              default=dtypes.InstrumentationType.No_Instrumentation)

    omp_num_threads = Property(dtype=int,
                               default=0,
                               desc="Number of OpenMP threads executing the Map",
                               serialize_if=lambda m: m.schedule in dtypes.CPU_SCHEDULES)
    omp_schedule = EnumProperty(dtype=dtypes.OMPScheduleType,
                                default=dtypes.OMPScheduleType.Default,
                                desc="OpenMP schedule {static, dynamic, guided}",
                                serialize_if=lambda m: m.schedule in dtypes.CPU_SCHEDULES)
    omp_chunk_size = Property(dtype=int,
                              default=0,
                              desc="OpenMP schedule chunk size",
                              serialize_if=lambda m: m.schedule in dtypes.CPU_SCHEDULES)

    gpu_block_size = ListProperty(element_type=int,
                                  default=None,
                                  allow_none=True,
                                  desc="GPU kernel block size",
                                  serialize_if=lambda m: m.schedule in dtypes.GPU_SCHEDULES)

    gpu_launch_bounds = Property(dtype=str,
                                 default="0",
                                 desc="GPU kernel launch bounds. A value of -1 disables the statement, 0 (default) "
                                 "enables the statement if block size is not symbolic, and any other value "
                                 "(including tuples) sets it explicitly.",
                                 serialize_if=lambda m: m.schedule in dtypes.GPU_SCHEDULES)

    gpu_maxnreg = Property(dtype=int,
                           default=0,
                           desc="Maximum number of registers per thread for GPU kernel",
                           serialize_if=lambda m: m.schedule in dtypes.GPU_SCHEDULES)

    gpu_force_syncthreads = Property(dtype=bool, desc="Force a call to the __syncthreads for the map", default=False)

    def __init__(self,
                 label,
                 params,
                 ndrange,
                 schedule=dtypes.ScheduleType.Default,
                 unroll=False,
                 collapse=1,
                 fence_instrumentation=False,
                 debuginfo=None):
        super(Map, self).__init__()

        # Assign properties
        self.label = label
        self.schedule = schedule
        self.unroll = unroll
        self.collapse = collapse
        self.params = params
        self.range = ndrange
        self.debuginfo = debuginfo
        self._fence_instrumentation = fence_instrumentation

    def __str__(self):
        return self.label + "[" + ", ".join(
            ["{}={}".format(i, r)
             for i, r in zip(self._params, [sbs.Range.dim_to_string(d) for d in self._range])]) + "]"

    def __repr__(self):
        return type(self).__name__ + ' (' + self.__str__() + ')'

    def validate(self, sdfg, state, node):
        if not dtypes.validate_name(self.label):
            raise NameError(f'Invalid map name "{self.label}"')
        if self.get_param_num() == 0:
            raise ValueError('There must be at least one parameter in a map.')
        if self.get_param_num() != self.range.dims():
            raise ValueError(f'There are {self.get_param_num()} parameters but the range'
                             f' has {self.range.dims()} dimensions.')

        # The only thing that makes sense, at least on GPU and CPU is a positive
        #  increment and a positive range of iteration. We could handle sequential
        #  Maps a bit more liberal, but we should probably not.
        if any((ss <= 0) == True for ss in self.range.size()):
            # The CPU and GPU backend tolerate such maps.
            warnings.warn(f'The iteration range of Map {self.label} is {self.range}, which contains a zero'
                          ' or negative sized range, which is allowed but not recommended.'
                          ' The Map will be turned into a no-ops.')
        if any((inc <= 0) == True for (_, _, inc) in self.range):
            # Should this be turned into an error?
            warnings.warn(f'An increment of Map {self.label} was negative, which is allowerd'
                          ' but probably not useful.')

    def get_param_num(self):
        """ Returns the number of map dimension parameters/symbols. """
        return len(self.params)


# Indirect Map properties to MapEntry and MapExit
MapEntry = indirect_properties(Map, lambda obj: obj.map)(MapEntry)

# ------------------------------------------------------------------------------


@dace.serialize.serializable
class ConsumeEntry(EntryNode):
    """ Node that opens a Consume scope.

        :see: Consume
    """

    def __init__(self, consume: 'Consume', dynamic_inputs=None):
        super(ConsumeEntry, self).__init__(dynamic_inputs or set())
        if consume is None:
            raise ValueError("Consume for ConsumeEntry can not be None.")
        self._consume = consume
        self.add_in_connector('IN_stream')
        self.add_out_connector('OUT_stream')

    @staticmethod
    def from_json(json_obj, context=None):
        c = Consume("", ['i', 1], None)
        ret = ConsumeEntry(consume=c)

        try:
            # Set map reference to map exit
            try:
                nid = int(json_obj['scope_exit'])
            except KeyError:
                # Backwards compatibility
                nid = int(json_obj['scope_exits'][0])
            except TypeError:
                nid = None

            if nid is not None:
                exit_node = context['sdfg_state'].node(nid)
                exit_node.consume = c
        except graph.NodeNotFoundError:  # Exit node has a higher node ID
            # Connection of the scope nodes handled in ConsumeExit
            pass

        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    @property
    def map(self):
        return self._consume.as_map()

    @property
    def consume(self):
        return self._consume

    @consume.setter
    def consume(self, val):
        self._consume = val

    def __str__(self):
        return str(self.consume)

    @property
    def free_symbols(self) -> Set[str]:
        dyn_inputs = set(c for c in self.in_connectors if not c.startswith('IN_'))
        result = set(self._consume.num_pes.free_symbols)
        if self._consume.condition is not None:
            result |= set(self._consume.condition.get_free_symbols())
        return result - dyn_inputs

    def new_symbols(self, sdfg, state, symbols) -> Dict[str, dtypes.typeclass]:
        result = {}
        # Add PE index
        result[self._consume.pe_index] = infer_expr_type(self._consume.num_pes, symbols)

        # Add dynamic inputs
        dyn_inputs = set(c for c in self.in_connectors if not c.startswith('IN_'))

        # Try to get connector type from connector
        for e in state.in_edges(self):
            if e.dst_conn in dyn_inputs:
                result[e.dst_conn] = (self.in_connectors[e.dst_conn] or sdfg.arrays[e.data.data].dtype)

        return result


@dace.serialize.serializable
class ConsumeExit(ExitNode):
    """ Node that closes a Consume scope.

        :see: Consume
    """

    def __init__(self, consume: 'Consume'):
        super(ConsumeExit, self).__init__()
        if consume is None:
            raise ValueError("Consume for ConsumeExit can not be None.")
        self._consume = consume

    @staticmethod
    def from_json(json_obj, context=None):
        try:
            # Set consume reference to entry node
            entry_node = context['sdfg_state'].node(int(json_obj['scope_entry']))
            ret = ConsumeExit(consume=entry_node.consume)
        except (IndexError, TypeError, graph.NodeNotFoundError):
            # Entry node has a higher ID than exit node
            # Connection of the scope nodes handled in ConsumeEntry
            ret = ConsumeExit(Consume("", ['i', 1], None))

        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    @property
    def map(self):
        return self._consume.as_map()

    @property
    def consume(self):
        return self._consume

    @consume.setter
    def consume(self, val):
        self._consume = val

    @property
    def schedule(self):
        return self._consume.schedule

    @schedule.setter
    def schedule(self, val):
        self._consume.schedule = val

    @property
    def label(self):
        return self._consume.label

    def __str__(self):
        return str(self.consume)


@make_properties
class Consume(object):
    """ Consume is a scope, like `Map`, that is a part of the parametric
        graph extension of the SDFG. It creates a producer-consumer
        relationship between the input stream and the scope subgraph. The
        subgraph is scheduled to a given number of processing elements
        for processing, and they will try to pop elements from the input
        stream until a given quiescence condition is reached. """

    # Properties
    label = Property(dtype=str, desc="Name of the consume node")
    pe_index = Property(dtype=str, desc="Processing element identifier")
    num_pes = SymbolicProperty(desc="Number of processing elements", default=1)
    condition = CodeProperty(desc="Quiescence condition", allow_none=True, default=None)
    schedule = EnumProperty(dtype=dtypes.ScheduleType, desc="Consume schedule", default=dtypes.ScheduleType.Default)
    chunksize = Property(dtype=int, desc="Maximal size of elements to consume at a time", default=1)
    debuginfo = DebugInfoProperty()
    is_collapsed = Property(dtype=bool, desc="Show this node/scope/state as collapsed", default=False)

    instrument = EnumProperty(dtype=dtypes.InstrumentationType,
                              desc="Measure execution statistics with given method",
                              default=dtypes.InstrumentationType.No_Instrumentation)

    def as_map(self):
        """ Compatibility function that allows to view the consume as a map,
            mainly in memlet propagation. """
        return Map(self.label, [self.pe_index], sbs.Range([(0, self.num_pes - 1, 1)]), self.schedule)

    def __init__(self, label, pe_tuple, condition, schedule=dtypes.ScheduleType.Default, chunksize=1, debuginfo=None):
        super(Consume, self).__init__()

        # Properties
        self.label = label
        self.pe_index, self.num_pes = pe_tuple
        self.condition = condition
        self.schedule = schedule
        self.chunksize = chunksize
        self.debuginfo = debuginfo

    def __str__(self):
        if self.condition is not None:
            return ("%s [%s=0:%s], Condition: %s" %
                    (self._label, self.pe_index, self.num_pes, CodeProperty.to_string(self.condition)))
        else:
            return ("%s [%s=0:%s]" % (self._label, self.pe_index, self.num_pes))

    def validate(self, sdfg, state, node):
        if not dtypes.validate_name(self.label):
            raise NameError('Invalid consume name "%s"' % self.label)

    def get_param_num(self):
        """ Returns the number of consume dimension parameters/symbols. """
        return 1


# Redirect Consume properties to ConsumeEntry and ConsumeExit
ConsumeEntry = indirect_properties(Consume, lambda obj: obj.consume)(ConsumeEntry)

# ------------------------------------------------------------------------------


# Based on https://stackoverflow.com/a/2020083/6489142
def full_class_path(cls_or_obj: Union[type, object]):
    if isinstance(cls_or_obj, type):
        cls = cls_or_obj
    else:
        cls = type(cls_or_obj)
    module = cls.__module__
    if module is None or module == str.__class__.__module__:
        return cls.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + cls.__name__


@make_properties
class LibraryNode(CodeNode):

    name = Property(dtype=str, desc="Name of node")
    implementation = LibraryImplementationProperty(dtype=str,
                                                   allow_none=True,
                                                   desc=("Which implementation this library node will expand into."
                                                         "Must match a key in the list of possible implementations."))
    schedule = EnumProperty(dtype=dtypes.ScheduleType,
                            desc="If set, determines the default device mapping of "
                            "the node upon expansion, if expanded to a nested SDFG.",
                            default=dtypes.ScheduleType.Default)
    debuginfo = DebugInfoProperty()

    def __init__(self, name, *args, schedule=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.label = name
        self.schedule = schedule or dtypes.ScheduleType.Default

    # Overrides subclasses to return LibraryNode as their JSON type
    @property
    def __jsontype__(self):
        return 'LibraryNode'

    @property
    def has_side_effects(self) -> bool:
        """
        Returns True if this library node has other side effects (e.g., calling stateful libraries, communicating)
        when expanded.
        This method is meant to be extended by subclasses.
        """
        return False

    def to_json(self, parent):
        jsonobj = super().to_json(parent)
        jsonobj['classpath'] = full_class_path(self)
        return jsonobj

    @classmethod
    def from_json(cls, json_obj, context=None):
        if cls == LibraryNode:
            clazz = pydoc.locate(json_obj['classpath'])
            if clazz is None:
                warnings.warn(f'Could not find class "{json_obj["classpath"]}" while deserializing. Falling back '
                              'to UnregisteredLibraryNode.')
                return UnregisteredLibraryNode.from_json(json_obj, context)
            return clazz.from_json(json_obj, context)
        else:  # Subclasses are actual library nodes
            # Initialize library node without calling constructor
            ret = cls.__new__(cls)
            dace.serialize.set_properties_from_json(ret, json_obj, context=context)
            return ret

    def expand(self, state_or_sdfg, state_or_impl=None, **kwargs) -> str:
        """ Create and perform the expansion transformation for this library
            node.

            This method supports two interfaces:
            1. New interface: expand(state, implementation=None, **kwargs)
            2. Old interface: expand(sdfg, state, **kwargs) [for backward compatibility]

            :param state_or_sdfg: Either a ControlFlowBlock (new interface) or SDFG (old interface)
            :param state_or_impl: Either implementation name (new interface) or SDFGState (old interface)
            :param kwargs: Additional expansion arguments
            :return: the name of the expanded implementation

            Examples:
                # New interface (recommended):
                result = node.expand(state, 'pure')

                # Old interface (still supported):
                result = node.expand(sdfg, state)
        """
        from dace.transformation.transformation import ExpandTransformation  # Avoid import loop
        import warnings

        # Handle both old and new signatures for backward compatibility
        from dace.sdfg.state import SDFGState

        if isinstance(state_or_sdfg, SDFGState):
            # New interface: expand(state, implementation=None, **kwargs)
            actual_state = state_or_sdfg
            sdfg = actual_state.parent
            implementation = state_or_impl
            expansion_kwargs = kwargs
        else:
            # Old interface: expand(sdfg, state, **kwargs)
            warnings.warn("The expand(sdfg, state) interface is deprecated. Use expand(state, implementation) instead.",
                          DeprecationWarning,
                          stacklevel=2)
            sdfg = state_or_sdfg
            actual_state = state_or_impl
            expansion_kwargs = kwargs
            implementation = None
            # In old interface, implementation might be passed as keyword arg
            if 'implementation' in kwargs:
                implementation = kwargs.pop('implementation')

        # Use provided implementation or fall back to current node implementation
        if implementation is not None:
            target_implementation = implementation
        else:
            target_implementation = self.implementation
        library_name = getattr(type(self), '_dace_library_name', '')
        try:
            if library_name:
                config_implementation = Config.get("library", library_name, "default_implementation")
            else:
                config_implementation = None
        except KeyError:
            # Non-standard libraries are not defined in the config schema, and
            # thus might not exist in the config.
            config_implementation = None
        if config_implementation is not None:
            try:
                config_override = Config.get("library", library_name, "override")
                if config_override and target_implementation in self.implementations:
                    if target_implementation is not None:
                        warnings.warn("Overriding explicitly specified "
                                      "implementation {} for {} with {}.".format(target_implementation, self.label,
                                                                                 config_implementation))
                    target_implementation = config_implementation
            except KeyError:
                config_override = False
        # If not explicitly set, try the node default
        if target_implementation is None:
            target_implementation = type(self).default_implementation
            # If no node default, try library default
            if target_implementation is None:
                import dace.library  # Avoid cyclic dependency
                lib = dace.library._DACE_REGISTERED_LIBRARIES[type(self)._dace_library_name]
                target_implementation = lib.default_implementation
                # Try the default specified in the config
                if target_implementation is None:
                    target_implementation = config_implementation
                    # Otherwise we don't know how to expand
                    if target_implementation is None:
                        raise ValueError("No implementation or default implementation specified.")
        if target_implementation not in self.implementations.keys():
            raise KeyError("Unknown implementation for node {}: {}".format(type(self).__name__, target_implementation))
        transformation_type = type(self).implementations[target_implementation]
        cfg_id = actual_state.parent_graph.cfg_id
        state_id = actual_state.block_id
        subgraph = {transformation_type._match_node: actual_state.node_id(self)}
        transformation: ExpandTransformation = transformation_type()
        transformation.setup_match(sdfg, cfg_id, state_id, subgraph, 0)
        if not transformation.can_be_applied(actual_state, 0, sdfg):
            raise RuntimeError("Library node expansion applicability check failed.")
        sdfg.append_transformation(transformation)
        transformation.apply(actual_state, sdfg, **expansion_kwargs)
        return target_implementation

    @classmethod
    def register_implementation(cls, name, transformation_type):
        """Register an implementation to belong to this library node type."""
        cls.implementations[name] = transformation_type
        transformation_type._match_node = cls

    @property
    def free_symbols(self) -> Set[str]:
        fsyms = super(LibraryNode, self).free_symbols
        for p, v in self.properties():
            if isinstance(p, SymbolicProperty) and issymbolic(v):
                fsyms.update((str(s) for s in v.free_symbols))
        return fsyms


class UnregisteredLibraryNode(LibraryNode):

    original_json = {}

    def __init__(self, json_obj={}, label=None):
        self.original_json = json_obj
        super().__init__(label)

    def to_json(self, parent):
        jsonobj = dcpy(self.original_json)
        curjson = super().to_json(parent)

        # Start with original json, then update the modified parts
        for pname, prop in curjson.items():
            if isinstance(prop, dict):  # Dictionary property update (e.g., attributes)
                jsonobj[pname].update(curjson[pname])
            else:  # Direct property update
                jsonobj[pname] = curjson[pname]

        return jsonobj

    @classmethod
    def from_json(cls, json_obj, context=None):
        ret = cls(json_obj=json_obj, label=json_obj['attributes']['name'])
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def expand(self, sdfg, state, *args, **kwargs):
        raise TypeError(f'Cannot expand unregistered library node "{self.name}"')
