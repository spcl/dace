""" Contains classes implementing the different types of nodes of the stateful
    dataflow multigraph representation. """

import ast
from copy import deepcopy as dcpy
import itertools
import dace.serialize
from typing import Set
from dace.graph import dot, graph
from dace.frontend.python.astutils import unparse
from dace.properties import (
    Property, CodeProperty, LambdaProperty, ParamsProperty, RangeProperty,
    DebugInfoProperty, SetProperty, make_properties, indirect_properties,
    DataProperty, SymbolicProperty, ListProperty, SDFGReferenceProperty)
from dace.frontend.operations import detect_reduction_type
from dace import data, subsets as sbs, dtypes

# -----------------------------------------------------------------------------


@make_properties
class Node(object):
    """ Base node class. """

    in_connectors = SetProperty(
        str, default=set(), desc="A set of input connectors for this node.")
    out_connectors = SetProperty(
        str, default=set(), desc="A set of output connectors for this node.")

    def __init__(self, in_connectors=None, out_connectors=None):
        self.in_connectors = in_connectors or set()
        self.out_connectors = out_connectors or set()

    def __str__(self):
        if hasattr(self, 'label'):
            return self.label
        else:
            return type(self).__name__

    def validate(self, sdfg, state):
        pass

    def to_json(self, parent):
        labelstr = str(self)
        typestr = str(type(self).__name__)

        scope_entry_node = parent.entry_node(self)
        if scope_entry_node is not None:
            ens = parent.exit_nodes(parent.entry_node(self))
            scope_exit_nodes = [str(parent.node_id(x)) for x in ens]
            scope_entry_node = str(parent.node_id(scope_entry_node))
        else:
            scope_entry_node = None
            scope_exit_nodes = []

        retdict = {
            "type": typestr,
            "label": labelstr,
            "attributes": dace.serialize.all_properties_to_json(self),
            "id": parent.node_id(self),
            "scope_entry": scope_entry_node,
            "scope_exits": scope_exit_nodes
        }
        return retdict

    def __repr__(self):
        return type(self).__name__ + ' (' + self.__str__() + ')'

    def add_in_connector(self, connector_name: str):
        """ Adds a new input connector to the node. The operation will fail if
            a connector (either input or output) with the same name already
            exists in the node.

            :param connector_name: The name of the new connector.
            :return: True if the operation is successful, otherwise False.
        """

        if (connector_name in self.in_connectors
                or connector_name in self.out_connectors):
            return False
        connectors = self.in_connectors
        connectors.add(connector_name)
        self.in_connectors = connectors
        return True

    def add_out_connector(self, connector_name: str):
        """ Adds a new output connector to the node. The operation will fail if
            a connector (either input or output) with the same name already
            exists in the node.

            :param connector_name: The name of the new connector.
            :return: True if the operation is successful, otherwise False.
        """

        if (connector_name in self.in_connectors
                or connector_name in self.out_connectors):
            return False
        connectors = self.out_connectors
        connectors.add(connector_name)
        self.out_connectors = connectors
        return True

    def remove_in_connector(self, connector_name: str):
        """ Removes an input connector from the node.
            :param connector_name: The name of the connector to remove.
            :return: True if the operation was successful.
        """

        if connector_name in self.in_connectors:
            connectors = self.in_connectors
            connectors.remove(connector_name)
            self.in_connectors = connectors
        return True

    def remove_out_connector(self, connector_name: str):
        """ Removes an output connector from the node.
            :param connector_name: The name of the connector to remove.
            :return: True if the operation was successful.
        """

        if connector_name in self.out_connectors:
            connectors = self.out_connectors
            connectors.remove(connector_name)
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

    def next_connector(self) -> str:
        """ Returns the next unused connector ID (as a string). Used for
            filling connectors when adding edges to scopes. """
        return str(self._next_connector_int())

    def last_connector(self) -> str:
        """ Returns the last used connector ID (as a string). Used for
            filling connectors when adding edges to scopes. """
        return str(self._next_connector_int() - 1)


# ------------------------------------------------------------------------------


@make_properties
class AccessNode(Node):
    """ A node that accesses data in the SDFG. Denoted by a circular shape. """

    access = Property(
        choices=dtypes.AccessType,
        desc="Type of access to this array",
        default=dtypes.AccessType.ReadWrite)
    setzero = Property(dtype=bool, desc="Initialize to zero", default=False)
    debuginfo2 = DebugInfoProperty()
    data = DataProperty(desc="Data (array, stream, scalar) to access")

    def __init__(self,
                 data,
                 access=dtypes.AccessType.ReadWrite,
                 debuginfo=None):
        super(AccessNode, self).__init__()

        # Properties
        self.debuginfo2 = debuginfo
        self.access = access
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
        node._access = self._access
        node._data = self._data
        node._setzero = self._setzero
        node._in_connectors = self._in_connectors
        node._out_connectors = self._out_connectors
        node.debuginfo2 = dcpy(self.debuginfo2)
        return node

    @property
    def label(self):
        return self.data

    def __label__(self, sdfg, state):
        return self.data

    def desc(self, sdfg):
        from dace.sdfg import SDFGState, ScopeSubgraphView
        if isinstance(sdfg, (SDFGState, ScopeSubgraphView)):
            sdfg = sdfg.parent
        return sdfg.arrays[self.data]

    def draw_node(self, sdfg, graph):
        desc = self.desc(sdfg)
        if isinstance(desc, data.Stream):
            return dot.draw_node(
                sdfg, graph, self, shape="oval", style='dashed')
        elif desc.transient:
            return dot.draw_node(sdfg, graph, self, shape="oval")
        else:
            return dot.draw_node(sdfg, graph, self, shape="oval", style='bold')

    def validate(self, sdfg, state):
        if self.data not in sdfg.arrays:
            raise KeyError('Array "%s" not found in SDFG' % self.data)


# ------------------------------------------------------------------------------


class CodeNode(Node):
    """ A node that contains runnable code with acyclic external data
        dependencies. May either be a tasklet or a nested SDFG, and
        denoted by an octagonal shape. """
    pass


@make_properties
class Tasklet(CodeNode):
    """ A node that contains a tasklet: a functional computation procedure
        that can only access external data specified using connectors.

        Tasklets may be implemented in Python, C++, or any supported
        language by the code generator.
    """

    label = Property(dtype=str, desc="Name of the tasklet")
    code = CodeProperty(desc="Tasklet code", default="")
    code_global = CodeProperty(
        desc="Global scope code needed for tasklet execution", default="")
    code_init = CodeProperty(
        desc="Extra code that is called on DaCe runtime initialization",
        default="")
    code_exit = CodeProperty(
        desc="Extra code that is called on DaCe runtime cleanup", default="")
    location = Property(
        dtype=str, desc="Tasklet execution location descriptor")
    debuginfo = DebugInfoProperty()

    instrument = Property(
        choices=dtypes.InstrumentationType,
        desc="Measure execution statistics with given method",
        default=dtypes.InstrumentationType.No_Instrumentation)

    def __init__(self,
                 label,
                 inputs=None,
                 outputs=None,
                 code="",
                 language=dtypes.Language.Python,
                 code_global="",
                 code_init="",
                 code_exit="",
                 location="-1",
                 debuginfo=None):
        super(Tasklet, self).__init__(inputs or set(), outputs or set())

        # Properties
        self.label = label
        # Set the language directly
        #self.language = language
        self.code = {'code_or_block': code, 'language': language}

        self.location = location
        self.code_global = {'code_or_block': code_global, 'language': language}
        self.code_init = {'code_or_block': code_init, 'language': language}
        self.code_exit = {'code_or_block': code_exit, 'language': language}
        self.debuginfo = debuginfo

    @property
    def language(self):
        return self._code['language']

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Tasklet("dummylabel")
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    @property
    def name(self):
        return self._label

    def draw_node(self, sdfg, graph):
        return dot.draw_node(sdfg, graph, self, shape="octagon")

    def validate(self, sdfg, state):
        if not data.validate_name(self.label):
            raise NameError('Invalid tasklet name "%s"' % self.label)
        for in_conn in self.in_connectors:
            if not data.validate_name(in_conn):
                raise NameError('Invalid input connector "%s"' % in_conn)
        for out_conn in self.out_connectors:
            if not data.validate_name(out_conn):
                raise NameError('Invalid output connector "%s"' % out_conn)

    def __str__(self):
        if not self.label:
            return "--Empty--"
        else:
            return self.label


@make_properties
class EmptyTasklet(Tasklet):
    """ A special tasklet that contains no code. Used for filling empty states
        in an SDFG. """

    def __init__(self, label=""):
        super(EmptyTasklet, self).__init__(label)

    def draw_node(self, sdfg, graph):
        return dot.draw_node(sdfg, graph, self, style="invis", shape="octagon")

    def validate(self, sdfg, state):
        pass

    @staticmethod
    def from_json(json_obj, context=None):
        ret = EmptyTasklet("dummylabel")
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret


# ------------------------------------------------------------------------------


@make_properties
class NestedSDFG(CodeNode):
    """ An SDFG state node that contains an SDFG of its own, runnable using
        the data dependencies specified using its connectors.

        It is encouraged to use nested SDFGs instead of coarse-grained tasklets
        since they are analyzable with respect to transformations.

        @note: A nested SDFG cannot create recursion (one of its parent SDFGs).
    """

    label = Property(dtype=str, desc="Name of the SDFG")
    # NOTE: We cannot use SDFG as the type because of an import loop
    sdfg = SDFGReferenceProperty(desc="The SDFG", allow_none=True)
    schedule = Property(
        dtype=dtypes.ScheduleType,
        desc="SDFG schedule",
        choices=dtypes.ScheduleType,
        from_string=lambda x: dtypes.ScheduleType[x],
        default=dtypes.ScheduleType.Default)
    location = Property(dtype=str, desc="SDFG execution location descriptor")
    debuginfo = DebugInfoProperty()
    is_collapsed = Property(
        dtype=bool,
        desc="Show this node/scope/state as collapsed",
        default=False)

    instrument = Property(
        choices=dtypes.InstrumentationType,
        desc="Measure execution statistics with given method",
        default=dtypes.InstrumentationType.No_Instrumentation)

    def __init__(self,
                 label,
                 sdfg,
                 inputs: Set[str],
                 outputs: Set[str],
                 schedule=dtypes.ScheduleType.Default,
                 location="-1",
                 debuginfo=None):
        super(NestedSDFG, self).__init__(inputs, outputs)

        # Properties
        self.label = label
        self.sdfg = sdfg
        self.schedule = schedule
        self.location = location
        self.debuginfo = debuginfo

    @staticmethod
    def from_json(json_obj, context=None):
        from dace import SDFG  # Avoid import loop

        # We have to load the SDFG first.
        ret = NestedSDFG("nolabel", SDFG('nosdfg'), set(), set())

        dace.serialize.set_properties_from_json(ret, json_obj, context)

        if context and 'sdfg_state' in context:
            ret.sdfg.parent = context['sdfg_state']
        if context and 'sdfg' in context:
            ret.sdfg.parent_sdfg = context['sdfg']

        ret.sdfg.update_sdfg_list([])

        return ret

    def draw_node(self, sdfg, graph):
        return dot.draw_node(sdfg, graph, self, shape="doubleoctagon")

    def __str__(self):
        if not self.label:
            return "SDFG"
        else:
            return self.label

    def validate(self, sdfg, state):
        if not data.validate_name(self.label):
            raise NameError('Invalid nested SDFG name "%s"' % self.label)
        for in_conn in self.in_connectors:
            if not data.validate_name(in_conn):
                raise NameError('Invalid input connector "%s"' % in_conn)
        for out_conn in self.out_connectors:
            if not data.validate_name(out_conn):
                raise NameError('Invalid output connector "%s"' % out_conn)

        # Recursively validate nested SDFG
        self.sdfg.validate()


# ------------------------------------------------------------------------------


# Scope entry class
class EntryNode(Node):
    """ A type of node that opens a scope (e.g., Map or Consume). """

    def validate(self, sdfg, state):
        self.map.validate(sdfg, state, self)


# ------------------------------------------------------------------------------


# Scope exit class
class ExitNode(Node):
    """ A type of node that closes a scope (e.g., Map or Consume). """

    def validate(self, sdfg, state):
        self.map.validate(sdfg, state, self)


# ------------------------------------------------------------------------------


@dace.serialize.serializable
class MapEntry(EntryNode):
    """ Node that opens a Map scope.
        @see: Map
    """

    def __init__(self, map, dynamic_inputs=None):
        super(MapEntry, self).__init__(dynamic_inputs or set())
        if map is None:
            raise ValueError("Map for MapEntry can not be None.")
        self._map = map

    @staticmethod
    def from_json(json_obj, context=None):
        m = Map("", [], [])
        ret = MapEntry(map=m)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, val):
        self._map = val

    def draw_node(self, sdfg, graph):
        if self.is_collapsed:
            return dot.draw_node(sdfg, graph, self, shape="hexagon")
        return dot.draw_node(sdfg, graph, self, shape="trapezium")

    def __str__(self):
        return str(self.map)


@dace.serialize.serializable
class MapExit(ExitNode):
    """ Node that closes a Map scope.
        @see: Map
    """

    def __init__(self, map):
        super(MapExit, self).__init__()
        if map is None:
            raise ValueError("Map for MapExit can not be None.")
        self._map = map

    @staticmethod
    def from_json(json_obj, context=None):
        # Set map reference to map entry
        entry_node = context['sdfg_state'].node(int(json_obj['scope_entry']))

        ret = MapExit(map=entry_node.map)
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

    def draw_node(self, sdfg, graph):
        return dot.draw_node(sdfg, graph, self, shape="invtrapezium")

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
    params = ParamsProperty(desc="Mapped parameters")
    range = RangeProperty(
        desc="Ranges of map parameters", default=sbs.Range([]))
    schedule = Property(
        dtype=dtypes.ScheduleType,
        desc="Map schedule",
        choices=dtypes.ScheduleType,
        from_string=lambda x: dtypes.ScheduleType[x],
        default=dtypes.ScheduleType.Default)
    is_async = Property(dtype=bool, desc="Map asynchronous evaluation")
    unroll = Property(dtype=bool, desc="Map unrolling")
    flatten = Property(dtype=bool, desc="Map loop flattening")
    debuginfo = DebugInfoProperty()
    is_collapsed = Property(
        dtype=bool,
        desc="Show this node/scope/state as collapsed",
        default=False)

    instrument = Property(
        choices=dtypes.InstrumentationType,
        desc="Measure execution statistics with given method",
        default=dtypes.InstrumentationType.No_Instrumentation)

    def __init__(self,
                 label,
                 params,
                 ndrange,
                 schedule=dtypes.ScheduleType.Default,
                 unroll=False,
                 is_async=False,
                 flatten=False,
                 fence_instrumentation=False,
                 debuginfo=None):
        super(Map, self).__init__()

        # Assign properties
        self.label = label
        self.schedule = schedule
        self.unroll = unroll
        self.is_async = is_async
        self.flatten = flatten
        self.params = params
        self.range = ndrange
        self.debuginfo = debuginfo
        self._fence_instrumentation = fence_instrumentation

    def __str__(self):
        return self.label + "[" + ", ".join([
            "{}={}".format(i, r)
            for i, r in zip(self._params,
                            [sbs.Range.dim_to_string(d) for d in self._range])
        ]) + "]"

    def validate(self, sdfg, state, node):
        if not data.validate_name(self.label):
            raise NameError('Invalid map name "%s"' % self.label)

    def get_param_num(self):
        """ Returns the number of map dimension parameters/symbols. """
        return len(self.params)


# Indirect Map properties to MapEntry and MapExit
MapEntry = indirect_properties(Map, lambda obj: obj.map)(MapEntry)

# ------------------------------------------------------------------------------


@dace.serialize.serializable
class ConsumeEntry(EntryNode):
    """ Node that opens a Consume scope.
        @see: Consume
    """

    def __init__(self, consume, dynamic_inputs=None):
        super(ConsumeEntry, self).__init__(dynamic_inputs or set())
        if consume is None:
            raise ValueError("Consume for ConsumeEntry can not be None.")
        self._consume = consume
        self._map_depth = 0
        self.add_in_connector('IN_stream')
        self.add_out_connector('OUT_stream')

    @staticmethod
    def from_json(json_obj, context=None):
        c = Consume("", ['i', 1], None)
        ret = ConsumeEntry(consume=c)
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

    def draw_node(self, sdfg, graph):
        if self.is_collapsed:
            return dot.draw_node(
                sdfg, graph, self, shape="hexagon", style='dashed')
        return dot.draw_node(
            sdfg, graph, self, shape="trapezium", style='dashed')

    def __str__(self):
        return str(self.consume)


@dace.serialize.serializable
class ConsumeExit(ExitNode):
    """ Node that closes a Consume scope.
        @see: Consume
    """

    def __init__(self, consume):
        super(ConsumeExit, self).__init__()
        if consume is None:
            raise ValueError("Consume for ConsumeExit can not be None.")
        self._consume = consume

    @staticmethod
    def from_json(json_obj, context=None):
        # Set map reference to entry node
        entry_node = context['sdfg_state'].node(int(json_obj['scope_entry']))

        ret = ConsumeExit(consume=entry_node.consume)
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

    def draw_node(self, sdfg, graph):
        return dot.draw_node(
            sdfg, graph, self, shape="invtrapezium", style='dashed')

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
    condition = CodeProperty(desc="Quiescence condition", allow_none=True)
    schedule = Property(
        dtype=dtypes.ScheduleType,
        desc="Consume schedule",
        choices=dtypes.ScheduleType,
        from_string=lambda x: dtypes.ScheduleType[x],
        default=dtypes.ScheduleType.Default)
    chunksize = Property(
        dtype=int,
        desc="Maximal size of elements to consume at a time",
        default=1)
    debuginfo = DebugInfoProperty()
    is_collapsed = Property(
        dtype=bool,
        desc="Show this node/scope/state as collapsed",
        default=False)

    instrument = Property(
        choices=dtypes.InstrumentationType,
        desc="Measure execution statistics with given method",
        default=dtypes.InstrumentationType.No_Instrumentation)

    def as_map(self):
        """ Compatibility function that allows to view the consume as a map,
            mainly in memlet propagation. """
        return Map(self.label, [self.pe_index],
                   sbs.Range([(0, self.num_pes - 1, 1)]), self.schedule)

    def __init__(self,
                 label,
                 pe_tuple,
                 condition,
                 schedule=dtypes.ScheduleType.Default,
                 chunksize=1,
                 debuginfo=None):
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
                    (self._label, self.pe_index, self.num_pes,
                     CodeProperty.to_string(self.condition)))
        else:
            return (
                "%s [%s=0:%s]" % (self._label, self.pe_index, self.num_pes))

    def validate(self, sdfg, state, node):
        if not data.validate_name(self.label):
            raise NameError('Invalid consume name "%s"' % self.label)

    def get_param_num(self):
        """ Returns the number of consume dimension parameters/symbols. """
        return 1


# Redirect Consume properties to ConsumeEntry and ConsumeExit
ConsumeEntry = indirect_properties(Consume,
                                   lambda obj: obj.consume)(ConsumeEntry)

# ------------------------------------------------------------------------------


@make_properties
class Reduce(Node):
    """ An SDFG node that reduces an N-dimensional array to an
        (N-k)-dimensional array, with a list of axes to reduce and
        a reduction binary function. """

    # Properties
    axes = ListProperty(element_type=int, allow_none=True)
    wcr = LambdaProperty(default='lambda a,b: a')
    identity = Property(dtype=object, allow_none=True)
    schedule = Property(
        dtype=dtypes.ScheduleType,
        desc="Reduction execution policy",
        choices=dtypes.ScheduleType,
        from_string=lambda x: dtypes.ScheduleType[x],
        default=dtypes.ScheduleType.Default)
    debuginfo = DebugInfoProperty()

    instrument = Property(
        choices=dtypes.InstrumentationType,
        desc="Measure execution statistics with given method",
        default=dtypes.InstrumentationType.No_Instrumentation)

    def __init__(self,
                 wcr,
                 axes,
                 wcr_identity=None,
                 schedule=dtypes.ScheduleType.Default,
                 debuginfo=None):
        super(Reduce, self).__init__()
        self.wcr = wcr  # type: ast._Lambda
        self.axes = axes
        self.identity = wcr_identity
        self.schedule = schedule
        self.debuginfo = debuginfo

    def draw_node(self, sdfg, state):
        return dot.draw_node(sdfg, state, self, shape="invtriangle")

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Reduce("(lambda a, b: (a + b))", None)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def __str__(self):
        # Autodetect reduction type
        redtype = detect_reduction_type(self.wcr)
        if redtype == dtypes.ReductionType.Custom:
            wcrstr = unparse(ast.parse(self.wcr).body[0].value.body)
        else:
            wcrstr = str(redtype)
            wcrstr = wcrstr[wcrstr.find('.') + 1:]  # Skip "ReductionType."

        return 'Op: {op}, Axes: {axes}'.format(
            axes=('all' if self.axes is None else str(self.axes)), op=wcrstr)

    def __label__(self, sdfg, state):
        # Autodetect reduction type
        redtype = detect_reduction_type(self.wcr)
        if redtype == dtypes.ReductionType.Custom:
            wcrstr = unparse(ast.parse(self.wcr).body[0].value.body)
        else:
            wcrstr = str(redtype)
            wcrstr = wcrstr[wcrstr.find('.') + 1:]  # Skip "ReductionType."

        return 'Op: {op}\nAxes: {axes}'.format(
            axes=('all' if self.axes is None else str(self.axes)), op=wcrstr)
