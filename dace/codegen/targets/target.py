# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import aenum
import os
import shutil  # which
from typing import Dict, Tuple
import warnings

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.sdfg.utils import dfs_topological_sort
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.registry import extensible_enum, make_registry


@make_registry
class TargetCodeGenerator(object):
    """ Interface dictating functions that generate code for:
          * Array allocation/deallocation/initialization/copying
          * Scope (map, consume) code generation
    """
    def get_generated_codeobjects(self):
        """ Returns a list of generated `CodeObject` classes corresponding
            to files with generated code. If an empty list is returned
            (default) then this code generator does not create new files.
            @see: CodeObject
        """
        return []

    @property
    def has_initializer(self):
        """ Returns True if the target generates a `__dace_init_<TARGET>`
            function that should be called on initialization. """
        return False

    @property
    def has_finalizer(self):
        """ Returns True if the target generates a `__dace_exit_<TARGET>`
            function that should be called on finalization. """
        return False

    def generate_state(self, sdfg, state, function_stream, callsite_stream):
        """ Generates code for an SDFG state, outputting it to the given
            code streams.
            :param sdfg: The SDFG to generate code from.
            :param state: The SDFGState to generate code from.
            :param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        pass

    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream,
                       callsite_stream):
        """ Generates code for an SDFG state scope (from a scope-entry node
            to its corresponding scope-exit node), outputting it to the given
            code streams.
            :param sdfg: The SDFG to generate code from.
            :param dfg_scope: The `ScopeSubgraphView` to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def generate_node(self, sdfg, dfg, state_id, node, function_stream,
                      callsite_stream):
        """ Generates code for a single node, outputting it to the given
            code streams.
            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param node: The node to generate code from.
            :param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream,
                       callsite_stream):
        """ Generates code for allocating an array, outputting to the given
            code streams.
            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param node: The data node to generate allocation for.
            :param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        """ Generates code for deallocating an array, outputting to the given
            code streams.
            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param node: The data node to generate deallocation for.
            :param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge,
                    function_stream, callsite_stream):
        """ Generates code for copying memory, either from a data access
            node (array/stream) to another, a code node (tasklet/nested
            SDFG) to another, or a combination of the two.
            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param src_node: The source node to generate copy code for.
            :param dst_node: The destination node to generate copy code for.
            :param edge: The edge representing the copy (in the innermost
                         scope, adjacent to either the source or destination
                         node).
            :param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')


class IllegalCopy(TargetCodeGenerator):
    """ A code generator that is triggered when invalid copies are specified
        by the SDFG. Only raises an exception on failure. """
    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge,
                    function_stream, callsite_stream):
        raise TypeError('Illegal copy! (from ' + str(src_node) + ' to ' +
                        str(dst_node) + ')')


@extensible_enum
class DefinedType(aenum.AutoNumberEnum):
    """ Data types for `DefinedMemlets`.
        @see: DefinedMemlets
    """
    Pointer = ()
    Scalar = ()
    Stream = ()
    StreamArray = ()
    FPGA_ShiftRegister = ()
    ArrayInterface = ()


class DefinedMemlets:
    """ Keeps track of the type of defined memlets to ensure that they are
        referenced correctly in nested scopes and SDFGs. """
    def __init__(self):
        self._scopes = [(None, {}, True)]

    def enter_scope(self, parent, can_access_parent=True):
        self._scopes.append((parent, {}, can_access_parent))

    def exit_scope(self, parent):
        expected, _, _ = self._scopes.pop()
        if expected != parent:
            raise ValueError(
                "Exited scope {} mismatched current scope {}".format(
                    parent.name, expected.name))

    def has(self, name):
        try:
            self.get(name)
            return True
        except KeyError:
            return False

    def get(self, name: str, ancestor: int = 0) -> Tuple[DefinedType, str]:
        for _, scope, can_access_parent in reversed(self._scopes):
            if ancestor > 0:
                ancestor -= 1
                continue
            if name in scope:
                return scope[name]
            if not can_access_parent:
                break
        raise KeyError("Variable {} has not been defined".format(name))

    def add(self,
            name: str,
            dtype: DefinedType,
            ctype: str,
            ancestor: int = 0,
            allow_shadowing: bool = False):
        if not isinstance(name, str):
            raise TypeError('Variable name type cannot be %s' %
                            type(name).__name__)

        for _, scope, can_access_parent in reversed(self._scopes):
            if name in scope:
                err_str = "Shadowing variable {} from type {} to {}".format(
                    name, scope[name], dtype)
                if (allow_shadowing or dace.config.Config.get_bool(
                        "compiler", "allow_shadowing")):
                    if not allow_shadowing:
                        print("WARNING: " + err_str)
                else:
                    raise dace.codegen.codegen.CodegenError(err_str)
            if not can_access_parent:
                break
        self._scopes[-1 - ancestor][1][name] = (dtype, ctype)


#############################################################################


class TargetDispatcher(object):
    """ Dispatches sub-SDFG generation (according to scope),
        storage<->storage copies, and storage<->tasklet copies to targets. """
    def __init__(self):
        self._used_targets = set()
        self._used_environments = set()

        # type: Dict[dace.dtypes.InstrumentationType, InstrumentationProvider]
        self.instrumentation = {}

        self._array_dispatchers = {
        }  # Type: dtypes.StorageType -> TargetCodeGenerator
        self._map_dispatchers = {
        }  # Type: dtypes.ScheduleType -> TargetCodeGenerator
        self._copy_dispatchers = {}  # Type: (dtypes.StorageType src,
        #                                     dtypes.StorageType dst,
        #                                     dtypes.ScheduleType dst_schedule)
        #                                     -> List[(predicate, TargetCodeGenerator)]
        self._generic_copy_dispatchers = {}  # Type: (dtypes.StorageType src,
        #                                     dtypes.StorageType dst,
        #                                     dtypes.ScheduleType dst_schedule)
        #                                     -> TargetCodeGenerator
        self._node_dispatchers = []  # [(predicate, dispatcher)]
        self._generic_node_dispatcher = None  # Type: TargetCodeGenerator
        self._state_dispatchers = []  # [(predicate, dispatcher)]
        self._generic_state_dispatcher = None  # Type: TargetCodeGenerator

        self._defined_vars = DefinedMemlets()

    @property
    def defined_vars(self):
        """ Returns a list of defined variables.
            @rtype: DefinedMemlets
        """
        return self._defined_vars

    @property
    def used_targets(self):
        """ Returns a list of targets (code generators) that were triggered
            during generation. """
        return self._used_targets

    @property
    def used_environments(self):
        """ Returns a list of environments required to build and run the code.
        """
        return self._used_environments

    def register_state_dispatcher(self, dispatcher, predicate=None):
        """ Registers a code generator that processes a single state, calling
            `generate_state`.
            :param dispatcher: The code generator to use.
            :param predicate: A lambda function that accepts the SDFG and
                              state, and triggers the code generator when True
                              is returned. If None, registers `dispatcher`
                              as the default state dispatcher.
            @see: TargetCodeGenerator
        """

        if not hasattr(dispatcher, "generate_state"):
            raise TypeError("State dispatcher \"{}\" does not "
                            "implement \"generate_state\"".format(dispatcher))
        if predicate is None:
            self._generic_state_dispatcher = dispatcher
        else:
            self._state_dispatchers.append((predicate, dispatcher))

    def get_generic_state_dispatcher(self):
        """ Returns the default state dispatcher. """
        return self._generic_state_dispatcher

    def get_predicated_state_dispatchers(self):
        """ Returns a list of state dispatchers with predicates. """
        return list(self._state_dispatchers)

    def register_node_dispatcher(self, dispatcher, predicate=None):
        """ Registers a code generator that processes a single node, calling
            `generate_node`.
            :param dispatcher: The code generator to use.
            :param predicate: A lambda function that accepts the SDFG, state,
                              and node, and triggers the code generator when
                              True is returned. If None, registers `dispatcher`
                              as the default node dispatcher.
            @see: TargetCodeGenerator
        """
        if not hasattr(dispatcher, "generate_node"):
            raise TypeError("Node dispatcher must "
                            "implement \"generate_node\"")
        if predicate is None:
            self._generic_node_dispatcher = dispatcher
        else:
            self._node_dispatchers.append((predicate, dispatcher))

    def get_generic_node_dispatcher(self):
        """ Returns the default node dispatcher. """
        return self._generic_node_dispatcher

    def get_predicated_node_dispatchers(self):
        """ Returns a list of node dispatchers with predicates. """
        return list(self._node_dispatchers)

    def register_map_dispatcher(self, schedule_type, func):
        """ Registers a function that processes a scope, used when calling
            `dispatch_subgraph` and `dispatch_scope`.
            :param schedule_type: The scope schedule that triggers `func`.
            :param func: A TargetCodeGenerator object that contains an
                         implementation of `generate_scope`.
            @see: TargetCodeGenerator
        """
        if isinstance(schedule_type, list):
            for stype in schedule_type:
                self.register_map_dispatcher(stype, func)
            return

        if not isinstance(schedule_type, dtypes.ScheduleType): raise TypeError
        if not isinstance(func, TargetCodeGenerator): raise TypeError
        if schedule_type in self._map_dispatchers:
            raise ValueError('Schedule already mapped to ' +
                             str(self._map_dispatchers[schedule_type]))
        self._map_dispatchers[schedule_type] = func

    def register_array_dispatcher(self, storage_type, func):
        """ Registers a function that processes data allocation,
            initialization, and deinitialization. Used when calling
            `dispatch_allocate/deallocate/initialize`.
            :param storage_type: The data storage type that triggers `func`.
            :param func: A TargetCodeGenerator object that contains an
                         implementation of data memory management functions.
            @see: TargetCodeGenerator
        """
        if isinstance(storage_type, list):
            for stype in storage_type:
                self.register_array_dispatcher(stype, func)
            return

        if not isinstance(storage_type, dtypes.StorageType): raise TypeError
        if not isinstance(func, TargetCodeGenerator): raise TypeError
        self._array_dispatchers[storage_type] = func

    def register_copy_dispatcher(self,
                                 src_storage,
                                 dst_storage,
                                 dst_schedule,
                                 func,
                                 predicate=None):
        """ Registers code generation of data-to-data (or data from/to
            tasklet, if src/dst storage is StorageType.Register) copy
            functions. Can also be target-schedule specific, or
            dst_schedule=None if the function will be invoked on any schedule.
            :param src_storage: The source data storage type that triggers
                                `func`.
            :param dst_storage: The destination data storage type that
                                triggers `func`.
            :param dst_schedule: An optional destination scope schedule type
                                 that triggers `func`.
            :param func: A TargetCodeGenerator object that contains an
                         implementation of `copy_memory`.
            :param predicate: A lambda function that accepts the SDFG, state,
                              and source and destination nodes, and triggers
                              the code generator when True is returned. If
                              None, always dispatches with this dispatcher.
            @see: TargetCodeGenerator
        """

        if not isinstance(src_storage, dtypes.StorageType): raise TypeError
        if not isinstance(dst_storage, dtypes.StorageType): raise TypeError
        if (dst_schedule is not None
                and not isinstance(dst_schedule, dtypes.ScheduleType)):
            raise TypeError
        if not isinstance(func, TargetCodeGenerator): raise TypeError

        dispatcher = (src_storage, dst_storage, dst_schedule)
        if predicate is None:
            self._generic_copy_dispatchers[dispatcher] = func
            return

        if dispatcher not in self._copy_dispatchers:
            self._copy_dispatchers[dispatcher] = []

        self._copy_dispatchers[dispatcher].append((predicate, func))

    def dispatch_state(self, sdfg, state, function_stream, callsite_stream):
        """ Dispatches a code generator for an SDFG state. """

        self.defined_vars.enter_scope(state)
        # Check if the state satisfies any predicates that delegate to a
        # specific code generator
        satisfied_dispatchers = [
            dispatcher for pred, dispatcher in self._state_dispatchers
            if pred(sdfg, state) is True
        ]
        num_satisfied = len(satisfied_dispatchers)
        if num_satisfied > 1:
            raise RuntimeError(
                "Multiple predicates satisfied for {}: {}".format(
                    state,
                    ", ".join([type(x).__name__
                               for x in satisfied_dispatchers])))
        elif num_satisfied == 1:
            satisfied_dispatchers[0].generate_state(sdfg, state,
                                                    function_stream,
                                                    callsite_stream)
        else:  # num_satisfied == 0
            # Otherwise use the generic code generator (CPU)
            self._generic_state_dispatcher.generate_state(
                sdfg, state, function_stream, callsite_stream)
        self.defined_vars.exit_scope(state)

    def dispatch_subgraph(self,
                          sdfg,
                          dfg,
                          state_id,
                          function_stream,
                          callsite_stream,
                          skip_entry_node=False):
        """ Dispatches a code generator for a scope subgraph of an
            `SDFGState`. """

        start_nodes = list(v for v in dfg.nodes()
                           if len(list(dfg.predecessors(v))) == 0)

        # Mark nodes to skip in order to be able to skip
        nodes_to_skip = set()

        if skip_entry_node:
            assert len(start_nodes) == 1
            nodes_to_skip.add(start_nodes[0])

        for v in dfs_topological_sort(dfg, start_nodes):
            if v in nodes_to_skip:
                continue

            if isinstance(v, nodes.MapEntry):
                scope_subgraph = sdfg.node(state_id).scope_subgraph(v)

                self.dispatch_scope(v.map.schedule, sdfg, scope_subgraph,
                                    state_id, function_stream, callsite_stream)

                # Skip scope subgraph nodes
                nodes_to_skip.update(scope_subgraph.nodes())
            else:
                self.dispatch_node(sdfg, dfg, state_id, v, function_stream,
                                   callsite_stream)

    def dispatch_node(self, sdfg, dfg, state_id, node, function_stream,
                      callsite_stream):
        """ Dispatches a code generator for a single node. """

        # If this node depends on any environments, register this for
        # generating header code later
        if hasattr(node, "environments"):
            self._used_environments |= node.environments

        # Check if the node satisfies any predicates that delegate to a
        # specific code generator
        satisfied_dispatchers = [
            dispatcher for pred, dispatcher in self._node_dispatchers
            if pred(sdfg, node)
        ]
        num_satisfied = len(satisfied_dispatchers)
        if num_satisfied > 1:
            raise RuntimeError(
                "Multiple predicates satisfied for {}: {}".format(
                    node,
                    ", ".join([type(x).__name__
                               for x in satisfied_dispatchers])))
        elif num_satisfied == 1:
            self._used_targets.add(satisfied_dispatchers[0])
            satisfied_dispatchers[0].generate_node(sdfg, dfg, state_id, node,
                                                   function_stream,
                                                   callsite_stream)
        else:  # num_satisfied == 0
            # Otherwise use the generic code generator (CPU)
            self._used_targets.add(self._generic_node_dispatcher)
            self._generic_node_dispatcher.generate_node(sdfg, dfg, state_id,
                                                        node, function_stream,
                                                        callsite_stream)

    def dispatch_scope(self, map_schedule, sdfg, sub_dfg, state_id,
                       function_stream, callsite_stream):
        """ Dispatches a code generator function for a scope in an SDFG
            state. """

        entry_node = sub_dfg.source_nodes()[0]
        self.defined_vars.enter_scope(entry_node)
        self._used_targets.add(self._map_dispatchers[map_schedule])
        self._map_dispatchers[map_schedule].generate_scope(
            sdfg, sub_dfg, state_id, function_stream, callsite_stream)
        self.defined_vars.exit_scope(entry_node)

    def dispatch_allocate(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):
        """ Dispatches a code generator for data allocation. """

        nodedesc = node.desc(sdfg)
        storage = (nodedesc.storage if not isinstance(node, nodes.Tasklet) else
                   dtypes.StorageType.Register)
        self._used_targets.add(self._array_dispatchers[storage])

        self._array_dispatchers[storage].allocate_array(sdfg, dfg, state_id,
                                                        node, function_stream,
                                                        callsite_stream)

    def dispatch_deallocate(self, sdfg, dfg, state_id, node, function_stream,
                            callsite_stream):
        """ Dispatches a code generator for a data deallocation. """

        nodedesc = node.desc(sdfg)
        storage = (nodedesc.storage if not isinstance(node, nodes.Tasklet) else
                   dtypes.StorageType.Register)
        self._used_targets.add(self._array_dispatchers[storage])

        self._array_dispatchers[storage].deallocate_array(
            sdfg, dfg, state_id, node, function_stream, callsite_stream)

    # Dispatches copy code for a memlet
    def dispatch_copy(self, src_node, dst_node, edge, sdfg, dfg, state_id,
                      function_stream, output_stream):
        """ Dispatches a code generator for a memory copy operation. """

        if isinstance(src_node, nodes.CodeNode):
            src_storage = dtypes.StorageType.Register
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, nodes.CodeNode):
            dst_storage = dtypes.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        if (isinstance(src_node, nodes.Tasklet)
                and not isinstance(dst_node, nodes.Tasklet)):
            # Special case: Copying from a tasklet to an array, schedule of
            # the copy is in the copying tasklet
            dst_schedule_node = dfg.scope_dict()[src_node]
        else:
            dst_schedule_node = dfg.scope_dict()[dst_node]

        if dst_schedule_node is not None:
            dst_schedule = dst_schedule_node.map.schedule
        else:
            dst_schedule = None

        if (src_storage, dst_storage, dst_schedule) in self._copy_dispatchers:
            disp = (src_storage, dst_storage, dst_schedule)
        elif (src_storage, dst_storage, None) in self._copy_dispatchers:
            disp = (src_storage, dst_storage, None)
        else:
            disp = None

        if disp is not None:
            # Check if the state satisfies any predicates that delegate to a
            # specific code generator
            satisfied_dispatchers = [
                dispatcher for pred, dispatcher in self._copy_dispatchers[disp]
                if pred(sdfg, dfg, src_node, dst_node) is True
            ]
        else:
            satisfied_dispatchers = []
        num_satisfied = len(satisfied_dispatchers)
        if num_satisfied > 1:
            raise RuntimeError(
                "Multiple predicates satisfied for copy: {}".format(", ".join(
                    [type(x).__name__ for x in satisfied_dispatchers])))
        elif num_satisfied == 1:
            target = satisfied_dispatchers[0]
        else:  # num_satisfied == 0
            # Otherwise use the generic copy dispatchers
            if (src_storage, dst_storage,
                    dst_schedule) in self._generic_copy_dispatchers:
                target = self._generic_copy_dispatchers[(src_storage,
                                                         dst_storage,
                                                         dst_schedule)]
            elif (src_storage, dst_storage,
                  None) in self._generic_copy_dispatchers:
                target = self._generic_copy_dispatchers[(src_storage,
                                                         dst_storage, None)]
            else:
                raise RuntimeError(
                    'Copy dispatcher for %s->%s with schedule %s' %
                    (str(src_storage), str(dst_storage), str(dst_schedule)) +
                    ' not found')

        # Dispatch copy
        self._used_targets.add(target)
        target.copy_memory(sdfg, dfg, state_id, src_node, dst_node, edge,
                           function_stream, output_stream)


def make_absolute(path):
    if os.path.isfile(path):
        if os.path.isabs(path):
            # Path is absolute, we're happy
            return path
        else:
            # Path is relative: make it absolute
            return os.path.abspath(path)
    else:
        # This is not a path, probably just an executable name, such
        # as "g++". Try to find it on the PATH
        executable = shutil.which(path)
        if not executable:
            executable = path
            warnings.warn("Could not find executable \"{}\"".format(path))
        return executable.replace('\\', '/')
