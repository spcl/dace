import os
import shutil  # which
import dace
from dace import types
from dace.graph import nodes, nxutil


class TargetCodeGenerator(object):
    """ Interface dictating functions that generate code for:
          * Array allocation/deallocation/initialization/copying
          * Scope (map, consume) code generation
    """

    def get_generated_codeobjects(self):
        """ Returns a list of generated `CodeObject` classes corresponding
            to files with generated code.
            @see: CodeObject
        """
        raise NotImplementedError('Abstract class')

    @property
    def has_initializer(self):
        """ Returns True if the target generates a `__dace_init_<TARGET>` 
            function that should be called on initialization. """
        raise NotImplementedError('Abstract class')

    @property
    def has_finalizer(self):
        """ Returns True if the target generates a `__dace_exit_<TARGET>` 
            function that should be called on finalization. """
        raise NotImplementedError('Abstract class')

    def generate_state(self, sdfg, state, function_stream, callsite_stream):
        """ Generates code for an SDFG state, outputting it to the given 
            code streams.
            @param sdfg: The SDFG to generate code from.
            @param state: The SDFGState to generate code from.
            @param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            @param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream,
                       callsite_stream):
        """ Generates code for an SDFG state scope (from a scope-entry node
            to its corresponding scope-exit node), outputting it to the given 
            code streams.
            @param sdfg: The SDFG to generate code from.
            @param dfg_scope: The `ScopeSubgraphView` to generate code from.
            @param state_id: The node ID of the state in the given SDFG.
            @param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            @param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def generate_node(self, sdfg, dfg, state_id, node, function_stream,
                      callsite_stream):
        """ Generates code for a single node, outputting it to the given 
            code streams.
            @param sdfg: The SDFG to generate code from.
            @param dfg: The SDFG state to generate code from.
            @param state_id: The node ID of the state in the given SDFG.
            @param node: The node to generate code from.
            @param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            @param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream,
                       callsite_stream):
        """ Generates code for allocating an array, outputting to the given 
            code streams.
            @param sdfg: The SDFG to generate code from.
            @param dfg: The SDFG state to generate code from.
            @param state_id: The node ID of the state in the given SDFG.
            @param node: The data node to generate allocation for.
            @param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            @param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def initialize_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        """ Generates code for initializing an array, outputting to the given 
            code streams.
            @param sdfg: The SDFG to generate code from.
            @param dfg: The SDFG state to generate code from.
            @param state_id: The node ID of the state in the given SDFG.
            @param node: The data node to generate initialization for.
            @param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            @param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        """ Generates code for deallocating an array, outputting to the given 
            code streams.
            @param sdfg: The SDFG to generate code from.
            @param dfg: The SDFG state to generate code from.
            @param state_id: The node ID of the state in the given SDFG.
            @param node: The data node to generate deallocation for.
            @param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            @param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge,
                    function_stream, callsite_stream):
        """ Generates code for copying memory, either from a data access 
            node (array/stream) to another, a code node (tasklet/nested 
            SDFG) to another, or a combination of the two.
            @param sdfg: The SDFG to generate code from.
            @param dfg: The SDFG state to generate code from.
            @param state_id: The node ID of the state in the given SDFG.
            @param src_node: The source node to generate copy code for.
            @param dst_node: The destination node to generate copy code for.
            @param edge: The edge representing the copy (in the innermost
                         scope, adjacent to either the source or destination
                         node).
            @param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            @param callsite_stream: A `CodeIOStream` object that points
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


class DefinedType(dace.types.AutoNumber):
    """ Data types for `DefinedMemlets`.
        @see: DefinedMemlets
    """
    Pointer = ()
    ArrayView = ()
    Scalar = ()
    ScalarView = ()
    Stream = ()
    StreamArray = ()


class DefinedMemlets:
    """ Keeps track of the type of defined memlets to ensure that they are
        referenced correctly in nested scopes and SDFGs. """

    def __init__(self):
        self._scopes = [(None, {})]

    def enter_scope(self, parent):
        self._scopes.append((parent, {}))

    def exit_scope(self, parent):
        expected, _ = self._scopes.pop()
        if expected != parent:
            raise ValueError(
                "Exited scope {} mismatched current scope {}".format(
                    parent.name, expected.name))

    def get(self, name):
        for _, scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        raise KeyError("Variable {} has not been defined".format(name))

    def add(self, name, connector_type):
        if not isinstance(name, str):
            raise TypeError(
                'Variable name type cannot be %s' % type(name).__name__)

        for _, scope in reversed(self._scopes):
            if name in scope:
                err_str = "Shadowing variable {} from type {} to {}".format(
                    name, scope[name], connector_type)
                if dace.config.Config.get_bool("compiler", "allow_shadowing"):
                    print("WARNING: " + err_str)
                else:
                    raise dace.codegen.codegen.CodegenError(err_str)
        self._scopes[-1][1][name] = connector_type


#############################################################################


class TargetDispatcher(object):
    """ Dispatches sub-SDFG generation (according to scope),
        storage<->storage copies, and storage<->tasklet copies to targets. """

    def __init__(self):
        self._used_targets = set()

        self._array_dispatchers = {
        }  # Type: types.StorageType -> TargetCodeGenerator
        self._map_dispatchers = {
        }  # Type: types.ScheduleType -> TargetCodeGenerator
        self._copy_dispatchers = {}  # Type: (types.StorageType src,
        #                                     types.StorageType dst,
        #                                     types.ScheduleType dst_schedule)
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

    def register_state_dispatcher(self, dispatcher, predicate=None):
        """ Registers a code generator that processes a single state, calling
            `generate_state`.
            @param dispatcher: The code generator to use.
            @param predicate: A lambda function that accepts the SDFG and 
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
            @param dispatcher: The code generator to use.
            @param predicate: A lambda function that accepts the SDFG, state,
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
            @param schedule_type: The scope schedule that triggers `func`.
            @param func: A TargetCodeGenerator object that contains an 
                         implementation of `generate_scope`.
            @see: TargetCodeGenerator
        """
        if isinstance(schedule_type, list):
            for stype in schedule_type:
                self.register_map_dispatcher(stype, func)
            return

        if not isinstance(schedule_type, types.ScheduleType): raise TypeError
        if not isinstance(func, TargetCodeGenerator): raise TypeError
        if schedule_type in self._map_dispatchers:
            raise ValueError('Schedule already mapped to ' +
                             str(self._map_dispatchers[schedule_type]))
        self._map_dispatchers[schedule_type] = func

    def register_array_dispatcher(self, storage_type, func):
        """ Registers a function that processes data allocation,   
            initialization, and deinitialization. Used when calling
            `dispatch_allocate/deallocate/initialize`.
            @param storage_type: The data storage type that triggers `func`.
            @param func: A TargetCodeGenerator object that contains an 
                         implementation of data memory management functions.
            @see: TargetCodeGenerator
        """
        if isinstance(storage_type, list):
            for stype in storage_type:
                self.register_array_dispatcher(stype, func)
            return

        if not isinstance(storage_type, types.StorageType): raise TypeError
        if not isinstance(func, TargetCodeGenerator): raise TypeError
        self._array_dispatchers[storage_type] = func

    def register_copy_dispatcher(self, src_storage, dst_storage, dst_schedule,
                                 func):
        """ Registers code generation of data-to-data (or data from/to 
            tasklet, if src/dst storage is StorageType.Register) copy 
            functions. Can also be target-schedule specific, or 
            dst_schedule=None if the function will be invoked on any schedule.
            @param src_storage: The source data storage type that triggers 
                                `func`.
            @param dst_storage: The destination data storage type that 
                                triggers `func`.
            @param dst_schedule: An optional destination scope schedule type 
                                 that triggers `func`.
            @param func: A TargetCodeGenerator object that contains an 
                         implementation of `copy_memory`.
            @see: TargetCodeGenerator            
        """

        if not isinstance(src_storage, types.StorageType): raise TypeError
        if not isinstance(dst_storage, types.StorageType): raise TypeError
        if (dst_schedule is not None
                and not isinstance(dst_schedule, types.ScheduleType)):
            raise TypeError
        if not isinstance(func, TargetCodeGenerator): raise TypeError

        self._copy_dispatchers[(src_storage, dst_storage, dst_schedule)] = \
            func

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
                    state, ", ".join(
                        [type(x).__name__ for x in satisfied_dispatchers])))
        elif num_satisfied == 1:
            satisfied_dispatchers[0].generate_state(
                sdfg, state, function_stream, callsite_stream)
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

        start_nodes = list(
            v for v in dfg.nodes() if len(list(dfg.predecessors(v))) == 0)

        # Mark nodes to skip in order to be able to skip
        nodes_to_skip = set()

        if skip_entry_node:
            assert len(start_nodes) == 1
            nodes_to_skip.add(start_nodes[0])

        for v in nxutil.dfs_topological_sort(dfg, start_nodes):
            if v in nodes_to_skip:
                continue

            if isinstance(v, nodes.MapEntry):
                scope_subgraph = sdfg.find_state(state_id).scope_subgraph(v)

                # Propagate parallelism
                if dfg.is_parallel():
                    scope_subgraph.set_parallel_parent(dfg.get_parallel_parent)

                assert not dfg.is_parallel() or scope_subgraph.is_parallel()
                self.dispatch_scope(v.map.schedule, sdfg, scope_subgraph,
                                    state_id, function_stream, callsite_stream)

                # Skip scope subgraph nodes
                #print(scope_subgraph.nodes())
                nodes_to_skip.update(scope_subgraph.nodes())
            else:
                self.dispatch_node(sdfg, dfg, state_id, v, function_stream,
                                   callsite_stream)

    def dispatch_node(self, sdfg, dfg, state_id, node, function_stream,
                      callsite_stream):
        """ Dispatches a code generator for a single node. """

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
                    node, ", ".join(
                        [type(x).__name__ for x in satisfied_dispatchers])))
        elif num_satisfied == 1:
            self._used_targets.add(satisfied_dispatchers[0])
            satisfied_dispatchers[0].generate_node(
                sdfg, dfg, state_id, node, function_stream, callsite_stream)
        else:  # num_satisfied == 0
            # Otherwise use the generic code generator (CPU)
            self._used_targets.add(self._generic_node_dispatcher)
            self._generic_node_dispatcher.generate_node(
                sdfg, dfg, state_id, node, function_stream, callsite_stream)

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
                   types.StorageType.Register)
        self._used_targets.add(self._array_dispatchers[storage])

        self._array_dispatchers[storage].allocate_array(
            sdfg, dfg, state_id, node, function_stream, callsite_stream)

    def dispatch_initialize(self, sdfg, dfg, state_id, node, function_stream,
                            callsite_stream):
        """ Dispatches a code generator for a data initialization. """

        nodedesc = node.desc(sdfg)
        storage = (nodedesc.storage if not isinstance(node, nodes.Tasklet) else
                   types.StorageType.Register)
        self._used_targets.add(self._array_dispatchers[storage])
        self._array_dispatchers[storage].initialize_array(
            sdfg, dfg, state_id, node, function_stream, callsite_stream)

    def dispatch_deallocate(self, sdfg, dfg, state_id, node, function_stream,
                            callsite_stream):
        """ Dispatches a code generator for a data deallocation. """

        nodedesc = node.desc(sdfg)
        storage = (nodedesc.storage if not isinstance(node, nodes.Tasklet) else
                   types.StorageType.Register)
        self._used_targets.add(self._array_dispatchers[storage])

        self._array_dispatchers[storage].deallocate_array(
            sdfg, dfg, state_id, node, function_stream, callsite_stream)

    # Dispatches copy code for a memlet
    def dispatch_copy(self, src_node, dst_node, edge, sdfg, dfg, state_id,
                      function_stream, output_stream):
        """ Dispatches a code generator for a memory copy operation. """

        if isinstance(src_node, nodes.CodeNode):
            src_storage = types.StorageType.Register
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, nodes.CodeNode):
            dst_storage = types.StorageType.Register
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
            target = self._copy_dispatchers[(src_storage, dst_storage,
                                             dst_schedule)]
            self._used_targets.add(target)
            target.copy_memory(sdfg, dfg, state_id, src_node, dst_node, edge,
                               function_stream, output_stream)
        elif (src_storage, dst_storage, None) in self._copy_dispatchers:
            target = self._copy_dispatchers[(src_storage, dst_storage, None)]
            self._used_targets.add(target)
            target.copy_memory(sdfg, dfg, state_id, src_node, dst_node, edge,
                               function_stream, output_stream)
        else:
            raise RuntimeError('Copy dispatcher for %s->%s with schedule %s' %
                               (str(src_storage), str(dst_storage),
                                str(dst_schedule)) + ' not found')


def make_absolute(path):
    if os.path.isfile(path):
        if os.path.isabs(path):
            # Path is abolute, we're happy
            return path
        else:
            # Path is relative: make it absolute
            return os.path.abspath(path)
    else:
        # This is not a path, probably just an executable name, such
        # as "g++". Try to find it on the PATH
        executable = shutil.which(path)
        if not executable:
            raise ValueError("Could not find executable \"{}\"".format(path))
        return executable
