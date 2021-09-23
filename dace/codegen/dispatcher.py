# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains the DaCe code generator target dispatcher, which is responsible for
flexible code generation with multiple backends by dispatching certain
functionality to registered code generators based on user-defined predicates.
"""
from dace.codegen.prettycode import CodeIOStream
import aenum
from dace import config, data as dt, dtypes, nodes, registry
from dace.codegen import exceptions as cgx, prettycode
from dace.codegen.targets import target
from dace.sdfg import utils as sdutil, SDFG, SDFGState, ScopeSubgraphView
from typing import Dict, Tuple


@registry.extensible_enum
class DefinedType(aenum.AutoNumberEnum):
    """ Data types for `DefinedMemlets`.
        :see: DefinedMemlets
    """
    Pointer = ()
    Scalar = ()
    Stream = ()
    StreamArray = ()
    FPGA_ShiftRegister = ()
    ArrayInterface = ()


class DefinedMemlets:
    """ Keeps track of the type of defined memlets to ensure that they are
        referenced correctly in nested scopes and SDFGs.
        The ones defined in the first (top) scope, refer to global variables.
    """
    def __init__(self):
        self._scopes = [(None, {}, True), (None, {}, True)]

    def enter_scope(self, parent, can_access_parent=True):
        self._scopes.append((parent, {}, can_access_parent))

    def exit_scope(self, parent):
        expected, _, _ = self._scopes.pop()
        if expected != parent:
            raise ValueError(
                "Exited scope {} mismatched current scope {}".format(
                    parent.name, expected.name))

    def has(self, name, ancestor: int = 0):
        try:
            self.get(name, ancestor)
            return True
        except KeyError:
            return False

    def get(self,
            name: str,
            ancestor: int = 0,
            is_global: bool = False) -> Tuple[DefinedType, str]:
        last_visited_scope = None
        for parent, scope, can_access_parent in reversed(self._scopes):
            last_parent = parent
            last_visited_scope = scope
            if ancestor > 0:
                ancestor -= 1
                continue
            if name in scope:
                return scope[name]
            if not can_access_parent:
                break

        # Search among globally defined variables (top scope), if not already visited
        # TODO: The following change makes it so we look in all top scopes, not
        # just the very top-level one. However, ft we are in a nested SDFG,
        # then we must limit the search to that SDFG only. There is one
        # exception, when the data has Global or Persistent allocation lifetime.
        # Then, we expect it to be only in the very top-level scope.
        # if last_visited_scope != self._scopes[0]:
        #     if name in self._scopes[0][1]:
        #         return self._scopes[0][1][name]
        if is_global:
            last_parent = None
        if last_parent:
            if isinstance(last_parent, SDFGState):
                last_parent = last_parent.parent
        for parent, scope, _ in self._scopes:
            if not last_parent or parent == last_parent:
                if name in scope:
                    return scope[name]

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
                if (allow_shadowing or config.Config.get_bool(
                        "compiler", "allow_shadowing")):
                    if not allow_shadowing:
                        print("WARNING: " + err_str)
                else:
                    raise cgx.CodegenError(err_str)
            if not can_access_parent:
                break
        self._scopes[-1 - ancestor][1][name] = (dtype, ctype)

    def add_global(self, name: str, dtype: DefinedType, ctype: str):
        ''' Adds a global variable (top scope) '''
        if not isinstance(name, str):
            raise TypeError('Variable name type cannot be %s' %
                            type(name).__name__)

        self._scopes[0][1][name] = (dtype, ctype)
    
    def remove(self,
               name: str,
               ancestor: int = 0,
               is_global: bool = False) -> Tuple[DefinedType, str]:
        last_visited_scope = None
        for parent, scope, can_access_parent in reversed(self._scopes):
            last_parent = parent
            last_visited_scope = scope
            if ancestor > 0:
                ancestor -= 1
                continue
            if name in scope:
                del scope[name]
                return
            if not can_access_parent:
                break

        if is_global:
            last_parent = None
        if last_parent:
            if isinstance(last_parent, SDFGState):
                last_parent = last_parent.parent
        for parent, scope, _ in self._scopes:
            if not last_parent or parent == last_parent:
                if name in scope:
                    del scope[name]
                    return

        raise KeyError("Variable {} has not been defined".format(name))


#############################################################################


class TargetDispatcher(object):
    """ Dispatches sub-SDFG generation (according to scope),
        storage<->storage copies, and storage<->tasklet copies to targets. """
    def __init__(self, framecode):
        # Avoid import loop
        from dace.codegen.targets import framecode as fc

        self.frame: fc.DaCeCodeGenerator = framecode
        self._used_targets = set()
        self._used_environments = set()

        # type: Dict[dace.dtypes.InstrumentationType, InstrumentationProvider]
        self.instrumentation = {}

        self._array_dispatchers: Dict[
            dtypes.StorageType, target.TargetCodeGenerator] = {
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

        self._declared_arrays = DefinedMemlets()
        self._defined_vars = DefinedMemlets()

    @property
    def declared_arrays(self) -> DefinedMemlets:
        """ Returns a list of declared variables.
        
            This is used for variables that must have their declaration and
            allocation separate. It includes all such variables that have been
            declared by the dispatcher.
        """
        return self._declared_arrays

    @property
    def defined_vars(self) -> DefinedMemlets:
        """ Returns a list of defined variables.
        
            This includes all variables defined by the dispatcher.
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
        if not isinstance(func, target.TargetCodeGenerator): raise TypeError
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
        if not isinstance(func, target.TargetCodeGenerator): raise TypeError
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
        if not isinstance(func, target.TargetCodeGenerator): raise TypeError

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
                          skip_entry_node=False,
                          skip_exit_node=False):
        """ Dispatches a code generator for a scope subgraph of an
            `SDFGState`. """

        start_nodes = list(v for v in dfg.nodes()
                           if len(list(dfg.predecessors(v))) == 0)

        # Mark nodes to skip in order to be able to skip
        nodes_to_skip = set()

        if skip_entry_node:
            assert len(start_nodes) == 1
            nodes_to_skip.add(start_nodes[0])

        if skip_exit_node:
            exit_node = dfg.exit_node(start_nodes[0])
            nodes_to_skip.add(exit_node)

        for v in sdutil.dfs_topological_sort(dfg, start_nodes):
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
        state = sdfg.node(state_id)
        satisfied_dispatchers = [
            dispatcher for pred, dispatcher in self._node_dispatchers
            if pred(sdfg, state, node)
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

    def dispatch_allocate(self,
                          sdfg: SDFG,
                          dfg: ScopeSubgraphView,
                          state_id: int,
                          node: nodes.AccessNode,
                          datadesc: dt.Data,
                          function_stream: prettycode.CodeIOStream,
                          callsite_stream: prettycode.CodeIOStream,
                          declare: bool = True,
                          allocate: bool = True):
        """ Dispatches a code generator for data allocation. """
        self._used_targets.add(self._array_dispatchers[datadesc.storage])

        if datadesc.lifetime is dtypes.AllocationLifetime.Persistent:
            declaration_stream = CodeIOStream()
            callsite_stream = self.frame._initcode
        else:
            declaration_stream = callsite_stream

        if declare and not allocate:
            self._array_dispatchers[datadesc.storage].declare_array(
                sdfg, dfg, state_id, node, datadesc, function_stream,
                declaration_stream)
        elif allocate:
            self._array_dispatchers[datadesc.storage].allocate_array(
                sdfg, dfg, state_id, node, datadesc, function_stream,
                declaration_stream, callsite_stream)

    def dispatch_deallocate(self, sdfg: SDFG, dfg: ScopeSubgraphView,
                            state_id: int, node: nodes.AccessNode,
                            datadesc: dt.Data,
                            function_stream: prettycode.CodeIOStream,
                            callsite_stream: prettycode.CodeIOStream):
        """ Dispatches a code generator for a data deallocation. """
        self._used_targets.add(self._array_dispatchers[datadesc.storage])

        if datadesc.lifetime is dtypes.AllocationLifetime.Persistent:
            callsite_stream = self.frame._exitcode

        self._array_dispatchers[datadesc.storage].deallocate_array(
            sdfg, dfg, state_id, node, datadesc, function_stream,
            callsite_stream)

    # Dispatches copy code for a memlet
    def _get_copy_dispatcher(self, src_node, dst_node, edge, sdfg, dfg,
                             state_id, function_stream, output_stream):
        """
        (Internal) Returns a code generator that should be dispatched for a
        memory copy operation.
        """
        src_is_data, dst_is_data = False, False
        state_dfg = sdfg.node(state_id)

        if isinstance(src_node, nodes.CodeNode):
            src_storage = dtypes.StorageType.Register
        else:
            src_storage = src_node.desc(sdfg).storage
            src_is_data = True

        if isinstance(dst_node, nodes.CodeNode):
            dst_storage = dtypes.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage
            dst_is_data = True

        # Skip copies to/from views where edge matches
        if src_is_data and isinstance(src_node.desc(sdfg), dt.View):
            e = sdutil.get_view_edge(state_dfg, src_node)
            if e is edge:
                return None
        if dst_is_data and isinstance(dst_node.desc(sdfg), dt.View):
            e = sdutil.get_view_edge(state_dfg, dst_node)
            if e is edge:
                return None

        if (isinstance(src_node, nodes.Tasklet)
                and not isinstance(dst_node, nodes.Tasklet)):
            # Special case: Copying from a tasklet to an array, schedule of
            # the copy is in the copying tasklet
            dst_schedule_node = state_dfg.entry_node(src_node)
        else:
            dst_schedule_node = state_dfg.entry_node(dst_node)

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

        return target

    def dispatch_copy(self, src_node, dst_node, edge, sdfg, dfg, state_id,
                      function_stream, output_stream):
        """ Dispatches a code generator for a memory copy operation. """
        target = self._get_copy_dispatcher(src_node, dst_node, edge, sdfg, dfg,
                                           state_id, function_stream,
                                           output_stream)
        if target is None:
            return

        # Dispatch copy
        self._used_targets.add(target)
        target.copy_memory(sdfg, dfg, state_id, src_node, dst_node, edge,
                           function_stream, output_stream)

    # Dispatches definition code for a memlet that is outgoing from a tasklet
    def dispatch_output_definition(self, src_node, dst_node, edge, sdfg, dfg,
                                   state_id, function_stream, output_stream):
        """
        Dispatches a code generator for an output memlet definition in a
        tasklet.
        """
        target = self._get_copy_dispatcher(src_node, dst_node, edge, sdfg, dfg,
                                           state_id, function_stream,
                                           output_stream)
        # Dispatch
        self._used_targets.add(target)
        target.define_out_memlet(sdfg, dfg, state_id, src_node, dst_node, edge,
                                 function_stream, output_stream)
