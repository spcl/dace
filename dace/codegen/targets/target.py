# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
import shutil  # which
from typing import Dict, List, Tuple
import warnings

from dace import dtypes, memlet as mm, data as dt
from dace.sdfg import nodes, SDFG, SDFGState, ScopeSubgraphView, graph as gr
from dace.sdfg.utils import dfs_topological_sort
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.registry import extensible_enum, make_registry
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.codeobject import CodeObject


@make_registry
class TargetCodeGenerator(object):
    """ Interface dictating functions that generate code for:
          * Array allocation/deallocation/initialization/copying
          * Scope (map, consume) code generation
    """
    def get_generated_codeobjects(self) -> List[CodeObject]:
        """ Returns a list of generated `CodeObject` classes corresponding
            to files with generated code. If an empty list is returned
            (default) then this code generator does not create new files.
            @see: CodeObject
        """
        return []

    def on_target_used(self) -> None:
        """
        Called before generating frame code (headers / footers) on this target
        if it was dispatched for any reason. Can be used to set up state struct
        fields.
        """
        pass

    @property
    def has_initializer(self) -> bool:
        """ Returns True if the target generates a `__dace_init_<TARGET>`
            function that should be called on initialization. """
        return False

    @property
    def has_finalizer(self) -> bool:
        """ Returns True if the target generates a `__dace_exit_<TARGET>`
            function that should be called on finalization. """
        return False

    def generate_state(self, sdfg: SDFG, state: SDFGState,
                       function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream) -> None:
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

    def generate_scope(self, sdfg: SDFG, dfg_scope: ScopeSubgraphView,
                       state_id: int, function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream) -> None:
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

    def generate_node(self, sdfg: SDFG, dfg: SDFGState, state_id: int,
                      node: nodes.Node, function_stream: CodeIOStream,
                      callsite_stream: CodeIOStream) -> None:
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

    def declare_array(self, sdfg: SDFG, dfg: SDFGState, state_id: int,
                      node: nodes.Node, nodedesc: dt.Data,
                      global_stream: CodeIOStream,
                      declaration_stream: CodeIOStream) -> None:
        """ Generates code for declaring an array without allocating it,
            outputting to the given code streams.
            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param node: The data node to generate allocation for.
            :param nodedesc: The data descriptor to allocate.
            :param global_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param declaration_stream: A `CodeIOStream` object that points
                                       to the point of array declaration.
        """
        raise NotImplementedError('Abstract class')

    def allocate_array(self, sdfg: SDFG, dfg: SDFGState, state_id: int,
                       node: nodes.Node, nodedesc: dt.Data,
                       global_stream: CodeIOStream,
                       declaration_stream: CodeIOStream,
                       allocation_stream: CodeIOStream) -> None:
        """ Generates code for allocating an array, outputting to the given
            code streams.
            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param node: The data node to generate allocation for.
            :param nodedesc: The data descriptor to allocate.
            :param global_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param declaration_stream: A `CodeIOStream` object that points
                                       to the point of array declaration.
            :param allocation_stream: A `CodeIOStream` object that points
                                       to the call-site of array allocation.
        """
        raise NotImplementedError('Abstract class')

    def deallocate_array(self, sdfg: SDFG, dfg: SDFGState, state_id: int,
                         node: nodes.Node, nodedesc: dt.Data,
                         function_stream: CodeIOStream,
                         callsite_stream: CodeIOStream) -> None:
        """ Generates code for deallocating an array, outputting to the given
            code streams.
            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param node: The data node to generate deallocation for.
            :param nodedesc: The data descriptor to deallocate.
            :param function_stream: A `CodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `CodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def copy_memory(self, sdfg: SDFG, dfg: SDFGState, state_id: int,
                    src_node: nodes.Node, dst_node: nodes.Node,
                    edge: gr.MultiConnectorEdge[mm.Memlet],
                    function_stream: CodeIOStream,
                    callsite_stream: CodeIOStream) -> None:
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


def make_absolute(path: str) -> str:
    """ 
    Finds an executable and returns an absolute path out of it. Used when
    finding compiler executables.
    :param path: Executable name, relative path, or absolute path.
    :return: Absolute path pointing to the same file as ``path``.
    """
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
