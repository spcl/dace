# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.codegen.prettycode import CodeIOStream
from dace.dtypes import DataInstrumentationType, InstrumentationType
from dace.registry import make_registry
from typing import Dict, Type, Union

from dace.memlet import Memlet
from dace.sdfg import nodes, SDFG
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowRegion, SDFGState


@make_registry
class InstrumentationProvider(object):
    """ Instrumentation provider for SDFGs, states, scopes, and memlets. Emits
        code on event. """

    @staticmethod
    def get_provider_mapping(
    ) -> Dict[Union[InstrumentationType, DataInstrumentationType], Type['InstrumentationProvider']]:
        """
        Returns a dictionary that maps instrumentation types to provider
        class types, given the currently-registered extensions of this class.
        """
        # Special case for no instrumentation
        result = {InstrumentationType.No_Instrumentation: None, DataInstrumentationType.No_Instrumentation: None}

        # Create providers for extensions
        for provider, params in InstrumentationProvider.extensions().items():
            if params.get('type'):
                result[params['type']] = provider

        return result

    def _idstr(self, cfg: ControlFlowRegion, state: SDFGState, node: nodes.Node) -> str:
        """ Returns a unique identifier string from a node or state. """
        result = str(cfg.cfg_id)
        if state is not None:
            result += '_' + str(cfg.node_id(state))
            if node is not None:
                result += '_' + str(state.node_id(node))
        return result

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream, codegen) -> None:
        """ Event called at the beginning of SDFG code generation.

            :param sdfg: The generated SDFG object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
            :param codegen: An instance of the code generator.
        """
        pass

    def on_sdfg_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        """ Event called at the end of SDFG code generation.

            :param sdfg: The generated SDFG object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_state_begin(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, local_stream: CodeIOStream,
                       global_stream: CodeIOStream) -> None:
        """ Event called at the beginning of SDFG state code generation.

            :param sdfg: The generated SDFG object.
            :param cfg: The generated Control Flow Region object.
            :param state: The generated SDFGState object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_state_end(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, local_stream: CodeIOStream,
                     global_stream: CodeIOStream) -> None:
        """ Event called at the end of SDFG state code generation.

            :param sdfg: The generated SDFG object.
            :param cfg: The generated Control Flow Region object.
            :param state: The generated SDFGState object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_scope_entry(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.EntryNode,
                       outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        """ Event called at the beginning of a scope (on generating an
            EntryNode).

            :param sdfg: The generated SDFG object.
            :param cfg: The generated Control Flow Region object.
            :param state: The generated SDFGState object.
            :param node: The EntryNode object from which code is generated.
            :param outer_stream: Code generator for the internal code before
                                 the scope is opened.
            :param inner_stream: Code generator for the internal code within
                                 the scope (at the beginning).
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_scope_exit(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.ExitNode,
                      outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        """ Event called at the end of a scope (on generating an ExitNode).

            :param sdfg: The generated SDFG object.
            :param cfg: The generated Control Flow Region object.
            :param state: The generated SDFGState object.
            :param node: The ExitNode object from which code is generated.
            :param outer_stream: Code generator for the internal code after
                                 the scope is closed.
            :param inner_stream: Code generator for the internal code within
                                 the scope (at the end).
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_copy_begin(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, src_node: nodes.Node,
                      dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet], local_stream: CodeIOStream,
                      global_stream: CodeIOStream, copy_shape, src_strides, dst_strides) -> None:
        """ Event called at the beginning of generating a copy operation.

            :param sdfg: The generated SDFG object.
            :param cfg: The generated Control Flow Region object.
            :param state: The generated SDFGState object.
            :param src_node: The source node of the copy.
            :param dst_node: The destination node of the copy.
            :param edge: An edge in the memlet path of the copy.
            :param local_stream: Code generator for the internal code.
            :param global_stream: Code generator for global (external) code.
            :param copy_shape: Tuple representing the shape of the copy.
            :param src_strides: Element-skipping strides for each dimension of the copied source.
            :param dst_strides: Element-skipping strides for each dimension of the copied destination.
        """
        pass

    def on_copy_end(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, src_node: nodes.Node,
                    dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet], local_stream: CodeIOStream,
                    global_stream: CodeIOStream) -> None:
        """ Event called at the end of generating a copy operation.

            :param sdfg: The generated SDFG object.
            :param cfg: The generated Control Flow Region object.
            :param state: The generated SDFGState object.
            :param src_node: The source node of the copy.
            :param dst_node: The destination node of the copy.
            :param edge: An edge in the memlet path of the copy.
            :param local_stream: Code generator for the internal code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_node_begin(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.Node,
                      outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        """ Event called at the beginning of generating a node.

            :param sdfg: The generated SDFG object.
            :param cfg: The generated Control Flow Region object.
            :param state: The generated SDFGState object.
            :param node: The generated node.
            :param outer_stream: Code generator for the internal code before
                                 the scope is opened.
            :param inner_stream: Code generator for the internal code within
                                 the scope (at the beginning).
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_node_end(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.Node,
                    outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        """ Event called at the end of generating a node.

            :param sdfg: The generated SDFG object.
            :param cfg: The generated Control Flow Region object.
            :param state: The generated SDFGState object.
            :param node: The generated node.
            :param outer_stream: Code generator for the internal code after
                                 the scope is closed.
            :param inner_stream: Code generator for the internal code within
                                 the scope (at the end).
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_sdfg_init_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        """ Event called at the beginning of SDFG initialization code generation.

            :param sdfg: The generated SDFG object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_sdfg_init_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        """ Event called at the end of SDFG initialization code generation.

            :param sdfg: The generated SDFG object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_sdfg_exit_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        """ Event called at the beginning of SDFG exit code generation.

            :param sdfg: The generated SDFG object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_sdfg_exit_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        """ Event called at the end of SDFG exit code generation.

            :param sdfg: The generated SDFG object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_allocation_begin(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG],
                            stream: CodeIOStream) -> None:
        """ Event called at the beginning of an allocation code generation.

            :param sdfg: The generated SDFG object.
            :param stream: Code generator.
        """
        pass

    def on_allocation_end(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG],
                          lstream: CodeIOStream) -> None:
        """ Event called at the end of an allocation code generation.

            :param sdfg: The generated SDFG object.
            :param local_stream: Code generator.
        """
        pass

    def on_deallocation_begin(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG],
                              stream: CodeIOStream) -> None:
        """ Event called at the beginning of a deallocation code generation.

            :param sdfg: The generated SDFG object.
            :param local_stream: Code generator.
        """
        pass

    def on_deallocation_end(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG],
                            lstream: CodeIOStream) -> None:
        """ Event called at the end of a deallocation code generation.

            :param sdfg: The generated SDFG object.
            :param local_stream: Code generator.
        """
        pass
