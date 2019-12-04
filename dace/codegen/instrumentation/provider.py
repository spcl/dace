class InstrumentationProvider(object):
    """ Instrumentation provider for SDFGs, states, scopes, and memlets. Emits
        code on event, as well as a function that prints the report. """

    @staticmethod
    def unified_id(node_id, state_id):
        if node_id > 0x0FFFF:
            raise ValueError("Nodeid is too large to fit in 16 bits!")
        if state_id > 0x0FFFF:
            raise ValueError("Stateid is too large to fit in 16 bits!")
        return (int(state_id) << 16) | int(node_id)

    def report(self, local_stream, global_stream):
        """ Emits an instrumentation report code.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_sdfg_begin(self, sdfg, local_stream, global_stream):
        """ Event called at the beginning of SDFG code generation.
            :param sdfg: The generated SDFG object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_sdfg_end(self, sdfg, local_stream, global_stream):
        """ Event called at the end of SDFG code generation.
            :param sdfg: The generated SDFG object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_state_begin(self, sdfg, state, local_stream, global_stream):
        """ Event called at the beginning of SDFG state code generation.
            :param sdfg: The generated SDFG object.
            :param state: The generated SDFGState object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_state_end(self, sdfg, state, local_stream, global_stream):
        """ Event called at the end of SDFG state code generation.
            :param sdfg: The generated SDFG object.
            :param state: The generated SDFGState object.
            :param local_stream: Code generator for the in-function code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_scope_entry(self, sdfg, state, node, outer_stream, inner_stream,
                       global_stream):
        """ Event called at the beginning of a scope (on generating an
            EntryNode).
            :param sdfg: The generated SDFG object.
            :param state: The generated SDFGState object.
            :param node: The EntryNode object from which code is generated.
            :param outer_stream: Code generator for the internal code before
                                 the scope is opened.
            :param inner_stream: Code generator for the internal code within
                                 the scope (at the beginning).
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_scope_exit(self, sdfg, state, node, outer_stream, inner_stream,
                      global_stream):
        """ Event called at the end of a scope (on generating an ExitNode).
            :param sdfg: The generated SDFG object.
            :param state: The generated SDFGState object.
            :param node: The ExitNode object from which code is generated.
            :param outer_stream: Code generator for the internal code after
                                 the scope is closed.
            :param inner_stream: Code generator for the internal code within
                                 the scope (at the end).
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_copy_begin(self, sdfg, state, src_node, dst_node, edge,
                      local_stream, global_stream, copy_shape, src_strides,
                      dst_strides):
        """ Event called at the beginning of generating a copy operation.
            :param sdfg: The generated SDFG object.
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

    def on_copy_end(self, sdfg, state, src_node, dst_node, edge, local_stream,
                    global_stream):
        """ Event called at the end of generating a copy operation.
            :param sdfg: The generated SDFG object.
            :param state: The generated SDFGState object.
            :param src_node: The source node of the copy.
            :param dst_node: The destination node of the copy.
            :param edge: An edge in the memlet path of the copy.
            :param local_stream: Code generator for the internal code.
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_node_begin(self, sdfg, state, node, outer_stream, inner_stream,
                      global_stream):
        """ Event called at the beginning of generating a node.
            :param sdfg: The generated SDFG object.
            :param state: The generated SDFGState object.
            :param node: The generated node.
            :param outer_stream: Code generator for the internal code before
                                 the scope is opened.
            :param inner_stream: Code generator for the internal code within
                                 the scope (at the beginning).
            :param global_stream: Code generator for global (external) code.
        """
        pass

    def on_node_end(self, sdfg, state, node, outer_stream, inner_stream,
                    global_stream):
        """ Event called at the end of generating a node.
            :param sdfg: The generated SDFG object.
            :param state: The generated SDFGState object.
            :param node: The generated node.
            :param outer_stream: Code generator for the internal code after
                                 the scope is closed.
            :param inner_stream: Code generator for the internal code within
                                 the scope (at the end).
            :param global_stream: Code generator for global (external) code.
        """
        pass
