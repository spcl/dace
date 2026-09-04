# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Inline multi-state SDFGs. """

from copy import deepcopy as dc
import itertools
from typing import Any, Dict, List, Set

from dace import Memlet, symbolic, subsets
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg import InterstateEdge, SDFG, SDFGState
from dace.sdfg import utils as sdutil
from dace.sdfg.replace import replace_datadesc_names, replace_properties_dict
from dace.sdfg.tasklet_utils import tasklet_replace_code, token_replace_dict
from dace.transformation import transformation, helpers
from dace.properties import make_properties, CodeBlock
from dace import data
from dace.sdfg.state import LoopRegion, ReturnBlock


def _same_layout(outer_desc: data.Data, inner_desc: data.Data) -> bool:
    """Whether two descriptors carry the same shape and the same strides.

    ``Scalar`` reports strides as a list and ``Array`` as a tuple, and ``same_value`` counts the
    sequence type -- so a ``Scalar`` facing a length-1 ``Array``, the ordinary nested-SDFG boundary,
    would refuse the inline over a container type. Compare by value.
    """
    return (symbolic.same_value(tuple(outer_desc.shape), tuple(inner_desc.shape))
            and symbolic.same_value(tuple(outer_desc.strides), tuple(inner_desc.strides)))


def _disambiguate_code_connectors(nsdfg: SDFG, reserved_names: Set[str]) -> None:
    """Rename tasklet connectors that clash with outer-scope names.

    After inlining, a tasklet connector whose name coincides with an outer
    array, symbol, constant, or assignment target would fail validation (and
    could confuse code generation). Rename such connectors to fresh names and
    update the connecting edges and the tasklet code.

    Library nodes are exempt: their connector names are part of the node's
    interface (ONNX schema parameters, BLAS operand names), and expansions look
    them up by name. Renaming one silently unbinds the node from its schema --
    the name is not recoverable from connector order either, e.g. an ONNX Gemm
    carries ``B, A, C`` against schema order ``A, B, C``. A library-node
    connector is not emitted as an identifier at this point anyway: expansion
    turns it into a nested-SDFG boundary, which has its own scope.
    """
    for nstate in nsdfg.states():
        for node in list(nstate.nodes()):
            if not isinstance(node, nodes.Tasklet):
                continue
            used = set(node.in_connectors.keys()) | set(node.out_connectors.keys())
            renames: Dict[str, str] = {}
            for conn in used:
                if conn in reserved_names:
                    renames[conn] = data.find_new_name(conn, reserved_names | used)
            if not renames:
                continue
            node.in_connectors = {renames.get(k, k): v for k, v in node.in_connectors.items()}
            node.out_connectors = {renames.get(k, k): v for k, v in node.out_connectors.items()}
            for edge in list(nstate.in_edges(node)):
                if edge.dst_conn in renames:
                    helpers.redirect_edge(nstate, edge, new_dst_conn=renames[edge.dst_conn])
            for edge in list(nstate.out_edges(node)):
                if edge.src_conn in renames:
                    helpers.redirect_edge(nstate, edge, new_src_conn=renames[edge.src_conn])
            tasklet_replace_code(node, renames)
            # tasklet_replace_code only rewrites symbols on the right-hand side;
            # also rename connector names that appear as assignment targets.
            node.code = CodeBlock(token_replace_dict(node.code.as_string, renames), language=node.code.language)


@make_properties
@transformation.explicit_cf_compatible
class InlineMultistateSDFG(transformation.SingleStateTransformation):
    """
    Inlines a multi-state nested SDFG into a top-level SDFG. This only happens
    if the state has the nested SDFG node isolated (i.e., only containing it
    and input/output access nodes), and thus the state machines can be combined.
    """

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def annotates_memlets():
        return True

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.nested_sdfg)]

    @staticmethod
    def _check_strides(inner_strides: List[symbolic.SymbolicType], outer_strides: List[symbolic.SymbolicType],
                       memlet: Memlet, nested_sdfg: nodes.NestedSDFG) -> bool:
        """
        Returns True if the strides of the inner array can be matched
        to the strides of the outer array upon inlining. Takes into
        consideration memlet (un)squeeze and nested SDFG symbol mapping.

        :param inner_strides: The strides of the array inside the nested SDFG.
        :param outer_strides: The strides of the array in the external SDFG.
        :param nested_sdfg: Nested SDFG node with symbol mapping.
        :return: True if all strides match, False otherwise.
        """
        # Replace all inner symbols based on symbol mapping
        istrides = list(inner_strides)

        def replfunc(mapping):
            for i, s in enumerate(istrides):
                if symbolic.issymbolic(s):
                    istrides[i] = s.subs(mapping)

        symbolic.safe_replace(nested_sdfg.symbol_mapping, replfunc)

        if istrides == list(outer_strides):
            return True

        # Take unsqueezing into account
        dims_to_ignore = [i for i, s in enumerate(memlet.subset.size()) if s == 1]
        ostrides = [os for i, os in enumerate(outer_strides) if i not in dims_to_ignore]

        if len(ostrides) == 0:
            ostrides = [1]

        if len(ostrides) != len(istrides):
            return False

        return all(istr == ostr for istr, ostr in zip(istrides, ostrides))

    def can_be_applied(self, state: SDFGState, expr_index, sdfg: SDFG, permissive=False):
        nested_sdfg = self.nested_sdfg
        if nested_sdfg.no_inline:
            return False

        # Not nested in scope
        if state.entry_node(nested_sdfg) is not None:
            return False

        # Must be
        # - connected to access nodes only
        # - read full subsets
        # - not use views inside
        for edge in state.in_edges(nested_sdfg):
            if edge.data.data is None:
                return False

            if not isinstance(edge.src, nodes.AccessNode):
                return False

            if edge.data.subset != subsets.Range.from_array(sdfg.arrays[edge.data.data]):
                return False

            outer_desc = sdfg.arrays[edge.data.data]
            if isinstance(outer_desc, data.View):
                return False

            # We can not compare shapes directly, we have to consider the symbol map
            #  for that. Clone the descriptor because the operation is inplace.
            inner_desc = nested_sdfg.sdfg.arrays[edge.dst_conn].clone()
            symbolic.safe_replace(nested_sdfg.symbol_mapping, lambda m: replace_properties_dict(inner_desc, m))
            if not _same_layout(outer_desc, inner_desc):
                return False

        for edge in state.out_edges(nested_sdfg):
            if edge.data.data is None:
                return False

            if not isinstance(edge.dst, nodes.AccessNode):
                return False

            if edge.data.subset != subsets.Range.from_array(sdfg.arrays[edge.data.data]):
                return False

            outer_desc = sdfg.arrays[edge.data.data]
            if isinstance(outer_desc, data.View):
                return False

            inner_desc = nested_sdfg.sdfg.arrays[edge.src_conn].clone()
            symbolic.safe_replace(nested_sdfg.symbol_mapping, lambda m: replace_properties_dict(inner_desc, m))
            if not _same_layout(outer_desc, inner_desc):
                return False

        if not helpers.isolate_nested_sdfg(state, nsdfg_node=nested_sdfg, test_if_applicable=True):
            return False

        return True

    def apply(self, outer_state: SDFGState, sdfg: SDFG):
        nsdfg_node = self.nested_sdfg
        nsdfg: SDFG = nsdfg_node.sdfg

        # If the nested SDFG contains returns, ensure they are inlined first.
        has_return = False
        for blk in nsdfg.all_control_flow_blocks():
            if isinstance(blk, ReturnBlock):
                has_return = True
        if has_return:
            sdutil.inline_control_flow_regions(nsdfg, lower_returns=True)

        #######################################################
        # Collect and update top-level SDFG metadata

        # Global/init/exit code
        for loc, code in nsdfg.global_code.items():
            sdfg.append_global_code(code.code, loc)
        for loc, code in nsdfg.init_code.items():
            sdfg.append_init_code(code.code, loc)
        for loc, code in nsdfg.exit_code.items():
            sdfg.append_exit_code(code.code, loc)

        # Callbacks and other types
        sdfg._callback_mapping.update(nsdfg.callback_mapping)

        # Environments
        for nstate in nsdfg.states():
            for node in nstate.nodes():
                if isinstance(node, nodes.CodeNode):
                    node.environments |= nsdfg_node.environments

        # Symbols
        outer_symbols = {str(k): v for k, v in sdfg.symbols.items()}
        for ise in sdfg.all_interstate_edges():
            outer_symbols.update(ise.data.new_symbols(sdfg, outer_symbols))

        # Isolate the nested SDFG in a separate state.
        predecessor_state, nsdfg_state, successor_state = helpers.isolate_nested_sdfg(state=outer_state,
                                                                                      nsdfg_node=nsdfg_node)

        # Find original source/destination edges (there is only one edge per
        # connector, according to match)
        inputs: Dict[str, MultiConnectorEdge] = {}
        outputs: Dict[str, MultiConnectorEdge] = {}
        input_set: Dict[str, str] = {}
        output_set: Dict[str, str] = {}
        for e in nsdfg_state.in_edges(nsdfg_node):
            inputs[e.dst_conn] = e
            input_set[e.data.data] = e.dst_conn
        for e in nsdfg_state.out_edges(nsdfg_node):
            outputs[e.src_conn] = e
            output_set[e.data.data] = e.src_conn

        # Replace symbols using invocation symbol mapping.
        #
        # Split the mapping into IDENTITY (``inner_K = inner_K``) and
        # NON-IDENTITY (``inner_K = outer_expr``). Only the non-identity
        # entries change anything; substituting them inline would
        # propagate ``outer_expr`` into every memlet/condition that
        # referenced ``inner_K``. Instead we lower the non-identity
        # entries to interstate-edge ASSIGNMENTS on the edge
        # ``predecessor_state -> source`` (the edge that enters the
        # inlined SDFG), so the inner code keeps using ``inner_K`` and
        # the parent sees ``inner_K = outer_expr`` as a normal iedge
        # assignment. Inner symbols absent from the outer scope get
        # added to the outer SDFG's symbol table with their inner
        # type, preserving the strict-typing contract.
        identity_mapping: Dict[Any, Any] = {}
        non_identity_mapping: Dict[str, str] = {}
        for k, v in nsdfg_node.symbol_mapping.items():
            if str(k) == str(v):
                identity_mapping[k] = v
            else:
                non_identity_mapping[str(k)] = symbolic.symstr(v)
        # Two-step replacement (N -> __dacesym_N --> map[N]) for any
        # identity entries we want safe_replace's clash-handling for.
        if identity_mapping:
            symbolic.safe_replace(identity_mapping, nsdfg.replace_dict)

        #######################################################
        # Collect and modify interstate edges as necessary

        outer_assignments = set()
        for e in sdfg.all_interstate_edges():
            outer_assignments |= e.data.assignments.keys()
        for b in sdfg.all_control_flow_blocks():
            if isinstance(b, LoopRegion):
                if b.loop_variable is not None:
                    outer_assignments.add(b.loop_variable)

        inner_assignments = set()
        for e in nsdfg.all_interstate_edges():
            inner_assignments |= e.data.assignments.keys()
        for b in nsdfg.all_control_flow_blocks():
            if isinstance(b, LoopRegion):
                if b.loop_variable is not None:
                    inner_assignments.add(b.loop_variable)
        # The non-identity symbol_mapping keys are lowered below to interstate-edge assignments
        # (``inner_K = outer_expr``) planted on the edge entering the inlined SDFG, i.e. they become
        # symbols *defined* inside the inlined region. Treat them like inner assignments so that, if such a
        # key collides with a symbol used elsewhere in the outer SDFG (e.g. a callback of the same name used
        # by a sibling branch), it is disambiguated to a fresh name instead of hijacking the outer symbol
        # and dropping it from the compiled signature (which left an uninitialized callback pointer -> crash).
        inner_assignments |= set(non_identity_mapping.keys())

        allnames = set(outer_symbols.keys()) | set(sdfg.arrays.keys())
        assignments_to_replace = inner_assignments & (outer_assignments | allnames)
        # Inner symbols that received their value from an IDENTITY symbol-mapping entry
        # (outer ``K`` -> inner ``K``, lowered above via ``safe_replace`` as a no-op rename).
        # Renaming such a symbol on collision (below) severs that implicit outer->inner link.
        identity_names = {str(k) for k in identity_mapping}
        sym_replacements: Dict[str, str] = {}
        for assign in assignments_to_replace:
            newname = data.find_new_name(assign, allnames)
            allnames.add(newname)
            outer_symbols[newname] = nsdfg.symbols.get(assign, None)
            sym_replacements[assign] = newname
            # ``assign`` was inner == outer (identity map) but is now renamed to ``newname``; the
            # outer value no longer reaches the inlined region. Re-establish it as a non-identity
            # assignment ``newname = assign`` (planted on the entry edge below) so the inlined
            # region is initialized from the outer symbol -- otherwise ``newname`` is undefined
            # (e.g. an external loop-init symbol whose inner name collides with the outer one).
            if assign in identity_names:
                non_identity_mapping[assign] = assign
        nsdfg.replace_dict(sym_replacements)

        #######################################################
        # Collect and modify access nodes as necessary

        # Mapping from nested transient name to top-level name
        transients: Dict[str, str] = {}

        # All transients become transients of the parent (if data already
        # exists, find new name)
        for nstate in nsdfg.states():
            for node in nstate.nodes():
                if isinstance(node, nodes.AccessNode):
                    datadesc = nsdfg.arrays[node.data]
                    if node.data not in transients and datadesc.transient:
                        new_name = node.data
                        if (new_name in sdfg.arrays or new_name in outer_symbols or new_name in sdfg.constants):
                            new_name = f'{nsdfg.label}_{node.data}'

                        name = sdfg.add_datadesc(new_name, datadesc, find_new_name=True)
                        transients[node.data] = name

            # All transients of edges between code nodes are also added to parent
            for edge in nstate.edges():
                if (isinstance(edge.src, nodes.CodeNode) and isinstance(edge.dst, nodes.CodeNode)):
                    if edge.data.data is not None:
                        datadesc = nsdfg.arrays[edge.data.data]
                        if edge.data.data not in transients and datadesc.transient:
                            new_name = edge.data.data
                            if (new_name in sdfg.arrays or new_name in outer_symbols or new_name in sdfg.constants):
                                new_name = f'{nsdfg.label}_{edge.data.data}'

                            name = sdfg.add_datadesc(new_name, datadesc, find_new_name=True)
                            transients[edge.data.data] = name

        # All constants (and associated transients) become constants of the parent
        for cstname, (csttype, cstval) in nsdfg.constants_prop.items():
            if cstname in sdfg.constants:
                if cstname in transients:
                    newname = transients[cstname]
                else:
                    newname = sdfg.find_new_constant(cstname)
                    transients[cstname] = newname
                sdfg.constants_prop[newname] = (csttype, cstval)
            else:
                sdfg.constants_prop[cstname] = (csttype, cstval)

        #######################################################
        # Replace data on inlined SDFG nodes/edges

        # Replace data names with their top-level counterparts
        repldict = {}
        repldict.update(transients)
        repldict.update({k: v.data.data for k, v in itertools.chain(inputs.items(), outputs.items())})

        symbolic.safe_replace(repldict, lambda m: replace_datadesc_names(nsdfg, m), value_as_string=True)

        # Make unique names for all control-flow blocks
        node_names = set(cfr.label for cfr in sdfg.all_control_flow_blocks(recursive=True))
        for node in nsdfg.all_control_flow_blocks(recursive=True):
            if node.label in node_names:
                node_name = data.find_new_name(node.label, node_names)
                node.label = node_name
            node_names.add(node.label)

        # Rename code-node connectors that would clash with outer arrays, symbols,
        # constants, or interstate assignments once the nodes are in the parent.
        reserved_names = allnames | set(sdfg.constants) | set(sdfg.constants_prop.keys()) | outer_assignments
        _disambiguate_code_connectors(nsdfg, reserved_names)

        #######################################################
        # Add nested SDFG states into top-level SDFG

        outer_start_state = outer_state.parent_graph.start_block

        outer_state.parent_graph.add_nodes_from(nsdfg.nodes())
        for ise in nsdfg.edges():
            outer_state.parent_graph.add_edge(ise.src, ise.dst, ise.data)

        #######################################################
        # Reconnect inlined SDFG

        source = nsdfg.start_state
        sinks = nsdfg.sink_nodes()

        # Apply disambiguation rename to non-identity symbol-mapping keys
        # so the iedge assignments use the post-rename name.
        non_identity_mapping = {sym_replacements.get(k, k): v for k, v in non_identity_mapping.items()}

        # Reconnect state machine. For each edge ``predecessor -> nsdfg_state``
        # we redirect it to ``predecessor -> source``; while doing so, plant the
        # non-identity symbol_mapping entries as interstate-edge assignments
        # on that edge. The inner code keeps its inner-symbol names and the
        # parent's state machine binds them on entry.
        for e in outer_state.parent_graph.in_edges(nsdfg_state):
            new_data = e.data
            if non_identity_mapping:
                new_data = dc(new_data)
                # Existing assignments win (caller may have already bound
                # the same name); add only those that aren't already there.
                for sym, expr in non_identity_mapping.items():
                    if sym not in new_data.assignments:
                        new_data.assignments[sym] = expr
            outer_state.parent_graph.add_edge(e.src, source, new_data)
            # Add new symbols to the outer SDFG so the iedge assignments
            # validate against the outer scope.
            if non_identity_mapping:
                for sym in non_identity_mapping:
                    if sym not in sdfg.symbols:
                        inner_type = nsdfg.symbols.get(sym, None)
                        if inner_type is not None:
                            sdfg.add_symbol(sym, inner_type)
        for e in outer_state.parent_graph.out_edges(nsdfg_state):
            for sink in sinks:
                outer_state.parent_graph.add_edge(sink, e.dst, dc(e.data))
                # Redirect sink incoming edges with a `False` condition to e.dst (return statements)
                for e2 in outer_state.parent_graph.in_edges(sink):
                    if e2.data.condition_sympy() == False:
                        outer_state.parent_graph.add_edge(e2.src, e.dst, InterstateEdge())

        # Modify start state as necessary
        if outer_start_state is nsdfg_state:
            outer_state.parent_graph.start_block = outer_state.parent_graph.node_id(source)

        # TODO: Modify memlets by offsetting

        # Replace nested SDFG parents with new SDFG
        for nstate in nsdfg.states():
            nstate.sdfg = sdfg
            for node in nstate.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    node.sdfg.parent_sdfg = sdfg
                    node.sdfg.parent_nsdfg_node = node

        #######################################################
        # Remove nested SDFG and state
        outer_state.parent_graph.remove_node(nsdfg_state)

        sdfg.reset_cfg_list()

        return nsdfg.nodes()
