"""
    Class for defining the reversal of pure SDFG nodes: AccessNode, Tasklet, MapEntry/Exit, NestedSDFG
    Each method should return a tuple (reversed_node, BackwardResult)
"""
import copy
from typing import List, Tuple

# DaCe imports
import dace
import dace.sdfg.nodes as nodes
from dace import dtypes
from dace.sdfg import SDFGState
from dace.util import find_str_not_in_set

# Autodiff imports
from dace.autodiff.base_abc import BackwardResult, AutoDiffException
from dace.autodiff.utils import is_int_eq_value, invert_map_connector



class DaceNodeBackwardImplementations:
    def __init__(self, backward_pass_generator: 'BackwardPassGenerator'):
        self.bwd_engine = backward_pass_generator 
        pass
    
    def _reverse_NestedSDFG(
        self,
        forward_state: SDFGState,
        backward_state: SDFGState,
        node: nodes.NestedSDFG,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> Tuple[nodes.Node, BackwardResult]:
        reverse_nsdfg = dace.SDFG(node.sdfg.name + "_backward")
        
        # Create a new backward pass generator object for the nested SDFG
        gen = self.bwd_engine.__class__(sdfg=node.sdfg,
                                    given_gradients=given_gradients,
                                    required_gradients=required_gradients,
                                    backward_sdfg=reverse_nsdfg,
                                    overwrite_strategy=self.bwd_engine.strategy,
                                    data_to_recompute=self.bwd_engine.data_to_recompute)
        backward_result, _, backward_input_arrays = gen.backward()

        # we need to defer add edges until after the arrays have been added because creation of the nested
        # sdfg fails otherwise
        deferred_edges = []

        inputs = set(backward_result.given_grad_names[name] for name in sorted(given_gradients))
        # loop through the arrays that we need from the forward pass
        for name, desc in sorted(backward_input_arrays.items()):
            # if the name is not already passed to the reverse SDFG node ...
            if name not in required_gradients and name not in node.in_connectors:
                # ... this array needs to be forwarded out of the forward SDFG (i.e. it is an intermediate value)
                # 1) add it to the current SDFG, and to self.bwd_engine.backward_input_arrays
                # 2) add an out connector to the forward nested SDFG, add a write node to the current state, and an edge
                #    from the output to there
                # 3) add a read node to the backward state, and an edge into it

                desc = node.sdfg.arrays[name]

                # if the original view node is in the in-connector, no need to connect it, continue
                # if forwarded_name in node.in_connectors:
                #     continue

                # (1)
                new_name = find_str_not_in_set(set(self.bwd_engine.sdfg.arrays), name + "_forwarded")
                if new_name in self.bwd_engine.sdfg.arrays or new_name in self.bwd_engine.backward_input_arrays:
                    raise AutoDiffException(
                        "Attempted to create array with name '{}', but it already existed".format(new_name))

                self.bwd_engine.sdfg.add_datadesc(new_name, copy.deepcopy(desc))
                self.bwd_engine.backward_input_arrays[new_name] = copy.deepcopy(desc)

                if self.bwd_engine.separate_sdfgs:
                    to_add = copy.deepcopy(desc)
                    to_add.transient = False
                    self.bwd_engine.backward_sdfg.add_datadesc(new_name, to_add)

                # (2)
                node.sdfg.arrays[name].transient = False
                assert node.add_out_connector(name, force=True)
                write = forward_state.add_write(new_name)
                forward_state.add_edge(node, name, write, None, self.bwd_engine.sdfg.make_array_memlet(new_name))

                # (3)
                read = backward_state.add_read(new_name)
                deferred_edges.append(
                    dict(u=read,
                         u_connector=None,
                         v_connector=name,
                         memlet=self.bwd_engine.backward_sdfg.make_array_memlet(new_name)))
                inputs.add(name)
            else:
                inputs.add(name)

        outputs = set(backward_result.required_grad_names[name] for name in required_gradients)

        for inp in inputs:
            if inp in reverse_nsdfg.arrays:
                reverse_nsdfg.arrays[inp].transient = False
        for outp in outputs:
            if outp in reverse_nsdfg.arrays:
                reverse_nsdfg.arrays[outp].transient = False
        # Create the sdfg and return it
        nsdfg = backward_state.add_nested_sdfg(
            reverse_nsdfg,
            inputs=inputs,
            outputs=outputs,
        )

        # If any input connectors point to symbols
        for conn, _ in nsdfg.in_connectors.items():
            if conn in nsdfg.sdfg.symbols:
                # We need to add a new symbol and create a mapping
                new_symbol = find_str_not_in_set(nsdfg.sdfg.symbols, conn)
                nsdfg.sdfg.add_symbol(new_symbol, nsdfg.sdfg.symbols[conn])
                nsdfg.sdfg.replace(conn, new_symbol)
                nsdfg.symbol_mapping[new_symbol] = conn
                # Remove it from the symbol mapping too
                if conn in nsdfg.symbol_mapping:
                    del nsdfg.symbol_mapping[conn]
        for edge_args in deferred_edges:
            edge_args["v"] = nsdfg
            backward_state.add_edge(**edge_args)

        return nsdfg, BackwardResult(required_grad_names=backward_result.required_grad_names,
                                     given_grad_names=backward_result.given_grad_names)

    def _reverse_AccessNode(
        self,
        forward_state: SDFGState,
        backward_state: SDFGState,
        node: nodes.AccessNode,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> Tuple[nodes.Node, BackwardResult]:
        rev = nodes.AccessNode(self.bwd_engine.array_grad_name(node.data))

        # We want all gradient arrays to be initialized to zero
        # This is important for correct gradient accumulation
        rev.setzero = True
        backward_state.add_node(rev)
        required_grad_names = {None: None}
        given_grad_names = {None: None}

        if "views" in node.in_connectors:
            required_grad_names = {"views": "views"}
        if "views" in node.out_connectors:
            given_grad_names = {"views": "views"}

        return rev, BackwardResult(required_grad_names=required_grad_names, given_grad_names=given_grad_names)

    def _reverse_MapEntry(
        self,
        forward_state: SDFGState,
        backward_state: SDFGState,
        node: nodes.MapEntry,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> Tuple[nodes.Node, BackwardResult]:

        required_grad_names = {n: invert_map_connector(n) for n in required_gradients}
        given_grad_names = {n: invert_map_connector(n) for n in given_gradients}
        result = BackwardResult(required_grad_names=required_grad_names, given_grad_names=given_grad_names)
        rev = nodes.MapExit(self.bwd_engine.reverse_map[node.map])

        for _, conn in sorted(given_grad_names.items()):
            assert rev.add_in_connector(conn)

        for _, conn in sorted(required_grad_names.items()):
            assert rev.add_out_connector(conn)

        backward_state.add_node(rev)
        return rev, result

    def _reverse_MapExit(
        self,
        forward_state: SDFGState,
        backward_state: SDFGState,
        node: nodes.MapExit,
        given_gradients: List[str],
        required_gradients: List[str],
    ):
        self.bwd_engine.reverse_map[node.map] = copy.deepcopy(node.map)

        rev = nodes.MapEntry(self.bwd_engine.reverse_map[node.map])
        for conn in sorted(node.in_connectors):
            assert rev.add_in_connector(conn)

        for conn in sorted(node.out_connectors):
            assert rev.add_out_connector(conn)

        backward_state.add_node(rev)
        # yapf: disable
        return (
            rev,
            BackwardResult(required_grad_names={
                n: invert_map_connector(n)
                for n in required_gradients
            },
                given_grad_names={
                    n: invert_map_connector(n)
                    for n in given_gradients
                }),
        )
        # yapf: enable

    def _reverse_Tasklet(
        self,
        state: SDFGState,
        backward_state: SDFGState,
        tasklet: nodes.Tasklet,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> Tuple[nodes.Node, BackwardResult]:
        if tasklet.language is not dtypes.Language.Python:
            raise AutoDiffException("Expected tasklet with language Python, got language {}".format(tasklet.language))

        # tasklets should have scalar inputs (can be relaxed)
        for _, _, _, _, memlet in state.in_edges(tasklet):
            if memlet.data is not None:
                try:
                    is_int_eq_value(memlet.subset.num_elements(), 1)
                except AutoDiffException as e:
                    raise AutoDiffException(
                        "Autodiff only supported for tasklets with scalar inputs and outputs") from e

        for _, _, _, _, memlet in state.out_edges(tasklet):
            if memlet.data is not None:
                try:
                    is_int_eq_value(memlet.subset.num_elements(), 1)
                except AutoDiffException as e:
                    raise AutoDiffException(
                        "Autodiff only supported for tasklets with scalar inputs and outputs") from e

        code_str = tasklet.code.as_string

        # check if this is a conditional tasklet
        if self.bwd_engine._conditional_tasklet(tasklet):
            # we want to extract the if and else expressions and pass them to sympy
            if_expression, else_expression, conditional = self.bwd_engine._extract_conitional_expressions(tasklet)

            if_code, if_rev_inputs, if_rev_outputs, if_result = self.bwd_engine._differentiate_code_symbolically(
                if_expression, state, tasklet, given_gradients, required_gradients)

            if else_expression:
                else_code, else_rev_inputs, else_rev_outputs, else_result = self.bwd_engine._differentiate_code_symbolically(
                    else_expression, state, tasklet, given_gradients, required_gradients)
                assert else_rev_inputs == if_rev_inputs
                assert if_rev_outputs == else_rev_outputs
                assert else_result == if_result

            # prepare the tasklet code depending on the conditional type
            # add the same conditional to the if_code
            # first, add indentation
            if_code = if_code.replace("\n", "\n\t")
            if_code = f"if {conditional}:\n{if_code}"

            # add the conditional to the in connectors
            if_rev_inputs.add(conditional)
            joint_code = if_code

            if ":" not in code_str:
                # only an if in the original code
                assert else_expression
                else_code = else_code.replace("\n", "\n\t")
                else_code = f"else:\n{else_code}"
                joint_code = f"{if_code}\n{else_code}"

            # in case there are no out_connectors, we will zero out the assigned-to AccessNode
            if len(if_rev_outputs) == 0:
                if_rev_outputs = {"__zero_out_conn__"}

            rev = nodes.Tasklet("_" + tasklet.label + "_reverse_",
                                inputs=if_rev_inputs,
                                outputs=if_rev_outputs,
                                code=joint_code,
                                debuginfo=tasklet.debuginfo)

            result = if_result
        else:
            code, rev_inputs, rev_outputs, result = self.bwd_engine._differentiate_code_symbolically(
                code_str, state, tasklet, given_gradients, required_gradients)
            rev = nodes.Tasklet("_" + tasklet.label + "_reverse_",
                                inputs=rev_inputs,
                                outputs=rev_outputs,
                                code=code,
                                debuginfo=tasklet.debuginfo)
            backward_state.add_node(rev)
        return rev, result