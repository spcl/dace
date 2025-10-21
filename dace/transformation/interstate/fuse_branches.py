import copy
from sympy import pycode
import re
import sympy
import dace
from dace import properties, transformation
from dace import InterstateEdge
from dace.properties import CodeBlock
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
import dace.sdfg.utils as sdutil
import dace.sdfg.construction_utils as cutil
from typing import Tuple, Set, Union
from dace.sdfg.construction_utils import token_match, token_replace


def extract_bracket_content(s: str):
    pattern = r"<(\w+)>\[([^\]]*)\]"
    matches = re.findall(pattern, s)

    extracted = dict()
    for name, content in matches:
        if name in extracted:
            raise Exception("Repeated assignment in interstate edge, not supported")
        extracted[name] = content

    # Remove [...] from the string
    cleaned = re.sub(pattern, r"<\1>", s)

    return cleaned, extracted


def interstate_index_to_subset_str(expr: str) -> str:
    pattern = r'\[([^\]]+)\]'  # match [...] including inside content

    def replacer(match):
        inner = match.group(1).strip()
        # split by commas, handle arbitrary whitespace
        parts = [p.strip() for p in inner.split(',')]
        # expand each token into b:b:1 form
        expanded = ', '.join(f'{p}:{p}:1' for p in parts)
        return f'[{expanded}]'

    return re.sub(pattern, replacer, expr)


def symbol_is_used(cfg: ControlFlowRegion, symbol_name: str) -> bool:
    # Symbol can be read on an interstate edge, appear in a conditional block's conditions, loop regions condition / update
    # Appear in shape of an array, in the expression of maps or in taskelts, passed to nested SDFGs

    # Interstate edge reads
    for e in cfg.all_interstate_edges():
        for v in e.data.assignments.values():
            if token_match(v, symbol_name):
                return True

    # Conditional Block
    for cb in cfg.all_control_flow_blocks():
        if isinstance(cb, ConditionalBlock):
            for cond, _ in cb.branches:
                if cond is None:
                    continue
                if token_match(cond.as_string, symbol_name):
                    return True

    # Loop
    for lr in cfg.all_control_flow_regions():
        if isinstance(lr, LoopRegion):
            if token_match(f"{lr.init_statement} {lr.update_statement} {lr.loop_condition}", symbol_name):
                return True

    # Arrays
    for arr in cfg.sdfg.arrays.values():
        for dim, stride in zip(arr.shape, arr.strides):
            if token_match(f"{dim} {stride}", symbol_name):
                return True

    # Maps
    for state in cfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                for (b, e, s) in node.map.range:
                    if token_match(f"{b} {e} {s}", symbol_name):
                        return True

    # Takslets
    for state in cfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                if token_match(node.code.as_string, symbol_name):
                    return True

    # NestedSDFGs (symbol mapping)
    for state in cfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                for v in node.symbol_mapping.values():
                    if token_match(str(v), symbol_name):
                        return True

    return False


def remove_symbol_assignments(graph: ControlFlowRegion, sym_name: str):
    for e in graph.all_interstate_edges():
        new_assignments = dict()
        for k, v in e.data.assignments.items():
            if k != sym_name:
                new_assignments[k] = v
        e.data.assignments = new_assignments


def remove_node_redirect_in_edges_to(parent_graph: ControlFlowRegion, node: ControlFlowRegion,
                                     new_dst: ControlFlowRegion):
    ies = parent_graph.in_edges(node)
    parent_graph.remove_node(node)
    for ie in ies:
        parent_graph.add_edge(ie.src, new_dst, copy.deepcopy(ie.data))


@properties.make_properties
@dace.transformation.explicit_cf_compatible
class FuseBranches(transformation.MultiStateTransformation):
    """
    It translated couple of SIMT branches to a form compatible with SIMD instructions
    ```
    if (cond){
        out1[address1] = computation1(...);
    } else {
        out1[address1] = computation2(...);
    }
    ```

    If all the write sets in the left and right branches are the same,
    menaing the address1 == address1,
    we can transformation the if branch to:
    ```
    fcond = float(cond)
    out1[address1] = computation1(...) * fcond + (1.0 - fcond) * computation2(...);
    ```

    For single branch case:
    ```
    out1[address1] = computation1(...) * fcond + (1.0 - fcond) * out1[address1];
    ```

    Also supportes:
    ```
    if (cond){
        out1[address1] = computation1(...);
    } else {
        out1[address2] = computation2(...);
    }
    ```
    If `address1` and `address2` are completely disjoint the pass will handle this
    by chaining the if branches and treating them as conditionals with a single branch
    like.
    ```
    if (cond){
        out1[address1] = computation1(...);
    }
    bool new_cong = !cond;
    if (neg_cond) {
        out1[address2] = computation2(...);
    }
    ```

    The pass also supports multiple writes if the conditional has just one branch:
    ```
    if (cond){
        out1[address1] = computation1(...);
        out1[address2] = computation2(...);
    }
    ```


    This eliminates branching by duplicating the computation of each branch
    but makes it possible to vectorize the computation.
    """
    conditional = transformation.PatternNode(ConditionalBlock)
    parent_nsdfg_state = properties.Property(dtype=SDFGState, allow_none=True, default=None)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.conditional)]

    def _check_reuse(self, sdfg: dace.SDFG, orig_state: dace.SDFGState, diff_set: Set[str]):
        for graph in sdfg.all_control_flow_regions():
            if (not isinstance(graph, dace.SDFGState)) and orig_state in graph.all_states():
                continue
            if graph == orig_state:
                continue
            read_set, write_set = graph.read_and_write_sets()
            if any({k in read_set or k in write_set for k in diff_set}):
                return True

        return False

    def _move_interstate_assignment_to_state(self, state: dace.SDFGState, rhs: str, lhs: str):
        rhs_as_symexpr = dace.symbolic.SymExpr(rhs)
        free_vars = {str(sym) for sym in rhs_as_symexpr.free_symbols}

        cleaned, extracted_subsets = extract_bracket_content(rhs)

        arr_inputs = {var for var in free_vars if var in state.sdfg.arrays}
        state.sdfg.add_scalar(
            name="float_" + lhs,
            dtype=dace.dtypes.float64,
            storage=dace.dtypes.StorageType.Register,
            transient=True,
        )
        symbol_inputs = {var for var in free_vars if var not in state.sdfg.arrays}

        for arr_input in arr_inputs:
            cleaned = token_replace(cleaned, arr_input, f"_in_{arr_input}")
        cleaned = token_replace(cleaned, lhs, f"_out_float_{lhs}")

        cleaned_as_symexpr = dace.symbolic.SymExpr(cleaned)

        assert arr_inputs.union(symbol_inputs) == free_vars
        print(cleaned_as_symexpr)

        from sympy.printing.pycode import PythonCodePrinter

        class BracketFunctionPrinter(PythonCodePrinter):

            def _print_Function(self, expr):
                name = self._print(expr.func)
                args = ", ".join([self._print(arg) for arg in expr.args])
                return f"{name}[{args}]"

        printer = BracketFunctionPrinter({'strict': False})
        cleaned_symexpr_str = printer.doprint(cleaned_as_symexpr)

        tasklet = state.add_tasklet(name=f"ieassign_{lhs}_to_float_{lhs}_scalar",
                                    inputs={"_in_" + arr_input
                                            for arr_input in arr_inputs},
                                    outputs={"_out_float_" + lhs},
                                    code=f"_out_float_{lhs} = ({cleaned_symexpr_str})")

        for arr_name in arr_inputs:
            an_of_arr_name = {n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == arr_name}
            last_access = None
            if len(an_of_arr_name) == 0:
                last_access = state.add_access(arr_name)
            elif len(an_of_arr_name) == 1:
                last_access = an_of_arr_name.pop()
            else:
                # Get the last one
                # All access nodes of the same data should be in the same weakly connected component
                # Otherwise it would be code-gen dependent data race
                source_nodes = {n for n in state.nodes() if state.in_degree(n) == 0}
                ordered_an_of_arr_names = dict()

                last_accessess = dict()
                for src_node in source_nodes:
                    ordered_an_of_arr_names[src_node] = list(state.bfs_nodes(src_node))
                    access_to_arr_name = [
                        n for n in ordered_an_of_arr_names[src_node]
                        if isinstance(n, dace.nodes.AccessNode) and n.data == arr_name
                    ]
                    if len(access_to_arr_name) > 0:
                        last_accessess[src_node] = access_to_arr_name[-1]

                assert len(last_accessess) == 1
                last_access = next(iter(last_accessess.values()))

            # Create subsetexpr from access expr which [b1,b2,b3]
            # Needs to become [b1:b1:1, b2:b2:1, b3:b3:1]
            # If it is a scalar and does not have a subset expression to it just make it 0 per shape
            new_subset = extracted_subsets.get(arr_name, None)
            if new_subset is None:
                new_subset = "[" + ",".join(["0" for _ in state.sdfg.arrays[arr_name].shape]) + "]"
            subset_str = interstate_index_to_subset_str(new_subset)
            state.add_edge(last_access, None, tasklet, "_in_" + arr_name, dace.memlet.Memlet(f"{arr_name}{subset_str}"))

            arr = state.sdfg.arrays[arr_name]
            if arr.dtype == dace.bool:
                arr.dtype = dace.float64

        state.add_edge(tasklet, f"_out_float_{lhs}", state.add_access(f"float_{lhs}"), None,
                       dace.memlet.Memlet(f"float_{lhs}"))

    def _is_disjoint_subset(self, state0: SDFGState, state1: SDFGState) -> bool:
        state0_writes = set()
        state1_writes = set()
        state0_write_subsets = dict()
        state1_write_subsets = dict()
        read_sets0, write_sets0 = state0.read_and_write_sets()
        read_sets1, write_sets1 = state1.read_and_write_sets()  # fixed typo
        joint_writes = write_sets0.intersection(write_sets1)
        for write in joint_writes:
            state0_accesses = {n for n in state0.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write}
            state1_accesses = {n for n in state1.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write}

            for state_writes, state_accesses, state, state_write_subsets in [
                (state0_writes, state0_accesses, state0, state0_write_subsets),
                (state1_writes, state1_accesses, state1, state1_write_subsets)
            ]:
                state_write_edges = set()
                for an in state_accesses:
                    state_write_edges |= {e for e in state.in_edges(an) if e.data.data is not None}
                # If there are multiple write edges again we would need to know the order
                state_writes |= {e.data.data for e in state_write_edges}
                for e in state_write_edges:
                    if e.data.data is None:
                        continue
                    assert (e.data.subset.num_elements_exact() == 1)
                    if e.data.data not in state_write_subsets:
                        state_write_subsets[e.data.data] = set()
                    state_write_subsets[e.data.data].add(e.data.subset)

            # Build symmetric difference of subsets
            all_keys = set(state0_write_subsets) | set(state1_write_subsets)
            intersects = {k: False for k in all_keys}
            for name, subsets0 in state0_write_subsets.items():
                if name in state1_write_subsets:
                    subsets1 = state1_write_subsets[name]
                else:
                    subsets1 = set()
                for other_subset in subsets1:
                    for subset0 in subsets0:
                        if subset0.intersects(other_subset):
                            intersects[name] = True
            for name, subsets1 in state1_write_subsets.items():
                if name in state0_write_subsets:
                    subsets0 = state0_write_subsets[name]
                else:
                    subsets0 = set()
                for other_subset in subsets0:
                    for subset1 in subsets1:
                        if subset1.intersects(other_subset):
                            intersects[name] = True

            if not all(v is False for k, v in intersects.items()):
                return False

        return True

    def collect_all_accesses(self, state: dace.SDFGState):
        state_accesses = {n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode)}
        return state_accesses

    def collect_all_accesses(self, state: dace.SDFGState):
        state_accesses = {n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode)}
        return state_accesses

    def collect_accesses(self, state: dace.SDFGState, write: str):
        state_accesses = {n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write}
        return state_accesses

    def collect_write_accesses(self, state: dace.SDFGState, write: str):
        state_accesses = self.collect_accesses(state, write)
        state_write_accesses = {
            a
            for a in state_accesses if (state.in_degree(a) > 0 and any(
                e.data.data is not None for e in state.in_edges(a)) and (state.out_degree(a) == 0 or isinstance(
                    state.sdfg.arrays[a.data], dace.data.Array) or state.sdfg.arrays[a.data].transient is False))
        }
        return state_write_accesses

    def collect_ignored_write_accesses(self, state: dace.SDFGState):
        state_all_write_accesses = {
            a
            for a in state.nodes()
            if isinstance(a, dace.nodes.AccessNode) and (state.in_degree(a) > 0 and any(e.data.data is not None
                                                                                        for e in state.in_edges(a)))
        }
        state_all_considered_write_accesses = {
            a
            for a in state_all_write_accesses if (state.in_degree(a) > 0 and any(
                e.data.data is not None for e in state.in_edges(a)) and (state.out_degree(a) == 0 or isinstance(
                    state.sdfg.arrays[a.data], dace.data.Array) or state.sdfg.arrays[a.data].transient is False))
        }
        return state_all_write_accesses - state_all_considered_write_accesses

    def ignored_accesses_are_reused(self, state: dace.SDFGState):
        ignored_accesses = self.collect_ignored_write_accesses(state)
        ignored_data = {a.data for a in ignored_accesses}

        # Ensure the arrays are not used elsewhere
        # Check loopRegions, conditionalBlocks
        copy_sdfg = copy.deepcopy(state.sdfg)
        sdutil.set_nested_sdfg_parent_references(copy_sdfg)
        for state2 in copy_sdfg.all_states():
            if state2.label == state.label:
                for n in state2.nodes():
                    state2.remove_node(n)

        # Check if ignore data is used anywhere
        read_set, write_set = copy_sdfg.read_and_write_sets()

        ignored_in_read = read_set.intersection(ignored_data)
        ignored_in_write = read_set.intersection(ignored_data)
        if len(ignored_in_read) > 0:
            print(
                f"[can_be_applied] Ignored data ({ignored_data}) is used in the rest of the SDFG (read) {ignored_in_read}"
            )
        if len(ignored_in_write) > 0:
            print(
                f"[can_be_applied] Ignored data ({ignored_data}) is used in the rest of the SDFG (write) {ignored_in_write}"
            )

        return len(ignored_in_read) > 0 or len(ignored_in_write) > 0

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # print("[can_be_applied]")
        # Works for if-else branches or only if branches
        if len(self.conditional.branches) > 2:
            print("[can_be_applied] More than two branches – only supports if/else or single if.")
            return False

        if len(self.conditional.branches) == 2:
            tup0 = self.conditional.branches[0]
            tup1 = self.conditional.branches[1]
            (cond0, body0) = tup0[0], tup0[1]
            (cond1, body1) = tup1[0], tup1[1]
            if cond0 is not None and cond1 is not None:
                print("[can_be_applied] Both branches have conditions – not a simple if/else.")
                return False
            assert not (cond0 is None and cond1 is None)

            # Works if the branch bodies have a single state each
            for i, body in enumerate([body0, body1]):
                if len(body.nodes()) != 1 or not isinstance(body.nodes()[0], SDFGState):
                    print(f"[can_be_applied] Branch {i} does not have exactly one SDFGState node.")
                    return False

            # Check write sets are equivalent
            state0: SDFGState = body0.nodes()[0]
            state1: SDFGState = body1.nodes()[0]

            read_sets0, write_sets0 = state0.read_and_write_sets()
            read_sets1, write_sets1 = state1.read_and_write_sets()  # fixed typo

            joint_writes = write_sets0.intersection(write_sets1)
            diff_state0 = write_sets0.difference(write_sets1)
            diff_state1 = write_sets1.difference(write_sets0)

            # For joint writes ensure the write subsets are always the same

            for write in joint_writes:
                state0_accesses = {
                    n
                    for n in state0.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write
                }
                state1_accesses = {
                    n
                    for n in state1.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write
                }

                state0_write_accesses = self.collect_write_accesses(state0, write)
                state1_write_accesses = self.collect_write_accesses(state1, write)

                for state, accesses in [(state0, state0_accesses), (state1, state1_accesses)]:
                    for access in accesses:
                        arr = state.sdfg.arrays[access.data]
                        if arr.dtype not in dace.dtypes.FLOAT_TYPES and arr.dtype not in dace.dtypes.INT_TYPES:
                            print(
                                f"[can_be_applied] Storage of '{write.data}' is not a floating/int point type has type: {arr.dtype}"
                            )
                            return False

                state0_writes = set()
                state1_writes = set()
                state0_write_subsets = dict()
                state1_write_subsets = dict()
                subsets_disjoint = True
                for state_writes, state_accesses, state, state_write_subsets in [
                    (state0_writes, state0_accesses, state0, state0_write_subsets),
                    (state1_writes, state1_accesses, state1, state1_write_subsets)
                ]:
                    state_write_edges = set()
                    for an in state_accesses:
                        state_write_edges |= {e for e in state.in_edges(an) if e.data.data is not None}
                    # If there are multiple write edges again we would need to know the order
                    state_writes |= {e.data.data for e in state_write_edges}
                    for e in state_write_edges:
                        if e.data.data is None:
                            continue
                        if e.data.subset.num_elements_exact() != 1:
                            print(
                                f"[can_be_applied] All write edges need to have exactly one-element write '{write}' (edge {e} problematic)."
                            )
                        if e.data.data not in state_write_subsets:
                            state_write_subsets[e.data.data] = set()
                        state_write_subsets[e.data.data].add(e.data.subset)

                subsets_disjoint = self._is_disjoint_subset(state0, state1)

                # If there are more than one writes we can't fuse them together without knowing how to order
                # Unless the subset is disjoint
                if (any(len(v) > 1 for k, v in state0_write_subsets.items())
                        or any(len(v) > 1 for k, v in state1_write_subsets.items())):
                    if not subsets_disjoint:
                        print(state0_write_subsets)
                        print(state1_write_subsets)
                        print(
                            f"[can_be_applied] Multiple write edges for '{write}' in one branch and subsets not disjoint."
                        )
                        return False

                # If the subset of each branch is different then we can't fuse either
                if state0_writes != state1_writes:
                    if not subsets_disjoint:
                        print(
                            f"[can_be_applied] Write subsets differ (and not disjoint) for '{write}' between branches.")
                        return False

                if len(state0_write_accesses) > 1 or len(state1_write_accesses) > 1:
                    if not subsets_disjoint:
                        print(
                            f"[can_be_applied] Multiple write accesses AccessNodes found for '{write}' in one of the branches."
                        )
                        return False

            # If diff states only have transient scalars or arrays it is probably ok (permissive)
            if diff_state0 or diff_state1:
                if self._check_reuse(sdfg, state0, diff_state0):
                    print(f"[can_be_applied] Branch 0 writes to non-reusable data: {diff_state0}")
                    return False
                if self._check_reuse(sdfg, state1, diff_state1):
                    print(f"[can_be_applied] Branch 1 writes to non-reusable data: {diff_state1}")
                    return False

            if self.ignored_accesses_are_reused(state0):
                return False
            if self.ignored_accesses_are_reused(state1):
                return False

        elif len(self.conditional.branches) == 1:
            tup0: Tuple[properties.CodeBlock, ControlFlowRegion] = self.conditional.branches[0]
            cond0, body0 = tup0[0], tup0[1]

            # Works if the branch body has a single state
            if len(body0.nodes()) != 1 or not isinstance(body0.nodes()[0], SDFGState):
                print("[can_be_applied] Single branch does not have exactly one SDFGState node.")
                return False

            # Check write sets are equivalent
            state0: SDFGState = body0.nodes()[0]

            read_sets0, write_sets0 = state0.read_and_write_sets()

            # For joint writes ensure the write subsets are always the same
            for write in write_sets0:
                state0_accesses = self.collect_accesses(state0, write)
                state0_write_accesses = self.collect_write_accesses(state0, write)

                for state, accesses in [
                    (state0, state0_accesses),
                ]:
                    for access in accesses:
                        arr = state.sdfg.arrays[access.data]
                        if arr.dtype not in dace.dtypes.FLOAT_TYPES and arr.dtype not in dace.dtypes.INT_TYPES:
                            print(
                                f"[can_be_applied] Storage of '{write.data}' is not a floating/int point type has type: {arr.dtype}"
                            )
                            return False

                state0_writes = set()
                for state_writes, state_accesses, state in [(state0_writes, state0_accesses, state0)]:
                    state_write_edges = set()
                    for an in state_accesses:
                        state_write_edges |= {e for e in state.in_edges(an) if e.data.data is not None}
                    # If there are multiple write edges again we would need to know the order
                    # Ok - if single take the last one
                    #if len(state_write_edges) > 1:
                    #    print(f"[can_be_applied] Multiple write edges for '{write}' in one branch.")
                    #    return False
                    state_writes |= {e.data.data for e in state_write_edges}
                    for e in state_write_edges:
                        if e.data.data is None:
                            continue
                        if e.data.subset.num_elements_exact() != 1:
                            print(
                                f"[can_be_applied] All write edges need to have exactly one-element write '{write}' (edge {e} problematic)."
                            )

                if self.ignored_accesses_are_reused(state0):
                    return False
        # If nsdfg all out edges need to have size-1 subsets
        if self.parent_nsdfg_state is not None:
            parent_nsdfg_node = self.conditional.parent_graph.sdfg.parent_nsdfg_node
            for oe in self.parent_nsdfg_state.out_edges(parent_nsdfg_node):
                if oe.data.data is None:
                    continue
                #if not isinstance(oe.data.subset, dace.subsets.Range):
                #    print("[can_be_applied] If working in a nested SDFG all out edges need to be single-subset edges and have type susbet Range")
                #    print(f"[can_be_applied] {oe.data.subset}")
                #r: dace.subsets.Range = oe.data.subset
                #if r.num_elements_exact() != 1:
                #    print("[can_be_applied] If working in a nested SDFG all out edges need to be single-subset edges and have type susbet Range")
                #    print(f"[can_be_applied] {oe.data.subset}")
                pass

        # TODO:
        # Check that transients are not used in any other state
        #print(
        #    f"[can_be_applied] The check for the writing states for transient data within the branches has not been implemented yet!"
        #)

        print(f"[can_be_applied] to {self.conditional} is True")
        return True

    def _scalar_is_assigned_symbolic_value(self, state: SDFGState,
                                           node: dace.nodes.AccessNode) -> Union[None, Tuple[str, dace.nodes.Tasklet]]:
        # Check if the scalars are really needed
        # If scalar has single source, which is a tasklet with 0 in degree then it is using some symbol value
        # then we can use the symbolic expression instead
        if state.in_degree(node) != 1:
            return None
        ies = state.in_edges(node)
        if len(ies) != 1:
            return None
        ie = ies[0]
        if not isinstance(ie.src, dace.nodes.Tasklet):
            return None
        t = ie.src
        if state.in_degree(t) > 0:
            return None
        if len(t.in_connectors) > 0:
            return None
        return t.code.as_string.split("=")[-1].strip(), t

    def _try_simplify_combine_tasklet(self, state: SDFGState, node: dace.nodes.Tasklet):
        if node.language != dace.dtypes.Language.Python:
            return

        for ie in state.in_edges(node):
            if isinstance(ie.src, dace.nodes.AccessNode):
                rettup = self._scalar_is_assigned_symbolic_value(state, ie.src)
                if rettup is not None:
                    rhs_str: str = rettup[0]
                    tasklet: dace.nodes.Tasklet = rettup[1]
                    state.remove_node(tasklet)
                    node.remove_in_connector(ie.dst_conn)
                    lhs, rhs = node.code.as_string.split("=")
                    lhs = lhs.strip()
                    rhs = rhs.strip()
                    rhs_expr = dace.symbolic.SymExpr(rhs)
                    code = sympy.nsimplify(rhs_expr)
                    # Use rational until the very end and then call evalf to get rational to flaot to avoid accumulating errors
                    code = sympy.nsimplify(code.subs(ie.dst_conn, rhs_str)).evalf()
                    new_code_str = lhs + " = " + pycode(code)
                    node.code = CodeBlock(new_code_str)

                    state.remove_edge(ie)
                    if state.degree(ie.src) == 0:
                        state.remove_node(ie.src)

    def _try_fuse(self, graph: ControlFlowRegion, new_state: SDFGState, cond_prep_state: SDFGState):
        if len({
                t
                for t in cond_prep_state.nodes()
                if isinstance(t, dace.nodes.Tasklet) or isinstance(t, dace.nodes.NestedSDFG)
        }) == 1:
            assign_tasklets = {
                t
                for t in cond_prep_state.nodes() if isinstance(t, dace.nodes.Tasklet) and t.label.startswith("ieassign")
            }

            assert cond_prep_state.out_degree(next(iter(assign_tasklets))) == 1
            assign_tasklet = assign_tasklets.pop()
            oe = cond_prep_state.out_edges(assign_tasklet)[0]

            dst_data = oe.dst.data

            source_nodes = [
                n for n in new_state.nodes()
                if isinstance(n, dace.nodes.AccessNode) and n.data == dst_data and new_state.in_degree(n) == 0
            ]

            ies = cond_prep_state.in_edges(assign_tasklet)

            cp_tasklet = copy.deepcopy(assign_tasklet)
            new_state.add_node(cp_tasklet)
            tmp_name, tmp_arr = new_state.sdfg.add_scalar(name="tmp_ieassign",
                                                          dtype=new_state.sdfg.arrays[dst_data].dtype,
                                                          transient=True,
                                                          find_new_name=True)
            tmp_an = new_state.add_access(tmp_name)
            new_state.add_edge(cp_tasklet, next(iter(cp_tasklet.out_connectors)), tmp_an, None,
                               dace.memlet.Memlet(tmp_an.data))
            for src_node in source_nodes:
                at = new_state.add_tasklet(name="assign", inputs={"_in"}, outputs={"_out"}, code="_out = _in")
                new_state.add_edge(tmp_an, None, at, "_in", dace.memlet.Memlet(tmp_an.data))
                new_state.add_edge(at, "_out", src_node, None, dace.memlet.Memlet(src_node.data))

            for ie in ies:
                assert isinstance(ie.src, dace.nodes.AccessNode)
                assert cp_tasklet in new_state.nodes()
                new_ie_src = new_state.add_access(ie.src.data)
                new_state.add_edge(new_ie_src, ie.src_conn, cp_tasklet, ie.dst_conn, copy.deepcopy(ie.data))
                cond_prep_state.remove_edge(ie)
                if cond_prep_state.degree(ie.src) == 0:
                    cond_prep_state.remove_node(ie.src)
            cond_prep_state.remove_edge(oe)
            cond_prep_state.remove_node(oe.dst)
            cond_prep_state.remove_node(assign_tasklet)

            # If no nodes inside the state and if it only connects to the new state and has no interstate assignments we can delete it
            if len(cond_prep_state.nodes()) == 0:
                if graph.out_degree(cond_prep_state) == 1 and {new_state
                                                               } == {e.dst
                                                                     for e in graph.out_edges(cond_prep_state)}:
                    oes = graph.out_edges(cond_prep_state)
                    ies = graph.in_edges(cond_prep_state)
                    all_assignments = [oe.data.assignments for oe in oes]
                    if all({d == dict() for d in all_assignments}):
                        graph.remove_node(cond_prep_state)
                        for ie in ies:
                            graph.add_edge(ie.src, new_state, copy.deepcopy(ie.data))

    def _extract_condition_var_and_assignment(self, graph: ControlFlowRegion) -> Tuple[str, str]:
        non_none_conds = [cond for cond, _ in self.conditional.branches if cond is not None]
        assert len(non_none_conds) == 1
        cond = non_none_conds.pop()
        cond_code_str = cond.as_string
        cond_code_symexpr = dace.symbolic.SymExpr(cond_code_str)

        assert len(cond_code_symexpr.free_symbols) == 1

        # We either have if (cond_var) or if (cond_expr)
        # If cond_var there needs to be an assignment
        # If cond expression we assume that is the assignment
        if (str(next(iter(cond_code_symexpr.free_symbols))) == cond_code_str
                or (f"({next(iter(cond_code_symexpr.free_symbols))})" == cond_code_str)):
            just_var = True
        else:
            just_var = False

        cond_var = str(next(iter(cond_code_symexpr.free_symbols)))
        cond_assignment = None
        for ie in graph.in_edges(self.conditional):
            for k, v in ie.data.assignments.items():
                if k == cond_var:
                    cond_assignment = v
                    break

        if cond_assignment is None:
            if just_var is False:
                if " == " in pycode(cond_code_symexpr):
                    cond_assignment = pycode(cond_code_symexpr)
                    cond_var = f"_if_cond_{self.conditional.label}"
                else:  # Possible an array value is being checked
                    cond_assignment = f"{pycode(cond_code_symexpr)} == 1"
                    cond_var = f"_if_cond_{self.conditional.label}"
            else:
                cond_assignment = pycode(cond_code_symexpr)
                cond_var = f"_if_cond_{self.conditional.label}"
        assert cond_assignment is not None

        return cond_var, cond_assignment

    def _get_prev_state_or_create(self, graph):
        cond_prep_state = None
        srcs = {e.src for e in graph.in_edges(self.conditional) if isinstance(e.src, dace.SDFGState)}
        if len(srcs) > 0:
            cond_prep_state = srcs.pop()
        else:
            cond_prep_state = graph.add_state_before(self.conditional, f"cond_prep_for_fused_{self.conditional}")

        assert cond_prep_state is not None
        return cond_prep_state

    def _generate_identity_write(self, state: SDFGState, arr_name: str, subset: dace.subsets.Range):
        accessed_data = {n.data for n in state.nodes() if isinstance(n, dace.nodes.AccessNode)}
        if arr_name in accessed_data:
            print(
                "Adding_a_new identity assign - even though present. Allowed if the ConditionalBlock only has 1 branch."
            )

        an1 = state.add_access(arr_name)
        an2 = state.add_access(arr_name)

        node_labels = {n.label for n in state.nodes()}
        candidate = "identity_assign_0"
        i = 0
        while candidate in node_labels:
            i += 1
            candidate = f"identity_assign_{i}"

        assign_t = state.add_tasklet(name=candidate, inputs={"_in"}, outputs={"_out"}, code="_out = _in")

        subset_str = ",".join([f"{b}:{e+1}:{s}" for (b, e, s) in subset])

        state.add_edge(an1, None, assign_t, "_in", dace.memlet.Memlet(f"{arr_name}[{subset_str}]"))
        state.add_edge(assign_t, "_out", an2, None, dace.memlet.Memlet(f"{arr_name}[{subset_str}]"))

        return [an1, assign_t, an2]

    ni = 0

    def _split_branches(self, parent_graph: ControlFlowRegion, if_block: ConditionalBlock):
        # Create two new conditional blocks with single branches each
        tup0 = if_block.branches[0]
        tup1 = if_block.branches[1]
        (cond0, body0) = tup0[0], tup0[1]
        (cond1, body1) = tup1[0], tup1[1]

        cond = cond0 if cond0 is not None else cond1
        body = body0 if cond0 is None else body1

        if_block.remove_branch(body)
        assert cond.language == dace.dtypes.Language.Python

        if_out_edges = parent_graph.out_edges(if_block)

        new_if_block = ConditionalBlock(label=f"{if_block.label}_negated", sdfg=parent_graph.sdfg, parent=parent_graph)
        new_if_block.add_branch(condition=CodeBlock(f"(negated_cond_{self.ni}) == 1"), branch=body)

        parent_graph.add_node(new_if_block)

        for oe in if_out_edges:
            oe.src = new_if_block

        parent_graph.add_edge(
            if_block, new_if_block,
            dace.sdfg.InterstateEdge(assignments={f"negated_cond_{self.ni}": f"not ({cond.as_string})"}))
        self.ni += 1

        parent_graph.reset_cfg_list()

        return if_block, new_if_block

    def try_clean(self, graph: ControlFlowRegion, sdfg: SDFG):
        # Some patterns that we can clean

        # Pattern 1
        # If start block is a state and is empty and has assignments only,
        # We can add them before
        assert graph == self.conditional.parent_graph
        assert sdfg == self.conditional.parent_graph.sdfg

        if len(self.conditional.branches) == 1:
            cond, body = self.conditional.branches[0]

            first_start_block = body.start_block
            start_block = body.start_block
            # While empty state with assignments
            while (len(start_block.nodes()) == 0 and isinstance(start_block, dace.SDFGState)
                   and body.out_degree(start_block) == 1):

                assert body.in_degree(start_block) == 0
                # RM and copy assignments
                oe = body.out_edges(start_block)[0]
                assignments = oe.data.assignments
                body.remove_node(start_block)
                # Update start block
                start_block = oe.dst

                # Add state before if cond
                graph.add_state_before(self.conditional,
                                       label=start_block.label + "_p",
                                       assignments=copy.deepcopy(assignments),
                                       is_start_block=graph.start_block == self.conditional)

            if start_block != first_start_block:
                # Rm and Add conditional to get the start block correct
                oes = body.out_edges(start_block)
                cpnode = copy.deepcopy(start_block)
                body.remove_node(start_block)
                body.add_node(cpnode, is_start_block=True)

                for oe in oes:
                    body.add_edge(cpnode, oe.dst, copy.deepcopy(oe.data))

            graph.reset_cfg_list()

            # Pattern 2
            # If all top-level nodes are connected through empty interstate edges
            # And we have a line graph, put each state to the same if condition
            nodes = list(body.bfs_nodes())
            in_degree_leq_one = all({body.in_degree(n) <= 1 for n in nodes})
            out_degree_leq_one = all({body.out_degree(n) <= 1 for n in nodes})
            edges = body.edges()
            all_edges_empty = all({e.data.assignments == dict() for e in edges})

            if in_degree_leq_one and out_degree_leq_one and all_edges_empty:
                # Put all nodes into their own if condition
                node_to_add_after = self.conditional
                # First node gets to stay
                for ci, node in enumerate(nodes[1:]):
                    body.remove_node(node)

                    parent_graph = self.conditional.parent_graph

                    copy_conditional = ConditionalBlock(label=self.conditional.label + f"_v_{ci}",
                                                        sdfg=self.conditional.sdfg,
                                                        parent=parent_graph)
                    cfg = ControlFlowRegion(label=self.conditional.label + f"_v_{ci}_body",
                                            sdfg=self.conditional.sdfg,
                                            parent=copy_conditional)
                    cfg.add_node(copy.deepcopy(node))
                    copy_conditional.add_branch(condition=copy.deepcopy(cond), branch=cfg)

                    parent_graph.add_node(copy_conditional, False, False)

                    for oe in parent_graph.out_edges(node_to_add_after):
                        parent_graph.remove_edge(oe)
                        parent_graph.add_edge(copy_conditional, oe.dst, copy.deepcopy(oe.data))
                    parent_graph.add_edge(node_to_add_after, copy_conditional, InterstateEdge())

                    node_to_add_after = copy_conditional

            graph.sdfg.reset_cfg_list()
            sdutil.set_nested_sdfg_parent_references(graph.sdfg)

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        # If CFG has 1 or two branches
        # If two branches then the write sets to sink nodes are the same

        # Strategy copy the nodes of the states to the new state
        # If we have 1 state we could essentially mimic the same behaviour
        # by making so such that the state only has copies for the writes

        cond_var, cond_assignment = self._extract_condition_var_and_assignment(graph)

        if len(self.conditional.branches) == 2:
            tup0 = self.conditional.branches[0]
            tup1 = self.conditional.branches[1]
            (cond0, body0) = tup0[0], tup0[1]
            (cond1, body1) = tup1[0], tup1[1]

            state0: SDFGState = body0.nodes()[0]
            state1: SDFGState = body1.nodes()[0]

            # Disjoint subsets do not require the combine tasklet.
            # Therefore we need to split:
            # if (cond1) {
            #   body1
            # } else {
            #   body2
            # }
            # into two sequential ifs:
            # if (cond1) {
            #   body1
            # }
            # neg_cond1 = !cond1
            # if (!cond1) {
            #   body2
            # }
            # And thus we can use twice the single branch imlpementaiton
            if self._is_disjoint_subset(state0, state1):  # Then we need to sequentialize branches
                first_if, second_if = self._split_branches(parent_graph=graph, if_block=self.conditional)
                t1 = FuseBranches()
                t1.conditional = first_if
                assert t1.can_be_applied(graph=graph, expr_index=0, sdfg=sdfg)
                t2 = FuseBranches()
                t2.conditional = second_if
                assert t2.can_be_applied(graph=graph, expr_index=0, sdfg=sdfg)
                t1.apply(graph=graph, sdfg=sdfg)
                t2.apply(graph=graph, sdfg=sdfg)
                # Create two single branch SDFGs
                # Then call apply on each one of them
                return

        cond_prep_state = self._get_prev_state_or_create(graph)
        self._move_interstate_assignment_to_state(cond_prep_state, cond_assignment, cond_var)

        if len(self.conditional.branches) == 2:
            tup0 = self.conditional.branches[0]
            tup1 = self.conditional.branches[1]
            (cond0, body0) = tup0[0], tup0[1]
            (cond1, body1) = tup1[0], tup1[1]

            state0: SDFGState = body0.nodes()[0]
            state1: SDFGState = body1.nodes()[0]

            if self._is_disjoint_subset(state0, state1):
                raise NotImplementedError("TODO")

            new_state = dace.SDFGState(f"fused_{state0.label}_and_{state1.label}")
            state0_to_new_state_node_map = cutil.copy_state_contents(state0, new_state)
            state1_to_new_state_node_map = cutil.copy_state_contents(state1, new_state)

            # State1, State0 write sets are data names which should be present in the new state too
            read_sets0, write_sets0 = state0.read_and_write_sets()
            read_sets0, write_sets1 = state1.read_and_write_sets()

            joint_writes = write_sets0.intersection(write_sets1)

            graph.add_node(new_state)
            for ie in graph.in_edges(self.conditional):
                graph.add_edge(ie.src, new_state, copy.deepcopy(ie.data))
            for oe in graph.out_edges(self.conditional):
                graph.add_edge(new_state, oe.dst, copy.deepcopy(oe.data))

            for write in joint_writes:
                state0_write_accesses = self.collect_write_accesses(state0, write)
                state1_write_accesses = self.collect_write_accesses(state1, write)

                state0_write_accesses_in_new_state = {state0_to_new_state_node_map[n] for n in state0_write_accesses}
                state1_write_accesses_in_new_state = {state1_to_new_state_node_map[n] for n in state1_write_accesses}

                # If it was a single branch we support multiple writes (just fuse last one)
                # But this is two branches so we need to ensure single write access nodes
                # We only support multiple-writes if the branch has only an if or else
                # Sequentializing the branches would fix this issue (but would come at a performance cost)
                assert len(state0_write_accesses_in_new_state) == 1, f"len({state0_write_accesses_in_new_state}) != 1"
                assert len(state1_write_accesses_in_new_state) == 1, f"len({state1_write_accesses_in_new_state}) != 1"

                state0_in_new_state_write_access = state0_write_accesses_in_new_state.pop()
                state1_in_new_state_write_access = state1_write_accesses_in_new_state.pop()

                # 1. Add a temporary scalar between state0's write to array (tmp1)
                # 2. Add a temporary scalar between state1's write to array (tmp2)
                # 3. Add the combine tasklet which will be perform float_cond1 * tmp1 + (1- float_cond1) * tmp2
                # 4. Redirect the writes to the access node with the temporary scalar for each each state
                # 5. Rm write of state 1
                # 6. Redirect tmp scalars to the combine tasklet and then that to to the old write
                # And which writes it to the access node coming from state0
                # 7. Rm unused access nodes of state 1
                tmp1_name = "if_body_tmp"
                tmp2_name = "else_body_tmp"
                arr = new_state.sdfg.arrays[state0_in_new_state_write_access.data]
                # 1
                tmp1_name, tmp1_scalar = new_state.sdfg.add_scalar(
                    name=tmp1_name,
                    dtype=arr.dtype,
                    storage=dace.dtypes.StorageType.Default,
                    transient=True,
                    find_new_name=True,
                )
                # 2
                tmp2_name, tmp2_scalar = new_state.sdfg.add_scalar(
                    name=tmp2_name,
                    dtype=arr.dtype,
                    storage=dace.dtypes.StorageType.Default,
                    transient=True,
                    find_new_name=True,
                )
                # 3
                combine_tasklet = new_state.add_tasklet(
                    name=f"combine_branch_values_for_{write}",
                    inputs={"_in_left", "_in_right", "_in_factor"},
                    outputs={"_out"},
                    code=f"_out = (_in_factor * _in_left) + ((1.0 - _in_factor) * _in_right)")
                # 4
                ies = new_state.in_edges(state0_in_new_state_write_access)
                assert len(ies) == 1
                ie = ies[0]
                tmp1_access = new_state.add_access(tmp1_name)
                new_state.add_edge(ie.src, ie.src_conn, tmp1_access, None, dace.memlet.Memlet(f"{tmp1_name}"))
                new_state.remove_edge(ie)
                ies = new_state.in_edges(state1_in_new_state_write_access)
                assert len(ies) == 1
                ie = ies[0]
                tmp2_access = new_state.add_access(tmp2_name)
                new_state.add_edge(ie.src, ie.src_conn, tmp2_access, None, dace.memlet.Memlet(f"{tmp2_name}"))
                # 5
                new_state.remove_edge(ie)
                # 6
                float_cond_access = new_state.add_access("float_" + cond_var)
                for tmp_access, connector in [(tmp1_access, "_in_left"), (tmp2_access, "_in_right"),
                                              (float_cond_access, "_in_factor")]:
                    new_state.add_edge(tmp_access, None, combine_tasklet, connector,
                                       dace.memlet.Memlet(tmp_access.data))

                new_state.add_edge(combine_tasklet, "_out", state0_in_new_state_write_access, None,
                                   dace.memlet.Memlet(state0_in_new_state_write_access.data))
                new_state.remove_node(state1_in_new_state_write_access)

                self._try_simplify_combine_tasklet(new_state, combine_tasklet)
        else:
            assert len(self.conditional.branches) == 1
            tup0 = self.conditional.branches[0]
            (cond0, body0) = tup0[0], tup0[1]

            state0: SDFGState = body0.nodes()[0]
            state1 = SDFGState("tmp_branch", sdfg=state0.sdfg)

            new_state = dace.SDFGState(f"fused_{state0.label}_and_{state1.label}")
            state0_to_new_state_node_map = cutil.copy_state_contents(state0, new_state)

            read_sets0, write_sets0 = state0.read_and_write_sets()
            joint_writes = {
                n
                for n in write_sets0
                if isinstance(state0.sdfg.arrays[n], dace.data.Array) or state0.sdfg.arrays[n].transient is False
            }

            #print(joint_writes)
            #raise Exception(joint_writes)
            new_joint_writes = copy.deepcopy(joint_writes)
            new_reads = dict()
            for write in joint_writes:
                state0_write_accesses = self.collect_write_accesses(state0, write)
                state0_write_accesses_in_new_state = {state0_to_new_state_node_map[n] for n in state0_write_accesses}

                for state0_write_access in state0_write_accesses:
                    ies = state0.in_edges(state0_write_access)
                    assert len(ies) == 1
                    ie = ies[0]
                    if ie.data.data is not None:
                        an1, tasklet, an2 = self._generate_identity_write(state1, write, ie.data.subset)
                        new_reads[state0_write_access] = (write, ie.data, (an1, tasklet, an2))

            # Copy over all identify writes
            state1_to_new_state_node_map = cutil.copy_state_contents(state1, new_state)
            read_sets0, write_sets1 = state1.read_and_write_sets()
            joint_writes = new_joint_writes

            # New joint writes require adding reads to a previously output-only connector
            if new_reads:
                if graph.sdfg.parent_nsdfg_node is not None:
                    parent_nsdfg_node = graph.sdfg.parent_nsdfg_node
                    parent_nsdfg_state = self.parent_nsdfg_state
                    for new_read_name, new_read_memlet, nodes in new_reads.values():
                        if new_read_name not in parent_nsdfg_node.in_connectors:
                            write_edges = set(
                                parent_nsdfg_state.out_edges_by_connector(parent_nsdfg_node, new_read_name))
                            assert len(write_edges) == 1, f"{write_edges} of new_read: {new_read_name}"
                            write_edge = write_edges.pop()
                            write_subset: dace.subsets.Range = write_edge.data.subset
                            assert write_subset.num_elements_exact() == 1
                            cutil.insert_non_transient_data_through_parent_scopes(
                                non_transient_data={write_edge.data.data},
                                nsdfg_node=parent_nsdfg_node,
                                parent_graph=parent_nsdfg_state,
                                parent_sdfg=parent_nsdfg_state.sdfg,
                                add_to_output_too=False,
                                add_with_exact_subset=True,
                                exact_subset=copy.deepcopy(write_subset))

            graph.add_node(new_state)
            for ie in graph.in_edges(self.conditional):
                graph.add_edge(ie.src, new_state, copy.deepcopy(ie.data))
            for oe in graph.out_edges(self.conditional):
                graph.add_edge(new_state, oe.dst, copy.deepcopy(oe.data))
            for write in joint_writes:
                state0_write_accesses = self.collect_write_accesses(state0, write)

                #state0_write_accesses_in_new_state = {state0_to_new_state_node_map[n] for n in state0_write_accesses}

                # For each combining needed, we generate a tasklet on state1
                # Then we copy it over to the new state to combine them together
                # Because of this we check the new_reads and its mapping to the new state
                for i, (state0_write_access) in enumerate(state0_write_accesses):
                    new_read_name, new_read_memlet, nodes = new_reads[state0_write_access]
                    state1_in_new_state_write_access: dace.nodes.AccessNode = state1_to_new_state_node_map[nodes[-1]]
                    state0_in_new_state_write_access = state0_to_new_state_node_map[state0_write_access]
                    assert state0_in_new_state_write_access in new_state.nodes()
                    assert state1_in_new_state_write_access in new_state.nodes()

                    # 1. Add a temporary scalar between state0's write to array (tmp1)
                    # 2. Add a temporary scalar between state1's write to array (tmp2)
                    # 3. Add the combine tasklet which will be perform float_cond1 * tmp1 + (1- float_cond1) * tmp2
                    # 4. Redirect the writes to the access node with the temporary scalar for each each state
                    # 5. Rm write of state 1
                    # 6. Redirect tmp scalars to the combine tasklet and then that to to the old write
                    # And which writes it to the access node coming from state0
                    # 7. Rm unused access nodes of state 1
                    tmp1_name = f"if_body_tmp_{i}"
                    tmp2_name = f"else_body_tmp_{i}"
                    arr = new_state.sdfg.arrays[state0_in_new_state_write_access.data]
                    # 1
                    tmp1_name, tmp1_scalar = new_state.sdfg.add_scalar(
                        name=tmp1_name,
                        dtype=arr.dtype,
                        storage=dace.dtypes.StorageType.Default,
                        transient=True,
                        find_new_name=True,
                    )
                    # 2
                    tmp2_name, tmp2_scalar = new_state.sdfg.add_scalar(
                        name=tmp2_name,
                        dtype=arr.dtype,
                        storage=dace.dtypes.StorageType.Default,
                        transient=True,
                        find_new_name=True,
                    )
                    # 3
                    combine_tasklet = new_state.add_tasklet(
                        name=f"combine_branch_values_for_{write}_{i}",
                        inputs={"_in_left", "_in_right", "_in_factor"},
                        outputs={"_out"},
                        code=f"_out = (_in_factor * _in_left) + ((1.0 - _in_factor) * _in_right)")
                    # 4
                    ies = new_state.in_edges(state0_in_new_state_write_access)
                    assert len(ies) == 1
                    ie = ies[0]
                    tmp1_access = new_state.add_access(tmp1_name)
                    new_state.add_edge(ie.src, ie.src_conn, tmp1_access, None, dace.memlet.Memlet(f"{tmp1_name}"))
                    new_state.remove_edge(ie)
                    ies = new_state.in_edges(state1_in_new_state_write_access)
                    assert len(ies) == 1, f"{ies} | {state1_in_new_state_write_access} | {state1.nodes()}"
                    ie = ies[0]
                    tmp2_access = new_state.add_access(tmp2_name)
                    new_state.add_edge(ie.src, ie.src_conn, tmp2_access, None, dace.memlet.Memlet(f"{tmp2_name}"))
                    # 5
                    new_state.remove_edge(ie)
                    # 6
                    float_cond_access = new_state.add_access("float_" + cond_var)
                    for tmp_access, connector in [(tmp1_access, "_in_left"), (tmp2_access, "_in_right"),
                                                  (float_cond_access, "_in_factor")]:
                        new_state.add_edge(tmp_access, None, combine_tasklet, connector,
                                           dace.memlet.Memlet(tmp_access.data))

                    ies = state0.in_edges(state0_write_access)
                    assert len(ies) == 1
                    ie = ies[0]
                    new_state.add_edge(
                        combine_tasklet, "_out", state0_in_new_state_write_access, None,
                        dace.memlet.Memlet(data=state0_in_new_state_write_access.data,
                                           subset=copy.deepcopy(ie.data.subset)))
                    #new_state.remove_node(state1_in_new_state_write_access)

                    # If we detect a previous right, for correctness we need to connect to that
                    self._connect_rhs_identity_assignment_to_previous_read(new_state=new_state,
                                                                           rhs_access=tmp2_access,
                                                                           data=state0_in_new_state_write_access.data,
                                                                           combine_tasklet=combine_tasklet,
                                                                           skip_set={tmp2_access})

                    #self._try_simplify_combine_tasklet(new_state, combine_tasklet)
                    new_state.remove_node(state1_in_new_state_write_access)

        # If the symbol is not used anymore
        conditional_strs = {cond.as_string for cond, _ in self.conditional.branches if cond is not None}
        conditional_symbols = set()
        graph.remove_node(self.conditional)

        for cond_str in conditional_strs:
            conditional_symbols = conditional_symbols.union(
                {str(s)
                 for s in dace.symbolic.SymExpr(cond_str).free_symbols})

        # Then name says symbols but could be an array too
        for sym_name in conditional_symbols:
            if not symbol_is_used(graph, sym_name):
                remove_symbol_assignments(graph, sym_name)
                if isinstance(graph, dace.SDFG):
                    if sym_name in graph.symbols:
                        graph.remove_symbol(sym_name)
                        if graph.parent_nsdfg_node is not None:
                            if sym_name in graph.parent_nsdfg_node.symbol_mapping:
                                del graph.parent_nsdfg_node.symbol_mapping[sym_name]

        for ie in graph.in_edges(new_state):
            # If ie.src is empty and ie.data.assignments is empty remove ie.src
            if len(ie.data.assignments) == 0 and isinstance(ie.src, dace.SDFGState) and len(ie.src.nodes()) == 0:
                remove_node_redirect_in_edges_to(graph, ie.src, new_state)

        self._try_fuse(graph, new_state, cond_prep_state)

        graph.sdfg.reset_cfg_list()
        sdutil.set_nested_sdfg_parent_references(graph.sdfg)

        graph.sdfg.validate()

    def _find_previous_write(self, state: dace.SDFGState, sink: dace.nodes.Tasklet, data: str,
                             skip_set: Set[dace.nodes.Node]):
        nodes_to_check = {ie.src for ie in state.in_edges(sink) if ie.src not in skip_set}
        while nodes_to_check:
            node_to_check = nodes_to_check.pop()
            if nodes_to_check in skip_set:
                continue
            if isinstance(node_to_check, dace.nodes.AccessNode) and node_to_check.data == data:
                return node_to_check
            nodes_to_check = nodes_to_check.union(
                {ie.src
                 for ie in state.in_edges(node_to_check) if ie.src not in skip_set})
        return None

    def _connect_rhs_identity_assignment_to_previous_read(self, new_state: dace.SDFGState,
                                                          rhs_access: dace.nodes.AccessNode, data: str,
                                                          combine_tasklet: dace.nodes.Tasklet,
                                                          skip_set: Set[dace.nodes.Node]):
        identity_rhs_access = self._find_previous_write(new_state, rhs_access, data, set())
        assert identity_rhs_access is not None
        previous_write = self._find_previous_write(new_state, combine_tasklet, data,
                                                   skip_set.union({identity_rhs_access, rhs_access}))

        if previous_write is not None:
            assert identity_rhs_access != previous_write

            # Rm identity rhs access, rewrite the edge
            assert len(new_state.out_edges(identity_rhs_access)) == 1
            oe = new_state.out_edges(identity_rhs_access)[0]
            new_state.remove_edge(oe)
            assert oe.src_conn is None
            new_state.add_edge(previous_write, oe.src_conn, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))
            if new_state.degree(oe.src) == 0:
                new_state.remove_node(oe.src)
