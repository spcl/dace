import ast
import copy
import numpy
from sympy import Eq, Equality, Function, Integer, preorder_traversal, pycode
import re
import sympy
import dace
from dace import properties, transformation
from dace import InterstateEdge
from dace.dtypes import typeclass
from dace.properties import CodeBlock
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
import dace.sdfg.utils as sdutil
import dace.sdfg.construction_utils as cutil
from typing import Tuple, Set, Union
from dace.sdfg.construction_utils import token_match, token_replace
from dace.symbolic import pystr_to_symbolic
from dace.transformation.interstate.state_fusion import StateFusion
from dace.transformation.passes import FuseStates


def extract_bracket_content(s: str):
    pattern = r"(\w+)\[([^\]]*)\]"
    matches = re.findall(pattern, s)

    extracted = dict()
    for name, content in matches:
        if name in extracted:
            raise Exception("Repeated assignment in interstate edge, not supported")
        extracted[name] = "[" + content + "]"

    # Remove [...] from the string
    cleaned = re.sub(pattern, r"\1", s)

    return cleaned, extracted


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

    return False


def remove_symbol_assignments(graph: ControlFlowRegion, sym_name: str):
    for e in graph.all_interstate_edges():
        new_assignments = dict()
        for k, v in e.data.assignments.items():
            if k != sym_name:
                new_assignments[k] = v
        e.data.assignments = new_assignments


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
        # Parse rhs of itnerstate edge assignment to a symbolic expression
        # array accesses are shown as functions e.g. arr1[i, j] is treated as a function arr1(i, j)
        # do not consider functions that are native python operators as functions (e.g. AND)
        rhs_as_symexpr = dace.symbolic.SymExpr(rhs)
        free_vars = {str(sym)
                     for sym in rhs_as_symexpr.free_symbols}.union({
                         str(node.func)
                         for node in preorder_traversal(rhs_as_symexpr) if isinstance(node, Function)
                     }) - {"AND", "OR", "NOT"}

        # For array accesses such as arr1(i, j) get the dictionary tha tmaps name to accesses {arr1: [i,j]}
        # And remove the array accesses
        cleaned, extracted_subsets = extract_bracket_content(rhs)

        # Collect inputs we need
        arr_inputs = {var for var in free_vars if var in state.sdfg.arrays}
        # Generate the scalar for the float constant
        float_lhs_name, float_lhs = state.sdfg.add_scalar(
            name="float_" + lhs,
            dtype=dace.dtypes.float64,
            storage=dace.dtypes.StorageType.Register,
            transient=True,
            find_new_name=True,
        )

        # For all non-symbol (=array/scalar) inputs replace the name with the connector version
        # Connector version is `_in_{arr_input}_{offset}` (to not repeat the connectors)
        symbol_inputs = {var for var in free_vars if var not in state.sdfg.arrays}
        for i, arr_input in enumerate(arr_inputs):
            cleaned = token_replace(cleaned, arr_input, f"_in_{arr_input}_{i}")

        # cleaned = token_replace(cleaned, lhs, f"_out_float_{lhs}")
        assert arr_inputs.union(symbol_inputs) == free_vars

        tasklet = state.add_tasklet(name=f"ieassign_{lhs}_to_{float_lhs_name}_scalar",
                                    inputs={f"_in_{arr_input}_{i}"
                                            for i, arr_input in enumerate(arr_inputs)},
                                    outputs={f"_out_{float_lhs_name}"},
                                    code=f"_out_{float_lhs_name} = ({cleaned})")

        for i, arr_name in enumerate(arr_inputs):
            an_of_arr_name = {n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == arr_name}
            last_access = None
            if len(an_of_arr_name) == 0:
                last_access = state.add_access(arr_name)
            elif len(an_of_arr_name) == 1:
                last_access = an_of_arr_name.pop()
            else:
                # Get the last sink access node, if exists, use to avoide data races.
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

            # Use interstate edge's access expression if exists, otherwise use [0,...,0]
            new_subset = extracted_subsets.get(arr_name, None)
            if new_subset is None:
                new_subset = "[" + ",".join(["0" for _ in state.sdfg.arrays[arr_name].shape]) + "]"
            #subset_str = interstate_index_to_subset_str(new_subset)
            state.add_edge(last_access, None, tasklet, "_in_" + arr_name + f"_{i}",
                           dace.memlet.Memlet(f"{arr_name}{new_subset}"))

            # Convert boolean to float type
            arr = state.sdfg.arrays[arr_name]
            if arr.dtype == dace.bool:
                arr.dtype = dace.float64

        state.add_edge(tasklet, f"_out_{float_lhs_name}", state.add_access(float_lhs_name), None,
                       dace.memlet.Memlet(float_lhs_name))

        return float_lhs_name

    def _is_disjoint_subset(self, state0: SDFGState, state1: SDFGState) -> bool:
        state0_writes = set()
        state1_writes = set()
        state0_write_subsets = dict()
        state1_write_subsets = dict()
        read_sets0, write_sets0 = state0.read_and_write_sets()
        read_sets1, write_sets1 = state1.read_and_write_sets()
        joint_writes = write_sets0.intersection(write_sets1)

        # Remove ignored writes
        ignores_writes = self.collect_ignored_writes(state0).union(self.collect_ignored_writes(state1))
        joint_writes = joint_writes.difference(ignores_writes)

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
            try:
                all_keys = set(state0_write_subsets) | set(state1_write_subsets)
                intersects = {k: False for k in all_keys}
                for name, subsets0 in state0_write_subsets.items():
                    if name in state1_write_subsets:
                        subsets1 = state1_write_subsets[name]
                    else:
                        subsets1 = set()
                    for other_subset in subsets1:
                        for subset0 in subsets0:
                            print(subset0, type(subset0), other_subset, type(other_subset))
                            if subset0.intersects(other_subset):
                                intersects[name] = True
                for name, subsets1 in state1_write_subsets.items():
                    if name in state0_write_subsets:
                        subsets0 = state0_write_subsets[name]
                    else:
                        subsets0 = set()
                    for other_subset in subsets0:
                        for subset1 in subsets1:
                            print(subset1, type(subset1), other_subset, type(other_subset))
                            if subset1.intersects(other_subset):
                                intersects[name] = True

                if not all(v is False for k, v in intersects.items()):
                    return False
            except Exception as e:
                print(f"Intersects call resulted in an exception: {e}")
                return False

        return True

    def collect_accesses(self, state: dace.SDFGState, name: str):
        """Return all AccessNodes in the state that access a given data name."""
        return {n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == name}

    def collect_write_accesses(self, state: dace.SDFGState, name: str):
        """
        Return AccessNodes that write to a given data name.

        A node is considered a write access if it:
        - Has incoming edges (i.e., receives data)
        - At least one incoming edge carries actual data
        - Either has no outgoing edges, or refers to a non-transient or array type
        """
        accesses = self.collect_accesses(state, name)
        result = set()
        for a in accesses:
            has_incoming_data = any(e.data.data is not None for e in state.in_edges(a))
            array = state.sdfg.arrays[a.data]
            if (state.in_degree(a) > 0 and has_incoming_data
                    and (state.out_degree(a) == 0 or isinstance(array, dace.data.Array) or array.transient is False)):
                result.add(a)
        return result

    def collect_ignored_write_accesses(self, state: dace.SDFGState):
        """
        Return AccessNodes that are *write accesses* but do not meet the criteria
        for considered writes - see collect write access for that is considered a
        write.
        """
        all_write_accesses = {
            a
            for a in state.nodes()
            if isinstance(a, dace.nodes.AccessNode) and state.in_degree(a) > 0 and any(e.data.data is not None
                                                                                       for e in state.in_edges(a))
        }

        considered_write_accesses = {
            a
            for a in all_write_accesses
            if (state.out_degree(a) == 0 or isinstance(state.sdfg.arrays[a.data], dace.data.Array)
                or not state.sdfg.arrays[a.data].transient)
        }

        return all_write_accesses - considered_write_accesses

    def collect_ignored_writes(self, state: dace.SDFGState):
        ignored_write_access_nodes = self.collect_ignored_write_accesses(state)
        return {an.data for an in ignored_write_access_nodes}

    def ignored_accesses_are_reused(self, states: Set[dace.SDFGState]):
        """
        Check if ignored write accesses are used elsewhere in the SDFG.
        Implement it by remove the state and then running and checking the
        read write sets of the SDFG

        Returns:
            True if any ignored data is later read or written in another state.
        """
        ignored_accesses = set()
        for state in states:
            ignored_accesses = ignored_accesses.union(self.collect_ignored_write_accesses(state))
        ignored_data = {a.data for a in ignored_accesses}

        # Work on a copy to safely remove nodes
        copy_sdfg = copy.deepcopy(state.sdfg)
        sdutil.set_nested_sdfg_parent_references(copy_sdfg)

        # Remove all nodes from the target state in the copy
        labels = {(s.label, s.parent_graph.label, s.sdfg.label) for s in states}
        for st in copy_sdfg.all_states():
            label_tuple = (st.label, st.parent_graph.label, st.sdfg.label)
            if label_tuple in labels:
                for n in list(st.nodes()):
                    st.remove_node(n)

        read_set, write_set = copy_sdfg.read_and_write_sets()
        ignored_in_read = read_set & ignored_data
        ignored_in_write = write_set & ignored_data

        if ignored_in_read:
            print(f"[can_be_applied] Ignored data ({ignored_data}) is used elsewhere (read): {ignored_in_read}")
        if ignored_in_write:
            print(f"[can_be_applied] Ignored data ({ignored_data}) is used elsewhere (write): {ignored_in_write}")

        return bool(ignored_in_read or ignored_in_write)

    def symbol_reused_outside_conditional(self, sym_name: str):
        copy_sdfg = copy.deepcopy(self.conditional.sdfg)
        sdutil.set_nested_sdfg_parent_references(copy_sdfg)

        # Remove all nodes from the target state in the copy
        conditional_label_tuple = (self.conditional.label, self.conditional.parent_graph.label
                                   if self.conditional.parent_graph is not None else "", self.conditional.sdfg.label)
        for st in copy_sdfg.all_control_flow_regions():
            label_tuple = (st.label, st.parent_graph.label if st.parent_graph is not None else "", st.sdfg.label)
            if label_tuple == conditional_label_tuple:
                ies = st.parent_graph.in_edges(st)
                oes = st.parent_graph.out_edges(st)
                empty_state = st.parent_graph.add_state(label=f"empty_replacement_{st.label}",
                                                        is_start_block=st.start_block)
                st.parent_graph.remove_node(st)
                for ie in ies:
                    st.parent_graph.add_edge(ie.src, empty_state, copy.deepcopy(ie.data))
                for oe in oes:
                    st.parent_graph.add_edge(empty_state, oe.dst, copy.deepcopy(oe.data))
                break

        print(symbol_is_used(copy_sdfg, sym_name))
        return symbol_is_used(copy_sdfg, sym_name)

    def only_top_level_tasklets(self, graph: ControlFlowRegion):
        checked_at_least_one_tasklet = False
        # Can be applied should ensure this
        assert set(graph.all_states()) == set(graph.nodes())
        for state in graph.all_states():
            # The function to get parent map and loop scopes is expensive so lets try map-libnodes first entry first
            for node in state.nodes():
                if isinstance(node, (dace.nodes.MapEntry, dace.nodes.LibraryNode)):
                    return False
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    parent_maps = cutil.get_parent_map_and_loop_scopes(root_sdfg=graph.sdfg,
                                                                       node=node,
                                                                       parent_state=state)
                    checked_at_least_one_tasklet = True
                    if len(parent_maps) > 0:
                        return False

        # If no tasklet has been checked
        return checked_at_least_one_tasklet

    def add_conditional_write_combination(self, new_state: dace.SDFGState,
                                          state0_in_new_state_write_access: dace.nodes.AccessNode,
                                          state1_in_new_state_write_access: dace.nodes.AccessNode,
                                          cond_var_as_float_name: str, write_name: str, index: int):
        """
        Adds a conditional write combination mechanism to merge outputs from two branches.

        This function creates temporary scalars and a combine tasklet to perform:
        result = float_cond * tmp1 + (1 - float_cond) * tmp2

        Args:
            new_state: The SDFG state where the combination will be added
            state0_write_access: Write access node from state0 (original state)
            state1_write_access: Write access node from state1 (original state)
            state0_in_new_state_write_access: Write access node from state0 in new_state
            state1_in_new_state_write_access: Write access node from state1 in new_state
            cond_var_as_float_name: Name of the float condition variable
            write_name: Base name for the write operation (used in naming)
            index: Index for unique naming of temporaries

        Returns:
            tuple: (combine_tasklet, tmp1_access, tmp2_access, float_cond_access)
        """

        # Prepare write subset for non-tasklet inputs
        ies0 = new_state.in_edges(state0_in_new_state_write_access)
        ies1 = new_state.in_edges(state1_in_new_state_write_access)
        assert len(ies0) == len(ies1)
        assert len(ies0) == 1

        ie0, ie1 = ies0[0], ies1[0]
        assert ie0.data.subset == ie1.data.subset
        write_subset: dace.subsets.Range = ie0.data.subset
        assert write_subset.num_elements_exact() == 1

        # Generate unique names for temporary scalars
        tmp1_name = f"if_body_tmp_{index}"
        tmp2_name = f"else_body_tmp_{index}"

        # Get array information for dtype
        arr = new_state.sdfg.arrays[state0_in_new_state_write_access.data]

        # 1. Add temporary scalar between state0's write to array (tmp1)
        tmp1_name, tmp1_scalar = new_state.sdfg.add_scalar(
            name=tmp1_name,
            dtype=arr.dtype,
            storage=dace.dtypes.StorageType.Default,
            transient=True,
            find_new_name=True,
        )

        # 2. Add temporary scalar between state1's write to array (tmp2)
        tmp2_name, tmp2_scalar = new_state.sdfg.add_scalar(
            name=tmp2_name,
            dtype=arr.dtype,
            storage=dace.dtypes.StorageType.Default,
            transient=True,
            find_new_name=True,
        )

        # 3. Add the combine tasklet which performs: float_cond1 * tmp1 + (1 - float_cond1) * tmp2
        combine_tasklet = new_state.add_tasklet(
            name=f"combine_branch_values_for_{write_name}_{index}",
            inputs={"_in_left", "_in_right", "_in_factor"},
            outputs={"_out"},
            code="_out = (_in_factor * _in_left) + ((1.0 - _in_factor) * _in_right)")

        # 4. Redirect the writes to access nodes with temporary scalars for each state
        # Handle state0's write
        ies = new_state.in_edges(state0_in_new_state_write_access)
        assert len(ies) == 1, f"Expected 1 input edge to state0 write access, got {len(ies)}"
        ie = ies[0]
        tmp1_access = new_state.add_access(tmp1_name)
        new_state.add_edge(ie.src, ie.src_conn, tmp1_access, None, dace.memlet.Memlet(f"{tmp1_name}"))
        new_state.remove_edge(ie)

        # Handle state1's write
        ies = new_state.in_edges(state1_in_new_state_write_access)
        assert len(ies) == 1, f"Expected 1 input edge to state1 write access, got {len(ies)}: {ies}"
        ie = ies[0]
        tmp2_access = new_state.add_access(tmp2_name)
        new_state.add_edge(ie.src, ie.src_conn, tmp2_access, None, dace.memlet.Memlet(f"{tmp2_name}"))

        # 5. Remove write of state1
        new_state.remove_edge(ie)

        # 6. Redirect tmp scalars to the combine tasklet and then to the old write
        float_cond_access = new_state.add_access(cond_var_as_float_name)

        # Connect inputs to combine tasklet
        for tmp_access, connector in [(tmp1_access, "_in_left"), (tmp2_access, "_in_right"),
                                      (float_cond_access, "_in_factor")]:
            new_state.add_edge(tmp_access, None, combine_tasklet, connector, dace.memlet.Memlet(tmp_access.data))

        # Connect combine tasklet output to the final write access
        new_state.add_edge(
            combine_tasklet, "_out", state0_in_new_state_write_access, None,
            dace.memlet.Memlet(data=state0_in_new_state_write_access.data, subset=copy.deepcopy(ie.data.subset)))

        return combine_tasklet, tmp1_access, tmp2_access, float_cond_access

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Works for if-else branches or only if branches
        # Sanity checks for the sdfg and graph parameters
        assert sdfg == self.conditional.sdfg
        assert graph == self.conditional.parent_graph

        if sdfg.parent_nsdfg_node is not None:
            if self.parent_nsdfg_state is None:
                print("[can_be_applied] Nested SDFGs need to provide the parent state of the parent nsdfg node")
                return False

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

            # Need to consist of top level tasklets
            all_top_level0 = self.only_top_level_tasklets(body0)
            if not all_top_level0:
                print(f"[can_be_applied] All tasklets need to be top level. Not the case for body {body0}.")
                return False
            all_top_level1 = self.only_top_level_tasklets(body1)
            if not all_top_level1:
                print(f"[can_be_applied] All tasklets need to be top level. Not the case for body {body1}.")
                return False

            read_sets0, write_sets0 = state0.read_and_write_sets()
            read_sets1, write_sets1 = state1.read_and_write_sets()  # fixed typo

            joint_writes = write_sets0.intersection(write_sets1)
            diff_state0 = write_sets0.difference(write_sets1)
            diff_state1 = write_sets1.difference(write_sets0)

            # For joint writes ensure the write subsets are always the same

            for write in joint_writes:
                state0_accesses = self.collect_accesses(state0, write)
                state1_accesses = self.collect_accesses(state1, write)

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
                        if e.data.data not in state_write_subsets:
                            state_write_subsets[e.data.data] = set()
                        state_write_subsets[e.data.data].add(e.data.subset)

                subsets_disjoint = self._is_disjoint_subset(state0, state1)
                if (len(state0_write_subsets.items()) == 1 and len(state1_write_subsets.items()) == 1):
                    if not subsets_disjoint:
                        # If data are the same ok, otherwise not
                        k1, v1 = next(iter(state0_write_subsets.items()))
                        k2, v2 = next(iter(state1_write_subsets.items()))
                        if k1 != k2 or v1 != v2:
                            return False

                # If there are more than one writes we can't fuse them together without knowing how to order
                # Unless the subset is disjoint
                if (any(len(v) > 1 for k, v in state0_write_subsets.items())
                        or any(len(v) > 1 for k, v in state1_write_subsets.items())):
                    if not subsets_disjoint:
                        #print(state0_write_subsets)
                        #print(state1_write_subsets)
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

            if not permissive:
                if self.ignored_accesses_are_reused({state0, state1}):
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

            # Need to consist of top level tasklets
            all_top_level0 = self.only_top_level_tasklets(body0)
            if not all_top_level0:
                print(f"[can_be_applied] All tasklets need to be top level. Not the case for body {body0}.")
                return False

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

                    state_writes |= {e.data.data for e in state_write_edges}
                    for e in state_write_edges:
                        if e.data.data is None:
                            continue
                        if e.data.subset.num_elements_exact() != 1:
                            print(
                                f"[can_be_applied] All write edges need to have exactly one-element write '{write}' (edge {e} problematic)."
                            )

                if not permissive:
                    if self.ignored_accesses_are_reused({state0}):
                        return False

        if permissive is False:
            if self.condition_has_map_param():
                print(
                    "[can_be_applied] Map parameter is used conditional. This will likely result in out-of-bounds accesses. Enable permissive if you want to risk it"
                )
                return False

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
        cond_code_symexpr = pystr_to_symbolic(cond_code_str, simplify=False)
        #assert len(cond_code_symexpr.free_symbols) == 1, f"{cond_code_symexpr}, {cond_code_symexpr.free_symbols}"

        # Find values assigned to the symbols
        #print("Condition as symexpr:", cond_code_symexpr)
        free_syms = {str(s).strip() for s in cond_code_symexpr.free_symbols if str(s) in graph.sdfg.symbols}
        #print("free_syms:", free_syms)
        sym_val_map = dict()
        symbolic_sym_val_map = dict()
        nodes_to_check = {self.conditional}

        # Do reverse BFS from the sink node to get all possible interstate assignments
        while nodes_to_check:
            node_to_check = nodes_to_check.pop()
            ies = {ie for ie in graph.in_edges(node_to_check)}
            for ie in ies:
                #print(f"Check node: {node_to_check}, edge: {ie.data.assignments}")
                for k, v in ie.data.assignments.items():
                    if k in free_syms and k not in sym_val_map:
                        # in case if Eq((x + 1 > b), 1) sympy will have a problem
                        expr = pystr_to_symbolic(v, simplify=False)
                        #if isinstance(expr, Eq):
                        sym_val_map[k] = v
                        #    symbolic_sym_val_map[k] = expr
                        # else: if not an Equality expression, subs / simplify will cause issue
                        # As relationals can't be integers in Sympy system
                        # But DaCe uses 1 == True, 0 == False
            nodes_to_check = nodes_to_check.union({ie.src for ie in ies})

        # The symbol can in free symbols
        assigned_syms = {k for k in sym_val_map}
        unassigned_syms = free_syms - assigned_syms
        #print("assigned_syms:", assigned_syms)
        #print("unassigned_syms:", unassigned_syms)
        #print("sym_val_map:", sym_val_map)

        # If 1 free symbol, easy it means it is condition variable,
        # otherwise get the left most
        if len(cond_code_symexpr.free_symbols) == 1:
            cond_var = str(next(iter(cond_code_symexpr.free_symbols)))
        else:
            cond_var = cutil.token_split_variable_names(cond_code_str).pop()

        # If the sym_map has any functions, then we need to drop, e.g. array access
        new_sym_val_map = dict()
        for k, v in sym_val_map.items():
            vv = dace.symbolic.SymExpr(v)
            funcs = [e for e in vv.atoms(Function)]
            #print(f"Functions in {v} are {funcs}")
            if len(funcs) == 0:
                new_sym_val_map[str(k)] = str(v)
        sym_val_map = new_sym_val_map

        # Subsitute using replace dict to avoid problems
        self.conditional.replace_dict(sym_val_map)

        #print(
        #    f"Subs {cond_code_symexpr}, with sym map ({sym_val_map}) -> {cond_code_symexpr.subs(sym_val_map)} | {cond_code_symexpr.xreplace(symbolic_sym_val_map)}"
        #)
        #cond_assignment = pycode(cond_code_symexpr.subs(sym_val_map))
        new_conds = {c.as_string for c, b in self.conditional.branches if c is not None}
        new_cond = new_conds.pop()
        #raise Exception(cond_var, new_cond)

        return cond_var, new_cond

    def _generate_identity_write(self, state: SDFGState, arr_name: str, subset: dace.subsets.Range):
        accessed_data = {n.data for n in state.nodes() if isinstance(n, dace.nodes.AccessNode)}
        #if arr_name in accessed_data:
        #    print(
        #        "Adding_a_new identity assign - even though present. Allowed if the ConditionalBlock only has 1 branch."
        #    )

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

        # Get the condition assignment of the if-block to copy the symbol type
        # We add its negation to the new branch (e.g. expr == 0 instead of expr == 1 which is the usual one)
        cond_var, cond_assignment = self._extract_condition_var_and_assignment(if_block)

        new_if_block.add_branch(condition=CodeBlock(f"({cond_assignment}) == 0"), branch=body)

        parent_graph.add_node(new_if_block)

        for oe in if_out_edges:
            parent_graph.remove_edge(oe)
            parent_graph.add_edge(new_if_block, oe.dst, copy.deepcopy(oe.data))

        # Do not use negation assignments={f"{negated_name}": f"not ({cond.as_string})"}
        # Crates issue when simplifying with sympy
        parent_graph.add_edge(if_block, new_if_block, dace.sdfg.InterstateEdge())

        parent_graph.reset_cfg_list()

        return if_block, new_if_block

    def move_interstate_assignments_from_empty_start_states_to_front_of_conditional(self, graph: ControlFlowRegion,
                                                                                    conditional: ConditionalBlock):
        # Pattern 1
        # If start block is a state and is empty and has assignments only,
        # We can add them before
        applied = False

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

            graph.sdfg.reset_cfg_list()
            sdutil.set_nested_sdfg_parent_references(graph.sdfg)
            graph.sdfg.validate()

            applied = True
        return applied

    def duplicate_condition_across_all_top_level_nodes_if_line_graph_and_empty_interstate_edges(
            self, graph: ControlFlowRegion, conditional: ConditionalBlock):
        # Pattern 2
        # If all top-level nodes are connected through empty interstate edges
        # And we have a line graph, put each state to the same if condition
        applied = False
        if len(self.conditional.branches) == 1:
            cond, body = self.conditional.branches[0]

            nodes = list(body.bfs_nodes())
            in_degree_leq_one = all({body.in_degree(n) <= 1 for n in nodes})
            out_degree_leq_one = all({body.out_degree(n) <= 1 for n in nodes})
            edges = body.edges()
            # Can support if they are not fully empty
            all_edges_empty = all({e.data.assignments == dict() for e in edges})

            if in_degree_leq_one and out_degree_leq_one:  #and all_edges_empty:
                # Put all nodes into their own if condition
                node_to_add_after = self.conditional
                # First node gets to stay
                for ci, node in enumerate(nodes[1:]):
                    # Get edge data to copy

                    if not all_edges_empty:
                        cfg_in_edges = body.in_edges(node)
                        assert len(cfg_in_edges) <= 1, f"{cfg_in_edges}"
                        cfg_in_edge = cfg_in_edges[0] if len(cfg_in_edges) == 1 else None
                        cfg_out_edges = body.out_edges(node)
                        assert len(cfg_out_edges) <= 1, f"{cfg_out_edges}"
                        cfg_out_edge = cfg_out_edges[0] if len(cfg_out_edges) == 1 else None

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

                    # Find the edge between the
                    parent_graph.add_edge(node_to_add_after, copy_conditional, InterstateEdge())

                    if not all_edges_empty:
                        if cfg_in_edge is not None:
                            pre_assign = parent_graph.add_state_before(
                                state=copy_conditional,
                                label=f"pre_assign_{copy_conditional.label}",
                                is_start_block=parent_graph.start_block == copy_conditional,
                                assignments=cfg_in_edge.data.assignments)
                        if cfg_out_edge is not None:
                            post_assign = parent_graph.add_state_after(
                                state=copy_conditional,
                                label=f"post_assign_{copy_conditional.label}",
                                is_start_block=False,
                                assignments=cfg_out_edge.data.assignments)
                            node_to_add_after = post_assign
                        else:
                            node_to_add_after = copy_conditional
                    else:
                        node_to_add_after = copy_conditional
                applied = True

                graph.sdfg.reset_cfg_list()
                sdutil.set_nested_sdfg_parent_references(graph.sdfg)
                graph.sdfg.validate()
        return applied

    def sequentialize_if_else_branch_if_disjoint_subsets(self, graph: ControlFlowRegion):
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
        if self.can_be_applied(graph=graph, expr_index=0, sdfg=graph.sdfg, permissive=False):
            if len(self.conditional.branches) == 2:
                tup0 = self.conditional.branches[0]
                tup1 = self.conditional.branches[1]
                (cond0, body0) = tup0[0], tup0[1]
                (cond1, body1) = tup1[0], tup1[1]
                state0: SDFGState = body0.nodes()[0]
                state1: SDFGState = body1.nodes()[0]
                if self._is_disjoint_subset(state0, state1):  # Then we need to sequentialize branches
                    first_if, second_if = self._split_branches(parent_graph=graph, if_block=self.conditional)
                    return first_if, second_if
        return None, None

    def demote_branch_only_symbols_appearing_only_a_single_branch_to_scalars_and_try_fuse(
            self, graph: ControlFlowRegion, sdfg: dace.SDFG):
        applied = False
        for branch, body in self.conditional.branches:
            # 2 states, first state empty and only thing is interstate assignments
            print(
                len(body.nodes()) == 2, all({isinstance(n, dace.SDFGState)
                                             for n in body.nodes()}), len(body.start_block.nodes()), len(body.edges()))
            if (len(body.nodes()) == 2 and all({isinstance(n, dace.SDFGState)
                                                for n in body.nodes()}) and len(body.start_block.nodes()) == 0
                    and len(body.edges()) == 1):

                edge = body.edges()[0]
                # If symbol not used anywhere else
                symbols_reused = False
                symbols_defined = set()
                for k, v in edge.data.assignments.items():
                    symbols_reused |= self.symbol_reused_outside_conditional(k)
                    symbols_defined.add(k)
                    if symbols_reused:
                        break

                #print(symbols_reused)
                print(symbols_defined)

                if len(symbols_defined) > 1:
                    print("Not well tested: More than one symbol in the clean-up is not well tested")
                    continue

                applied = True

                if not symbols_reused:
                    # Then demote all symbols
                    if len(symbols_defined) == 0:
                        # First state is empty then
                        start_block, other_state = list(body.bfs_nodes())[0:2]
                        assert body.start_block == start_block
                        body.remove_node(body.start_block)
                        body.remove_node(other_state)
                        body.add_node(other_state, is_start_block=True)
                        continue
                    assert len(symbols_defined) == 1
                    for i, symbol_str in enumerate(symbols_defined):
                        # It might be that the symbol is not defined (defined through an interstate edge)
                        if symbol_str not in sdfg.symbols:
                            sdfg.add_symbol(symbol_str, dace.float64)
                        sdutil.demote_symbol_to_scalar(sdfg, symbol_str)
                        # Get edges of the first nodes
                        edge0, edge1 = list(body.all_edges(*(list(body.bfs_nodes()))[0:2]))
                        # assert edge0.data.assignments == dict()
                        # assert edge1.data.assignments == dict()
                        for k, v in edge1.data.assignments.items():
                            assert k not in edge0.data.assignments
                            edge0.data.assignments[k] = v

                        # State fusion will fail but we know it is fine
                        # Copy all access nodes to the next state, connect the sink node from prev. state
                        # to the next state
                        body.reset_cfg_list()

                        assignment_state, other_state = list(body.bfs_nodes())[1:3]
                        node_map = cutil.copy_state_contents(assignment_state, other_state)
                        # Multiple symbols -> multiple sink nodes

                        sink_nodes = {n for n in assignment_state.nodes() if assignment_state.out_degree(n) == 0}
                        #print("Sink nodes:", sink_nodes, " of:", assignment_state.nodes())

                        for sink_node in sink_nodes:
                            sink_data = sink_node.data
                            sink_node_in_other_state = node_map[sink_node]

                            # Find matching source nodes with same name
                            source_nodes = {
                                n
                                for n in other_state.nodes() if isinstance(n, dace.nodes.AccessNode)
                                and n.data == sink_data and n not in node_map.values()
                            }

                            # Reconnect edges to the new source node
                            for source_node in source_nodes:
                                out_edges = other_state.out_edges(source_node)
                                for out_edge in out_edges:
                                    other_state.remove_edge(out_edge)
                                    other_state.add_edge(sink_node_in_other_state, out_edge.src_conn, out_edge.dst,
                                                         out_edge.dst_conn, copy.deepcopy(out_edge.data))
                                other_state.remove_node(source_node)

                        # Remove both nodes to change the start block
                        # Old node is not needed enaymore
                        if i != len(symbols_defined) - 1:
                            #body.remove_node(body.start_block)
                            oes = body.out_edges(body.start_block)
                            assert len(oes) == 1
                            #body.remove_node(other_state)
                            body.remove_node(assignment_state)
                            #body.add_node(other_state, is_start_block=False)
                            body.add_edge(body.start_block, other_state, copy.deepcopy(oes.pop().data))
                        else:
                            body.remove_node(body.start_block)
                            body.remove_node(other_state)
                            body.remove_node(assignment_state)
                            body.add_node(other_state, is_start_block=True)
        return applied

    def demote_branch_only_symbols_appering_on_both_branches_to_scalars_and_try_fuse(
            self, graph: ControlFlowRegion, sdfg: dace.SDFG):
        applied = False
        if len(self.conditional.branches) != 2:
            return False
        (cond0, body0), (cond1, body1) = self.conditional.branches
        # 2 states, first state empty and only thing is interstate assignments
        if (len(body0.nodes()) == 2 and all({isinstance(n, dace.SDFGState)
                                             for n in body0.nodes()}) and len(body0.start_block.nodes()) == 0
                and len(body0.edges()) == 1 and len(body1.nodes()) == 2
                and all({isinstance(n, dace.SDFGState)
                         for n in body1.nodes()}) and len(body1.start_block.nodes()) == 0 and len(body1.edges()) == 1):
            edge0 = body0.edges()[0]
            edge1 = body1.edges()[0]

            # If symbol not used anywhere else
            symbols_defined0 = set()
            symbols_defined1 = set()
            for k, v in edge0.data.assignments.items():
                symbols_defined0.add(k)
            for k, v in edge1.data.assignments.items():
                symbols_defined1.add(k)

            if symbols_defined1 != symbols_defined0:
                return False

            symbols_defined = symbols_defined0

            # Then demote all symbols
            for symbol_str in symbols_defined:
                sdutil.demote_symbol_to_scalar(sdfg, symbol_str)

            # Get edges coming and out from the first two nodes
            edge0_0, edge0_1 = list(body0.all_edges(*(list(body0.bfs_nodes()))[0:2]))
            edge1_0, edge1_1 = list(body1.all_edges(*(list(body1.bfs_nodes()))[0:2]))
            assert edge0_0.data.assignments == dict()
            assert edge1_1.data.assignments == dict()
            assert edge0_1.data.assignments == dict()
            assert edge1_0.data.assignments == dict()

            # State fusion will fail but we know it is fine
            # Copy all access nodes to the next state, connect the sink node from prev. state
            # to the next state
            body0.reset_cfg_list()
            body1.reset_cfg_list()

            for body in [body0, body1]:
                #print("CCC", body)
                assignment_state, other_state = list(body.bfs_nodes())[1:3]
                node_map = cutil.copy_state_contents(assignment_state, other_state)
                # Multiple symbols -> multiple sink nodes

                sink_nodes = {n for n in assignment_state.nodes() if assignment_state.out_degree(n) == 0}
                #print("Sink nodes:", sink_nodes, " of:", assignment_state.nodes())

                for sink_node in sink_nodes:
                    sink_data = sink_node.data
                    sink_node_in_other_state = node_map[sink_node]

                    # Find matching source nodes with same name
                    source_nodes = {
                        n
                        for n in other_state.nodes()
                        if isinstance(n, dace.nodes.AccessNode) and n.data == sink_data and n not in node_map.values()
                    }

                    # Reconnect edges to the new source node
                    for source_node in source_nodes:
                        if source_node == sink_node_in_other_state:
                            continue
                        out_edges = other_state.out_edges(source_node)
                        for out_edge in out_edges:
                            other_state.remove_edge(out_edge)
                            other_state.add_edge(sink_node_in_other_state, out_edge.src_conn, out_edge.dst,
                                                 out_edge.dst_conn, copy.deepcopy(out_edge.data))

                        other_state.remove_node(source_node)

                # Remove both nodes to change the start block
                # Old node is not needed enaymore
                body.remove_node(body.start_block)
                body.remove_node(other_state)
                body.remove_node(assignment_state)
                body.add_node(other_state, is_start_block=True)

            FuseStates().apply_pass(self.conditional.sdfg, {})
            applied = True
        return applied

    def try_clean(self, graph: ControlFlowRegion, sdfg: SDFG):
        assert graph == self.conditional.parent_graph
        assert sdfg == self.conditional.parent_graph.sdfg
        # Some patterns that we can clean
        applied = False
        applied |= self.move_interstate_assignments_from_empty_start_states_to_front_of_conditional(
            graph, self.conditional)
        sdfg.validate()
        applied |= self.duplicate_condition_across_all_top_level_nodes_if_line_graph_and_empty_interstate_edges(
            graph, self.conditional)
        sdfg.validate()

        # Implicitly done by can-be-applied
        # self.sequentialize_if_else_branch_if_disjoint_subsets(graph)

        applied |= self.demote_branch_only_symbols_appering_on_both_branches_to_scalars_and_try_fuse(graph, sdfg)
        sdfg.validate()

        applied |= self.demote_branch_only_symbols_appearing_only_a_single_branch_to_scalars_and_try_fuse(graph, sdfg)
        sdfg.validate()

        return applied

    _processed_tasklets = set()

    def make_division_tasklets_safe_for_unconditional_execution(
        self,
        state: dace.SDFGState,
        precision: typeclass,
    ):

        tasklets = {n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and n not in self._processed_tasklets}
        self._processed_tasklets = self._processed_tasklets.union(tasklets)

        def _add_eps(expr_str: str, eps: str):
            eps_node = ast.Name(id=eps, ctx=ast.Load())

            class DivEps(ast.NodeTransformer):

                def visit_BinOp(self, node):
                    self.generic_visit(node)
                    if isinstance(node.op, ast.Div):
                        print(
                            f"Changing {tasklet_code_str} to have +{eps} to avoid NaN/inf floating-point exception and its propagation!"
                        )
                        node.right = ast.BinOp(left=node.right, op=ast.Add(), right=eps_node)
                    return node

            tree = ast.parse(expr_str, mode='exec')
            tree = DivEps().visit(tree)
            return ast.unparse(tree).strip()

        if precision == dace.float64:
            eps = numpy.finfo(numpy.float64).eps
        else:
            eps = numpy.finfo(numpy.float32).eps

        has_division = False
        for tasklet in tasklets:
            tasklet_code_str = tasklet.code.as_string
            if tasklet.code.language != dace.dtypes.Language.Python:
                continue
            e = str(eps)
            new_code = _add_eps(tasklet_code_str, e)
            if new_code != tasklet.code.as_string:
                tasklet_code_str = CodeBlock(new_code)
                tasklet.code = tasklet_code_str
                has_division = True

        return has_division

    def condition_has_map_param(self):
        # Can be applied should ensure this
        root_sdfg = self.conditional.sdfg if self.parent_nsdfg_state is None else self.parent_nsdfg_state.sdfg

        all_parent_maps_and_loops = cutil.get_parent_map_and_loop_scopes(root_sdfg=root_sdfg,
                                                                         node=self.conditional,
                                                                         parent_state=None)

        all_params = set()
        for map_or_loop in all_parent_maps_and_loops:
            if isinstance(map_or_loop, dace.nodes.MapEntry):
                all_params = all_params.union(map_or_loop.map.params)
            else:
                assert isinstance(map_or_loop, LoopRegion)
                all_params.add(map_or_loop.loop_variable)

        graph = self.conditional.parent_graph

        non_none_conds = [cond for cond, _ in self.conditional.branches if cond is not None]
        assert len(non_none_conds) == 1
        cond = non_none_conds.pop()
        cond_code_str = cond.as_string
        cond_code_symexpr = pystr_to_symbolic(cond_code_str, simplify=False)
        #assert len(cond_code_symexpr.free_symbols) == 1, f"{cond_code_symexpr}, {cond_code_symexpr.free_symbols}"

        # Find values assigned to the symbols
        #print("Condition as symexpr:", cond_code_symexpr)
        free_syms = {str(s).strip() for s in cond_code_symexpr.free_symbols if str(s) in graph.sdfg.symbols}

        nodes_to_check = {self.conditional}
        while nodes_to_check:
            node_to_check = nodes_to_check.pop()
            ies = {ie for ie in graph.in_edges(node_to_check)}
            for ie in ies:
                #print(f"Check node: {node_to_check}, edge: {ie.data.assignments}")
                for k, v in ie.data.assignments.items():
                    if k in free_syms:
                        # in case if Eq((x + 1 > b), 1) sympy will have a problem
                        expr = pystr_to_symbolic(v, simplify=False)
                        for new_free_sym in expr.free_symbols:
                            if str(new_free_sym) in graph.sdfg.symbols:
                                free_syms.add(str(new_free_sym))
            nodes_to_check = nodes_to_check.union({ie.src for ie in ies})

        print(f"Free syms related to the condition assignment are: {free_syms}")
        print(f"All parent map and loop iterators are: {all_params}")

        print(f"Foudn folllowing map params: {all_params.intersection(free_syms)}")

        return all_params.intersection(free_syms) != set()

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        # If CFG has 1 or two branches
        # If two branches then the write sets to sink nodes are the same

        # Strategy copy the nodes of the states to the new state
        # If we have 1 state we could essentially mimic the same behaviour
        # by making so such that the state only has copies for the writes
        assert graph == self.conditional.parent_graph
        cond_var, cond_assignment = self._extract_condition_var_and_assignment(graph)
        #print("CV", cond_var, cond_assignment)
        orig_cond_var = cond_var

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
            first_if, second_if = self.sequentialize_if_else_branch_if_disjoint_subsets(graph)

            if first_if is not None and second_if is not None:
                #print("Disjoint subsets - split the branching if-else two separate single state branches and execute")
                #print(f"Applying to first_if: {first_if}")
                t1 = FuseBranches()
                t1.conditional = first_if
                if sdfg.parent_nsdfg_node is not None:
                    t1.parent_nsdfg_state = self.parent_nsdfg_state
                # Right now can_be_applied False because the is reused, but we do not care about
                # this type of a reuse - so call permissive=True
                assert t1.can_be_applied(graph=graph, expr_index=0, sdfg=sdfg, permissive=True)
                t1.apply(graph=graph, sdfg=sdfg)

                t2 = FuseBranches()
                t2.conditional = second_if
                if sdfg.parent_nsdfg_node is not None:
                    t2.parent_nsdfg_state = self.parent_nsdfg_state
                #print(f"Applying to second_if: {second_if}")
                # Right now can_be_applied False because the is reused, but we do not care about
                # this type of a reuse - so call permissive=True
                assert t2.can_be_applied(graph=graph, expr_index=0, sdfg=sdfg, permissive=True)
                t2.apply(graph=graph, sdfg=sdfg)
                # Create two single branch SDFGs
                # Then call apply on each one of them
                return

        cond_prep_state = graph.add_state_before(self.conditional,
                                                 f"cond_prep_for_fused_{self.conditional}",
                                                 is_start_block=graph.in_degree(self.conditional) == 0)
        cond_var_as_float_name = self._move_interstate_assignment_to_state(cond_prep_state, cond_assignment, cond_var)

        if len(self.conditional.branches) == 2:
            tup0 = self.conditional.branches[0]
            tup1 = self.conditional.branches[1]
            (cond0, body0) = tup0[0], tup0[1]
            (cond1, body1) = tup1[0], tup1[1]

            state0: SDFGState = body0.nodes()[0]
            state1: SDFGState = body1.nodes()[0]

            if self._is_disjoint_subset(state0, state1):
                raise Exception("The case shoudl have been handled by branch split")

            new_state = dace.SDFGState(f"fused_{state0.label}_and_{state1.label}")
            state0_to_new_state_node_map = cutil.copy_state_contents(state0, new_state)
            state1_to_new_state_node_map = cutil.copy_state_contents(state1, new_state)

            # State1, State0 write sets are data names which should be present in the new state too
            read_sets0, write_sets0 = state0.read_and_write_sets()
            read_sets0, write_sets1 = state1.read_and_write_sets()

            joint_writes = write_sets0.intersection(write_sets1)

            # Remove ignored writes from the set
            ignored_writes0 = self.collect_ignored_writes(state0)
            ignored_writes1 = self.collect_ignored_writes(state1)
            ignored_writes = ignored_writes0.union(ignored_writes1)
            joint_writes = joint_writes.difference(ignored_writes)

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
                assert len(state0_write_accesses) == 1

                state0_write_access = state0_write_accesses.pop()
                state0_in_new_state_write_access = state0_write_accesses_in_new_state.pop()
                state1_in_new_state_write_access = state1_write_accesses_in_new_state.pop()

                combine_tasklet, tmp1_access, tmp2_access, float_cond_access = self.add_conditional_write_combination(
                    new_state=new_state,
                    state0_in_new_state_write_access=state0_in_new_state_write_access,
                    state1_in_new_state_write_access=state1_in_new_state_write_access,
                    cond_var_as_float_name=cond_var_as_float_name,
                    write_name=write,
                    index=0)

                new_state.remove_node(state1_in_new_state_write_access)
                float_type = new_state.sdfg.arrays[float_cond_access.data].dtype
                has_divisions = self.make_division_tasklets_safe_for_unconditional_execution(new_state, float_type)
                if not has_divisions:
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
            joint_writes = write_sets0.difference(self.collect_ignored_writes(state0))

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

            # New joint writes require adding reads to a previously output-only connector,
            # we nede to do this if the data is not transient
            if new_reads:
                if graph.sdfg.parent_nsdfg_node is not None:
                    parent_nsdfg_node = graph.sdfg.parent_nsdfg_node
                    parent_nsdfg_state = self.parent_nsdfg_state
                    for i, (new_read_name, new_read_memlet, nodes) in enumerate(new_reads.values()):
                        assert new_read_name in graph.sdfg.arrays
                        new_read_is_transient = graph.sdfg.arrays[new_read_name].transient
                        if new_read_is_transient:
                            continue
                        if new_read_name not in parent_nsdfg_node.in_connectors:
                            write_edges = set(
                                parent_nsdfg_state.out_edges_by_connector(parent_nsdfg_node, new_read_name))
                            assert len(write_edges) == 1, f"{write_edges} of new_read: {new_read_name}"
                            write_edge = write_edges.pop()
                            write_subset: dace.subsets.Range = write_edge.data.subset
                            # This is not necessarily true because the subset connection can be the full set
                            # assert write_subset.num_elements_exact() == 1, f"{new_read_name}: {write_subset}: {write_subset.num_elements_exact()}, ()"
                            use_exact_subset = write_subset.num_elements_exact() == 1
                            print(i)
                            cutil.insert_non_transient_data_through_parent_scopes(
                                non_transient_data={write_edge.data.data},
                                nsdfg_node=parent_nsdfg_node,
                                parent_graph=parent_nsdfg_state,
                                parent_sdfg=parent_nsdfg_state.sdfg,
                                add_to_output_too=False,
                                add_with_exact_subset=use_exact_subset,
                                exact_subset=copy.deepcopy(write_subset),
                                nsdfg_connector_name=new_read_name)

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

                    # If state 1 access should have setzero defined to be true to avoid writing trash
                    state1_in_new_state_write_access.setzero = True

                    combine_tasklet, tmp1_access, tmp2_access, float_cond_access = self.add_conditional_write_combination(
                        new_state=new_state,
                        state0_in_new_state_write_access=state0_in_new_state_write_access,
                        state1_in_new_state_write_access=state1_in_new_state_write_access,
                        cond_var_as_float_name=cond_var_as_float_name,
                        write_name=write,
                        index=i)

                    # If we detect a previous right, for correctness we need to connect to that
                    self._connect_rhs_identity_assignment_to_previous_read(new_state=new_state,
                                                                           rhs_access=tmp2_access,
                                                                           data=state0_in_new_state_write_access.data,
                                                                           combine_tasklet=combine_tasklet,
                                                                           skip_set={tmp2_access})

                    new_state.remove_node(state1_in_new_state_write_access)
                    float_type = new_state.sdfg.arrays[float_cond_access.data].dtype

                    has_divisions = self.make_division_tasklets_safe_for_unconditional_execution(new_state, float_type)

                    if not has_divisions:
                        self._try_simplify_combine_tasklet(new_state, combine_tasklet)

        # If the symbol is not used anymore
        conditional_strs = {cond.as_string for cond, _ in self.conditional.branches if cond is not None}
        conditional_symbols = set()
        graph.remove_node(self.conditional)

        for cond_str in conditional_strs:
            conditional_symbols = conditional_symbols.union(
                {str(s)
                 for s in dace.symbolic.SymExpr(cond_str).free_symbols})
        conditional_symbols.add(orig_cond_var)

        # Then name says symbols but could be an array too
        for sym_name in conditional_symbols:
            if not symbol_is_used(graph.sdfg, sym_name):
                remove_symbol_assignments(graph.sdfg, sym_name)
                if sym_name in graph.sdfg.symbols:
                    graph.sdfg.remove_symbol(sym_name)
                    if graph.sdfg.parent_nsdfg_node is not None:
                        if sym_name in graph.sdfg.parent_nsdfg_node.symbol_mapping:
                            del graph.sdfg.parent_nsdfg_node.symbol_mapping[sym_name]

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
