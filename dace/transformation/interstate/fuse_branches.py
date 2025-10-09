from collections import defaultdict
from sympy import pycode, re
import dace
from dace import properties, transformation
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
import dace.sdfg.utils as sdutil
import dace.sdfg.construction_utils as cutil
from typing import Tuple, Set


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


def token_replace(code: str, src: str, dst: str) -> str:
    # Split while keeping delimiters
    tokens = re.split(r'([()\[\]])', code)

    # Replace tokens that exactly match src
    tokens = [dst if token.strip() == src else token for token in tokens]

    # Recombine everything
    return ''.join(tokens).strip()


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

@properties.make_properties
@dace.transformation.explicit_cf_compatible
class FuseBranches(transformation.MultiStateTransformation):
    """
    We have the pattern:
    ```
    if (cond){
        out1[address1] = computation1(...)
    } else {
        out1[address2] = computation2(...)
    }
    ```

    If all the write sets in the left and right branches are the same,
    menaing the address1 == address2,
    we can transformation the if branch to:
    ```
    fcond = float(cond)
    out1[address1] = computation1(...) * fcond + (1.0 - fcond) * computation2(...)
    ```

    For single branch case:
    ```
    out1[address1] = computation1(...) * fcond + (1.0 - fcond) * out1[address1]
    ```

    This eliminates branching by duplicating the computation of each branch
    but makes it possible to vectorize the computation.
    """
    conditional = transformation.PatternNode(ConditionalBlock)

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
        )
        symbol_inputs = {var for var in free_vars if var not in state.sdfg.arrays}

        for arr_input in arr_inputs:
            cleaned = token_replace(cleaned, arr_input, f"_in_{arr_input}")
        cleaned = token_replace(cleaned, arr_input, f"_out_float_{lhs}")

        cleaned_as_symexpr = dace.symbolic.SymExpr(cleaned)

        assert arr_inputs.union(symbol_inputs) == free_vars

        tasklet = state.add_tasklet(
            name=f"ieassign_{lhs}_to_float_{lhs}_scalar",
            inputs={"_in_" + arr_input for arr_input in arr_inputs},
            outputs={"_out_float_" + lhs},
            code=pycode(cleaned_as_symexpr)
        )

        for arr_name in arr_inputs:
            an_of_arr_name = {n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == arr_name}
            last_access = None
            if len(an_of_arr_name) == 0:
                last_access = state.add_access(arr_name)
            elif len(an_of_arr_name) == 1:
                last_access = an_of_arr_name.pop()
            else:
                # Get the lass one
                # All access nodes of the same data should be in the same weakly connected component
                # Otherwise it would be code-gen dependent data race
                source_nodes = {n for n in state.nodes() if state.in_degree(n) == 0}
                ordered_an_of_arr_names = dict()

                last_accessess = dict()
                for src_node in source_nodes:
                    ordered_an_of_arr_names[src_node] = list(state.bfs_nodes(src_node))
                    access_to_arr_name = [n for n in ordered_an_of_arr_names[src_node] if isinstance(n, dace.nodes.AccessNode) and n.data == arr_name]
                    if len(access_to_arr_name) > 0:
                        last_accessess[src_node] = access_to_arr_name[-1]

                assert len(last_accessess) == 1
                last_access = next(iter(last_accessess.values()))

            # Create subsetexpr from access expr which [b1,b2,b3]
            # Needs to become [b1:b1:1, b2:b2:1, b3:b3:1]
            subset_str = interstate_index_to_subset_str(extracted_subsets[arr_name])
            state.add_edge(
                last_access, None,
                tasklet, arr_name,
                dace.memlet.Memlet(f"{arr_name}{subset_str}")
            )
        state.add_edge(
            tasklet, f"_out_float_{rhs}",
            state.add_access(f"float_{lhs}"), None,
            dace.memlet.Memlet(f"float_{lhs}")
        )

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        print("C")
        assert False
        raise Exception("x")

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
                state0_accesses = {n for n in state0.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write}
                state1_accesses = {n for n in state1.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write}

                # If there are more than one writes we can't fuse them together without knowing how to order
                if len(state0_accesses) > 1 or len(state1_accesses) > 1:
                    print(f"[can_be_applied] Multiple write AccessNodes found for '{write}' in one of the branches.")
                    return False

                state0_writes = set()
                state1_writes = set()
                for state_writes, state_accesses, state in [
                    (state0_writes, state0_accesses, state0),
                    (state1_writes, state1_accesses, state1)
                ]:
                    state_write_edges = set()
                    for an in state_accesses:
                        state_write_edges |= {e for e in state.in_edges(an) if e.data.data is not None}
                    # If there are multiple write edges again we would need to know the order
                    if len(state_write_edges) > 1:
                        print(f"[can_be_applied] Multiple write edges for '{write}' in one branch.")
                        return False
                    state_writes |= {e.data.data for e in state_write_edges}

                # If the subset of each branch is different then we can't fuse either
                if state0_writes != state1_writes:
                    print(f"[can_be_applied] Write subsets differ for '{write}' between branches.")
                    return False

            # If diff states only have transient scalars or arrays it is probably ok (permissive)
            if diff_state0 or diff_state1:
                if self._check_reuse(sdfg, state0, diff_state0):
                    print(f"[can_be_applied] Branch 0 writes to non-reusable data: {diff_state0}")
                    return False
                if self._check_reuse(sdfg, state1, diff_state1):
                    print(f"[can_be_applied] Branch 1 writes to non-reusable data: {diff_state1}")
                    return False

        elif len(self.conditional.branches) == 1:
            tup0: Tuple[properties.CodeBlock, ControlFlowRegion] = self.conditional.branches[0]
            cond0, body0 = tup0[0], tup0[1]

            # Works if the branch body has a single state
            if len(body0.nodes()) != 1 or not isinstance(body0.nodes()[0], SDFGState):
                print("[can_be_applied] Single branch does not have exactly one SDFGState node.")
                return False

        return True


    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        # If CFG has 1 or two branches
        # If two branches then the write sets to sink nodes are the same

        # Strategy copy the nodes of the states to the new state
        if len(self.conditional.branches) == 2:

            tup0 = self.conditional.branches[0]
            tup1 = self.conditional.branches[1]
            (cond0, body0) = tup0[0], tup0[1]
            (cond1, body1) = tup1[0], tup1[1]
            
            cond_code = cond0 if cond0 is not None else cond1

            state0: SDFGState = body0.nodes()[0]
            state1: SDFGState = body1.nodes()[0]

            new_state = dace.SDFGState(f"fused_{state0.label}_and_{state1.label}")
            state0_to_new_state_node_map = cutil.copy_state_contents(state0, new_state)
            state1_to_new_state_node_map = cutil.copy_state_contents(state1, new_state)

            cond_prep_state = None
            srcs = {e.src for e in graph.in_edges(self.conditional) if isinstance(e.src, dace.SDFGState)}
            if len(srcs) > 0:
                cond_prep_state = srcs.pop()
            else:
                cond_prep_state = graph.add_state_before(self.conditional, f"cond_prep_for_fused_{state0.label}_and_{state1.label}")

            assert cond_prep_state is not None
            # TODO normalize conditions before

            cond_code_str = cond_code.as_string
            cond_code_symexpr = dace.symbolic.SymExpr(cond_code_str)

            assert len(cond_code_symexpr.free_symbols) == 1
            assert (str(next(iter(cond_code_symexpr.free_symbols))) == cond_code_str or
                    (f"({next(iter(cond_code_symexpr.free_symbols))})" == cond_code_str))

            cond_var = str(next(iter(cond_code_symexpr.free_symbols)))
            cond_assignment = None
            for ie in graph.in_edges(self.conditional):
                for k, v in ie.data.assignments:
                    if k == cond_var:
                        cond_assignment = v
                        break
            assert cond_assignment is not None

            self._move_interstate_assignment_to_state(cond_prep_state, cond_assignment, cond_var)
            graph.sdfg.save("a.sdfg")

            raise Exception("uwu")


            # State1, State0 write sets are data names which should be present in the new state too
            read_sets0, write_sets0 = state0.read_and_write_sets()
            read_sets0, write_sets1 = state1.read_and_write_stes()

            joint_writes = write_sets0.intersection(write_sets1)

            for write in joint_writes:
                state0_accesses = {n for n in new_state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write}
                state1_accesses = {n for n in state1.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write}

                state0_accesses_in_new_state = {state0_to_new_state_node_map[n] for n in state0_accesses}
                state1_accesses_in_new_state = {state1_to_new_state_node_map[n] for n in state1_accesses}

                assert len(state0_accesses_in_new_state) == 1
                assert len(state1_accesses_in_new_state) == 1

                state0_in_new_state_access = state0_accesses_in_new_state.pop()
                state1_in_new_state_access = state1_accesses_in_new_state.pop()







            # Need to fuse left and right states

        pass