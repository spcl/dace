import copy
from sympy import pycode
import re

import sympy
import dace
from dace import properties, transformation
from dace.properties import CodeBlock
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
import dace.sdfg.utils as sdutil
import dace.sdfg.construction_utils as cutil
from typing import Tuple, Set, Union

from dace.transformation.interstate.state_fusion import StateFusion


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
            transient=True,
        )
        symbol_inputs = {var for var in free_vars if var not in state.sdfg.arrays}

        for arr_input in arr_inputs:
            cleaned = token_replace(cleaned, arr_input, f"_in_{arr_input}")
        cleaned = token_replace(cleaned, lhs, f"_out_float_{lhs}")

        cleaned_as_symexpr = dace.symbolic.SymExpr(cleaned)

        assert arr_inputs.union(symbol_inputs) == free_vars

        tasklet = state.add_tasklet(name=f"ieassign_{lhs}_to_float_{lhs}_scalar",
                                    inputs={"_in_" + arr_input
                                            for arr_input in arr_inputs},
                                    outputs={"_out_float_" + lhs},
                                    code=f"_out_float_{lhs} = dace.float64({pycode(cleaned_as_symexpr)})")

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
        state.add_edge(tasklet, f"_out_float_{lhs}", state.add_access(f"float_{lhs}"), None,
                       dace.memlet.Memlet(f"float_{lhs}"))

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
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

                # If there are more than one writes we can't fuse them together without knowing how to order
                if len(state0_accesses) > 1 or len(state1_accesses) > 1:
                    print(f"[can_be_applied] Multiple write AccessNodes found for '{write}' in one of the branches.")
                    return False

                for state, accesses in [(state0, state0_accesses), (state1, state1_accesses)]:
                    for access in accesses:
                        arr = state.sdfg.arrays[access.data]
                        if arr.dtype not in dace.dtypes.FLOAT_TYPES:
                            print(f"[can_be_applied] Storage of '{write.data}' is not a floating point type.")
                            return False

                state0_writes = set()
                state1_writes = set()
                for state_writes, state_accesses, state in [(state0_writes, state0_accesses, state0),
                                                            (state1_writes, state1_accesses, state1)]:
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

        print(f"[can_be_applied] to {self.conditional}")
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
                #new_state.add_access(ie.src.data)
                new_state.add_edge(ie.src, ie.src_conn, cp_tasklet, ie.dst_conn, copy.deepcopy(ie.data))
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
            assert just_var is False
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
            return

        an1 = state.add_access(arr_name)
        an2 = state.add_access(arr_name)
        assign_t = state.add_tasklet(name="identity_assign", inputs={"_in"}, outputs={"_out"}, code="_out = _in")

        subset_str = ",".join([f"{b}:{e+1}:{s}" for (b, e, s) in subset])

        state.add_edge(an1, None, assign_t, "_in", dace.memlet.Memlet(f"{arr_name}[{subset_str}]"))
        state.add_edge(assign_t, "_out", an2, None, dace.memlet.Memlet(f"{arr_name}[{subset_str}]"))

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        # If CFG has 1 or two branches
        # If two branches then the write sets to sink nodes are the same

        # Strategy copy the nodes of the states to the new state
        # If we have 1 state we could essentially mimic the same behaviour
        # by making so such that the state only has copies for the writes

        cond_var, cond_assignment = self._extract_condition_var_and_assignment(graph)
        cond_prep_state = self._get_prev_state_or_create(graph)
        self._move_interstate_assignment_to_state(cond_prep_state, cond_assignment, cond_var)

        if len(self.conditional.branches) == 2:
            tup0 = self.conditional.branches[0]
            tup1 = self.conditional.branches[1]
            (cond0, body0) = tup0[0], tup0[1]
            (cond1, body1) = tup1[0], tup1[1]

            state0: SDFGState = body0.nodes()[0]
            state1: SDFGState = body1.nodes()[0]

            new_state = dace.SDFGState(f"fused_{state0.label}_and_{state1.label}")
            state0_to_new_state_node_map = cutil.copy_state_contents(state0, new_state)
            state1_to_new_state_node_map = cutil.copy_state_contents(state1, new_state)

            # State1, State0 write sets are data names which should be present in the new state too
            read_sets0, write_sets0 = state0.read_and_write_sets()
            read_sets0, write_sets1 = state1.read_and_write_sets()

            joint_writes = write_sets0.intersection(write_sets1)
        else:
            assert len(self.conditional.branches) == 1
            tup0 = self.conditional.branches[0]
            (cond0, body0) = tup0[0], tup0[1]

            state0: SDFGState = body0.nodes()[0]
            state1 = SDFGState("tmp_branch", sdfg=state0.sdfg)

            new_state = dace.SDFGState(f"fused_{state0.label}_and_{state1.label}")
            state0_to_new_state_node_map = cutil.copy_state_contents(state0, new_state)

            read_sets0, write_sets0 = state0.read_and_write_sets()
            joint_writes = write_sets0

            for write in joint_writes:
                state0_write_accesses = {
                    n
                    for n in state0.nodes()
                    if isinstance(n, dace.nodes.AccessNode) and n.data == write and state0.in_degree(n) > 0
                }
                state0_write_accesses_in_new_state = {state0_to_new_state_node_map[n] for n in state0_write_accesses}

                assert len(state0_write_accesses_in_new_state) == 1

                state0_in_new_state_write_access = state0_write_accesses_in_new_state.pop()

                ies = new_state.in_edges(state0_in_new_state_write_access)
                assert len(ies) == 1
                ie = ies[0]
                if ie.data.data is not None:
                    self._generate_identity_write(state1, write, ie.data.subset)

            # Copy over all identify writes
            state1_to_new_state_node_map = cutil.copy_state_contents(state1, new_state)
            read_sets0, write_sets1 = state1.read_and_write_sets()

        graph.add_node(new_state)
        for ie in graph.in_edges(self.conditional):
            graph.add_edge(ie.src, new_state, copy.deepcopy(ie.data))
        for oe in graph.out_edges(self.conditional):
            graph.add_edge(new_state, oe.dst, copy.deepcopy(oe.data))

        for write in joint_writes:
            state0_write_accesses = {
                n
                for n in state0.nodes()
                if isinstance(n, dace.nodes.AccessNode) and n.data == write and state0.in_degree(n) > 0
            }
            state1_write_accesses = {
                n
                for n in state1.nodes()
                if isinstance(n, dace.nodes.AccessNode) and n.data == write and state1.in_degree(n) > 0
            }

            state0_write_accesses_in_new_state = {state0_to_new_state_node_map[n] for n in state0_write_accesses}
            state1_write_accesses_in_new_state = {state1_to_new_state_node_map[n] for n in state1_write_accesses}

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
                new_state.add_edge(tmp_access, None, combine_tasklet, connector, dace.memlet.Memlet(tmp_access.data))
            new_state.add_edge(combine_tasklet, "_out", state0_in_new_state_write_access, None,
                               dace.memlet.Memlet(state0_in_new_state_write_access.data))
            new_state.remove_node(state1_in_new_state_write_access)

            self._try_simplify_combine_tasklet(new_state, combine_tasklet)
        graph.sdfg.save("x1.sdfg")
        graph.remove_node(self.conditional)
        graph.sdfg.save("x2.sdfg")
        self._try_fuse(graph, new_state, cond_prep_state)
        graph.sdfg.save("x3.sdfg")
        graph.sdfg.validate()
