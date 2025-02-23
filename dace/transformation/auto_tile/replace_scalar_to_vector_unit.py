# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import copy
from typing import Callable
import dace
from dace.properties import DictProperty, Property, SetProperty, make_properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation

import operator


@make_properties
class ReplaceScalarToVectorUnit(transformation.SingleStateTransformation):
    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    supported_operators = DictProperty(
        key_type=type(operator.add),
        value_type=str,
        default={
            operator.add: "add",
        },
    )
    operator_impl = DictProperty(
        key_type=str,
        value_type=str,
        default={"add": "{c} = Add({a},{b})", "gemm": "{d} = GEMM({a},{b},{c})"},
    )
    mmu_size = Property(dtype=int, default=32)
    vector_unit_size = Property(dtype=int, default=32)

    def __init__(self):
        super().__init__()

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.device_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        dev_entry = self.device_map_entry
        dev_exit = state.exit_node(dev_entry)
        nodes_to_check = state.all_nodes_between(dev_entry, dev_exit)

        # If we encounter a tasklet, we need start the check
        found_mult_add_assign_patterns = set()
        for n in nodes_to_check:
            if isinstance(n, dace.nodes.Tasklet):
                assert state.entry_node(n) is not None

                mult_add_assign_pattern, nodetuple = self.is_mult_add_assign_pattern(
                    sdfg, state, n
                )
                if mult_add_assign_pattern:
                    found_mult_add_assign_patterns.add(nodetuple)

        for nodetuple in found_mult_add_assign_patterns:
            n = nodetuple[0]
            entry = state.entry_node(n)
            outer_entry = state.entry_node(entry)

            if len(entry.map.params) == 1 and len(outer_entry.map.params) == 2:
                # Remove inner map and replace the nodes with a GEMM tasklet
                purpose_dict = self.try_deduce_input_purpose(
                    state, outer_entry, entry, nodetuple
                )
                self.device_map_entry.purpose_dict = purpose_dict
                self.add_gemm_tasklet_rm_scalar_unit_nodes(
                    state, entry, nodetuple, purpose_dict
                )
                self.increase_map_step_size(entry, self.mmu_size)
                self.increase_map_step_size(outer_entry, self.mmu_size)
                params_to_update = set(entry.map.params).union(outer_entry.map.params)
                params_to_update = [dace.symbol(p) for p in params_to_update]
                # self.increase_memlet_range(state, entry, params_to_update, factor=self.mmu_size)
                self.propagate_compute_unit(state, dev_entry, entry, "MMU")

        nodes_to_check = state.all_nodes_between(dev_entry, dev_exit)
        found_assign_patterns = set()
        for n in nodes_to_check:
            if isinstance(n, dace.nodes.Tasklet):
                # Assign pattern after mult_add_assign as assign is part of it
                assign_pattern, nodetuple = self.is_add_assign_pattern(sdfg, state, n)

                if assign_pattern:
                    found_assign_patterns.add(nodetuple)

        entry_nodes = set()
        for n, v, dst_tasklet in found_assign_patterns:
            scope_node = state.scope_dict()[n]
            if scope_node is not None and scope_node in entry_nodes:
                raise Exception(
                    "Multiple add-assign patterns in the same first-level scope is not supported currently"
                )

        for nodetuple in found_assign_patterns:
            map_entry = state.scope_dict()[nodetuple[0]]
            vars_used_in_contig_access_dict = self.find_variables_used_in_contig_access(
                sdfg, state, map_entry, nodetuple
            )
            vars_used_in_contig_access = set(
                [k for k, v in vars_used_in_contig_access_dict.items() if v]
            )
            if len(vars_used_in_contig_access) > 1:
                raise Exception(
                    "Multiple variables used in contiguous access is not supported currently"
                )
            elif len(vars_used_in_contig_access) == 0:
                raise Exception(
                    "No variable used in contiguous access is not supported currently"
                )

            # Update the map range
            self.add_non_scalar_unit_tasklet_rm_scalar_unit_nodes(
                state, map_entry, nodetuple, "add", {}
            )
            self.increase_map_step_size(
                map_entry, self.vector_unit_size, params=vars_used_in_contig_access
            )
            self.propagate_compute_unit(state, dev_entry, map_entry, "VECTOR")
            # self.increase_memlet_range(state, map_entry, vars_used_in_contig_access, factor=self.vector_unit_size)

    def propagate_compute_unit(self, state : dace.SDFGState, dev_entry: dace.nodes.Node, first_map: dace.nodes.EntryNode, unit_type: str):
        entry_node = first_map
        distance = 0
        while entry_node:
            # The property is a FrozenSet, so we need to copy it
            if unit_type not in entry_node.computational_units:
                entry_node.computational_units[unit_type] = distance
            else:
                if entry_node.computational_units[unit_type] != distance:
                    raise Exception("Multiple distances to the tasklets requiring same computational units in the same scope is not supported")
            entry_node = state.entry_node(entry_node)
            distance += 1


    def _param_appears(self, param_i, ranges):
        for r in ranges:
            b, e, s = r
            print(b.free_symbols, e.free_symbols, s.free_symbols, param_i)
            if (
                param_i in b.free_symbols
                or param_i in e.free_symbols
                or param_i in s.free_symbols
            ):
                return True
        return False

    def try_deduce_input_purpose(
        self,
        state: dace.SDFGState,
        outer_entry: dace.nodes.MapEntry,
        entry: dace.nodes.MapEntry,
        nodetuple,
    ):
        print(nodetuple)
        purpose_dict = dict()
        param_i, param_j = [dace.symbol(strsym) for strsym in outer_entry.map.params]
        print(type(param_i), type(param_j))
        print(outer_entry, entry)
        for oe in state.out_edges(outer_entry):
            print(oe, oe.data)
            ranges = oe.data.subset
            i_appears = self._param_appears(param_i, ranges)
            j_appears = self._param_appears(param_j, ranges)
            if i_appears and not j_appears:
                purpose_dict[oe.data.data] = "A"
            elif j_appears and not i_appears:
                purpose_dict[oe.data.data] = "B"
            elif i_appears and j_appears:
                purpose_dict[oe.data.data] = "acc"
            else:
                raise Exception("Could not deduce purpose of input")
        for oe in state.out_edges(state.exit_node(outer_entry)):
            ranges = oe.data.subset
            i_appears = self._param_appears(param_i, ranges)
            j_appears = self._param_appears(param_j, ranges)
            if oe.data.data not in purpose_dict:
                purpose_dict[oe.data.data] = "C"
            for oe2 in state.out_edges(oe.dst):
                ranges = oe2.data.subset
                i_appears = self._param_appears(param_i, ranges)
                j_appears = self._param_appears(param_j, ranges)
                if oe.data.data not in purpose_dict:
                    purpose_dict[oe.data.data] = "C"
        return purpose_dict

    def find_variables_used_in_contig_access(
        self, sdfg: SDFG, state: dace.SDFGState, map_entry, nodetuple
    ):
        n, v, dst_tasklet = nodetuple
        print(state.scope_dict())
        scope_node = state.scope_dict()[n]
        assert scope_node == map_entry

        # Find all variables used in the access nodes
        params = [dace.symbol(param) for param in map_entry.map.params]

        map_exit = state.exit_node(map_entry)
        edges = state.all_edges(*state.all_nodes_between(map_entry, map_exit))

        used_in_contig_dimensions = {k: False for k in params}

        for edge in edges:
            _, _, _, _, memlet = edge
            if memlet.data is not None and memlet.subset is not None:
                # Get array
                # If a variable from param is used in an expression
                # at the dimension where stride is == 1, then
                # we have a variable used in contiguous access
                arr = sdfg.arrays[memlet.data]
                strides = arr.strides

                for stride, r in zip(strides, memlet.subset):
                    if stride != 1:
                        continue

                    for sub_expr in r:
                        for param in params:
                            if param in sub_expr.free_symbols:
                                used_in_contig_dimensions[param] = True

        print(used_in_contig_dimensions)
        return used_in_contig_dimensions

    def add_gemm_tasklet_rm_scalar_unit_nodes(
        self, state, map_entry, nodetuple, purpose_dict
    ):
        self.add_non_scalar_unit_tasklet_rm_scalar_unit_nodes(
            state, map_entry, nodetuple, "gemm", purpose_dict
        )

    def add_non_scalar_unit_tasklet_rm_scalar_unit_nodes(
        self,
        state: dace.SDFGState,
        map_entry: dace.nodes.MapEntry,
        nodetuple,
        tasklet_type,
        purpose_dict,
    ):
        map_exit = state.exit_node(map_entry)
        scalar_unit_nodes = state.all_nodes_between(map_entry, map_exit)

        # Already asserted that only one pattern is within a scope
        # Move all out edges of the map entry to the new tasklet, all in edges of map exit
        # need to come from taskelt
        # The range of the memlet will be scaled later

        # Collect inputs and outputs
        ins = set()
        outs = set()
        conn_map = dace.OrderedDict()
        data_map = dace.OrderedDict()
        _i = 0
        for oe in state.out_edges(map_entry):
            vc = oe.dst_conn
            v = oe.dst
            if isinstance(v, dace.nodes.Tasklet):
                if v == nodetuple[0]:
                    conn_id = _i
                    ins.add(vc)
                    conn_map[(v, vc)] = vc
                    assert len(list(state.in_edges_by_connector(v, vc))) == 1
                    data_map[vc] = list(
                        state.in_edges_by_connector(v, vc)
                    )[0].data.data
                    _i += 1
                elif v == nodetuple[2]:
                    ins.add("_in_acc")
                    conn_map[(v, vc)] = "_in_acc"
                    assert len(list(state.in_edges_by_connector(v, vc))) == 1
                    data_map[f"_in_acc"] = list(state.in_edges_by_connector(v, vc))[
                        0
                    ].data.data
                else:
                    raise Exception("uwu1")
        _i = 0
        for ie in state.in_edges(map_exit):
            uc = ie.src_conn
            u = ie.src
            if isinstance(u, dace.nodes.Tasklet):
                if u == nodetuple[-1]:
                    conn_id = _i
                    outs.add(uc)
                    conn_map[(u, uc)] = uc
                    assert len(list(state.out_edges_by_connector(u, uc))) == 1
                    data_map[uc] = list(
                        state.out_edges_by_connector(u, uc)
                    )[0].data.data
                    _i += 1
                else:
                    raise Exception("uwu2")
        purpose_map = dace.OrderedDict({k: purpose_dict[v] for k, v in data_map.items() if v in purpose_dict})
        print(ins, outs, conn_map, data_map, purpose_map)

        if len(purpose_map) != 0:
            assert tasklet_type == "gemm"
            res = None
            a = None
            b = None
            acc = None
            for k, v in purpose_map.items():
                if "out" in k:
                    if v == "acc":
                        res = k
                else:
                    if v == "A":
                        a = k
                    elif v == "B":
                        b = k
                    elif v == "acc":
                        acc = k
            tasklet_template = self.operator_impl[tasklet_type]
            instanciated_tasklet = tasklet_template.format(a=a, b=b, c=acc, d=res)
        else:
            tasklet_template = self.operator_impl[tasklet_type]
            sorted_format_in_dict = {k: v for k in sorted(data_map) if "in" in k}
            sorted_format_out_dict = {k: v for k in sorted(data_map) if "out" in k}
            name = 'a'
            combined_dict = dace.OrderedDict()
            for k, v in sorted_format_in_dict.items():
                combined_dict [name] = k
                name = chr(ord(name) + 1)
            for k, v in sorted_format_out_dict.items():
                combined_dict [name] = k
                name = chr(ord(name) + 1)
            instanciated_tasklet = tasklet_template.format(**combined_dict)
        tasklet = dace.nodes.Tasklet(
            str(tasklet_type),
            ins,
            outs,
            instanciated_tasklet,
        )
        #if len(purpose_map) == 0:
        #    raise Exception(instanciated_tasklet, data_map, purpose_map, conn_map)

        for ic in ins:
            tasklet.add_in_connector(ic)
        for oc in outs:
            tasklet.add_out_connector(oc)

        for oe in state.out_edges(map_entry):
            u, uc, v, vc, memlet = oe
            if isinstance(v, dace.nodes.Tasklet):
                state.add_edge(
                    u,
                    uc,
                    tasklet,
                    conn_map[(v, vc)],
                    copy.deepcopy(memlet),
                )
        for ie in state.in_edges(map_exit):
            u, uc, v, vc, memlet = ie
            if isinstance(u, dace.nodes.Tasklet):
                state.add_edge(
                    tasklet,
                    conn_map[(u, uc)],
                    v,
                    vc,
                    copy.deepcopy(memlet),
                )

        for nd in scalar_unit_nodes:
            if (
                nd == map_entry
                or nd == map_exit
                or nd not in nodetuple
                or nd == tasklet
            ):
                continue
            state.remove_node(nd)

    def increase_memlet_range(
        self, state: dace.SDFGState, entry, params_to_update, factor
    ):
        edges = state.all_edges(*state.all_nodes_between(entry, state.exit_node(entry)))
        for e in edges:
            _, _, _, _, memlet = e
            if memlet.data is not None:
                new_subsets = []
                for r in memlet.subset:
                    _r = copy.deepcopy(r)
                    beg, end, step = _r
                    for p in params_to_update:
                        beg = beg.subs(p, p * factor)
                        end = end.subs(p, p * factor)
                        step = step.subs(p, p * factor)
                    _r = (beg, end, step)
                    new_subsets.append(_r)
                _subset = dace.subsets.Range(new_subsets)
                memlet.subset = _subset

    def increase_map_step_size(self, map_entry, step_size, params=None):
        # Increase the step size of the map
        _map = map_entry.map
        new_ranges = []
        for param, r in zip(_map.params, _map.range):
            if params is None:
                beg, end, step = r
                step *= step_size
                new_ranges.append((beg, end, step))
            else:
                if dace.symbol(param) in params:
                    beg, end, step = r
                    step *= step_size
                    new_ranges.append((beg, end, step))
                else:
                    new_ranges.append(r)
        _map.range = dace.subsets.Range(new_ranges)

    def extract_operators_from_code(self, codelist):
        for code in codelist:
            tree = ast.parse(code)
            operators = dict()
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp):
                    op_type = type(node.op)
                    if op_type in operators:
                        operators[op_type] += 1
                    else:
                        operators[op_type] = 1
            return operators
        return dict()

    def check_is_assignment(self, codelist):
        for code in codelist:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    # Check if the assignment is of the form Name = Name
                    if isinstance(node.targets[0], ast.Name) and isinstance(
                        node.value, ast.Name
                    ):
                        return True
        return False

    def is_add_assign_pattern(self, sdfg, state, n):
        # Pattern is:
        # Tasklet (add) -> AccessNode -> Tasklet (assign)
        # Or tasklet (add) -> Tasklet (assign)
        if not isinstance(n, dace.nodes.Tasklet):
            return False, ()
        operators = self.extract_operators_from_code(n.code.code)
        assert len(operators) <= 1
        # Check if there is an operator
        if len(operators) == 0:
            return False, ()
        if len(operators) == 1:
            # Check if operator is add
            found_operator = list(operators.keys())[0]
            if found_operator == ast.Add:
                # Continue checking next steps
                out_edges = state.out_edges(n)
                # No outgoing edge or too many return false
                if len(out_edges) != 1:
                    return False, ()
                v = out_edges[0].dst
                dst_tasklet = None
                # If access node contiue to next step
                if isinstance(v, dace.nodes.AccessNode):
                    out_edges = state.out_edges(v)
                    if len(out_edges) != 1:
                        return False
                    vv = out_edges[0].dst
                    if isinstance(vv, dace.nodes.Tasklet):
                        dst_tasklet = vv
                    else:
                        return False, ()
                else:
                    if not isinstance(v, dace.nodes.Tasklet):
                        return False, ()
                    else:
                        dst_tasklet = v
                print(self.check_is_assignment(dst_tasklet.code.code))
                if self.check_is_assignment(dst_tasklet.code.code):
                    return True, (n, v, dst_tasklet)
                else:
                    return False, ()
        return False, ()

    def is_mult_add_assign_pattern(self, sdfg, state, n):
        # Pattern is:
        # Tasklet (mult) -> AccessNode -> Add-Assign pattern
        # Or Tasklet (mult) -> Add-Assign pattern
        if not isinstance(n, dace.nodes.Tasklet):
            return False, ()
        operators = self.extract_operators_from_code(n.code.code)
        assert len(operators) <= 1
        # Check if there is an operator
        if len(operators) == 0:
            return False, ()
        if len(operators) == 1:
            # Check if operator is mult
            found_operator = list(operators.keys())[0]
            if found_operator == ast.Mult:
                # Continue checking next steps
                out_edges = state.out_edges(n)
                # No outgoing edge or too many return false
                if len(out_edges) != 1:
                    return False, ()
                v = out_edges[0].dst
                dst_tasklet = None
                # If access node contiue to next step
                if isinstance(v, dace.nodes.AccessNode):
                    out_edges = state.out_edges(v)
                    if len(out_edges) != 1:
                        return False
                    vv = out_edges[0].dst
                    if isinstance(vv, dace.nodes.Tasklet):
                        dst_tasklet = vv
                    else:
                        return False, ()
                    add_assign_pattern, node_tuple = self.is_add_assign_pattern(
                        sdfg, state, dst_tasklet
                    )
                    if add_assign_pattern:
                        return True, (n, v, *node_tuple)
                    else:
                        return False, ()
                else:
                    if not isinstance(v, dace.nodes.Tasklet):
                        return False, ()
                    else:
                        dst_tasklet = v
                        add_assign_pattern, node_tuple = self.is_add_assign_pattern(
                            sdfg, state, dst_tasklet
                        )
                        if add_assign_pattern:
                            return True, (n, None, *node_tuple)
                        else:
                            return False, ()
        return False, ()

    @staticmethod
    def annotates_memlets():
        return True
