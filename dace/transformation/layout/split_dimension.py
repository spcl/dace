import dace
from typing import Dict, List, Any, Tuple
from dace.sdfg.graph import Edge, EdgeT
from dace.transformation import pass_pipeline as ppl
from dataclasses import dataclass
import copy
import re
import sympy
from sympy import simplify


@dataclass
class SplitDimensions(ppl.Pass):

    def _block_maps():
        pass

    def _simplify_memlets():
        pass

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes & ppl.Modifies.Memlets & ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def __init__(self, split_map: Dict[str, Tuple[List[bool], List[int]]], verbose: bool = False):
        self._split_map = split_map
        self._verbose = verbose

    def _split_dimension(sefl, arr: dace.data.Data, dim_expr: dace.symbolic.SymExpr | dace.symbolic.symbol | int,
                         factor: int):
        # If we can't divide the block properly we need 1 extra block, therefore use int_ceil
        if isinstance(dim_expr, dace.symbolic.symbol):
            return dace.symbolic.SymExpr(f"int_ceil({dim_expr}, {factor})")
        elif isinstance(dim_expr, int):
            # If the length is an integer, ensure we evaluate an integer
            if dim_expr % factor == 0:
                return (dim_expr // factor)
            else:
                return ((dim_expr // factor) + 1)
        elif isinstance(dim_expr, dace.symbolic.SymExpr):
            # Same threament for SymExpr as Symbol
            divisible = simplify(dim_expr.expr % factor) == 0
            if divisible:
                return dace.symbolic.SymExpr(f"({dim_expr} / {factor})")
            else:
                return dace.symbolic.SymExpr(f"int_ceil({dim_expr}, {factor})")
        else:
            raise ValueError(f"Dimension in array.shape must be int, "
                             f"symbol or symexpr {arr} ({arr.shape}) dimension "
                             f"{dim_expr} is {type(dim_expr)}")

    def _split_range_expr(self, b: dace.symbolic.SymExpr, e: dace.symbolic.SymExpr, s: dace.symbolic.SymExpr,
                          factor: int, is_perfect_match: bool, inner_block_replacement_map: Dict[str, str]):
        step_expr = s / factor
        try:
            int_step_expr = int(step_expr)
            if int_step_expr == 0:
                int_step_expr += 1
            step_expr = int_step_expr
        except Exception as exc:
            step_expr = f"max({step_expr}, 1)"

        if is_perfect_match:
            symbolified_inner_block_replacement_map = {
                dace.symbolic.symbol(k): dace.symbolic.SymExpr(v)
                for k, v in inner_block_replacement_map.items()
            }
            new_b = b.subs(symbolified_inner_block_replacement_map)
            new_e = e.subs(symbolified_inner_block_replacement_map)
            return (dace.symbolic.SymExpr(f"({new_b / factor})"), dace.symbolic.SymExpr(f"({new_e / factor})"),
                    dace.symbolic.SymExpr(f"{step_expr}"))
        else:
            return (dace.symbolic.SymExpr(f"int_floor({b}, {factor})"),
                    dace.symbolic.SymExpr(f"int_floor({e}, {factor})"), dace.symbolic.SymExpr(f"{step_expr}"))

    def _modulo_range_expr(self, b: dace.symbolic.SymExpr, e: dace.symbolic.SymExpr, s: dace.symbolic.SymExpr,
                           factor: int, is_perfect_match: bool, inner_block_replacement_map: Dict[str, str]):
        if is_perfect_match is True:
            symbolified_inner_block_replacement_map = {
                dace.symbolic.symbol(k): dace.symbolic.SymExpr(k) - dace.symbolic.SymExpr(v)
                for k, v in inner_block_replacement_map.items()
            }
            # If the previous access is a range that is not 1 then we overapproximate to full subset?
            range_len = ((e + 1) - b) // s
            if range_len == 1:
                assert e == b
                assert s == 1
                new_b = b.subs(symbolified_inner_block_replacement_map)
                new_e = e.subs(symbolified_inner_block_replacement_map)
                new_s = dace.symbolic.SymExpr(f"{s}")
            else:
                # Simplify a subset that access >1 element to access the whole block. The under approximation
                # won't hurt (hopefully) the data anaylsis,and since for those subsets the start pointer is returned, and we have
                # an access on the previous dimension this should not change the pointer address
                new_b = b.subs(symbolified_inner_block_replacement_map)
                new_e = e.subs(symbolified_inner_block_replacement_map)
                new_s = dace.symbolic.SymExpr(f"{s}")
            return (new_b, new_e, new_s)
        else:
            # If the previous access is a range that is not 1 then we overapproximate to full subset?
            range_len = ((e + 1) - b) // s
            if range_len == 1:
                assert e == b
                assert s == 1
                new_b = dace.symbolic.SymExpr(f"(({b}) % {factor})")
                new_e = dace.symbolic.SymExpr(f"(({e}) % {factor})")
                new_s = dace.symbolic.SymExpr(f"{s}")
            else:
                # Simplify a subset that access >1 element to access the whole block. The under approximation
                # won't hurt (hopefully) the data anaylsis,and since for those subsets the start pointer is returned, and we have
                # an access on the previous dimension this should not change the pointer address
                new_b = dace.symbolic.SymExpr(f"(({b}) % {factor})")
                new_e = dace.symbolic.SymExpr(f"(({e}) % {factor})")
                new_s = dace.symbolic.SymExpr(f"{s}")
            return (new_b, new_e, new_s)

    def _eval_map_range(self, range_exprs, inner_params, outer_map_params):
        """
        range_exprs: list of sympy expressions (e.g., [i, Min(i+8, N), 1])
        inner_params: list of strings, e.g., ["i", "j"]
        """
        # Map string params to symbols
        all_symbols = set().union(*[e.free_symbols for e in range_exprs])
        subs_dict = {}

        # Set inner parameters to 0
        for name in inner_params + outer_map_params:
            sym = dace.symbolic.symbol(name)
            subs_dict[sym] = 0

        # Set all other symbols to INT_MAX
        for sym in all_symbols:
            if sym not in subs_dict:
                subs_dict[sym] = 2**32 - 1

        # Substitute and simplify
        evaluated = [int(dace.symbolic.simplify(e.subs(subs_dict))) for e in range_exprs]
        b, e, s = evaluated
        return ((e + 1) - b), s == 1

    def _is_perfect_block_match(self, state: dace.SDFGState, edge: Edge[EdgeT], node: dace.nodes.Node,
                                block_shape: Tuple[int]):
        if self._verbose:
            print(f"[BlockMatch] Called for edge({edge}), node({node})")
        entry_node = state.entry_node(node)
        if entry_node is None:
            if self._verbose:
                print("[BlockMatch] No inner map found → node is not inside a map.")
            return False, None

        twice_entry_node = state.entry_node(entry_node)
        if twice_entry_node is None:
            if self._verbose:
                print(f"[BlockMatch] No outer map found → inner map not nested. {entry_node} -> {twice_entry_node}")
            return False, None

        if not isinstance(entry_node, dace.nodes.MapEntry) or not isinstance(twice_entry_node, dace.nodes.MapEntry):
            if self._verbose:
                print("[BlockMatch] One of the nodes is not a MapEntry.")
            return False, None

        inner_map_params, inner_map_range = entry_node.map.params, entry_node.map.range
        outer_map_params, outer_map_range = twice_entry_node.map.params, twice_entry_node.map.range

        if len(inner_map_params) != len(block_shape):
            if self._verbose:
                print(f"[BlockMatch] Inner map param count {len(inner_map_params)} "
                      f"≠ block_shape length {len(block_shape)}.")
            return False, None

        for (b, e, s), blk in zip(inner_map_range, block_shape):
            over_range, step_is_one = self._eval_map_range([b, e, s], inner_map_params, outer_map_params)
            if not step_is_one:
                if self._verbose:
                    print(f"[BlockMatch] Inner step size for {b}:{e}:{s} is not 1.")
                return False, None
            if over_range != blk:
                if self._verbose:
                    print(f"[BlockMatch] Inner range {over_range} ≠ block size {blk} "
                          f"for {b}:{e}:{s}. ({inner_map_range}) ({inner_map_params})")
                return False, None

        inner_block_replacement_map = dict()
        for p, (b, e, s), blk in zip(inner_map_params, inner_map_range, block_shape):
            over_range, step_is_one = self._eval_map_range([b, e, s], inner_map_params, outer_map_params)
            assert over_range == blk

            matching_step_size = None
            matching_outer_param = None
            for (ob, oe, os), outer_param in zip(outer_map_range, outer_map_params):
                if str(b) == outer_param:
                    if matching_step_size is not None:
                        if self._verbose:
                            print(f"[BlockMatch] Ambiguous match for {b}: "
                                  f"already matched with step {matching_step_size}.")
                        return False, None
                    matching_step_size = os
                    matching_outer_param = outer_param

            if matching_step_size is None:
                if self._verbose:
                    print(f"[BlockMatch] No outer param matched inner begin {b}.")
                return False, None

            if matching_step_size != blk:
                if self._verbose:
                    print(f"[BlockMatch] Outer step {matching_step_size} ≠ "
                          f"block size {blk} for inner {b}.")
                return False, None

            inner_block_replacement_map[p] = matching_outer_param

        if self._verbose:
            print("[BlockMatch] Perfect block match found.")
        return True, inner_block_replacement_map

    def _split_dimensions(self, arr: dace.data.Data, masks: List[int],
                          factors: List[bool]) -> List[dace.symbolic.symbol | int | dace.symbolic.SymExpr]:
        # Divide the dimensions (appending order will depend on the mode)
        # If mode is FACTOR then keep the dimensions for mask is false, replace dimension with factor, and then append the factor that has been divided
        # If mode is block divide as you iterate, and then append leater
        new_shape = []
        for (dim_len, mask, factor) in zip(arr.shape, masks, factors):
            if mask == False:
                new_shape.append(dim_len)
            else:
                new_shape.append(self._split_dimension(arr, dim_len, factor))
        for (dim_len, mask, factor) in zip(arr.shape, masks, factors):
            if mask == True:
                new_shape.append(factor)

        return new_shape

    def _split_range_expressions(self, subset: dace.subsets.Range, masks: List[int], factors: List[bool],
                                 edge: Edge[EdgeT], state: dace.SDFGState) -> dace.subsets.Range:
        new_range_list = []
        # Let's say we access a array of shape [N] in [0:64:2]
        # (Division is by default floor)
        # And then block with factor 16 then the new array looks like [N/16, 16]
        # and then the subset should look like [0/16:64/16:2/16][0:16:1]
        # Since the access aligns perfectly with subset, this works
        # In case of [0:50:2] then:
        # we get [0/16:50/46:2/16][0:16:2]
        # We need to overapproximate the access on the block as some accesses
        # are over full blocks and some over partial blocks
        #
        # Now consider the case [2:10:1] and we block using a block size of 16
        # [2/16:10/16:1] and [2:10:1] but because we know the offset.
        # so need divide all subsets before
        # Optimization 1: If the range covers the whole dimension, we can also covert the whole dimension
        # TODO
        # If not optimizations fit:
        src = edge.src
        dst = edge.dst
        node = None
        # The node passed to the perfect block match functionc can't be an entry or exit node
        # (preferred nodes are taskelts and access nodes)
        if not isinstance(src, (dace.nodes.EntryNode, dace.nodes.ExitNode)):
            node = src
        elif not isinstance(dst, (dace.nodes.EntryNode, dace.nodes.ExitNode)):
            node = dst
        if node is None and isinstance(dst, dace.nodes.EntryNode):
            node = dst
        if node is None and isinstance(src, dace.nodes.ExitNode):
            node = src
        #print(src, "|", dst, "|", node, "|", edge)

        # Get the block shape from factor and mask
        block_shape = [f for f, m in zip(factors, masks) if m is True]

        if node is None:
            is_perfect_match, repl_map = False, None
        else:
            is_perfect_match, repl_map = self._is_perfect_block_match(state, edge, node, block_shape)

        for ((b, e, s), mask, factor) in zip(subset, masks, factors):
            if mask == False:
                new_range_list.append((b, e, s))
            else:
                new_range_list.append(self._split_range_expr(b, e, s, factor, is_perfect_match, repl_map))
        for ((b, e, s), mask, factor) in zip(subset, masks, factors):
            if mask == True:
                new_b, new_e, new_s = self._modulo_range_expr(b, e, s, factor, is_perfect_match, repl_map)
                new_range_list.append((new_b, new_e, new_s))

        return dace.subsets.Range(new_range_list)

    def _replace_array(self, sdfg: dace.SDFG, arr_name: str, new_dimensions: List):
        arr = sdfg.arrays[arr_name]
        datadesc = copy.deepcopy(arr)
        datadesc.shape = tuple(new_dimensions)
        sdfg.remove_data(arr_name, validate=False)
        sdfg.add_array(
            name=arr_name,
            shape=new_dimensions,
            dtype=datadesc.dtype,
            transient=datadesc.transient,
            storage=datadesc.storage,
            lifetime=datadesc.lifetime,
            alignment=datadesc.alignment,
            debuginfo=datadesc.debuginfo,
            find_new_name=False,
        )

    def _replace_array_recursive(self, sdfg: dace.SDFG, arr_name: str, new_dimensions: List):
        self._replace_array(sdfg, arr_name, new_dimensions)
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    in_map = {ie.data.data: ie.dst_conn for ie in state.in_edges(node)}
                    if arr_name in in_map:
                        self._replace_array_recursive(node.sdfg, in_map[arr_name], new_dimensions)
                    out_map = {oe.data.data: oe.src_conn for oe in state.out_edges(node)}
                    if arr_name in out_map:
                        self._replace_array_recursive(node.sdfg, out_map[arr_name], new_dimensions)

    def _replace_memlets_recursive(self, sdfg: dace.SDFG, arr_name: str, masks, factors):
        # if [M, N, K] becomes [M, N/16, K/32, 16, 32]
        # then access [i, j, k] becomes [i, int_floor(j/16), int_floor(k/32), j%16, k%32]
        for state in sdfg.all_states():
            for edge in state.edges():
                if edge.data is not None and edge.data.data == arr_name:
                    # memlet expression is a range/subset expression
                    # we would have [(b,e,s)], if we have a range then:
                    # we would have [(floor(b/16), floor(e/16), s)]
                    new_range = self._split_range_expressions(edge.data.subset, masks, factors, edge, state)
                    new_memlet = dace.memlet.Memlet(data=edge.data.data,
                                                    subset=new_range,
                                                    other_subset=copy.deepcopy(edge.data.other_subset))
                    edge.data = new_memlet
                # Other subset issue be careful
                if isinstance(edge.dst, dace.nodes.AccessNode) and edge.dst.data == arr_name:
                    if edge.data is not None and edge.data.other_subset is not None:
                        # Update other subset
                        new_other_range = self._split_range_expressions(edge.data.other_subset, masks, factors, edge,
                                                                        state)
                        new_memlet = dace.memlet.Memlet(data=edge.data.data,
                                                        subset=copy.deepcopy(edge.data.subset),
                                                        other_subset=new_other_range)
                        edge.data = new_memlet
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    in_map = {ie.data.data: ie.dst_conn for ie in state.in_edges(node)}
                    if arr_name in in_map:
                        self._replace_memlets_recursive(node.sdfg, in_map[arr_name], masks, factors)
                    out_map = {oe.data.data: oe.src_conn for oe in state.out_edges(node)}
                    if arr_name in out_map:
                        self._replace_memlets_recursive(node.sdfg, out_map[arr_name], masks, factors)

    def _extract_indices(self, expr: str, name: str):
        # Find the part inside name[ ... ]
        m = re.search(rf'{re.escape(name)}\[(.*)\]', expr)
        if not m:
            return []

        inside = m.group(1)

        # Now split by commas that are not inside parentheses
        parts = []
        depth = 0
        current = []
        for ch in inside:
            if ch == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                current.append(ch)
        if current:
            parts.append(''.join(current).strip())

        return parts

    def _simplify_str(self, expr_str: str):
        simplified = []
        try:
            sympy_expr = sympy.sympify(expr_str, evaluate=True)
            simp = sympy.simplify(sympy_expr)
            simplified = str(simp)
        except Exception as ex:
            # If something goes wrong, keep original
            simplified = expr_str
        return simplified

    def _replace_indices(self, expr_str: str, name: str, new_parts: list[str]) -> str:
        """Replace the indices inside name[ ... ] with new_parts."""
        m = re.search(rf'{re.escape(name)}\[(.*)\]', expr_str)
        assert m

        return f"{name}[{', '.join(new_parts)}]"

    def _replace_interstate_edges_recursive(self, sdfg: dace.SDFG, arr_name: str, masks, factors):
        # if [M, N, K] becomes [M, N/16, K/32, 16, 32]
        # then access [i, j, k] becomes [i, int_floor(j/16), int_floor(k/32), j%16, k%32]
        # Also for interstate edges
        for edge in sdfg.all_interstate_edges():
            if edge.data is not None:
                assert isinstance(edge.data, dace.InterstateEdge)
                new_assignments = dict()
                for k, v in edge.data.assignments.items():
                    assert k != arr_name  # We replace arrays, they can't appear on the left side
                    v_split = self._extract_indices(v, arr_name)
                    block_shape = [f for m, f in zip(masks, factors) if m is True]
                    array_shape = sdfg.arrays[arr_name].shape
                    # It is either access to array matching the shape, or it is empty list because
                    # it is another array
                    assert len(v_split) + len(block_shape) == len(array_shape) or len(
                        v_split
                    ) == 0, f"{len(v_split)} (old access string length) + {len(block_shape)} (block length) is not equal to 0 or {len(array_shape)}"
                    if len(v_split) + len(block_shape) == len(array_shape):
                        # Ok, can split
                        new_split = []
                        for i, b in enumerate(v_split):
                            assert len(v_split) == len(masks)
                            assert len(v_split) == len(factors)
                            # Try to simplify access if possible, get a str back
                            new_split.append(self._simplify_str(f"(({b}) // {factors[i]})"))
                        for i, b in enumerate(v_split):
                            if masks[i] is True:
                                new_split.append(self._simplify_str(f"(({b}) % {factors[i]})"))
                        new_assignments[k] = self._replace_indices(expr_str=v, name=arr_name, new_parts=new_split)
                    else:
                        new_split = v_split
                        new_assignments[k] = v

                edge.data.assignments = new_assignments

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    in_map = {ie.data.data: ie.dst_conn for ie in state.in_edges(node)}
                    if arr_name in in_map:
                        self._replace_interstate_edges_recursive(node.sdfg, in_map[arr_name], masks, factors)
                    out_map = {oe.data.data: oe.src_conn for oe in state.out_edges(node)}
                    if arr_name in out_map:
                        self._replace_interstate_edges_recursive(node.sdfg, out_map[arr_name], masks, factors)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        # For each array in the list:
        # We have a mask for the dimensions to split e.g. we have tensor [M, N, K]
        # and the mask [F, T, T] and the corresponding factors [_, 16, 32] then the array will be
        # split to [M, N/16, K/32, 16, 32] if block_size mode is chosen, if factor mode is chosen
        # then it will be split to [M, 16, 32, N/16, K/32].
        # In case N % 16 != 0 then the symbol we need to pad it with (16 - (N % 16)).

        # This we will do on an interstate edge where we pad dimensions
        # But what if storage of the array is persistent / SDFG where it is allocated before padding?
        # Thus we need to replace the size expression E with floor(E / 16)
        # unless we can identify E % 16 = 0

        # 1. First pad dimensions of all arrays
        for array_name, (masks, factors) in self._split_map.items():
            arr = sdfg.arrays[array_name]
            if len(masks) != len(arr.shape) or len(factors) != len(arr.shape):
                raise ValueError("Mask and factors must have the same length as the number of dimensions of the array")
            new_shape = self._split_dimensions(arr, masks, factors)

            # Remove old shape and add the array with the new shape
            self._replace_array_recursive(sdfg, array_name, new_shape)
            self._replace_memlets_recursive(sdfg, array_name, masks, factors)
            self._replace_interstate_edges_recursive(sdfg, array_name, masks, factors)

        return 0
