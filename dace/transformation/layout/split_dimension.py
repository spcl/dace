import dace
from typing import Dict, List, Any, Tuple
from dace.transformation import pass_pipeline as ppl
from dataclasses import dataclass
import copy
from enum import Enum, auto
from sympy import Eq, simplify

@dataclass
class SplitDimensions(ppl.Pass):
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes & ppl.Modifies.Memlets & ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def __init__(self,
                 split_map: Dict[str, Tuple[List[int], List[int]]],
                ):
        self._split_map = split_map

    def _split_dimension(sefl, arr: dace.data.Data,
                         dim_expr: dace.symbolic.SymExpr | dace.symbolic.symbol | int,
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

    def _split_range_expr(self,
                          b: dace.symbolic.SymExpr,
                          e: dace.symbolic.SymExpr,
                          s: dace.symbolic.SymExpr,
                          factor: int):
        step_expr = s / factor
        try:
            int_step_expr = int(step_expr)
            if int_step_expr == 0:
                int_step_expr += 1
            step_expr = int_step_expr
        except Exception as exc:
            step_expr = f"max({step_expr}, 1)"

        return (
            dace.symbolic.SymExpr(f"int_floor({b}, {factor})"),
            dace.symbolic.SymExpr(f"int_floor({e}, {factor})"),
            dace.symbolic.SymExpr(f"{step_expr}")
        )

    def _module_range_expr(self,
                          b: dace.symbolic.SymExpr,
                          e: dace.symbolic.SymExpr,
                          s: dace.symbolic.SymExpr,
                          factor: int):
        # If the previous access is a range that is not 1 then we overapproximate to full subset?
        range_len = ((e+1)-b)//s
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
            # new_b = dace.symbolic.SymExpr(f"0")
            # new_e = dace.symbolic.SymExpr(f"{factor} - 1")
            # new_s = dace.symbolic.SymExpr(f"1")
            new_b = dace.symbolic.SymExpr(f"(({b}) % {factor})")
            new_e = dace.symbolic.SymExpr(f"(({e}) % {factor})")
            new_s = dace.symbolic.SymExpr(f"{s}")
        return (
            new_b,
            new_e,
            new_s
        )

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

    def _split_range_expressions(self, subset: dace.subsets.Range, masks: List[int],
                         factors: List[bool]) -> dace.subsets.Range:
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
        for ((b,e,s), mask, factor) in zip(subset, masks, factors):
            if mask == False:
                new_range_list.append((b,e,s))
            else:
                new_range_list.append(self._split_range_expr(b, e, s, factor))
        for ((b,e,s), mask, factor) in zip(subset, masks, factors):
            if mask == True:
                new_b, new_e, new_s = self._module_range_expr(b, e, s, factor)
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
                    in_map = {ie.data.data : ie.dst_conn for ie in state.in_edges(node)}
                    if arr_name in in_map:
                        self._replace_array_recursive(node.sdfg, in_map[arr_name], new_dimensions)
                    out_map = {oe.data.data : oe.src_conn for oe in state.out_edges(node)}
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
                    new_range = self._split_range_expressions(edge.data.subset, masks, factors)
                    new_memlet = dace.memlet.Memlet(data=edge.data.data, subset=new_range, other_subset=copy.deepcopy(edge.data.other_subset))
                    edge.data = new_memlet
                # Other subset issue be careful
                if isinstance(edge.dst, dace.nodes.AccessNode) and edge.dst.data == arr_name:
                    if edge.data is not None and edge.data.other_subset is not None:
                        # Update other subset
                        new_other_range = self._split_range_expressions(edge.data.other_subset, masks, factors)
                        new_memlet = dace.memlet.Memlet(data=edge.data.data, subset=copy.deepcopy(edge.data.subset), other_subset=new_other_range)
                        edge.data = new_memlet
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    in_map = {ie.data.data : ie.dst_conn for ie in state.in_edges(node)}
                    if arr_name in in_map:
                        self._replace_memlets_recursive(node.sdfg, in_map[arr_name], masks, factors)
                    out_map = {oe.data.data : oe.src_conn for oe in state.out_edges(node)}
                    if arr_name in out_map:
                        self._replace_memlets_recursive(node.sdfg, out_map[arr_name], masks, factors)


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
            #self._replace_interstate_edges_recursive(sdfg, array_name, new_shape)

        return 0

