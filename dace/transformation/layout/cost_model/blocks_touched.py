import dace
import sympy as sp
import math
from typing import Dict, List, Optional, Tuple



def build_symbol_map(
    loop_nests: Dict[str, dace.subsets.Range],
    access_subsets: Dict[str, List[dace.subsets.Range]],
) -> Dict[str, sp.Symbol]:
    """name -> SymPy symbol extracted from stored Range objects.
    Must extract rather than create: DaCe attaches assumptions that
    make freshly created symbols distinct objects, breaking .subs()."""
    symbol_map: Dict[str, sp.Symbol] = {}
    all_subsets = list(loop_nests.values()) + [r for rs in access_subsets.values() for r in rs]
    for subset in all_subsets:
        for b, e, s in subset:
            for expr in (b, e, s):
                if hasattr(expr, 'free_symbols'):
                    for fs in expr.free_symbols:
                        symbol_map[str(fs)] = fs
    return symbol_map


def compute_flat_address(subset: dace.subsets.Range, strides: Tuple) -> sp.Expr:
    """flat_address = sum over dims: begin_index * stride"""
    return sp.Add(*[b * stride for (b, e, s), stride in zip(subset.ranges, strides)])


def compute_address_delta(
    loop_symbol: str,
    subset: dace.subsets.Range,
    strides: Tuple,
    symbol_map: Dict[str, sp.Symbol],
    inner_loop_nests: Dict[str, dace.subsets.Range],
) -> sp.Expr:
    """
    Flat address delta when loop_symbol increments by 1.

    Innermost loop: delta = addr(sym=k+1) - addr(sym=k)
    Outer loop:     delta = addr(sym=k+1, inner=lower_bounds)
                          - addr(sym=k,   inner=upper_bounds)
    """
    sym = symbol_map.get(loop_symbol)

    def apply_subs(rng, target_val, inner_subs):
        new_ranges = []
        for (b, e, s) in rng.ranges:
            new_b, new_e = b, e
            for inner_name, inner_val in inner_subs.items():
                inner_sym = symbol_map.get(inner_name)
                if inner_sym is not None:
                    new_b = new_b.subs(inner_sym, inner_val)
                    new_e = new_e.subs(inner_sym, inner_val)
            if sym is not None:
                new_b = new_b.subs(sym, target_val)
                new_e = new_e.subs(sym, target_val)
            new_ranges.append((new_b, new_e, s))
        return dace.subsets.Range(new_ranges)

    inner_at_upper = {name: rng.ranges[0][1] for name, rng in inner_loop_nests.items()}
    inner_at_lower = {name: rng.ranges[0][0] for name, rng in inner_loop_nests.items()}
    next_sym       = sym + 1 if sym is not None else sp.Integer(0)

    curr_addr = compute_flat_address(apply_subs(subset, sym,      inner_at_upper), strides)
    next_addr = compute_flat_address(apply_subs(subset, next_sym, inner_at_lower), strides)

    return sp.simplify(next_addr - curr_addr)


def classify_block_behavior(delta: sp.Expr, block_size: int, has_inner_loops: bool) -> Dict:
    """
    delta = 0         -> 0 new blocks (full reuse)
    delta = k (const) -> 1 block / (block_size / gcd(k, block_size)) iters
    delta = symbolic  -> 1 block/iter (assumed >= block_size)
    """
    if delta == 0:
        return {"avg_new_blocks_per_iter": sp.Integer(0), "label": "full reuse"}

    if delta.is_number:
        abs_delta       = int(abs(delta))
        new_block_every = 1 if has_inner_loops else block_size // math.gcd(abs_delta, block_size)
        return {
            "avg_new_blocks_per_iter": sp.Rational(1, new_block_every),
            "label": f"delta={abs_delta}, 1 block/{new_block_every} iters",
        }

    return {
        "avg_new_blocks_per_iter": sp.Integer(1),
        "label": f"delta={delta} (symbolic), 1 block/iter",
    }


def analyze_block_traffic(
    state: dace.SDFGState,
    loop_nests: Dict[str, dace.subsets.Range],
    access_subsets: Dict[str, List[dace.subsets.Range]],
    block_size: int,
) -> Dict[str, List[Dict]]:
    """
    :param loop_nests:     Ordered dict outermost->innermost, 1D Range per loop.
    :param access_subsets: Array name -> list of Ranges (one per stencil point).
    :param block_size:     Cache line size in elements.
    :return:               Array name -> [{loop, avg_new_blocks_per_iter, label}] per depth.
    """
    arrays       = state.sdfg.arrays
    loop_symbols = list(loop_nests.keys())
    symbol_map   = build_symbol_map(loop_nests, access_subsets)

    print(f"Block size : {block_size} elements")
    print(f"Loop order : {' -> '.join(loop_symbols)}  (outermost -> innermost)\n")

    results: Dict[str, List[Dict]] = {}

    for arr_name, ranges in access_subsets.items():
        if arr_name not in arrays:
            raise KeyError(f"Array '{arr_name}' not found in SDFG")

        strides  = arrays[arr_name].strides
        arr_info = []
        print(f"=== {arr_name} ===")

        for depth, loop_sym in enumerate(loop_symbols):
            inner_syms       = loop_symbols[depth + 1:]
            inner_loop_nests = {s: loop_nests[s] for s in inner_syms}
            firing           = "every iter" if not inner_syms else f"fires when {', '.join(inner_syms)} reset"

            # Use worst-case (largest) delta across all points
            deltas = [compute_address_delta(loop_sym, r, strides, symbol_map, inner_loop_nests) for r in ranges]
            delta  = max(deltas, key=lambda d: abs(d) if d.is_number else sp.oo)

            info = {"loop": loop_sym, **classify_block_behavior(delta, block_size, bool(inner_syms))}
            arr_info.append(info)
            print(f"  [{loop_sym}] {firing}: {info['label']}  ->  avg {info['avg_new_blocks_per_iter']} blocks/iter")

        results[arr_name] = arr_info
        print()

    return results

if __name__ == "__main__":
    N, M, K = dace.symbol("N"), dace.symbol("M"), dace.symbol("K")
    sdfg = dace.SDFG("example")
    sdfg.add_array("A", shape=[M, N], dtype=dace.float64)
    sdfg.add_array("B", shape=[M, N], dtype=dace.float64)
    state = sdfg.add_state("s0")

    loop_nests = {
        "i": dace.subsets.Range([("0", "M-1", "1")]),   # outer
        "j": dace.subsets.Range([("0", "N-1", "1")]),   # inner
    }
    # A[i,j]: good row-major access
    # B[j,i]: column-major access (bad for inner loop j)
    access_subset = {
        "A": [dace.subsets.Range([("i", "i", "1"), ("j", "j", "1")])],
        "B": [dace.subsets.Range([("j", "j", "1"), ("i", "i", "1")])],
    }
    analyze_block_traffic(state, loop_nests, access_subset, block_size=8)
