import dace
import sympy as sp
import math
from typing import Dict, List, Optional, Tuple

# A flat address interval [lo, hi] (inclusive, in elements)
Interval = Tuple[sp.Expr, sp.Expr]


# ---------------------------------------------------------------------------
# Symbol utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Flat address interval computation
# ---------------------------------------------------------------------------

def compute_flat_interval(
    subset: dace.subsets.Range,
    strides: Tuple,
) -> Interval:
    """[begin, end] flat addresses for a single Range access."""
    lo = sp.Add(*[b * st for (b, e, s), st in zip(subset.ranges, strides)])
    hi = sp.Add(*[e * st for (b, e, s), st in zip(subset.ranges, strides)])
    return (sp.expand(lo), sp.expand(hi))


def substitute_into_interval(
    interval: Interval,
    substitutions: Dict[sp.Symbol, sp.Expr],
) -> Interval:
    """Apply sym -> val substitutions to both endpoints."""
    lo, hi = interval
    for sym, val in substitutions.items():
        lo = lo.subs(sym, val)
        hi = hi.subs(sym, val)
    return (sp.expand(lo), sp.expand(hi))


def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    """
    Merge a list of flat address intervals into disjoint sorted intervals.
    Only merges intervals whose bounds are numeric (can be compared).
    Symbolic intervals are kept as-is.
    """
    numeric, symbolic = [], []
    for iv in intervals:
        if iv[0].is_number and iv[1].is_number:
            numeric.append((int(iv[0]), int(iv[1])))
        else:
            symbolic.append(iv)

    numeric.sort()
    merged_numeric: List[Tuple[int, int]] = []
    for lo, hi in numeric:
        if merged_numeric and lo <= merged_numeric[-1][1] + 1:
            merged_numeric[-1] = (merged_numeric[-1][0], max(merged_numeric[-1][1], hi))
        else:
            merged_numeric.append((lo, hi))

    result: List[Interval] = [(sp.Integer(lo), sp.Integer(hi)) for lo, hi in merged_numeric]
    result.extend(symbolic)
    return result


def intervals_new_elements(
    curr: List[Interval],
    next_: List[Interval],
) -> List[Interval]:
    """
    Intervals in next_ that are not covered by curr.
    For symbolic intervals: if an interval shifted by a constant delta,
    the new portion is determined by that delta.
    Falls back to returning next_ unchanged for fully symbolic cases.
    """
    new_parts: List[Interval] = []
    for n_lo, n_hi in next_:
        # Try to find a matching curr interval (same symbolic structure, shifted)
        covered = False
        for c_lo, c_hi in curr:
            # Compute symbolic difference of endpoints
            lo_diff = sp.simplify(n_lo - c_lo)
            hi_diff = sp.simplify(n_hi - c_hi)
            if lo_diff == hi_diff and lo_diff.is_number:
                # Same interval shifted by a constant
                shift = int(lo_diff)
                if shift == 0:
                    covered = True  # exact same interval, nothing new
                    break
                elif shift > 0:
                    # New portion is the top `shift` elements
                    new_lo = sp.expand(c_hi + 1)
                    new_hi = n_hi
                    new_parts.append((new_lo, new_hi))
                    covered = True
                    break
                else:
                    # Shifted left: new portion at the bottom
                    new_lo = n_lo
                    new_hi = sp.expand(c_lo - 1)
                    new_parts.append((new_lo, new_hi))
                    covered = True
                    break
        if not covered:
            new_parts.append((n_lo, n_hi))
    return new_parts


def count_blocks(intervals: List[Interval], block_size: int) -> sp.Expr:
    """
    Count cache blocks touched by a list of flat address intervals.

    Numeric interval [lo, hi]: ceil((hi+1)/B) - floor(lo/B)  blocks
    Symbolic interval:         1 block assumed (conservative lower bound;
                               upper bound would require range size / B)
    """
    total: sp.Expr = sp.Integer(0)
    for lo, hi in intervals:
        if lo.is_number and hi.is_number:
            lo_i, hi_i = int(lo), int(hi)
            blocks = (hi_i // block_size) - (lo_i // block_size) + 1
            total += sp.Integer(blocks)
        else:
            # Symbolic: each distinct symbolic interval assumed 1 new block
            total += sp.Integer(1)
    return total


# ---------------------------------------------------------------------------
# Core per-loop analysis
# ---------------------------------------------------------------------------

def build_substitutions(
    target_sym: Optional[sp.Symbol],
    target_val: sp.Expr,
    inner_loop_nests: Dict[str, dace.subsets.Range],
    symbol_map: Dict[str, sp.Symbol],
    inner_at_upper: bool,
) -> Dict[sp.Symbol, sp.Expr]:
    """Build {symbol: value} substitution dict for curr or next iteration."""
    subs: Dict[sp.Symbol, sp.Expr] = {}
    for inner_name, inner_rng in inner_loop_nests.items():
        inner_sym = symbol_map.get(inner_name)
        if inner_sym is not None:
            bound_idx = 1 if inner_at_upper else 0  # 1=end(ub), 0=begin(lb)
            subs[inner_sym] = inner_rng.ranges[0][bound_idx]
    if target_sym is not None:
        subs[target_sym] = target_val
    return subs


def compute_new_blocks_for_loop(
    loop_symbol: str,
    depth: int,
    loop_symbols: List[str],
    loop_nests: Dict[str, dace.subsets.Range],
    ranges: List[dace.subsets.Range],
    strides: Tuple,
    symbol_map: Dict[str, sp.Symbol],
    block_size: int,
) -> sp.Expr:
    """
    New cache blocks loaded when loop_symbol increments by 1.

    curr state: loop_symbol=k,   inner vars at upper bounds (end of inner sweep)
    next state: loop_symbol=k+1, inner vars at lower bounds (start of inner sweep)

    For the innermost loop: inner_loop_nests is empty, so only loop_symbol changes.
    """
    sym = symbol_map.get(loop_symbol)
    inner_syms = loop_symbols[depth + 1:]
    inner_loop_nests = {s: loop_nests[s] for s in inner_syms}

    curr_subs = build_substitutions(sym, sym,         inner_loop_nests, symbol_map, inner_at_upper=True)
    next_subs = build_substitutions(sym, sym + 1 if sym is not None else sp.Integer(0),
                                    inner_loop_nests, symbol_map, inner_at_upper=False)

    curr_intervals = merge_intervals([
        substitute_into_interval(compute_flat_interval(r, strides), curr_subs)
        for r in ranges
    ])
    next_intervals = merge_intervals([
        substitute_into_interval(compute_flat_interval(r, strides), next_subs)
        for r in ranges
    ])

    new_parts = intervals_new_elements(curr_intervals, next_intervals)
    return count_blocks(new_parts, block_size)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def analyze_block_traffic(
    state: dace.SDFGState,
    loop_nests: Dict[str, dace.subsets.Range],
    access_subsets: Dict[str, List[dace.subsets.Range]],
    block_size: int,
) -> Dict[str, List[Dict]]:
    """
    Cache block traffic for every array at every loop level.

    :param loop_nests:     Ordered dict outermost->innermost, 1D Range per loop.
    :param access_subsets: Array name -> list of Ranges (one per stencil point).
    :param block_size:     Cache line size in elements.
    :return:               Array name -> list of {loop, new_blocks, firing} per depth.
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
        print(f"=== {arr_name}  strides={strides} ===")

        for depth, loop_sym in enumerate(loop_symbols):
            inner_syms = loop_symbols[depth + 1:]
            firing = "every iter" if not inner_syms else (
                "fires when " + ", ".join(f"{s}: ub->lb" for s in inner_syms)
            )
            new_blocks = compute_new_blocks_for_loop(
                loop_sym, depth, loop_symbols, loop_nests,
                ranges, strides, symbol_map, block_size,
            )
            info = {"loop": loop_sym, "new_blocks": new_blocks, "firing": firing}
            arr_info.append(info)
            print(f"  depth={depth}  [{loop_sym}] ({firing}): {new_blocks} new blocks/iter")

        results[arr_name] = arr_info
        print()

    return results


# ---------------------------------------------------------------------------
# Example: 2D Jacobi stencil
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N, M = dace.symbol("N"), dace.symbol("M")
    sdfg = dace.SDFG("jacobi2d")
    sdfg.add_array("A", shape=[M, N], dtype=dace.float64)  # write
    sdfg.add_array("B", shape=[M, N], dtype=dace.float64)  # read (stencil)
    state = sdfg.add_state("s0")

    # for i in [1, M-2]:
    #   for j in [1, N-2]:
    #     A[i,j] = 0.2 * (B[i,j] + B[i-1,j] + B[i+1,j] + B[i,j-1] + B[i,j+1])
    loop_nests = {
        "i": dace.subsets.Range([("1", "M-2", "1")]),
        "j": dace.subsets.Range([("1", "N-2", "1")]),
    }
    access_subsets = {
        # A is written at a single point
        "A": [dace.subsets.Range([("i", "i", "1"), ("j", "j", "1")])],
        # B has 5 stencil points
        "B": [
            dace.subsets.Range([("i",   "i",   "1"), ("j",   "j",   "1")]),  # center
            dace.subsets.Range([("i",   "i",   "1"), ("j-1", "j-1", "1")]),  # left
            dace.subsets.Range([("i",   "i",   "1"), ("j+1", "j+1", "1")]),  # right
            dace.subsets.Range([("i-1", "i-1", "1"), ("j",   "j",   "1")]),  # top
            dace.subsets.Range([("i+1", "i+1", "1"), ("j",   "j",   "1")]),  # bottom
        ],
    }
    analyze_block_traffic(state, loop_nests, access_subsets, block_size=8)