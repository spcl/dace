"""K6 one-shot classifier: triage TSVC + cloudsc kernels into T1 (full
knob matrix), T2 (medium — restricted), and T3 (smoke — vec_config only).

The classifier:

1. **Maps-produced gate**: runs each kernel through ``to_sdfg()`` +
   ``LoopToMap`` (+ ``MapCollapse``) and counts innermost maps. A kernel
   with zero maps after the pre-passes won't be vectorised — drop from
   all tiers.

2. **Hardness fingerprint** from the kernel body AST:

   - **Hard (T1)**: branches inside a loop, multi-statement loop bodies
     with cross-iter dep, reductions with conditionals, gather/scatter
     mixed with arithmetic.
   - **Medium (T2)**: single reduction OR single conditional OR indirect
     access, single-statement body.
   - **Simple (T3)**: pure elementwise (``c[i] = a[i] op b[i]`` shape),
     no branches, no reductions, no indirection.

3. **Pattern detection** (stencil): adjacent-element reads in a non-tiled
   dim — flagged as T1 regardless of other features (the heat3d /
   cloudsc / jacobi family stress test path).

Output: a triage table the harness reads to pick fixture coverage per
kernel.
"""
import ast
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import dace

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent


def _count_innermost_maps_in_sdfg(sdfg: dace.SDFG) -> int:
    """Apply LoopToMap pre-passes and count innermost MapEntry nodes."""
    try:
        from dace.transformation.interstate import LoopToMap
        sdfg.apply_transformations_repeated(LoopToMap, permissive=False, validate=False)
    except Exception:  # noqa: BLE001
        pass

    count = 0
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            count += 1
    return count


def _kernel_ast(module_ast: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    for node in ast.walk(module_ast):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _fingerprint_hardness(fn: ast.FunctionDef) -> Tuple[str, List[str]]:
    """Return (tier_estimate, feature_tags) from the AST."""
    features: List[str] = []
    has_branch = False
    has_indirect = False
    has_neighbor = False  # adjacent-element read pattern (stencil)
    reduction_count = 0
    statement_count_in_loop = 0
    for_count = 0
    augassign_count = 0
    has_reduction_call = False

    def walk_loops(node: ast.AST, in_loop: bool) -> None:
        nonlocal has_branch, has_indirect, has_neighbor, reduction_count, statement_count_in_loop, for_count, augassign_count, has_reduction_call
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                for_count += 1
                walk_loops(child, True)
                continue
            if in_loop:
                if isinstance(child, ast.If):
                    has_branch = True
                if isinstance(child, ast.AugAssign):
                    augassign_count += 1
                    reduction_count += 1
                if isinstance(child, ast.Assign):
                    statement_count_in_loop += 1
                if isinstance(child, ast.Subscript):
                    # data-dep gather: subscript with subscript inside (e.g., a[idx[i]])
                    if any(isinstance(grand, ast.Subscript) for grand in ast.walk(child.slice)):
                        has_indirect = True
                    # neighbor pattern: index expression of the form `i + N`
                    # where N is a small integer literal (-1, +1, +2)
                    for grand in ast.walk(child.slice):
                        if isinstance(grand, ast.BinOp) and isinstance(grand.op, (ast.Add, ast.Sub)):
                            if isinstance(grand.right, ast.Constant) and isinstance(grand.right.value, int):
                                if grand.right.value in (1, 2, 3):
                                    has_neighbor = True
            walk_loops(child, in_loop)

    walk_loops(fn, False)

    if has_branch:
        features.append("branch")
    if has_neighbor:
        features.append("stencil")
    if has_indirect:
        features.append("indirect")
    if reduction_count >= 1:
        features.append(f"reduction({reduction_count})")
    features.append(f"loops={for_count}")
    features.append(f"stmts_in_loop={statement_count_in_loop}")

    # Tier:
    # - T1: stencil OR (branch + reduction) OR (branch + indirect) OR multi-stmt
    # - T2: single reduction, single branch, single indirect, multiple loops
    # - T3: pure elementwise (single assign in single loop, no flags)
    if has_neighbor:
        return "T1", features
    if has_branch and (reduction_count or has_indirect):
        return "T1", features
    if statement_count_in_loop >= 3:
        return "T1", features
    if has_branch or reduction_count or has_indirect:
        return "T2", features
    if for_count <= 2 and statement_count_in_loop == 1:
        return "T3", features
    return "T2", features


def _try_count_maps(kernel) -> Optional[int]:
    try:
        sdfg = kernel.to_sdfg()
        return _count_innermost_maps_in_sdfg(sdfg)
    except Exception as e:  # noqa: BLE001
        return None


def classify_tsvc() -> List[Dict]:
    """Walk the TSVC corpus, classify each kernel."""
    sys.path.insert(0, str(REPO_ROOT))
    from tests.corpus.tsvc import tsvc
    src = (REPO_ROOT / "tests" / "corpus" / "tsvc.py").read_text()
    module = ast.parse(src)
    results: List[Dict] = []
    for regime in ("1d", "2d"):
        try:
            kernels = tsvc.collect(regime=regime)
        except Exception:  # noqa: BLE001
            continue
        for k in kernels:
            kernel_full = getattr(k.program, "name", None) or getattr(k, "name", None)
            if kernel_full is None:
                continue
            # ``tests_corpus_tsvc_s000_d_single`` -> ``s000_d_single``
            kernel_name = kernel_full.split("tsvc_", 1)[-1]
            fn_ast = _kernel_ast(module, kernel_name)
            if fn_ast is None:
                continue
            tier, features = _fingerprint_hardness(fn_ast)
            map_count = _try_count_maps(k.program)
            if map_count is None:
                tier_final = "ERROR"
            elif map_count == 0:
                tier_final = "NO_MAPS"
            else:
                tier_final = tier
            results.append({
                "name": kernel_name,
                "regime": regime,
                "tier": tier_final,
                "maps": map_count,
                "features": features,
            })
    return results


def main():
    results = classify_tsvc()
    by_tier: Dict[str, List[Dict]] = {}
    for r in results:
        by_tier.setdefault(r["tier"], []).append(r)
    print("=" * 78)
    print("TSVC kernel classification")
    print("=" * 78)
    for tier in ("T1", "T2", "T3", "NO_MAPS", "ERROR"):
        kernels = by_tier.get(tier, [])
        print(f"\n{tier}: {len(kernels)} kernels")
        if tier in ("T1", "NO_MAPS", "ERROR"):
            for r in sorted(kernels, key=lambda x: x["name"]):
                features = ", ".join(r["features"])
                print(f"  {r['name']:<22} maps={r['maps']:<3}  [{features}]")
        elif tier == "T2":
            # T2 just list names so the report stays readable
            names = sorted(r["name"] for r in kernels)
            for chunk in (names[i:i + 6] for i in range(0, len(names), 6)):
                print(f"  {', '.join(chunk)}")
        elif tier == "T3":
            names = sorted(r["name"] for r in kernels)
            for chunk in (names[i:i + 6] for i in range(0, len(names), 6)):
                print(f"  {', '.join(chunk)}")
    print()
    print(f"Total: {len(results)} kernels")
    print(f"  T1 (full matrix): {len(by_tier.get('T1', []))}")
    print(f"  T2 (medium):      {len(by_tier.get('T2', []))}")
    print(f"  T3 (simple):      {len(by_tier.get('T3', []))}")
    print(f"  NO_MAPS:          {len(by_tier.get('NO_MAPS', []))}  ← skip in all tiers")
    print(f"  ERROR:            {len(by_tier.get('ERROR', []))}")


if __name__ == "__main__":
    main()
