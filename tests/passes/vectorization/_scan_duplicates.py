"""K0 one-shot scanner: find near-duplicate tests in tests/passes/vectorization/.

Walks every test file, extracts:

* ``@dace.program`` decorated functions (kernels)
* ``_get_*_sdfg`` SDFG-construction helpers (also kernels)
* ``def test_*`` functions and the kernel each invokes via
  :func:`run_vectorization_test` (``dace_func=X``)

Then groups:

A. **Same-kernel-different-params** — tests that call the same ``dace_func``.
   Action per knob-hardening plan: FIXTURISE (collapse into one parametrised
   test, drop the standalone variants).

B. **Similar-kernel-different-bodies** — kernels whose AST fingerprint matches
   (loop nesting + statement opcode sequence + array-rank signature) but bodies
   differ in literals/names. Action: keep the harder one canonical,
   ``@pytest.mark.simple`` the simpler one(s).

Run with ``python -m tests.passes.vectorization._scan_duplicates``.
Not a test; not picked up by pytest collection.
"""
import ast
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).parent


def walk_test_files() -> List[pathlib.Path]:
    """Every test_*.py + _get_*_sdfg helper file under the vectorization tree."""
    return sorted(p for p in ROOT.rglob("*.py") if p.name.startswith(("test_", "helpers")) or p.name == "harness.py")


def _is_dace_program(decorator: ast.expr) -> bool:
    """Detect ``@dace.program`` and ``@dace.program(...)``."""
    if isinstance(decorator, ast.Attribute):
        return (isinstance(decorator.value, ast.Name) and decorator.value.id == "dace" and decorator.attr == "program")
    if isinstance(decorator, ast.Call):
        return _is_dace_program(decorator.func)
    return False


def _is_pytest_skip(decorator: ast.expr) -> bool:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Attribute):
        return decorator.attr in ("skip", "skipif")
    return False


def _statement_opcodes(fn: ast.FunctionDef) -> Tuple[str, ...]:
    """Linearised opcode sequence: each ``ast`` node type, depth-first.

    Ignores names and literal values so equivalent kernels under different
    variable names still match. Truncated at 200 nodes to keep the
    fingerprint comparison cheap.
    """
    seq: List[str] = []
    for node in ast.walk(fn):
        seq.append(type(node).__name__)
        if len(seq) >= 200:
            break
    return tuple(seq)


def _loop_structure(fn: ast.FunctionDef) -> Tuple[int, int]:
    """``(max_nest_depth, total_loops)`` — for/while/dace.map all count."""
    max_depth = 0
    total = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal max_depth, total
        if isinstance(node, (ast.For, ast.While)):
            total += 1
            max_depth = max(max_depth, depth + 1)
            for child in ast.walk(node):
                if child is node:
                    continue
                if isinstance(child, (ast.For, ast.While)):
                    visit(child, depth + 1)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, depth)

    visit(fn, 0)
    return (max_depth, total)


def _array_signature(fn: ast.FunctionDef) -> Tuple[int, ...]:
    """Tuple of array ranks inferred from annotations like ``dace.float64[N, N]``."""
    ranks: List[int] = []
    for arg in fn.args.args:
        ann = arg.annotation
        if isinstance(ann, ast.Subscript):
            # dace.float64[N, N] -> Subscript(value=Attribute(...), slice=Tuple|Name|BinOp)
            slc = ann.slice
            if isinstance(slc, ast.Tuple):
                ranks.append(len(slc.elts))
            else:
                ranks.append(1)
    return tuple(ranks)


def kernel_fingerprint(fn: ast.FunctionDef) -> Tuple:
    """Combined structural fingerprint."""
    return (_loop_structure(fn), _array_signature(fn), _statement_opcodes(fn))


def extract_kernel_from_test(test_fn: ast.FunctionDef) -> Optional[str]:
    """Return the name of the kernel ``test_fn`` invokes via
    :func:`run_vectorization_test`, or ``None`` if not a harness test."""
    for node in ast.walk(test_fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "run_vectorization_test":
                for kw in node.keywords:
                    if kw.arg == "dace_func":
                        v = kw.value
                        if isinstance(v, ast.Name):
                            return v.id
                        if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name):
                            return f"{v.value.id}.{v.attr}"
                        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name):
                            return f"{v.func.id}(...)"
    return None


def scan_file(path: pathlib.Path):
    """Returns (programs_dict, tests_dict, test_to_kernel_dict)."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return {}, {}, {}
    programs: Dict[str, ast.FunctionDef] = {}
    tests: Dict[str, ast.FunctionDef] = {}
    test_to_kernel: Dict[str, Optional[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            is_program = any(_is_dace_program(d) for d in node.decorator_list)
            is_helper = node.name.startswith("_get_") and node.name.endswith("_sdfg")
            if is_program or is_helper:
                programs[node.name] = node
            if node.name.startswith("test_"):
                tests[node.name] = node
                test_to_kernel[node.name] = extract_kernel_from_test(node)
    return programs, tests, test_to_kernel


def main():
    file_paths = walk_test_files()
    all_programs: Dict[Tuple[str, str], ast.FunctionDef] = {}  # (file, fn) -> ast
    all_tests: Dict[Tuple[str, str], ast.FunctionDef] = {}
    test_kernel: Dict[Tuple[str, str], Optional[str]] = {}
    for path in file_paths:
        rel = str(path.relative_to(ROOT))
        programs, tests, mapping = scan_file(path)
        for name, fn in programs.items():
            all_programs[(rel, name)] = fn
        for name, fn in tests.items():
            all_tests[(rel, name)] = fn
            test_kernel[(rel, name)] = mapping.get(name)
    print("=" * 78)
    print(f"K0 SCANNER REPORT")
    print(f"  Files scanned: {len(file_paths)}")
    print(f"  @dace.program kernels: {sum(1 for _,n in all_programs if not n.startswith('_get_'))}")
    print(f"  _get_*_sdfg helpers: {sum(1 for _,n in all_programs if n.startswith('_get_'))}")
    print(f"  Tests calling run_vectorization_test: {sum(1 for v in test_kernel.values() if v)}")
    print("=" * 78)
    print()

    # ---- Group A: same kernel, different test wrappers ----
    kernel_to_tests: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for (f, t), k in test_kernel.items():
        if k:
            kernel_to_tests[k].append((f, t))
    print("GROUP A — SAME KERNEL, DIFFERENT TEST WRAPPERS  (action: FIXTURISE)")
    print("-" * 78)
    a_groups = [(k, ts) for k, ts in kernel_to_tests.items() if len(ts) >= 2]
    for kernel, tests in sorted(a_groups, key=lambda x: -len(x[1])):
        print(f"\n  kernel: {kernel}  ({len(tests)} tests)")
        for f, t in sorted(tests):
            print(f"    {f}::{t}")
    if not a_groups:
        print("  (none)")
    print()

    # ---- Group B: structurally similar kernels ----
    fp_to_kernels: Dict[Tuple, List[Tuple[str, str]]] = defaultdict(list)
    for (f, name), fn in all_programs.items():
        fp = kernel_fingerprint(fn)
        fp_to_kernels[fp].append((f, name))
    print("GROUP B — STRUCTURALLY-SIMILAR KERNELS  (action: keep harder canonical, @simple the rest)")
    print("-" * 78)
    b_groups = [(fp, ks) for fp, ks in fp_to_kernels.items() if len(ks) >= 2]
    # Only show groups where the kernel names differ (else they're just multi-file copies)
    interesting = [g for g in b_groups if len({n for _, n in g[1]}) >= 2]
    for fp, kernels in sorted(interesting, key=lambda x: -len(x[1])):
        loop_struct, arr_sig, _ = fp
        print(f"\n  fingerprint: loops={loop_struct}, array_ranks={arr_sig}")
        for f, name in sorted(kernels):
            # Show which tests USE this kernel (to assess hardness)
            consumers = [f"{tf}::{tn}" for (tf, tn), k in test_kernel.items() if k == name]
            consumer_str = f"  used by: {len(consumers)} test(s)"
            print(f"    {f}::{name}{consumer_str}")
    if not interesting:
        print("  (none)")
    print()

    # ---- Group C: tests with identical kernel + nearly-identical bodies ----
    # Heuristic: tests in the same file calling the same kernel with bodies of
    # the same line-count (within 3 lines) are likely the wrapper-only variants
    # the plan calls out (test_v_const_subs_cpu vs test_v_const_subs_two_cpu etc).
    print("GROUP C — TEST WRAPPERS WITH SIMILAR BODIES + KERNELS  (suggest @simple gating)")
    print("-" * 78)
    by_file = defaultdict(list)
    for (f, t), fn in all_tests.items():
        if test_kernel[(f, t)] is None:
            continue
        nlines = (fn.end_lineno or fn.lineno) - fn.lineno
        by_file[f].append((t, test_kernel[(f, t)], nlines))
    any_c = False
    for f, entries in sorted(by_file.items()):
        # Same prefix tests with similar line counts
        prefix_groups: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        for t, k, nlines in entries:
            # Group by first 3 underscore-separated tokens
            tokens = t.split("_")
            prefix = "_".join(tokens[:3])
            prefix_groups[prefix].append((t, k, nlines))
        for prefix, grp in prefix_groups.items():
            if len(grp) < 2:
                continue
            line_counts = sorted({nl for _, _, nl in grp})
            if max(line_counts) - min(line_counts) > 5:
                continue
            kernels = {k for _, k, _ in grp}
            if len(kernels) <= 1:
                continue  # Group A covers this
            any_c = True
            print(f"\n  {f}  (prefix '{prefix}', ~{max(line_counts)} lines each)")
            for t, k, nl in sorted(grp):
                print(f"    {t}  -> kernel {k}  ({nl} lines)")
    if not any_c:
        print("  (none)")
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
