# cavity_flow — legacy vs experimental (readable) CPU codegen

NPBench `cavity_flow` (structured-grid CFD: pressure-Poisson + momentum update),
lowered through the same pipeline used everywhere in this project:

```
dace + simplify + LoopToMap + MapFusion
```

then generated once with `compiler.cpu.implementation = legacy` and once with
`experimental`. Both `.cpp` files (the CPU frame code object) are saved next to
this file.

## Lines of code

Both code generators run the same pipeline (`dace + simplify + LoopToMap + MapFusion
+ len-1 → scalar`), so the comparison isolates the code generator.

| codegen                         | total lines | non-blank | file |
|---------------------------------|------------:|----------:|------|
| legacy                          | 1844        | 1541      | `cavity_flow_legacy_cpu.cpp` |
| experimental (readable)         | **416**     | **365**   | `cavity_flow_experimental_cpu.cpp` |
| reduction                       | **−77 %**   | **−76 %** | |

## Why the experimental output is smaller / more readable

- **Connector-free single-line tasklets.** Each element-wise tasklet is one line
  `C[C_idx(i,j)] = A[A_idx(i,j)] op B[B_idx(i,j)];  // <label>` instead of a
  copy-in / compute / copy-out block. The legacy output has **252** `///////////////////`
  separator blocks with `__tmp` / `__out` register temporaries; the experimental
  output has **0**, and no redundant `// Tasklet code (...)` banner lines.
- **`const T x = expr;` for single-write scalars.** A scope-local scalar transient
  written exactly once (e.g. `const double dt_div_dx_2 = ...;`) is emitted as a fused
  const binding instead of a mutable `double x;` declaration followed by a separate
  assignment — **35** such bindings here.
- **Per-array `<arr>_idx(...)` index functions** — the flat-offset arithmetic
  appears once per array rather than inlined at every access; emitted as plain
  `static` functions (deduplicated in a post-pass, **no** `#ifndef` include-guard
  macros).
- **`dace::aligned_alloc<T>` / `dace::free`** heap wrappers instead of aligned
  `new[]` / `delete[]`.

## Correctness

Legacy and experimental are **bit-exact** on all outputs (`u`, `v`, `p`), so the
line-count reduction is pure presentation — no change to the computation.

```
u: bit-exact=True   v: bit-exact=True   p: bit-exact=True
```

## Reproduce

```bash
# regenerate both .cpp files + LoC (CPU, no compile needed):
PYTHONPATH=<repo> python - <<'PY'
from dace.config import set_temporary
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from tests.corpus.npbench import npbench
desc = npbench.collect('cavity_flow')[0]
for impl in ('legacy', 'experimental'):
    with set_temporary('compiler', 'cpu', 'implementation', value=impl):
        s = npbench.fresh_sdfg(desc); s.simplify()
        s.apply_transformations_repeated(LoopToMap); s.apply_transformations_repeated(MapFusion)
        ConvertLengthOneArraysToScalars(single_element=True, transient_only=True).apply_pass(s, {})
        s.simplify()
        s.name = f'cavity_flow_{impl}'
        code = '\n'.join((o.clean_code or o.code) for o in s.generate_code() if o.language == 'cpp')
        open(f'cavity_flow_{impl}_cpu.cpp', 'w').write(code)
        print(impl, len(code.splitlines()), 'lines')
PY
```
