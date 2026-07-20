# Phase B: native C++ graphlib backend (design only, not built)

Third `graph:backend:` value (e.g. `'dacegraph'`), same `GraphBackend` protocol as
`rustworkx_backend.py`. Zero call-site changes when it lands — callers only ever talk to
`dace.graphlib`'s dispatch layer, never a backend directly.

Standalone repo. Answers each gap found in the BoostX/DaCeX prior-art review (see project
memory `project_dacex_replacement_planning` / the approved implementation plan) point by point:

| BoostX gap | Fix here |
|---|---|
| Node props = untyped `nb::dict`, only ~5 algos got real speedup | Real typed C++ property storage (typed schema or `std::any`/variant store), no dict round-trip per access |
| Naive backtracking isomorphism matcher | Boost.Graph's own `vf2_subgraph_iso` |
| Untested hand-rolled Edmonds-Karp | Boost.Graph's `boykov_kolmogorov_max_flow`/`push_relabel_max_flow` — closes the ONE confirmed rustworkx gap (§7.2 of the plan), first backend that could accelerate `cutout.py`'s min-cut |
| Multigraphs punted to unaccelerated pure Python | Real multigraph support in the C++ core — required before this backend can matter for a future backend-aware `.nx` (dace's core IR is a multigraph) |
| Unbounded recursive postorder DFS in dominator calc (stack-overflow risk on deep CFGs) | Boost.Graph's iterative Lengauer-Tarjan |
| Assert-free 12-line test script | Differential tests vs real networkx, reusing the exact harness shape in `tests/graphlib/` (backend-agnostic by design) |
| No CI | CI on every push, same shape as this repo's `general-ci.yml`/`gpu-ci.yml` |
| Truncated LICENSE (claims MIT, no grant text) | Real, complete grant compatible with dace's BSD-3-Clause; verify Boost.Graph's own Boost-license compatibility explicitly in this repo's LICENSE/NOTICE |
| cibuildwheel scaffolding present but never run | Real wheels across `python_requires='>=3.10,<3.15'`'s matrix, so this becomes a normal `extras_require` entry like `fastgraph`, not a source build |

Binding: nanobind (BoostX's choice was reasonable — header-only, lower overhead than pybind11,
good buffer-protocol interop) — keep it.

Not started. No timeline. Revisit once Phase A's rustworkx backend has run in production long
enough to show where the remaining `.nx`-mediated cost actually is (see the "Phase C" note in
`dace/graphlib/__init__.py` and `rustworkx_backend._coerce`'s docstring for the specific,
uncached-conversion-cost gap this — or a cached backend-aware `.nx` — would need to close).
