# Semantic decision register

Every place where DaCe's documented intent, its code generator, and a clean
mathematical model disagree (or where behavior is undefined) is recorded here
together with the model chosen for the Lean formalization and the evidence.
Entries are referenced from the Lean sources as `Decisions.md#<id>`.

Status: **living document**, started 2026-10-16. Line references are to the
DaCe sources at the time of writing.

| Id | Topic | DaCe evidence | Model chosen |
|----|-------|---------------|--------------|
| D1 | Integer arithmetic | Codegen emits native C ints (default `int32`, `dace/symbolic.py:26`), wrapping and truncating `/`, `%`; SymPy expressions use exact rationals (`1/2*N`) | Idealized `ℤ` with exact rationals `ℚ` for symbolic division; `int_floor` = floor division (Python `//`), `int_ceil` = ceiling division, `Mod` = Python modulo (sign of divisor). Overflow and C truncation are out of scope (assumption; codegen's `int_floor` truncates for negative operands). |
| D2 | Multiple true interstate conditions | Doc: undefined (`doc/sdfg/ir.rst:151-152`); codegen takes the first true edge in out-edge order (`dace/codegen/control_flow.py:304-328`) | Nondeterministic choice among true edges. Codegen order is a refinement. |
| D3 | Map schedules | Maps are parallel scopes (`doc/sdfg/ir.rst:446-471`); `Sequential` lowers to ordered `for` loops (`cpu.py:2132-2148`) | `schedule` is semantically inert: every map is a parallel scope; iteration logs are merged, conflicts are an `error race`. Sequential execution is a refinement. |
| D4 | Concurrent components in a state | Parallel only under `openmp_sections` (`framecode.py:558-574`), else emitted sequentially | Components run in parallel (permission); each component in some topological order; logs merged. |
| D5 | Stream `buffer_size` | Initial queue capacity only; push never blocks or drops; pop blocks on empty (`stream.h:81-124`) | `buffer_size` advisory; `push` appends; `pop` on empty = `error stuckPop`. |
| D6 | WCR on streams | Doc forbids (`ir.rst:433-434`); not validated, silently ignored by codegen | Ill-formed (rejected by WF). |
| D7 | Stream-to-stream copy | `NotImplementedError` (`cpu.py:953-955`) | Ill-formed. |
| D8 | Consume early stop | Remaining elements stay in the queue; never-true condition on an empty stream spins | Remaining elements stay; divergence is the absence of an outcome. |
| D9 | `other_subset = None` on copies | Full range of the other array (`cpp.py:185-188`) | Normalized at import to the full range. |
| D10 | Copy element order | Row-major over the squeezed copy shape (`copy.h`) | Elementwise over `zip (points src) (points dst)` in lexicographic order. |
| D11 | References | May be set several times in a state (no validation); read before set rejected by validation | Last `set` in dataflow order wins; read of an unset reference = `error unsetRef`. |
| D12 | Dynamic map inputs | Scalars only in practice (`cpu.py:2021-2028`) | Scalar symbol bound before the iteration space is evaluated. |
| D13 | Unassigned tasklet outputs | Static scalar outputs are written back unconditionally (`cpu.py:1212-1258`); dynamic outputs are may-writes | WF requires static outputs to be written; dynamic outputs are `Option`. |
| D14 | Stream connector reads | Each read of a dynamic stream input pops (`cpp.py:1328`) | Modeled as such in the executable tasklet subset; opaque tasklets declare pop counts. |
| D15 | Nested SDFG descriptor `offset` | Excluded from `is_equivalent` (`dace/sdfg/dealias.py:1240-1250`) | Ignored; inner memlets address the outer container with identical indices. |
| D16 | Legacy (windowed) nested SDFGs | Converted by `convert_legacy_nested_sdfgs` (`dealias.py:1469`); not inlined otherwise | Rejected by the importer; only the no-view/no-squeeze contract is modeled. |
| D17 | View edge ambiguity | `get_view_edge` falls back to the input edge with a warning (`dace/sdfg/utils.py:1223-1225`) | Same resolution order, reproduced in WF. |
| D18 | `nosync` | GPU-only effect (`cuda.py:1799`) | Ignored. |
| D19 | `StateBoundaryNode` | Conversion artifact of tree→SDFG | No stree syntax; appears only in the correspondence relation. |
| D20 | Stree sibling order | SDFG→tree serializes a state by one topological order | Sibling order is one valid schedule; the correspondence accepts any topological order. |
| D21 | Structures / ContainerArrays | Member access `a.b` lowered to `a->b` | Full syntax and the generic `Loc` memory model; no theorems in this phase. |
| D22 | Symbolic expression parsing | DaCe stores sympy expressions as Python-syntax strings; serialized symbols carry a `$` prefix; `SymExpr(exact, approx)` pairs an expression with an over-approximation | Parsed by `Core/Parse` (both `$`-prefixed and bare symbols); `SymExpr` becomes `Expr.approx` and evaluates as its exact part; unknown function calls become opaque `app` nodes. |
| D23 | Floating point | Tasklets and symbolic expressions may use floats | Floats are an abstract sort interpreted by `Interp`; no IEEE semantics. |
| D24 | Custom WCR lambdas | Commutativity/associativity checked symbolically, may be indeterminate (`dace/frontend/operations.py:201-250`) | Assumed commutative and associative (hypothesis `WcrSem.Lawful` of the semantics). |
| D25 | Parallel pushes to one stream | Concurrent producers enqueue in a nondeterministic interleaving | Logs record pushes per execution; `merge` appends them in list order (one admissible interleaving). Pops require exclusive access. |
| D26 | WCR lambda representation | Stored as source text `lambda a, b: body`; built-ins detected by `detect_reduction_type` | Parsed once into `WcrOp.custom a b body` with structural detection of built-ins; the interpreter evaluates custom bodies by binding `a := old`, `b := new`. An undefined evaluation degrades to `new`. |
| D27 | Empty memlets | `Memlet()` with no data; codegen emits nothing, only orders the endpoints (`cpu.py:844-846`, `concurrent_subgraphs`) | A distinct edge payload `EdgeData.dep`: a happens-before edge that also locks a node into a scope; carries no data. |
