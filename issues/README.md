# Python frontend bug reports

Failures in the DaCe Python frontend and library expansions, found while porting a
NumPy HPC kernel corpus. Each `.md` is ready to paste into a GitHub issue; each has a
standalone reproducer under `repro/` that runs with no arguments and no environment
beyond a DaCe checkout on `PYTHONPATH`:

    PYTHONPATH=/path/to/dace python3 repro/01-symbolic-or.py

A reproducer exits 0 when every case still behaves as its `.md` documents, and 1
when something moved -- fixed upstream, or bit-rotted. 01 through 04 are parse-only
(`to_sdfg(simplify=False)`) and invoke no C++ compiler, `gemv-out-connector-clash`
stops after expansion, simplification and validation, and `linalg-solve-index-error`
stops after expansion alone (needs OpenBLAS or MKL findable by DaCe to reach that
expansion). 05 through 08 are about WRONG ANSWERS and generated code, so those four
build and run what they parse and do need a C++ compiler.

| # | slug | symptom | verified against |
| --- | --- | --- | --- |
| 01 | `symbolic-or` | `if N == 0 or N == 1:` over a symbol raises `TypeError: 'Equality' object is not iterable` from CPython's `ast`, because `RewriteSympyEquality.visit_Compare` returns a non-`AST` sympy object into a list-valued field. | `main` `d7efcef0c`, and `8d749ff2c` |
| 02 | `advanced-indexing-compound-index` | `p[int(cols[i])]` raises `AttributeError: 'str' object has no attribute '_fields'`: a replacement's list-of-names result is mistaken for a list-literal index, then fed to the AST visitor as a `str`. | `main` `d7efcef0c`, and `8d749ff2c` |
| 03 | `numpy-empty-dtype` | `np.empty(10)` raises `TypeError: _numpy_empty() missing 1 required positional argument: 'dtype'`; every sibling creation replacement defaults `dtype`. One-line fix, diff hunk in the `.md`, verified applied. | `main` `d7efcef0c`, and `8d749ff2c` |
| 04 | `arrayview-copy` | `path[:, 1].copy()` raises a bare `NotImplementedError` with an empty message: `_add_transient_data` dispatches on exact type, and a slice's descriptor is `ArrayView`, not one of its four registered keys. | `main` `d7efcef0c`, and `8d749ff2c` |
| 05 | `scalar-assign-aliases` | `b = a` on a transient SCALAR aliases the container instead of copying, so `b += ...` or a later `b = c` writes into `a`. Silent wrong answers: `s0 = s1 = 0.0` collapses an unrolled reduction's accumulators into one (`11x` over-count), and `pc = rp0` / `pc = rp1` in adjacent branches destroys `rp0`. | `main` `d7efcef0c`, and `extended` `ac9398525` |
| 06 | `scalar-reassign-narrows` | `udiff = 1` types the container `int64`; a later `udiff = <float>` is silently TRUNCATED into it, so `while udiff > 1e-3` exits after two trips. | `main` `d7efcef0c`, and `extended` `ac9398525` |
| 07 | `complex-integer-operators` | `dace/runtime/include/dace/complex.h` has mixed `operator*` and nothing else: `cmplx<double> / int64_t` fails to compile (a naive DFT twiddle hits it), and the `operator*` that exists takes `const int&`, so an `int64_t` operand is silently narrowed to 32 bits. | `main` `d7efcef0c`, and `extended` `ac9398525` |
| 08 | `unqualified-real-imag` | `np.real(x)` / `np.imag(x)` lower to a tasklet body spelling the bare name, and `cppunparse`'s `_renamed_funcs` maps only `re`/`im`, so the generated C++ calls an undeclared `real(...)`. A complex operand compiles by accident through ADL into `std`; a `float64` one does not compile at all. | `main` `a4740d4e7` |
| 09 | `openblas-mixed-install-headers` | With libopenblas-dev AND an `OPENBLAS_DIR` from-source install, `_mode()` picks `find_package` (env var cannot outrank ldconfig) but `cmake_includes()` is not mode-gated: distro libs + from-source headers in one build, threading probe dlopens the wrong `.so`, and the shared SONAME `libopenblas.so.0` makes the kernel die with `undefined symbol: gotoblas`. | `main` `a4740d4e7` |
| -- | `gemv-out-connector-clash` | `out[:] = A @ x` into an array named `out` raises `InvalidSDFGNodeError: Connector name 'out' is already used as a symbol, constant, or array name` once the pure BLAS expansion is inlined: five expansions name their zero-init tasklet connector `out` instead of `__out`. | `main` `d7efcef0c`, and `extended` `11004dab0` |
| -- | `linalg-solve-index-error` | `np.linalg.solve(A, b)` with a 1-D vector `b` raises `IndexError: list index out of range` from `Solve.validate()` once `expand_library_nodes()` picks an implementation: `shape_out[1]` assumes the RHS is always a matrix. `to_sdfg` alone does not trigger it. | `main` `d7efcef0c`, and `extended` `ac9398525` |

`d7efcef0c` is `spcl/dace` `main`. `8d749ff2c` is a downstream branch carrying later
frontend commits; all four reproduce there unchanged, with line numbers shifted. The
line numbers quoted in the `.md` files are `main`'s.

One caveat, and it is not a change to any of the four bugs. The downstream branch
moved during verification, to `9feb23929` ("Reduce argmax/argmin through two scalars
instead of a struct"). That commit breaks the *workaround* documented for case (b)
of issue 02: `tmp[:] = absU[:, j]; k = np.argmax(tmp)` now raises
`ValueError: View "tmp_0" already has both incoming and outgoing edges`. It still
works on `main`. Since `repro/02-*.py` is calibrated against `main`, it reports that
case as CHANGED and exits 1 when run on that branch; the three bug cases in it
continue to reproduce.
