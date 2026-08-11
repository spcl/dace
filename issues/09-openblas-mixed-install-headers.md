# 09 — OpenBLAS environment mixes distro libs with from-source headers; kernel cannot load

## Summary

With BOTH a distro OpenBLAS (libopenblas-dev, behind the update-alternatives
symlinks) and a from-source install pointed at by `OPENBLAS_DIR`, the OpenBLAS
environment links one implementation but compiles against the other's headers,
and its threading probe dlopens the wrong library. The compiled kernel then
fails to load at runtime.

## Mechanism (dace/libraries/blas/environments/openblas.py)

- `OpenBLAS._mode()` prefers `find_package` whenever `_system_blas_libs()`
  resolves lapacke/cblas/blas via ldconfig. An explicit `OPENBLAS_DIR` hit
  cannot outrank it — there is no way to force `direct_link` while the distro
  packages are installed.
- `cmake_libraries()` (find_package mode) hands over the distro sonames
  (`liblapacke.so.3`, `libblas.so.3`).
- `cmake_includes()` is NOT gated on `_mode()`: it still returns the
  from-source install's include dir when `OPENBLAS_DIR` is set. Result: distro
  libs + from-source headers in one build.
- `_warn_unless_openmp_threaded()` dlopens the from-source `.so` (the
  `OPENBLAS_DIR` one), reporting the threading flavor of a library the kernel
  does not link.
- Both libraries carry SONAME `libopenblas.so.0`. Once the loader has mapped
  one, the other's symbols are resolved against it: a kernel linked against
  Debian's `libblas.so.3` in this mixed state dies with
  `dlerror: undefined symbol: gotoblas`.

## Reproduce

Host with libopenblas-dev installed plus a from-source OpenBLAS (any recent
tag, e.g. v0.3.29) installed to `$PREFIX`:

```bash
OPENBLAS_DIR=$PREFIX python -c "
import dace, numpy as np
from dace.libraries.blas import Gemm
Gemm.default_implementation = 'OpenBLAS'

@dace.program
def g(A: dace.float64[64, 32], B: dace.float64[32, 48], C: dace.float64[64, 48]):
    C[:] = A @ B

A = np.random.rand(64, 32); B = np.random.rand(32, 48); C = np.zeros((64, 48))
g(A=A, B=B, C=C)
"
```

`_mode()` returns `find_package`, the binary links the distro libs, the
compile used `$PREFIX/include`, and loading fails (or, depending on load
order, silently binds the wrong implementation).

## Candidate fixes

1. Gate `cmake_includes()` on `_mode()` so find_package mode never leaks
   from-source headers.
2. Let an explicit `OPENBLAS_ENV_VARS` hit (`OPENBLAS_DIR` etc.) outrank the
   distro alternatives — the user set it deliberately.
3. Point `_warn_unless_openmp_threaded()` at the library `cmake_libraries()`
   actually selected.

## Found by

HPCAgent-Bench integration tests (tests/test_dace_openblas_link.py):
asserting via `ldd` which OpenBLAS the compiled kernel really binds. The
probe runs each install shape in its own process because of the shared
SONAME.
