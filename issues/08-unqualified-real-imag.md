# `np.real` / `np.imag` emit an UNQUALIFIED `real(...)` / `imag(...)` that nothing declares

## Environment

- dace: `spcl/dace` `main` @ `a4740d4e7` (`dace.__version__ == 2.0.0a5`).
- gcc 15.2.0, `-std=c++20`, Linux x86_64;
  Python 3.12.11, numpy 2.4.4
- Runnable reproducer: `repro/08-unqualified-real-imag.py`
  (`PYTHONPATH=/path/to/dace python3 repro/08-unqualified-real-imag.py`)

## Summary

`np.real(x)` / `np.imag(x)` lower to a tasklet whose body is the bare spelling of the function:

```python
# dace/frontend/python/replacements/pymath.py:77-91
@oprepo.replaces('numpy.real')
def _real(pv, sdfg, state, input):
    inptype = sdfg.arrays[input].dtype
    return simple_call(pv, sdfg, state, input, 'real', complex_to_scalar(inptype))
```

```python
# dace/frontend/python/replacements/utils.py:74 (and :86 for the mapped form)
tasklet = state.add_tasklet(func, ..., f'__out = {func}({inconn_name})')
```

`cppunparse` writes an unrecognised call name through verbatim, so the generated C++ contains
`real(<operand>)`. There is no `::real`, no `dace::real`, and the runtime headers declare no such
free function -- `dace/runtime/include/dace/math.h:633-640` declares `dace::math::re` / `::im`
instead. When the operand is a **real** type, ADL reaches no namespace at all and the translation
unit does not compile:

```
error: 'real' was not declared in this scope; did you mean 'std::real'?
```

A **complex** operand happens to compile, because `dace::complex128` is `std::complex<double>` and
ADL finds `std::real`. That is what makes the defect easy to miss: the same replacement is correct
by accident on the type it was written for and broken on every other one.

`cppunparse` already has the table this belongs in, and `real` / `imag` are simply absent from it:

```python
# dace/codegen/cppunparse.py:1198-1201
_renamed_funcs = {
    're': 'dace::math::re',
    'im': 'dace::math::im',
}
```

## Minimal reproducer

```python
import numpy as np
import dace

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def take_real(a: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = np.real(a[i])


take_real.to_sdfg(simplify=True).compile()
```

Generated C++ and the error:

```cpp
// take_real.cpp:17
const double out_slice = real(a[a_idx(i)]);  // real
```

```
take_real.cpp:17:34: error: 'real' was not declared in this scope; did you mean 'std::real'?
```

Swapping the annotation to `a: dace.complex128[N]` builds and runs. `np.imag` behaves identically,
with `imag` / `std::imag` in place of `real` / `std::real`.

## Expected vs actual

- **Expected:** the emitted call names a function that is in scope, for every operand type
  `np.real` accepts. numpy's own semantics are total -- `np.real(x) is x` and `np.imag(x) == 0`
  for a real `x` -- so the C++ should be too. Two changes together:

  1. Route the two spellings through the existing table, so nothing is emitted unqualified:

     ```python
     _renamed_funcs = {
         're': 'dace::math::re',
         'im': 'dace::math::im',
         'real': 'dace::math::re',
         'imag': 'dace::math::im',
     }
     ```

  2. Widen the helpers to arithmetic types, since today's trailing `decltype(z.real())` constrains
     them to complex operands and would reject `double` just as loudly:

     ```cpp
     template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
     DACE_CONSTEXPR DACE_HDFI T re(const T& z) { return z; }
     template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
     DACE_CONSTEXPR DACE_HDFI T im(const T&) { return T(0); }
     ```

  Emitting `std::real` / `std::imag` instead would also compile (C++11 gives them arithmetic
  overloads), but it would leave the CUDA path -- where the complex type is `thrust::complex` --
  reaching for the wrong namespace. `dace::math::re` / `::im` already handle both, which is why
  they exist.
- **Actual:** the generated translation unit does not compile whenever the operand is not complex.

Fixing this in the codegen rather than in the replacement is the smaller change, and it also covers
a hand-written tasklet body that spells `real(...)` instead of the `re(...)` the table already
knows -- the `real`/`imag` spelling is the one a reader arrives at from numpy, and it is silently
the broken one.

## Impact

Found while running a NumPy HPC kernel corpus through the frontend. Two kernels fail to build on
it, both reaching it through the same `np.linalg.eigh` desugar -- a Jacobi sweep that asks for
`np.real` / `np.imag` of a matrix element that is `float64` on this input:

- `rayleigh_ritz_rotation` -- 6 errors, all of this form, and nothing else:

  ```
  ..._rayleigh_ritz_rotation_dace_kernel.cpp:354:58: error: 'real' was not declared in this scope; did you mean 'std::real'?
  ..._rayleigh_ritz_rotation_dace_kernel.cpp:359:58: error: 'imag' was not declared in this scope; did you mean 'std::imag'?
  ..._rayleigh_ritz_rotation_dace_kernel.cpp:387:40: error: 'real' was not declared in this scope; did you mean 'std::real'?
  ..._rayleigh_ritz_rotation_dace_kernel.cpp:388:40: error: 'imag' was not declared in this scope; did you mean 'std::imag'?
  ..._rayleigh_ritz_rotation_dace_kernel.cpp:397:52: error: 'real' was not declared in this scope; did you mean 'std::real'?
  ..._rayleigh_ritz_rotation_dace_kernel.cpp:530:45: error: 'real' was not declared in this scope; did you mean 'std::real'?
  ```

  The offending line, with the operand's declaration one line above it -- a `double`, so the
  component read is the identity and the call is pure overhead even once it builds:

  ```cpp
  const double __eigh0_Cm_index = __eigh0_Cm[__eigh0_Cm_idx(__eigh0_pp, __eigh0_qq, k)];
  const double real___eigh0_Cm_slice = real(__eigh0_Cm_index);  // real
  ```

- `largest_eigenval` -- the same 6, same desugar, different line numbers.

**Workaround:** drop the call when the operand is real -- `x` for `np.real(x)` and `0.0` for
`np.imag(x)` is what numpy computes there. There is none for a generated program, which is
where both of these came from.
