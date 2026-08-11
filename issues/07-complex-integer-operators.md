# Runtime headers: `cmplx<T> / <integer>` does not compile, and `cmplx<T> * <integer>` silently narrows to 32 bits

## Environment

- dace: `spcl/dace` `main` @ `d7efcef0c96580e590caf7003c8320ba8908239c` (`dace.__version__ == 2.0.0a5`),
  and `extended` @ `ac939852548ac8aa43085690a6b32ec6c5887d2e`. `dace/runtime/include/dace/complex.h`
  is unchanged between them apart from formatting.
- gcc 15.2.0, `-std=c++20`, Linux x86_64; Python 3.12.11, numpy 2.4.4
- Runnable reproducer: `repro/07-complex-integer-operators.py`
  (`PYTHONPATH=/path/to/dace python3 repro/07-complex-integer-operators.py`)

## Summary

`dace/runtime/include/dace/complex.h` supplies exactly two mixed complex/integer operators:

```cpp
// dace/runtime/include/dace/complex.h:62-69
template <typename T> cmplx<T> operator*(const cmplx<T>& a, const int& b) { ... }
template <typename T> cmplx<T> operator*(const int& a, const cmplx<T>& b) { ... }
```

Two consequences, one loud and one silent.

**(a) `operator/` is missing entirely**, so any generated expression of the form
`complex_constant * <int symbol> / <int symbol>` fails to compile. `std::complex<T>`'s own
`operator/(const complex<_Tp>&, const _Tp&)` cannot help: `_Tp` deduces to `double` from the left
operand and to `long int` from the right, deduction conflicts, and there is no viable overload.

**(b) The `operator*` that does exist takes `const int&`, not a deduced integer type.** An `int64_t`
operand therefore binds by an implicit `long -> int` conversion and is truncated, with no
diagnostic. `(1+0j) * int64_t(3000000000)` evaluates to `-1294967296+0j`.

## Minimal reproducer

A naive DFT is enough for (a). The twiddle exponent is const-folded into one tasklet with the
symbols inlined, and the symbols are `int64_t`:

```python
N = dace.symbol('N', dtype=dace.int64)


@dace.program
def dft(x: dace.complex128[N], y: dace.complex128[N]):
    for k in range(N):
        y[k] = 0j
        for n in range(N):
            y[k] += x[n] * np.exp(-1j * (2.0 * 3.141592653589793 * k * n / N))


dft.to_sdfg(simplify=True).compile()
```

Generated C++ and the error:

```cpp
const dace::complex128 exp_expr_... =
    exp((((dace::complex128(-0.0, -6.28318530717959) * __ft0_k0) * __ft0_n0) / N));
```

```
error: no match for ‘operator/’ (operand types are ‘cmplx<double>’
       {aka ‘std::complex<double>’} and ‘int64_t’ {aka ‘long int’})
```

Note that the two multiplications on the same line **do** compile -- via the `const int&` overload
above, narrowing `__ft0_k0` and `__ft0_n0` to 32 bits on the way. That is defect (b), sitting one
token to the left of defect (a) in the same expression. For a transform larger than `2**31` points
the compile error is the lucky outcome; the multiplication would have returned a wrong twiddle
factor quietly.

For (b) on its own, straight against the header:

```cpp
#include <dace/dace.h>
int64_t k = 3000000000LL;
dace::complex128 c(1.0, 0.0);
auto r = c * k;      // r.real() == -1294967296.0
```

## Expected vs actual

- **Expected:** the mixed operators are complete (`+`, `-`, `*`, `/`, both operand orders) and
  templated on the integer type, so `int64_t` participates at full width. Promoting the integer to
  `T` before the operation is the whole fix:

  ```cpp
  template <typename T, typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
  cmplx<T> operator/(const cmplx<T>& a, const I& b) { return a / static_cast<T>(b); }
  ```

  (and likewise for `*`, replacing the two `const int&` overloads, plus `+` and `-` for symmetry).
- **Actual:** `/` does not compile; `*` compiles and truncates.

Fixing the header is preferable to casting in codegen: the Python-tasklet C++ unparser emits free
symbols by name and does not know the surrounding expression's dtype, so a cast would have to be
inserted at every mixed site. Note that the paths which *do* know the dtypes already cast correctly
-- the same kernel's `z[k] / N` statement is emitted as `z_index / dace::complex128(N)` and compiles
fine. Only the const-folded symbolic expression inside a replacement's argument is affected.

## Impact

Found while running a NumPy HPC kernel corpus through the frontend. `fft_1d` (`np.fft.fft` /
`np.fft.ifft` lowered to a naive DFT) fails to build for (a) with exactly two errors, both
`complex / int64_t` in the twiddle and nothing else. Defect (b) is latent in every complex kernel
whose problem size exceeds `2**31`.

**Workaround for (a):** keep the whole exponent REAL and only meet `1j` at the outermost level, so
the division never sees a complex operand:

```python
ang = -2.0 * 3.141592653589793 * k * n / N     # all-real: double / int64_t compiles
y[k] += x[n] * np.exp(1j * ang)
```

Verified: builds, and agrees with `np.fft.fft` to `1.9e-15`. Multiplying the divisor by `1.0`
instead (`/ (1.0 * N)`) does **not** work -- sympy folds the factor back out and the emitted C++ is
byte-identical to the failing form.

There is no workaround for (b) short of patching the header, since the narrowing overload is
selected without the user writing anything.
