# Python frontend: `b = a` on a scalar aliases the container, so a later write through either name corrupts the other

## Status

- **Fixed on `extended`** in `0d51c5c53` ("fix: copy scalars on assignment instead of aliasing them"):
  `_visit_assign` now takes the `_add_transient_data` copy path when the right-hand side descriptor is a
  `data.Scalar`; arrays keep aliasing. Regression tests in
  `tests/python_frontend/assignment_statements_test.py` (`test_scalar_assignment_copies`,
  `test_scalar_chained_assignment_copies`, `test_scalar_assignment_branch_dispatch`,
  `test_array_assignment_aliases`).
- **Still open on `main`**, reported upstream as https://github.com/spcl/dace/issues/2491.

## Environment

- dace: `spcl/dace` `main` @ `d7efcef0c96580e590caf7003c8320ba8908239c` (`dace.__version__ == 2.0.0a5`),
  and `extended` @ `ac939852548ac8aa43085690a6b32ec6c5887d2e`. The code path is byte-identical on both;
  every line number below is `main`'s.
- Python 3.12.11, numpy 2.4.4, sympy 1.14.0, Linux x86_64
- Runtime values, so a C++ compiler is involved. The descriptor probe is parse-only.
- Runnable reproducer: `repro/05-scalar-assign-aliases.py`
  (`PYTHONPATH=/path/to/dace python3 repro/05-scalar-assign-aliases.py`)

## Summary

`b = a`, where `a` names a transient **scalar**, does not copy the value -- it makes `b` a second
name for `a`'s data container. Python floats and ints are immutable, so this diverges from NumPy
semantics the moment anything writes through either name. There is no error, no warning, and the
SDFG is valid: the answer is just wrong.

```python
@dace.program
def plain(out: dace.float64[2]):
    a = 1.0
    b = a
    b += 1.0
    out[0] = a      # DaCe: 2.0     Python: 1.0
    out[1] = b      # DaCe: 2.0     Python: 2.0
```

## Two shapes it takes in real code

### (a) Chained initialisation collapses every accumulator into one

`s0 = s1 = ... = 0.0` is the standard way to open a hand-unrolled reduction. Desugared -- by hand or
by a source-to-source tool -- into a temp plus one assignment per target, all the accumulators become
the same container and the reduction over-counts by the unroll factor.

```python
@dace.program
def chained(a: dace.float64[N], out: dace.float64[1]):
    tmp = 0.0
    s0 = tmp
    s1 = tmp
    i = 0
    while i + 2 <= N:
        s0 += a[i + 0]
        s1 += a[i + 1]
        i += 2
    out[0] = s0 + s1
```

For `a = [1..8]`: DaCe returns `72.0`, NumPy returns `36.0` -- exactly `2x`, the unroll factor. At
11 accumulators it is `11x`.

### (b) A branch dispatch writes through the alias and destroys the source

This one is worse, because the corrupted value is one the program still needs.

```python
@dace.program
def branches(out: dace.float64[6]):
    rp0, rp1, rp2 = 10.0, 20.0, 30.0
    for idir in range(3):
        if idir == 0:
            pc = rp0        # pc becomes an ALIAS of rp0
        elif idir == 1:
            pc = rp1        # writes rp1's value INTO rp0
        else:
            pc = rp2        # writes rp2's value INTO rp0
        out[idir] = pc
    out[3], out[4], out[5] = rp0, rp1, rp2
```

Expected `[10, 20, 30, 10, 20, 30]`, actual `[10, 20, 30, 30, 20, 30]`. `rp0` is gone, replaced by
`rp2`. The dispatched reads (`out[0..2]`) are all correct, which is what makes this hard to spot:
only the *original* is damaged, and only after the loop. In `simplify=False` the SDFG has containers
`{out, pc, rp1, rp2}` -- there is no `rp0` left at all.

This is the exact shape of an `if idir == 0: center = center0 / span = span0 / product_center = rp0`
axis dispatch, which is idiomatic in ported C and Fortran. In a CP2K grid-integration kernel it
silently destroyed all three x-axis quantities and moved the final result by `4.3e+00`.

## Mechanism

`_visit_assign` (`dace/frontend/python/newast.py:3562`) ends its "target not yet defined" ladder with
a bare aliasing fall-through:

```python
# newast.py:3659-3665
elif not result_data.transient or result in self.sdfg.constants_prop:
    true_name, new_data = _add_transient_data(self, self.sdfg, result_data, dtype)
    self.variables[name] = true_name
    defined_vars[name] = true_name
else:
    self.variables[name] = result          # <-- alias, no copy
    defined_vars[name] = result
    continue
```

A **non**-transient result is copied through `_add_transient_data`; a transient one is aliased. That
distinction is right for arrays -- NumPy aliases arrays too -- and wrong for scalars, where Python
rebinds rather than mutates.

The second half is the guard that would otherwise have caught the reassignment in case (b):

```python
# newast.py:3565-3566
if (not is_return and isinstance(target, ast.Name) and true_name and not op
        and not isinstance(true_array, data.Scalar) and not (true_array.shape == (1, ))):
    ...
    raise DaceSyntaxError(self, target, 'Cannot reassign value to variable "{}"'.format(name))
```

Scalars are explicitly **exempted** from the reassign check, so `pc = rp1` on an aliased scalar falls
straight through to the memlet-writing path at `newast.py:3673+` and stores into `rp0`.

## Expected vs actual

- **Expected:** `b = a` on a scalar copies the value -- `b` gets its own container -- so neither
  `b += ...` nor a later `b = c` can be observed through `a`. That is what NumPy/Python do, and what
  `b = a` on a *non-transient* scalar already does via `_add_transient_data`.
- **Actual:** `b` aliases `a`. Every subsequent write through `b` lands in `a`.

Suggested shape of a fix, in order of how surgical it is:

1. In the `else` branch at `newast.py:3663`, take the `_add_transient_data` copy path when
   `result_data` is a `data.Scalar` (or `shape == (1,)`), and keep aliasing only for real arrays.
2. Failing that, drop `not isinstance(true_array, data.Scalar) and not (true_array.shape == (1,))`
   from the guard at `newast.py:3566` for the aliased case specifically, so that a reassignment
   through an alias is at least a `DaceSyntaxError` rather than a silent clobber. That turns (b) into
   a diagnostic, but leaves (a) wrong.

## Impact

Found while running a NumPy HPC kernel corpus through the frontend and grading it against the NumPy
reference. Three corpus kernels return wrong numbers for this one cause:

| kernel | output | worst absolute error | shape |
| --- | --- | --- | --- |
| `unroll_reduction_11_accs` | `out` | `1.12e+03` | (a), 11 accumulators -- exactly `11 * main + tail` |
| `s353_gather_reduction_unroll` | `b` | `5.41e+02` | (a), 7 accumulators -- exactly `7 * main + tail` |
| `cp2k_grid_integrate` | `hab` | `4.32e+00` | (b), an `idir` axis dispatch |

For `cp2k_grid_integrate`, rewriting the three aliasing lines as `center = center0 + 0`,
`span = span0 + 0`, `product_center = rp0 + 0.0` -- forcing a fresh container and changing nothing
else -- brings every intermediate and the final output to `8.9e-16` of NumPy.

**Workaround:** never write `b = a` for a scalar. Force a new container with a no-op operation
(`b = a + 0.0`) or repeat the initialiser at each target (`s0 = 0.0; s1 = 0.0`) instead of chaining.
