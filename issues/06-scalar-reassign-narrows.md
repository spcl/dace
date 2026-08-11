# Python frontend: a name first bound to an int keeps an integer container, and a later float assignment is silently truncated

## Environment

- dace: `spcl/dace` `main` @ `d7efcef0c96580e590caf7003c8320ba8908239c` (`dace.__version__ == 2.0.0a5`),
  and `extended` @ `ac939852548ac8aa43085690a6b32ec6c5887d2e`. Identical on both; line numbers are `main`'s.
- Python 3.12.11, numpy 2.4.4, sympy 1.14.0, Linux x86_64
- Runnable reproducer: `repro/06-scalar-reassign-narrows.py`
  (`PYTHONPATH=/path/to/dace python3 repro/06-scalar-reassign-narrows.py`)

## Summary

A scalar's dtype is fixed by its **first** assignment. A later assignment of a float to the same
name does not re-bind the name and does not widen the container -- it writes into the existing
integer container through an implicit narrowing cast. No warning is emitted.

The idiom this breaks is the convergence loop:

```python
udiff = 1                     # int, so `udiff` becomes an int64 scalar
while udiff > 0.001:
    ...
    udiff = <float expression>  # truncated to int64
```

Once the true residual drops below `1.0` the stored value is `0`, the condition `udiff > 0.001` is
false, and the loop exits -- typically after **two** trips regardless of the tolerance.

## Minimal reproducer

```python
@dace.program
def converge(a: dace.float64[N], trips: dace.int64[1], last: dace.float64[1]):
    udiff = 1
    n = 0
    while udiff > 0.001:
        a[:] = a[:] * 0.5
        udiff = 0.5 + 0.0 * a[0]   # a float, constant at 0.5, never below 1e-3
        n += 1
    trips[0] = n
    last[0] = udiff
```

Python never leaves this loop (`0.5 > 0.001` forever). DaCe returns `trips = 1`, `last = 0.0`.

## Mechanism

The first assignment types the container from the RHS (`dace/frontend/python/newast.py:3609-3625`):

```python
if result in self.sdfg.symbols:
    rtype = self.sdfg.symbols[result]
elif symbolic.issymbolic(result):
    rtype = sym_type(result)
else:
    rtype = type(result)                       # <-- Python `int` for `udiff = 1`
...
true_name, new_data = self.sdfg.add_scalar(true_name, ttype, transient=True, find_new_name=True)
```

The second assignment finds `udiff` in `defined_vars`, so `true_name` is set. The guard that raises
`Cannot reassign value to variable` (`newast.py:3565-3582`) explicitly **exempts** scalars:

```python
and not isinstance(true_array, data.Scalar) and not (true_array.shape == (1, ))
```

so control reaches the memlet-writing path, which stores the double-valued RHS into the `int64_t`
container. The parsed SDFG shows both halves plainly -- `udiff` (`int64_t`, what the loop condition
reads) and `udiff_0` (`double`, the RHS temp):

```
udiff containers: {'udiff': 'int64_t', 'udiff_0': 'double'}
loop condition reads: (udiff > 0.001)
```

and the generated graph is `_Div_ -> udiff_0 (double)` then `assign -> udiff (int64_t)`.

## Expected vs actual

- **Expected:** either Python's rebinding semantics (the name takes the new value and type), or a
  hard `DaceSyntaxError` naming the two dtypes. A silent narrowing double -> int64 on an assignment
  the user did not write a cast for is the one outcome that cannot be debugged from the output.
- **Actual:** silent truncation.

The scalar exemption in the reassign guard at `newast.py:3566` is the single line to revisit. Even
keeping the exemption for same-kind reassignment, a dtype change from float to int (or int to float)
should be rejected, or the container widened.

## Impact

Found while grading a NumPy HPC kernel corpus against its NumPy reference. `channel_flow` (a
Navier-Stokes channel solver, iterate-to-convergence on `udiff = (sum(u) - sum(un)) / sum(u)`) is
wrong by `4.41e-02` in `u`.

Localised exactly: with `ny = nx = 48`, `nit = 5`, DaCe's `u` is **bit-identical** to the NumPy
reference's `u` after **2** outer iterations, out of the several hundred NumPy needs. The residual
sequence is `1.0, 0.49997, 0.33330, ...` -- truncated to `1, 0, 0, ...`, so the loop takes the first
trip on the seeded `udiff = 1`, the second on the truncated `1`, and stops.

**Workaround:** spell the initialiser as a float -- `udiff = 1.0`.
