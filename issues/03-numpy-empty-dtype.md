# Python frontend: `np.empty(shape)` without `dtype` fails, unlike `np.zeros`/`np.ones` and unlike NumPy

This one is more useful as a PR than as an issue: the fix is a default value on one
parameter, and the exact hunk is at the bottom.

## Environment

- dace: `spcl/dace` `main` @ `d7efcef0c96580e590caf7003c8320ba8908239c` (`dace.__version__ == 2.0.0a5`)
- Python 3.12.11, numpy 2.4.4, sympy 1.14.0, Linux x86_64
- Parse-time only, no C++ compiler involved (`to_sdfg(simplify=False)`)
- Re-verified 2026-08-06: identical failure on that `main` and on a downstream branch carrying
  later frontend commits (`8d749ff2c`). Only line numbers shift; every line number below is `main`.
- Runnable reproducer, which also checks the resulting dtype once the patch is applied:
  `repro/03-numpy-empty-dtype.py`
  (`PYTHONPATH=/path/to/dace python3 repro/03-numpy-empty-dtype.py`)

## Minimal reproducer

```python
import numpy as np
import dace


@dace.program
def f(x: dace.float64[10]):
    y = np.empty(10)
    y[:] = 1.0
    x[:] = y


f.to_sdfg(simplify=False)
```

The identical program with `np.zeros(10)` (also no `dtype`) parses fine.

## Traceback tail

```
Exception _numpy_empty() missing 1 required positional argument: 'dtype' raised while parsing DaCe program:
  in File "repro.py", line 7
    y = np.empty(10)
Traceback (most recent call last):
  ...
  File ".../dace/frontend/python/newast.py", line 3490, in _visit_assign
    rval = self._gettype(node.value)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../dace/frontend/python/newast.py", line 5148, in _gettype
    operands = self.visit(opnode)
               ^^^^^^^^^^^^^^^^^^
  File ".../dace/frontend/python/newast.py", line 1238, in visit
    result = super().visit(node)
             ^^^^^^^^^^^^^^^^^^^
  File ".../python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File ".../dace/frontend/python/newast.py", line 4868, in visit_Call
    result = func(self, self.sdfg, self.last_block, *args, **keywords)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: _numpy_empty() missing 1 required positional argument: 'dtype'
```

## Mechanism

The `numpy.empty` replacement declares `dtype` as a **required positional
parameter** with no default:

```python
# dace/frontend/python/replacements/array_creation_dace.py:163-166
@oprepo.replaces('numpy.empty')
def _numpy_empty(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dtypes.typeclass):
    """ Creates an unitialized array of the specificied shape and dtype. """
    return _define_local(pv, sdfg, state, shape, dtype)
```

Every neighbouring creation replacement defaults it:

| replacement | location | `dtype` default |
| --- | --- | --- |
| `numpy.empty` | `array_creation_dace.py:163-166` | **none -- required** |
| `numpy.empty_like` | `array_creation_dace.py:169-175` | `None` |
| `numpy.zeros` | `array_creation.py:181-186` | `dtypes.float64` |
| `numpy.ones` | `array_creation.py:157-162` | `dtypes.float64` |
| `numpy.full` | `array_creation.py:42-48` | `None` |

Only `numpy.empty` is missing it, so `visit_Call` calls the replacement with one
argument and Python raises before DaCe can produce a useful diagnostic.

## Expected vs actual

- **Expected:** `np.empty(10)` allocates an uninitialised `float64[10]`, exactly as
  `numpy.empty(10)` does and exactly as DaCe's own `np.zeros(10)` already does.
- **Actual:** `TypeError: _numpy_empty() missing 1 required positional argument:
  'dtype'` -- a raw Python arity error naming a DaCe-internal function, for a call
  that is valid NumPy.

Fails identically with `simplify=True` -- the crash is during
`newast.parse_dace_program`, before any simplification pass runs.

## Fix

Give `dtype` the same default its siblings have. Written on one line the signature
is 123 characters, over the repo's 120-column `yapf` limit, so it wraps one
parameter per line -- which is how `_numpy_zeros` and `_numpy_ones` are already
formatted. The hunk below is `yapf --style=.style.yapf` clean.

```diff
--- a/dace/frontend/python/replacements/array_creation_dace.py
+++ b/dace/frontend/python/replacements/array_creation_dace.py
@@ -161,7 +161,11 @@
 
 
 @oprepo.replaces('numpy.empty')
-def _numpy_empty(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dtypes.typeclass):
+def _numpy_empty(pv: ProgramVisitor,
+                 sdfg: SDFG,
+                 state: SDFGState,
+                 shape: Shape,
+                 dtype: dtypes.typeclass = dtypes.float64):
     """ Creates an unitialized array of the specificied shape and dtype. """
     return _define_local(pv, sdfg, state, shape, dtype)
```

`dtypes` is already imported in that module (`array_creation_dace.py:9`), so the
hunk stands alone.

Verified against `main` @ `d7efcef0c` with the hunk applied:

- `np.empty(10)` parses, and the transient it creates is `float64` -- matching
  `numpy.empty`'s own default;
- `np.empty(10, dtype=np.int32)` still produces an `int32` transient, so the
  explicit-dtype path is unchanged.

## Impact

Hit while porting a NumPy HPC kernel corpus to DaCe; `np.empty(shape)` with no
`dtype` is idiomatic NumPy for scratch buffers.

**Workaround exists and is trivial:** pass the dtype explicitly,
`np.empty(10, dtype=np.float64)`, or use `np.zeros(10)`.
