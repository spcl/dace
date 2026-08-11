# Python frontend: `.copy()` on a sliced array raises a bare `NotImplementedError` with an empty message

## Environment

- dace: `spcl/dace` `main` @ `d7efcef0c96580e590caf7003c8320ba8908239c` (`dace.__version__ == 2.0.0a5`)
- Python 3.12.11, numpy 2.4.4, sympy 1.14.0, Linux x86_64
- Parse-time only, no C++ compiler involved (`to_sdfg(simplify=False)`)
- Re-verified 2026-08-06: identical failure on that `main` and on a downstream branch carrying
  later frontend commits (`8d749ff2c`). Only line numbers shift; every line number below is `main`.
- Runnable reproducer, which also prints the descriptor probe quoted below:
  `repro/04-arrayview-copy.py`
  (`PYTHONPATH=/path/to/dace python3 repro/04-arrayview-copy.py`)

## Minimal reproducer

```python
import dace


@dace.program
def f(path: dace.float64[10, 10], out: dace.float64[10]):
    out[:] = path[:, 1].copy()


f.to_sdfg(simplify=False)
```

Also fails for a contiguous row slice, `path[1, :].copy()`, and via the free
function, `np.copy(path[:, 1])`. Copying the **whole** array, `path.copy()`, works.

## Traceback tail

```
  File ".../dace/frontend/python/newast.py", line 4868, in visit_Call
    result = func(self, self.sdfg, self.last_block, *args, **keywords)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../dace/frontend/python/replacements/array_creation.py", line 39, in _ndarray_copy
    return _numpy_copy(pv, sdfg, state, arr)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../dace/frontend/python/replacements/array_creation.py", line 28, in _numpy_copy
    name, desc = _add_transient_data(pv, sdfg, sdfg.arrays[a])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../dace/frontend/python/newast.py", line 163, in _add_transient_data
    raise NotImplementedError
NotImplementedError
```

Note the exception carries **no message at all** -- no data name, no descriptor type,
no source line. On a large program this is close to undebuggable.

## Mechanism

A slice such as `path[:, 1]` is materialised by the frontend through
`SDFG.add_view` (`dace/frontend/python/newast.py:5346`, in `_add_read_slice`),
which produces a `data.ArrayView` (`dace/data/core.py:1277`,
`class ArrayView(Array, View)`).

`.copy()` dispatches correctly -- `_ndarray_copy` *is* registered for `View`
(`dace/frontend/python/replacements/array_creation.py:37-39`) -- and reaches
`_numpy_copy`, which calls `_add_transient_data`
(`dace/frontend/python/newast.py:158-163`). That function resolves the descriptor
handler with an **exact-type dict lookup**, `AddTransientMethods.get(type(sample_data))`
(`newast.py:161`, backed by the `datatype not in AddTransientMethods._methods` test
at `newast.py:115-118`). The dict is registered for exactly four keys --
`data.Scalar`, `data.Array`, `data.View`, `data.Stream`
(`newast.py:127`, `:134`, `:140`, `:146`).

`ArrayView` is a concrete subclass, not one of those four keys, so the lookup
returns `None` and the bare `raise NotImplementedError` at `newast.py:163` fires.
Verified at runtime:

```
descriptor handed to _add_transient_data: ArrayView
its MRO:                                  ['ArrayView', 'Array', 'Data', 'View', 'object']
AddTransientMethods keys:                 ['Scalar', 'Array', 'View', 'Stream']
```

## Expected vs actual

- **Expected:** `path[:, 1].copy()` produces a new transient holding a copy of the
  slice, matching NumPy semantics and matching what `path.copy()` already does.
- **Actual:** `NotImplementedError` with an empty message.

Two separable defects here, and the second is worth fixing on its own even if the
first is intentional:

1. The exact-type dispatch does not see subclasses. `ArrayView` has
   `[ArrayView, Array, Data, View, object]` as its MRO, so an MRO-ordered lookup
   would land on the `Array` handler -- which copies shape and dtype and is very
   likely the right behaviour for a copy of a view.
2. `raise NotImplementedError` with no argument gives the user nothing. It should
   at least name the descriptor type and the data, e.g.
   `raise NotImplementedError(f'Cannot create a transient like {type(sample_data).__name__} "{...}"')`.

Fails identically with `simplify=True` -- the crash is during
`newast.parse_dace_program`, before any simplification pass runs.

## Impact

Hit while porting a NumPy HPC kernel corpus to DaCe. `col = M[:, k].copy()` is the
idiomatic NumPy way to detach a column before mutating it, and it is exactly the
call that has to work for the copy to be meaningful.

**No clean workaround.** The obvious alternatives fail the same way
(`np.copy(path[:, 1])` hits the identical `_add_transient_data` path). The only
route we found is to allocate the temporary explicitly and assign into it:

```python
tmp = np.zeros(10, dtype=np.float64)
tmp[:] = path[:, 1]
```
