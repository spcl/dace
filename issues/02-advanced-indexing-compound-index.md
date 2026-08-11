# Python frontend: indexing with a computed index crashes with `AttributeError: 'str' object has no attribute '_fields'`

## Environment

- dace: `spcl/dace` `main` @ `d7efcef0c96580e590caf7003c8320ba8908239c` (`dace.__version__ == 2.0.0a5`)
- Python 3.12.11, numpy 2.4.4, sympy 1.14.0, Linux x86_64
- Parse-time only, no C++ compiler involved (`to_sdfg(simplify=False)`)
- Re-verified 2026-08-06: identical failure on that `main` and on a downstream branch carrying
  later frontend commits (`8d749ff2c`). Only line numbers shift; every line number below is `main`.
- Runnable reproducer with all three cases and the controls:
  `repro/02-advanced-indexing-compound-index.py`
  (`PYTHONPATH=/path/to/dace python3 repro/02-advanced-indexing-compound-index.py`)

## Minimal reproducer

```python
import dace


@dace.program
def f(p: dace.float64[10], cols: dace.int64[5], out: dace.float64[5]):
    for i in range(5):
        out[i] = p[int(cols[i])]


f.to_sdfg(simplify=False)
```

Two more spellings of the same crash:

```python
# (b) an index produced by a reduction
@dace.program
def g(U: dace.float64[10, 10], absU: dace.float64[10, 10], out: dace.float64[10]):
    for j in range(10):
        out[j] = U[np.argmax(absU[:, j]), j]


# (c) a list-literal index with a symbolic element
N = dace.symbol('N', dtype=dace.int64)


@dace.program
def h(x: dace.float64[10], out: dace.float64[2]):
    out[:] = x[[0, N]]        # -> AttributeError: 'symbol' object has no attribute '_fields'
```

A list-literal index whose elements are all plain numbers (`x[[0, 2, 4]]`) parses
fine, which is the discriminator: the crash needs a *non-`Number`* component.

## Traceback tail

```
  File ".../dace/frontend/python/newast.py", line 5540, in visit_Subscript
    return self._add_read_slice(array, node, expr)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../dace/frontend/python/newast.py", line 5280, in _add_read_slice
    return self._array_indirection_subgraph(rnode, expr)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../dace/frontend/python/newast.py", line 5839, in _array_indirection_subgraph
    output_shape = self._compute_output_shape_from_advanced_indexing(aname, expr)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../dace/frontend/python/newast.py", line 5667, in _compute_output_shape_from_advanced_indexing
    arrname = [v if isinstance(v, Number) else self._parse_value(v) for v in arrname]
                                               ^^^^^^^^^^^^^^^^^^^^
  File ".../dace/frontend/python/newast.py", line 1852, in _parse_value
    return str(self.visit(node))
               ^^^^^^^^^^^^^^^^
  File ".../dace/frontend/python/newast.py", line 1238, in visit
    result = super().visit(node)
             ^^^^^^^^^^^^^^^^^^^
  File ".../python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File ".../dace/frontend/python/astutils.py", line 527, in generic_visit
    for field, old_value in ast.iter_fields(node):
                            ^^^^^^^^^^^^^^^^^^^^^
  File ".../python3.12/ast.py", line 262, in iter_fields
    for field in node._fields:
                 ^^^^^^^^^^^^
AttributeError: 'str' object has no attribute '_fields'
```

## Mechanism

Replacements return their results as a Python **list of data names** (a ufunc may
have several outputs), so `_parse_subscript_slice`'s `_promote` helper
(`dace/frontend/python/newast.py:5380-5388`) hands back `['int_cols_slice']` for the
index expression `int(cols[i])`. `_fill_missing_slices` then matches that on
`isinstance(dim.id, (list, tuple))` -- the branch meant for a *Python list literal*
index, i.e. advanced indexing (`dace/frontend/python/memlet_parser.py:134-137`) --
and records `expr.arrdims = {0: ['int_cols_slice']}` (verified by inspecting the
frame).

`_compute_output_shape_from_advanced_indexing`
(`newast.py:5613`, offending line `newast.py:5667`) then walks that "literal"
element by element; `'int_cols_slice'` is not a `Number`, so it goes to
`_parse_value`, whose fallthrough is `str(self.visit(node))` (`newast.py:1852`).
`node` is already a `str`, `ProgramVisitor` has no `visit_str`, so dispatch lands in
`ExtNodeVisitor.generic_visit` (`dace/frontend/python/astutils.py:526-527`) which
calls `ast.iter_fields()` on a string.

Case (c) is the same line reached from a genuine list literal: `visit_List` yields
`[0, symbol('N')]`, and `_parse_value(symbol('N'))` produces
`'symbol' object has no attribute '_fields'`.

## Expected vs actual

- **Expected:** either the index expression is materialised into a scalar/array and
  used as an indirection (which DaCe already supports when the index is a plain
  array name, e.g. `p[cols]`), or a `DaceSyntaxError` naming the unsupported
  construct and the source line.
- **Actual:** an `AttributeError` from CPython's `ast` module, produced by feeding a
  `str` into the AST visitor. The message gives no hint about which index expression
  is at fault.

Fails identically with `simplify=True` -- the crash is during
`newast.parse_dace_program`, before any simplification pass runs.

## Impact

Hit by 3 kernels in a NumPy HPC kernel corpus we are porting to DaCe (pivot
selection: `U[np.argmax(absU[:, j]), j]`, and gather-with-cast:
`p[int(cols[idx])]`).

**Workaround:** hoist the index into its own statement first, so the subscript sees
a plain name rather than an expression. This is enough for case (a):

```python
k = int(cols[i])
out[i] = p[k]
```

For case (b) the plain hoist is *not* enough -- `k = np.argmax(absU[:, j])` then
fails with a separate error, `ValueError: View "absU_0" already has both incoming
and outgoing edges` (`newast.py:1329`, `_views_to_data`). The slice has to be
materialised into an explicit temporary first:

```python
tmp = np.zeros(10, dtype=np.float64)
tmp[:] = absU[:, j]
k = np.argmax(tmp)
out[j] = U[k, j]
```

That explicit-temporary workaround holds on `main` @ `d7efcef0c`. It does **not**
hold on the downstream branch used for the second verification: as of `9feb23929`
("Reduce argmax/argmin through two scalars instead of a struct", which rewrites
`dace/frontend/python/replacements/reduction.py`) it fails with
`ValueError: View "tmp_0" already has both incoming and outgoing edges` -- the same
`_views_to_data` check as the plain hoist, now reached one step earlier. So on that
branch case (b) currently has no workaround at all. This does not affect the bug
reported here, which reproduces on both.

For case (c) we found no workaround other than avoiding symbolic elements inside a
list-literal index.
