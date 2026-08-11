# Python frontend: `if sym == a or sym == b:` fails with `TypeError: 'Equality' object is not iterable`

## Environment

- dace: `spcl/dace` `main` @ `d7efcef0c96580e590caf7003c8320ba8908239c` (`dace.__version__ == 2.0.0a5`)
- Python 3.12.11, numpy 2.4.4, sympy 1.14.0, Linux x86_64
- Parse-time only, no C++ compiler involved (`to_sdfg(simplify=False)`)
- Re-verified 2026-08-06: identical failure on that `main` and on a downstream branch carrying
  later frontend commits (`8d749ff2c`). Only line numbers shift; every line number below is `main`.
- Runnable reproducer with all seven cases below: `repro/01-symbolic-or.py`
  (`PYTHONPATH=/path/to/dace python3 repro/01-symbolic-or.py`)

## Minimal reproducer

```python
import dace

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def symbolic_or(x: dace.float64[10]):
    if N == 0 or N == 1:
        x[:] = 1.0
    else:
        x[:] = 2.0


symbolic_or.to_sdfg(simplify=False)
```

The same program with a single comparison (`if N == 0:`) parses fine. So does the
`elif` spelling of the same condition (see *Workaround*).

## Traceback tail

```
  File ".../dace/frontend/python/parser.py", line 896, in _generate_pdp
    parsed_ast, closure = preprocessing.preprocess_dace_program(dace_func,
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../dace/frontend/python/preprocessing.py", line 1700, in preprocess_dace_program
    src_ast = ConditionalCodeResolver(resolved).visit(src_ast)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File ".../python3.12/ast.py", line 483, in generic_visit
    value = self.visit(value)
            ^^^^^^^^^^^^^^^^^
  File ".../python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File ".../dace/frontend/python/preprocessing.py", line 159, in visit_If
    test = RewriteSympyEquality(self.globals_and_locals).visit(node.test)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File ".../python3.12/ast.py", line 487, in generic_visit
    new_values.extend(value)
TypeError: 'Equality' object is not iterable
```

## Mechanism

`RewriteSympyEquality` is an `ast.NodeTransformer`, and its `visit_Compare`
(`dace/frontend/python/preprocessing.py:122-132`) returns a bare `sympy.Eq` /
`sympy.Ne` -- an object that is not an `ast.AST`. That is a violation of the
`NodeTransformer` contract: CPython's `generic_visit` treats a non-`AST` return
value coming from a **list-valued field** as a *sequence of replacement nodes* and
splices it with `new_values.extend(value)` (CPython `ast.py:487`).

When the `Compare` *is* the whole `if` test, the transformer's root return value is
consumed directly by `ConditionalCodeResolver.visit_If`
(`preprocessing.py:159-160`, which then hands it to `astutils.evalnode`) and never
reaches `generic_visit` -- which is why `if N == 0:` works. But `BoolOp.values` is a
list field, so `if N == 0 or N == 1:` routes the `Equality` into the `extend()`
above and raises.

The same root cause produces a second symptom on **single-`AST`-valued** fields,
where `generic_visit` does `setattr(node, field, new_node)` and plants the sympy
object in the tree, to be tripped over later. E.g. an `IfExp` in the test:

```python
if (N == 0) if N > 5 else (N == 1):   # -> AttributeError: 'ExtUnparser' object has no attribute '_Equality'
```

## Expected vs actual

- **Expected:** `if N == 0 or N == 1:` gets the same treatment as `if N == 0:` --
  folded to the taken branch when the symbol's value is known at parse time, and
  left as an indeterminate condition otherwise.
- **Actual:** `TypeError: 'Equality' object is not iterable`, raised from CPython's
  `ast` module, with no reference to the offending user line.

Both `or` and `and` reproduce (verified separately). Non-equality comparisons are
unaffected -- `if N < 1 or N > 5:` parses fine -- because `visit_Compare` only
short-circuits for `ast.Eq` / `ast.NotEq`. `if not (N == 0):` also happens to
survive.

The failure is in preprocessing, before any SDFG is built, so it is identical with
`simplify=True` and `simplify=False`.

## Possible direction

At minimum `visit_Compare` must not return a non-`AST` object when the `Compare` is
nested. Handling `ast.BoolOp` explicitly in `RewriteSympyEquality` -- mapping
`or`/`and` onto `sympy.Or`/`sympy.And` over the rewritten operands -- would let the
existing constant-folding in `visit_If` work for these conditions instead of only
for single comparisons.

## Impact

Hit by 6 kernels in a NumPy HPC kernel corpus we are porting to DaCe. The pattern
is common shape/rank dispatch, e.g. `if dim == 0 or dim == -2:`.

**Workaround exists:** rewrite the disjunction as an `elif` chain (or nested `if`s)
so that every `if` test is a single `Compare`:

```python
if N == 0:
    x[:] = 1.0
elif N == 1:
    x[:] = 1.0
else:
    x[:] = 2.0
```
