# SDFG Symbol Registry

Each `SDFG` owns a registry that uniquely identifies every symbol by its
name, DaCe dtype, and sympy assumptions. Two references to the same name
always resolve to the same `dace.symbolic.symbol` object — no more silent
identity drift between different parts of the SDFG.

## Invariants

- **`sdfg.symbols`** is a `SymbolDict` (not a `dict`). Reads behave like a
  dict; in-place mutation raises `RuntimeError` and points the caller at the
  API. Internal mutators inside `SDFG` write to `self._symbols._data`.
- **`sdfg._symbol_objects`** stores the canonical `symbolic.symbol` object
  per name. Only the API methods write to it.
- A symbol's identity is `(dtype, sympy assumptions)`. Re-registering a name
  with matching identity is idempotent; conflicting identity is rejected.

## API

| Method | Purpose |
| --- | --- |
| `sdfg.add_symbol(name, dtype, find_new_name=False, **assumptions)` | Register. Idempotent on match; raises `FileExistsError` on conflict unless `find_new_name=True`. |
| `sdfg.set_symbol_type(name, dtype)` | In-place dtype swap; rebuilds the canonical object. |
| `sdfg.remove_symbol(name)` | Drop name from `symbols` and `_symbol_objects`. |
| `sdfg.replace(name, new_name)` | Rename, rebuilding the canonical object under the new name. |
| `sdfg.get_symbol(name)` | Returns the canonical `symbolic.symbol`. Stable identity within one SDFG. |

## SymbolDict

Reads identical to `dict`: `[]`, `in`, iteration, `.items()`, `.keys()`,
`.values()`, `len()`, `==`, `dict(sd)`, `{**sd}`, `sd | other`, `other | sd`,
`sd.copy()` (returns a `SymbolDict`).

Mutations rejected with a single error pointing at the API: `[]=`, `del`,
`pop`, `popitem`, `update`, `setdefault` (new key), `clear`, `|=`.

The composition-over-`dict` design (rather than a `dict` subclass) is on
purpose — `dict` C-slot descriptors (`dict.__setitem__`, `dict.update`, ...)
bypass Python overrides on a `dict` subclass, so the only way to fully
intercept mutation is to not be a `dict` at all.

## Bypass coverage

Caught: every `[]=` / `del` / `pop` / `update` / `setdefault` / `clear`,
plus calling `dict.__setitem__(sd, ...)` and the other `dict.*` C-slots
directly (they fail with `TypeError` because `sd` is not a `dict`).

Not caught: `object.__setattr__(sdfg, '_symbols', plain_dict)` and
`sdfg.__dict__['_symbols'] = plain_dict`. These are explicit Python
introspection idioms; pure-Python code cannot prevent them. Out of scope.

## Cloning

`SymbolDict.__copy__` and `__deepcopy__` return fresh `SymbolDict`s. Local
accumulators that want a freely-mutable copy should call `dict(sdfg.symbols)`
explicitly.

## Nested SDFGs

Each SDFG owns its own registry. The same name in inner and outer SDFGs
yields two independent canonical objects — the registry is per-SDFG, not
per-process. Coordination across the parent/nested boundary (matching
dtypes, propagating renames) remains the caller's responsibility, the same
way DaCe has always treated nested-SDFG `symbol_mapping`.

`add_nested_sdfg` propagates free symbols from the parent to the inner SDFG
by calling `add_symbol` on the inner — fresh canonical object, with the
parent's dtype.
