# SDFG Symbol Registry

Each `SDFG` owns a guarded registry that uniquely identifies every symbol by
its name, DaCe dtype, and sympy assumptions. The registry exists so that two
references to the same name always resolve to the same `dace.symbolic.symbol`
object — preventing the same logical symbol from existing as multiple
inconsistent objects with conflicting types or assumptions.

## Invariants

- **`sdfg.symbols`** stores the dtype of every symbol the SDFG knows about.
  Direct mutation (`sdfg.symbols['N'] = ...`, `del sdfg.symbols['N']`,
  `pop`/`update`/`clear`) is rejected with a `RuntimeError` that names the
  official API.
- **`sdfg._symbol_objects`** stores the canonical `symbolic.symbol` object
  per name (its dtype plus its sympy assumptions: `integer`, `positive`,
  `nonnegative`, ...). Only `add_symbol` writes to it.
- A symbol's *identity* is its `(dtype, assumptions0)` tuple. Two adds with
  matching identity are idempotent; two adds of the same name with different
  identity are rejected.

## Official API

| Method | Purpose |
| --- | --- |
| `sdfg.add_symbol(name, dtype, find_new_name=False, **assumptions)` | Register a symbol. Idempotent on matching identity; raises `FileExistsError` on conflict unless `find_new_name=True` is passed (then an unused name is allocated). |
| `sdfg.set_symbol_type(name, dtype)` | In-place dtype swap for an already-registered symbol. The cached symbol object is rebuilt on next access. |
| `sdfg.remove_symbol(name)` | Drop the entry from both `symbols` and `_symbol_objects`, plus the corresponding parent `symbol_mapping` if nested. |
| `sdfg.replace(name, new_name)` / `sdfg.replace_dict(...)` | Rename: rewrites all references and rebuilds the canonical symbol object under the new name. |
| `sdfg.get_symbol(name)` | Returns the canonical `symbolic.symbol` object. Stable identity within one SDFG. |

Internal callers that need to bypass the guard — only the four methods above,
plus `replace_dict` — do so via the `self.symbols.allow_mutation()` context
manager. Outside that, every write attempt is a programming error.

## Why a guard, not a convention

Symbols flow through every layer of DaCe (frontend, transformations, codegen).
Before the registry, transformations updated `sdfg.symbols[name]` directly,
producing two failure modes:

1. **Identity drift** — two callers reused `'N'` with different dtypes or
   sympy assumptions; the resulting expressions were silently mixed.
2. **Stale renaming** — `add_symbol(..., find_new_name=True)` produced
   `N_0` even when a matching `N` already existed, leaving the SDFG with two
   logically-identical symbols.

Both failure modes manifest only at codegen time, far from the offending
write. The guard turns these into immediate, localized `RuntimeError`s.

## Nested SDFGs

Every SDFG owns its own registry. The same name in inner and outer SDFGs is
allowed and they remain independent objects. When `add_nested_sdfg` propagates
free symbols from the outer to the inner SDFG, it does so via `add_symbol`,
which preserves the inner registry's invariants.

## Cloning / serialization

`copy.copy` and `copy.deepcopy` of `_SymbolDict` deliberately return plain
`dict` instances so that local accumulators (validation, codegen) can mutate
freely. Cloning a whole SDFG goes through `SDFG.__deepcopy__`, which re-wraps
the cloned `_symbols` in a fresh guarded `_SymbolDict`.

## Bypass coverage

What the guard catches (raises `RuntimeError` / `TypeError`):

- `sdfg.symbols[k] = v`, `del sdfg.symbols[k]`, `sdfg.symbols.pop`,
  `popitem`, `update`, `setdefault` (new key), `clear`.
- `dict.__setitem__(sdfg.symbols, ...)` and the other `dict.*` C-slots —
  `_SymbolDict` is a `MutableMapping` (composition over `dict`), so the C-slot
  descriptors of `dict` simply don't apply.
- `sdfg._symbols = some_plain_mapping` — `SDFG.__setattr__` rejects writes
  that aren't a `_SymbolDict`. Use `sdfg.symbols = ...` (the property) for the
  rare legitimate full-replacement case.

What the guard does **not** catch (deliberate Python introspection):

- `object.__setattr__(sdfg, '_symbols', plain_dict)` — bypasses the class
  `__setattr__` by construction.
- `sdfg.__dict__['_symbols'] = plain_dict` — same.
- Any C-extension that pokes at the underlying object.

These paths require explicit introspection — they are not normal coding
patterns, and Python provides no in-process mechanism to prevent them. The
guard is designed to catch *accidental* misuse; code that goes out of its way
to bypass it is on its own to maintain the invariants.
