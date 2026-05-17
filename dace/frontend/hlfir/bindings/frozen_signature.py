"""Frozen SDFG signature  --  snapshotted at build time, verified at codegen.

At the moment the kernel SDFG leaves ``SDFGBuilder.build()``, we
capture its argument list + free symbols into a ``FrozenSignature``
and pin it on the SDFG (``sdfg._frozen_signature = fs``).  The
binding emitter downstream uses this snapshot, not the live SDFG, so
transformations that mutate the SDFG can't silently invalidate a
generated ``.f90`` wrapper.

At codegen time (see ``dace/codegen/codegen.py``), before the C++
header is emitted, we call ``fs.verify_against(sdfg)``.  Any drift
from the snapshot raises ``SignatureDriftError``  --  the contract is
compile-time, not SDFG-time.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple


class SignatureDriftError(RuntimeError):
    """Raised when the live SDFG's arglist / free_symbols disagrees with
    a ``FrozenSignature`` attached to it."""


@dataclass(frozen=True)
class FrozenArg:
    """One argument in the frozen signature.

    Fields:
        fortran_name: name as declared in the user's Fortran source.
        sdfg_name:    name DaCe sees (may differ after struct
                      flattening  --  e.g. ``st%u`` becomes ``st_u``).
        kind:         ``'array'`` | ``'scalar'`` | ``'symbol'``.
        dtype:        ``'float64'`` | ``'int32'`` | ``'complex128'`` | ...
        rank:         tensor rank (0 for scalars).
        shape:        symbolic extents in Fortran symbols.  Empty tuple
                      for scalars / symbols.
        intent:       ``'in'`` | ``'out'`` | ``'inout'`` | ``''``.
        from_struct_member: when this arg was extracted from a struct
                      dummy by ``hlfir-flatten-structs``, the original
                      Fortran expression (``st%u``).  ``None`` otherwise.
        layout:       ``'same'`` (caller + callee share layout  --  alias
                      via ``c_loc``) | ``'complex_split'`` (Fortran
                      complex split into two reals) | ``'transpose'`` /
                      similar.  The binding emitter picks its copy
                      strategy off this tag.
    """

    fortran_name: str
    sdfg_name: str
    kind: str
    dtype: str
    rank: int
    shape: Tuple[str, ...] = field(default_factory=tuple)
    intent: str = ''
    from_struct_member: Optional[str] = None
    layout: str = 'same'

    def to_dict(self) -> dict:
        d = asdict(self)
        # shape round-trips as a list in JSON; rebuild as tuple on load.
        d['shape'] = list(self.shape)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FrozenArg":
        d = dict(d)
        d['shape'] = tuple(d.get('shape', []))
        return cls(**d)


@dataclass(frozen=True)
class FrozenSignature:
    """Full snapshot of one entry subroutine's SDFG signature.

    ``args`` is ordered to match the generated C function
    ``__program_<entry>``'s parameter order (data args sorted, then
    scalars, then free symbols  --  the order DaCe's
    ``generate_headers`` emits).
    """

    entry: str  # 'compute_tendencies'
    mangled: str  # '_QPcompute_tendencies'
    args: Tuple[FrozenArg, ...]
    free_symbols: Tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = 1
    # Auto-detected Fortran module-global provenance for SDFG names
    # that are NOT outer dummies  --  free symbols (a scalar module
    # global lifted into a shape / bound) and module-global args
    # (the bridge ``intent=inout`` lift).  Maps ``sdfg_name ->
    # (module, entity)``.  The binding emitter merges this with any
    # hand-authored ``OriginalInterface.module_symbol_sources`` (the
    # explicit map wins on conflict) so no hand-authored list is
    # required for kernels the bridge can resolve on its own.
    module_symbol_origins: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    # ----- I/O ---------------------------------------------------------

    def to_json(self, path: str):
        with open(path, 'w') as fh:
            json.dump(
                {
                    'entry': self.entry,
                    'mangled': self.mangled,
                    'args': [a.to_dict() for a in self.args],
                    'free_symbols': list(self.free_symbols),
                    'schema_version': self.schema_version,
                    'module_symbol_origins': {
                        k: list(v)
                        for k, v in self.module_symbol_origins.items()
                    },
                },
                fh,
                indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FrozenSignature":
        with open(path) as fh:
            d = json.load(fh)
        return cls(
            entry=d['entry'],
            mangled=d['mangled'],
            args=tuple(FrozenArg.from_dict(a) for a in d['args']),
            free_symbols=tuple(d.get('free_symbols', [])),
            schema_version=d.get('schema_version', 1),
            module_symbol_origins={
                k: tuple(v)
                for k, v in d.get('module_symbol_origins', {}).items()
            },
        )

    # ----- Drift check -------------------------------------------------

    def verify_against(self, sdfg):
        """Compare the live ``sdfg.arglist()`` + free-symbol set against
        this snapshot.  Raise ``SignatureDriftError`` on any divergence.

        Checks:
        - Same set of argument names.
        - Same order of argument names.
        - Same dtype per argument.
        - Same set of free symbols.

        We DON'T check dimensionality invariants past order/dtype since
        symbolic shapes may canonicalise; downstream codegen will catch
        concrete mismatches when it assembles memlets.
        """
        live_arglist = sdfg.arglist()
        live_names = list(live_arglist.keys())
        snap_names = [a.sdfg_name for a in self.args]
        if live_names != snap_names:
            raise SignatureDriftError(f"signature drift on {self.entry!r}: "
                                      f"expected args {snap_names}, got {live_names}")

        # dtype per arg  --  guard against silent type change.
        for a in self.args:
            desc = live_arglist[a.sdfg_name]
            live_dtype = _dtype_string(desc)
            if live_dtype != a.dtype:
                raise SignatureDriftError(f"signature drift on {self.entry!r}: arg {a.sdfg_name!r} "
                                          f"dtype {a.dtype!r} in snapshot but {live_dtype!r} now")

        live_fs = set(str(s) for s in sdfg.free_symbols)
        snap_fs = set(self.free_symbols)
        if live_fs != snap_fs:
            raise SignatureDriftError(f"signature drift on {self.entry!r}: "
                                      f"expected free symbols {sorted(snap_fs)}, got {sorted(live_fs)}")


def _dtype_string(desc) -> str:
    """Stringify a DaCe data descriptor's dtype for comparison."""
    t = getattr(desc, 'dtype', None)
    if t is None:
        return '?'
    # dace.typeclass instances have a ``to_string``  --  fall back to repr.
    return getattr(t, 'to_string', lambda: str(t))()


def freeze_signature(sdfg, entry: str, mangled: str, args: Tuple[FrozenArg, ...]) -> FrozenSignature:
    """Build a ``FrozenSignature`` from a live SDFG + caller-supplied
    per-arg metadata (which captures the struct-flattening / layout
    hints the bridge observed during extraction)."""
    return FrozenSignature(
        entry=entry,
        mangled=mangled,
        args=args,
        free_symbols=tuple(str(s) for s in sorted(sdfg.free_symbols, key=str)),
    )
