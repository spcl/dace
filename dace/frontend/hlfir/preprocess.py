"""Optional Fortran-source-level pre-processor.

Some legacy weather/climate kernels (parts of ECRAD, ICON tendency
routines, the CloudSC microphysics loopnests) use an INTEGER flag as
the operand of an ``IF (...)`` -- e.g. ``IF (laericeauto) THEN ...``.
This was valid in older Fortran practice (and gfortran is still happy
with it) but flang-new-21 rejects it: the IF condition must be LOGICAL,
not INTEGER.

Because flang fails BEFORE producing HLFIR, the rewrite cannot live as
an HLFIR-level pass; it has to happen at the Fortran source layer
itself.  This module implements that rewrite as a single regex-driven
sweep over the input text.

We deliberately keep the rewriter narrow:

* Only ``IF (single_ident) THEN`` and ``IF (single_ident)`` shapes are
  rewritten -- anything more elaborate (``IF (a .AND. b)``,
  ``IF (.NOT. flag)``) parses fine on its own.
* The identifier must appear in an ``INTEGER ::`` declaration in the
  same source unit before we rewrite its IF guard.  Scalars with array
  bounds (``INTEGER, DIMENSION(...) :: x``) are skipped.

This is a pragmatic SED-style transform, NOT a Fortran parser; it's
brittle by construction and we'd prefer not to need it.  Use it only
when an upstream code base ships ECRAD-shape legacy patterns that we
can't reasonably patch in-tree.
"""
from __future__ import annotations

import re

_INTEGER_DECL_RE = re.compile(
    r"\bINTEGER\b(?:\s*\([^)]*\))?(?:\s*,\s*[A-Z_]+(?:\s*\([^)]*\))?)*\s*::\s*([^\n!]+)",
    re.IGNORECASE,
)
_BARE_IF_RE = re.compile(r"\b(IF\s*\(\s*)([A-Za-z_]\w*)(\s*\))", re.IGNORECASE)


def _collect_integer_scalar_names(source: str) -> set[str]:
    """Return the set of INTEGER scalar identifiers declared in
    ``source``.  Skip array declarations -- those can't be the bare
    operand of an ``IF`` anyway.  All names are lowercased for
    case-insensitive matching."""
    names: set[str] = set()
    for m in _INTEGER_DECL_RE.finditer(source):
        decl = m.group(1).split('!', 1)[0]
        for tok in decl.split(','):
            head = tok.strip().split('=', 1)[0].strip()
            # Skip array forms ("name(...)") and assumed-shape (":") --
            # an array can't be the bare argument of IF.
            if '(' in head:
                continue
            name = head.split()[0] if head else ''
            if name and name.replace('_', '').isalnum() and not name[0].isdigit():
                names.add(name.lower())
    return names


def preprocess_fortran(source: str) -> str:
    """Rewrite ``IF (intvar)`` to ``IF (intvar /= 0)`` for any INTEGER
    scalar declared in ``source``.

    Idempotent: a second invocation finds no bare-identifier IF guards
    left to rewrite and returns the input unchanged.
    """
    int_names = _collect_integer_scalar_names(source)
    if not int_names:
        return source

    def _rewrite(m: re.Match) -> str:
        ident = m.group(2)
        if ident.lower() in int_names:
            return f"{m.group(1)}{ident} /= 0{m.group(3)}"
        return m.group(0)

    return _BARE_IF_RE.sub(_rewrite, source)
