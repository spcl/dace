"""Direct-replacement intrinsics (SIZE / LBOUND / UBOUND / BIT_SIZE /
PRESENT / ALLOCATED).

Phase 4 — each folds into the enclosing expression as a constant or a
simple interstate-edge assignment, mirroring
``dace/frontend/fortran/intrinsics/direct_replacements.py`` on the
legacy frontend.  Today this package only carries an empty
``DIRECT_INTRINSICS`` set so callers can query against it without
special-casing the "not yet implemented" state.
"""
from __future__ import annotations

DIRECT_INTRINSICS: set[str] = set()
