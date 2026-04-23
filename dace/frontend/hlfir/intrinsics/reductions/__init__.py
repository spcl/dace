"""Whole-array reductions (SUM / PRODUCT / MINVAL / MAXVAL / COUNT / ALL /
ANY / DOT_PRODUCT).  Each will become a ``standard.Reduce`` library node
via ``state.add_reduce(wcr, axes, identity)``.

Phase 2 — today this package only carries an empty ``REDUCTION_INTRINSICS``
registry so the top-level ``is_reduction`` helper can stay family-agnostic.
When Phase 2 lands, individual intrinsic files (e.g. ``scalar_reductions.py``)
fill this dict.
"""
from __future__ import annotations

from dace.frontend.hlfir.intrinsics.base import ReductionIntrinsic
from dace.frontend.hlfir.intrinsics.reductions.scalar_reductions import (
    SCALAR_REDUCTIONS, )

REDUCTION_INTRINSICS: dict[str, ReductionIntrinsic] = {**SCALAR_REDUCTIONS}


def lookup(name: str) -> ReductionIntrinsic | None:
    return REDUCTION_INTRINSICS.get(name)
