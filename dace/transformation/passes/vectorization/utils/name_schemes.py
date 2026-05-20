# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Centralised synthetic-name schemes for the vectorization pipeline.

A single owner for each class of minted identifier avoids the silent
mismatch where one site composes a name and another parses it differently:

- ``LaneIdScheme`` — per-lane symbols ``<base>_laneid_<i>`` (1D path).
- ``VecNameScheme`` — vector buffers ``<base>_vec`` / ``<base>_vec_<i>`` /
  ``<base>_vec_k``.
- ``PackedNameScheme`` — scatter / gather buffers ``<base>_packed``.
- ``TileLaneScheme`` — multi-dim per-tile-lane recognizer
  ``<base>_tilelane_<i_0>x<i_1>x..._<i_{K-1}>`` (v2 audit-only); the
  K=1 legacy ``LaneIdScheme`` form is also accepted by
  :meth:`TileLaneScheme.is_tilelane` so a single helper covers both.
- ``TileNameScheme`` — v2 tile-transient names (``<base>_tile``,
  ``<base>_tile_idx``, ``_tile_iter_mask``, ``<base>_tile_cond_mask``).
- ``CORE_MAP_PARAM_PREFIX`` — prefix marking outer map parameters minted
  by the SVE tile-by-cores step (so the loop-conversion pass can find
  the right outer scope later).

``_packed`` is reserved for scatter/gather; ``_vec`` is used for everything
else in the 1D path; ``_tile`` is reserved for the v2 multi-dim path so
both pipelines can coexist in one SDFG without name collisions. Inout
connectors must use the same suffix on both directions, so callers
classify by access kind, not by direction.
"""
import re
from typing import Iterable, Iterator, Optional, Tuple

import dace
from dace.sdfg import SDFG

CORE_MAP_PARAM_PREFIX = "core"


class LaneIdScheme:
    """Owner of the per-lane symbol scheme ``<base>_laneid_<i>``.

    A symbol that already encodes its lane (parses non-trivially) is treated
    as fixed, which keeps the lane-expansion passes idempotent.
    """

    SUFFIX = "_laneid_"
    _PARSE_RE = re.compile(r"^(.*)_laneid_(\d+)$")

    @staticmethod
    def make(base: str, lane: int) -> str:
        """Build the lane-encoded name ``<base>_laneid_<lane>``.

        :param base: Un-encoded symbol base.
        :param lane: Lane index.
        :returns: The lane-encoded name.
        """
        return f"{base}{LaneIdScheme.SUFFIX}{lane}"

    @staticmethod
    def parse(name: str) -> Optional[Tuple[str, int]]:
        """Peel the trailing lane off a lane-encoded name.

        :param name: Name to parse.
        :returns: ``(base, lane)`` if ``name`` ends with ``_laneid_<digits>``,
            else ``None``. Nested forms peel one level only
            (``foo_laneid_3_laneid_0`` -> ``("foo_laneid_3", 0)``); call
            repeatedly until ``None`` for the original base.
        """
        m = LaneIdScheme._PARSE_RE.match(name)
        if m is None:
            return None
        return m.group(1), int(m.group(2))

    @staticmethod
    def is_laneid(name: str) -> bool:
        """Check whether ``name`` is lane-encoded.

        :param name: Name to test.
        :returns: True iff ``name`` matches ``<base>_laneid_<digits>``.
        """
        return LaneIdScheme.parse(name) is not None


class VecNameScheme:
    """Owner of the ``_vec``-family vector-buffer name scheme.

    Three shapes share the ``_vec`` root:

    - ``<base>_vec`` — default contiguous vector buffer.
    - ``<base>_vec_<i>`` — per-subset variant when one base array sees
      multiple distinct subsets inside a NestedSDFG.
    - ``<base>_vec_k`` — per-NSDFG ``(vector_width,)`` transient minted by
      ``prepare_vectorized_array``.

    Inout connectors route through ``make`` on both directions so the
    names agree.
    """

    SUFFIX = "_vec"
    K_SUFFIX = "_vec_k"
    _INDEXED_RE = re.compile(r"^(.*)_vec_(\d+)$")

    @staticmethod
    def make(base: str) -> str:
        """Build the plain vector name ``<base>_vec``.

        :param base: Buffer base name.
        :returns: The plain vector name.
        """
        return f"{base}{VecNameScheme.SUFFIX}"

    @staticmethod
    def make_indexed(base: str, index: int) -> str:
        """Build the per-subset variant ``<base>_vec_<index>``.

        :param base: Buffer base name.
        :param index: Subset disambiguation index.
        :returns: The indexed vector name.
        """
        return f"{base}_vec_{index}"

    @staticmethod
    def make_k(base: str) -> str:
        """Build the per-NSDFG ``(vector_width,)`` name ``<base>_vec_k``.

        :param base: Buffer base name.
        :returns: The ``_vec_k`` name.
        """
        return f"{base}{VecNameScheme.K_SUFFIX}"

    @staticmethod
    def is_vec(name: str) -> bool:
        """Check whether ``name`` belongs to the ``_vec`` family.

        :param name: Name to test.
        :returns: True iff ``name`` ends with ``_vec``, ``_vec_<digits>``, or
            ``_vec_k``. Does not match ``_packed``.
        """
        if name.endswith(VecNameScheme.SUFFIX):
            return True
        if name.endswith(VecNameScheme.K_SUFFIX):
            return True
        return VecNameScheme._INDEXED_RE.match(name) is not None


class PackedNameScheme:
    """Owner of the ``<base>_packed`` scheme for scatter / gather paths.

    ``_packed`` is reserved for scatter / gather access patterns; the
    classifier and producer both route through this scheme so they agree.
    """

    SUFFIX = "_packed"

    @staticmethod
    def make(base: str) -> str:
        """Build the packed name ``<base>_packed``.

        :param base: Buffer base name.
        :returns: The packed name.
        """
        return f"{base}{PackedNameScheme.SUFFIX}"

    @staticmethod
    def is_packed(name: str) -> bool:
        """Check whether ``name`` is a packed buffer name.

        :param name: Name to test.
        :returns: True iff ``name`` ends with ``_packed``.
        """
        return name.endswith(PackedNameScheme.SUFFIX)


class TileLaneScheme:
    """Recognizer for multi-dim per-tile-lane names.

    The v2 K-dim path **never mints** per-lane scalars — tile lib nodes
    carry the lane offset implicitly. This scheme exists only so the
    audit pass :func:`assert_no_laneid_in_tile_path` can flag any leak
    from a buggy prep pass. K=1 form (the legacy ``LaneIdScheme``) is
    also recognized via :meth:`is_tilelane` so one helper covers both.

    :cvar SUFFIX: The infix used between the base and the per-dim lane
        indices.
    """

    SUFFIX = "_tilelane_"
    _PARSE_RE = re.compile(r"^(.*)_tilelane_((?:\d+x)*\d+)$")

    @staticmethod
    def make(base: str, indices: Tuple[int, ...]) -> str:
        """Build the multi-dim lane-encoded name.

        :param base: Un-encoded symbol base.
        :param indices: Per-dim lane indices, innermost-last.
        :returns: ``<base>_tilelane_<i_0>x<i_1>x..._<i_{K-1}>``.
        :raises ValueError: If ``indices`` is empty.
        """
        if not indices:
            raise ValueError("TileLaneScheme.make: indices must be non-empty")
        return f"{base}{TileLaneScheme.SUFFIX}{'x'.join(str(i) for i in indices)}"

    @staticmethod
    def parse(name: str) -> Optional[Tuple[str, Tuple[int, ...]]]:
        """Peel the trailing multi-dim lane coordinate off ``name``.

        :param name: Name to parse.
        :returns: ``(base, indices)`` if ``name`` matches
            ``<base>_tilelane_<digits>(x<digits>)*``; else ``None``.
        """
        m = TileLaneScheme._PARSE_RE.match(name)
        if m is None:
            return None
        return m.group(1), tuple(int(s) for s in m.group(2).split("x"))

    @staticmethod
    def is_tilelane(name: str) -> bool:
        """Check whether ``name`` is per-lane encoded under any K.

        Returns True for the v2 multi-dim form ``<base>_tilelane_<...>``
        and for the legacy 1D ``<base>_laneid_<i>`` form, so a single
        check covers both.

        :param name: Name to test.
        :returns: True iff ``name`` is a per-lane name in either scheme.
        """
        return (
            TileLaneScheme.parse(name) is not None
            or LaneIdScheme.is_laneid(name)
        )


class TileNameScheme:
    """Owner of the v2 tile-transient names.

    The K-dim path mints multi-dim arrays (register tiles, mask tiles,
    gather/scatter index tiles) — not per-lane scalars — and the names
    are reserved here so the audit pass can distinguish v2 transients
    from leftover 1D-path ``_iter_mask`` / ``_vec`` / ``_packed`` names.

    :cvar TILE_SUFFIX: Suffix for a register-tile transient.
    :cvar IDX_SUFFIX: Suffix for a gather / scatter index tile.
    :cvar ITER_MASK: Reserved name for the iteration-mask transient
        produced by ``TileMaskGen`` inside a masked-remainder body.
    :cvar COND_MASK_SUFFIX: Suffix for branch-condition mask transients
        produced by post-MVP ``TileMaskGen`` (paired with merge).
    """

    TILE_SUFFIX = "_tile"
    IDX_SUFFIX = "_tile_idx"
    ITER_MASK = "_tile_iter_mask"
    COND_MASK_SUFFIX = "_tile_cond_mask"

    @staticmethod
    def make_tile(base: str) -> str:
        """Build the tile-transient name ``<base>_tile``.

        :param base: Buffer base name.
        :returns: ``<base>_tile``.
        """
        return f"{base}{TileNameScheme.TILE_SUFFIX}"

    @staticmethod
    def make_idx(base: str) -> str:
        """Build the gather/scatter index-tile name ``<base>_tile_idx``.

        :param base: Buffer base name.
        :returns: ``<base>_tile_idx``.
        """
        return f"{base}{TileNameScheme.IDX_SUFFIX}"

    @staticmethod
    def make_cond_mask(base: str) -> str:
        """Build the branch-condition mask name ``<base>_tile_cond_mask``.

        :param base: Source-branch base name.
        :returns: ``<base>_tile_cond_mask``.
        """
        return f"{base}{TileNameScheme.COND_MASK_SUFFIX}"

    @staticmethod
    def is_tile_transient(name: str) -> bool:
        """Check whether ``name`` belongs to the v2 tile-transient family.

        :param name: Name to test.
        :returns: True iff ``name`` ends with any of the v2 tile-name
            suffixes, or equals ``_tile_iter_mask`` exactly.
        """
        if name == TileNameScheme.ITER_MASK:
            return True
        for suffix in (
            TileNameScheme.TILE_SUFFIX,
            TileNameScheme.IDX_SUFFIX,
            TileNameScheme.COND_MASK_SUFFIX,
        ):
            if name.endswith(suffix):
                return True
        return False


def _walk_sdfgs(sdfg: SDFG) -> Iterator[SDFG]:
    """Yield ``sdfg`` plus every ``NestedSDFG`` body reachable from it.

    :param sdfg: Root SDFG.
    :yields: ``sdfg`` first, then each inner SDFG body in depth-first
        order. Covers SDFGs nested inside NestedSDFG nodes
        (``all_sdfgs_recursive`` only descends control-flow regions,
        not NSDFG nodes).
    """
    yield sdfg
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                yield from _walk_sdfgs(node.sdfg)


def _iter_all_symbol_strings(sdfg: SDFG) -> Iterator[str]:
    """Yield every name the K-dim path might leak a per-lane scalar
    through: array names, SDFG symbols, tasklet connector names, and
    every interstate-edge assignment target. Walks recursively through
    every nested SDFG body.

    :param sdfg: SDFG to walk.
    :yields: Names that the audit pass should classify.
    """
    for inner in _walk_sdfgs(sdfg):
        yield from inner.arrays.keys()
        yield from inner.symbols.keys()
        for state in inner.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    yield from node.in_connectors.keys()
                    yield from node.out_connectors.keys()
        for ise in inner.all_interstate_edges():
            yield from ise.data.assignments.keys()


def assert_no_laneid_in_tile_path(sdfg: SDFG) -> None:
    """Audit-only: refuse any per-lane encoded name in the K-dim path.

    The v2 multi-dim track relies on lib-node emission carrying lane
    offsets implicitly; a leaked ``<base>_laneid_<i>`` or
    ``<base>_tilelane_<...>`` is a sign that some prep pass accidentally
    fanned out to per-lane scalars. Loud failure here keeps the design
    invariant honest. Runs at orchestrator exit.

    :param sdfg: SDFG to audit (walked recursively, including nested
        SDFGs and interstate-edge assignments).
    :raises AssertionError: If any classified name matches
        :meth:`TileLaneScheme.is_tilelane`.
    """
    leaks = sorted({n for n in _iter_all_symbol_strings(sdfg) if TileLaneScheme.is_tilelane(n)})
    if leaks:
        raise AssertionError(
            f"K-dim tile path leaked {len(leaks)} per-lane scalar(s): {leaks!r}. "
            f"Lib-node emission must carry lane offsets implicitly — check the prep passes."
        )
