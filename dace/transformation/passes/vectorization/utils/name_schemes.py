# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Centralised synthetic-name schemes for the vectorization pipeline.

One owner per class of minted identifier, avoiding the silent mismatch where one site composes a
name and another parses it differently:

- ``LaneIdScheme`` — per-tile-lane symbols. Canonical (Option B) form = chain of per-dim chunks
  ``<base>_lane<d>id_<n>`` (one chunk per tile dim, source order); legacy 1D form
  ``<base>_laneid_<n>`` (no dim index) parsed for back-compat as ``<base>_lane0id_<n>``.
- ``TileNameScheme`` — K-dim tile-transient names (``<base>_tile``, ``<base>_tile_idx``,
  ``_tile_iter_mask``, ``<base>_tile_cond_mask``).
"""
import re
from typing import Iterable, Iterator, List, Optional, Tuple

import dace
from dace.sdfg import SDFG


class LaneIdScheme:
    """Owner of the unified per-tile-lane symbol scheme.

    Canonical (Option B) form: ``<base>_lane<d>id_<n>`` per tile dimension ``d``, chained for K>=2
    (one chunk per dim, source order):

    - K=1 dim 0 lane 3: ``a_lane0id_3``.
    - K=2 dim 0 lane 3, dim 1 lane 5: ``a_lane0id_3_lane1id_5``.

    Legacy 1D form ``<base>_laneid_<n>`` parsed as ``<base>_lane0id_<n>``, so callers under either
    scheme classify uniformly. Emit new names via :meth:`make_dim` / :meth:`make_multi`; :meth:`make`
    keeps the single-arg legacy emit form for back-compat with 1D callers (A.1: infra only; emitters
    switch in A.2).

    :cvar LEGACY_SUFFIX: Legacy 1D infix ``_laneid_``. Reserved for substring searches in callers
        pre-dating Option B; new code should use :meth:`is_lane_fanned` or :meth:`parse`.
    """

    LEGACY_SUFFIX = "_laneid_"

    # Trailing-chunk matcher anchored to end of string (chunked form).
    _CHUNK_TAIL_RE = re.compile(r"_lane(\d+)id_(\d+)$")
    # Legacy 1D form, full name.
    _LEGACY_RE = re.compile(r"^(.*)_laneid_(\d+)$")
    # Substring scanner: matches a lane chunk anywhere in a string (canonical or legacy form). Used
    # by audit passes scanning memlet subset strings / interstate expressions for accidentally
    # fanned-out lanes without parsing the whole symbol.
    LANE_INFIX_RE = re.compile(r"_lane(?:\d+id|id)_\d+")

    @staticmethod
    def make(base: str, lane: int) -> str:
        """Build the legacy 1D lane-encoded name ``<base>_laneid_<lane>``.

        Single-int back-compat shape used by every 1D emitter today. New callers use :meth:`make_dim`
        / :meth:`make_multi` so the name carries the dim index. A.2 switches legacy emitters over.

        :param base: Un-encoded symbol base.
        :param lane: Lane index in the single (innermost) tile dim.
        :returns: ``<base>_laneid_<lane>``.
        """
        return f"{base}{LaneIdScheme.LEGACY_SUFFIX}{lane}"

    @staticmethod
    def make_dim(base: str, dim: int, lane: int) -> str:
        """Build a single Option B chunk ``<base>_lane<dim>id_<lane>``.

        :param base: Un-encoded symbol base.
        :param dim: Tile-dim index (0 outermost → innermost).
        :param lane: Lane index in dim ``dim``.
        :returns: ``<base>_lane<dim>id_<lane>``.
        """
        return f"{base}_lane{dim}id_{lane}"

    @staticmethod
    def make_multi(base: str, chunks: Iterable[Tuple[int, int]]) -> str:
        """Build a multi-dim lane-encoded name with one chunk per dim.

        :param base: Un-encoded symbol base.
        :param chunks: Iterable of ``(dim, lane)`` pairs in source (chain)
            order. Empty chunks yield ``base`` unchanged.
        :returns: ``<base>_lane<d_0>id_<n_0>_lane<d_1>id_<n_1>...``.
        """
        return base + "".join(f"_lane{d}id_{n}" for d, n in chunks)

    @staticmethod
    def parse(name: str) -> Optional[Tuple[str, int]]:
        """Peel one trailing lane chunk off ``name`` (legacy 1D shape).

        Accepts canonical ``<base>_lane<d>id_<n>`` AND legacy ``<base>_laneid_<n>``; both return
        ``(base_without_the_chunk, lane)``. Peels one level only — for the full base use
        :meth:`base_of`.

        :param name: Name to parse.
        :returns: ``(base_without_last_chunk, lane)`` if ``name`` ends with a recognised chunk; else
            ``None``.
        """
        m = LaneIdScheme._CHUNK_TAIL_RE.search(name)
        if m is not None and m.end() == len(name):
            return name[:m.start()], int(m.group(2))
        lm = LaneIdScheme._LEGACY_RE.match(name)
        if lm is not None:
            return lm.group(1), int(lm.group(2))
        return None

    @staticmethod
    def parse_chunks(name: str) -> Optional[Tuple[str, Tuple[Tuple[int, int], ...]]]:
        """Strip every trailing lane chunk off ``name``.

        Walks the trailing-chunk shape repeatedly so ``a_lane0id_3_lane1id_5`` decomposes to
        ``("a", ((0, 3), (1, 5)))``. Legacy 1D form = one ``(0, n)`` chunk.

        :param name: Name to parse.
        :returns: ``(base, chunks)`` where ``chunks`` = per-dim ``(dim, lane)`` tuple in source
            order; ``None`` if ``name`` carries no recognised chunk.
        """
        peeled: List[Tuple[int, int]] = []
        remaining = name
        while True:
            m = LaneIdScheme._CHUNK_TAIL_RE.search(remaining)
            if m is not None and m.end() == len(remaining):
                peeled.append((int(m.group(1)), int(m.group(2))))
                remaining = remaining[:m.start()]
                continue
            lm = LaneIdScheme._LEGACY_RE.match(remaining)
            if lm is not None:
                peeled.append((0, int(lm.group(2))))
                remaining = lm.group(1)
                continue
            break
        if not peeled:
            return None
        peeled.reverse()
        return remaining, tuple(peeled)

    @staticmethod
    def is_laneid(name: str) -> bool:
        """Whether ``name`` carries any lane chunk. Back-compat alias of :meth:`is_lane_fanned` for
        1D callers.

        :param name: Name to test.
        :returns: True iff ``name`` matches the canonical chunked form or legacy ``_laneid_<digits>``.
        """
        return LaneIdScheme.parse(name) is not None

    @staticmethod
    def is_lane_fanned(name: str) -> bool:
        """Whether ``name`` carries any per-tile-lane chunk.

        Audit-pass entrypoint (canonical + legacy forms). Replaces the deleted
        ``TileLaneScheme.is_tilelane``.

        :param name: Name to test.
        :returns: True iff at least one ``_lane<d>id_<n>`` chunk (or legacy ``_laneid_<n>``) present.
        """
        return LaneIdScheme.parse(name) is not None

    @staticmethod
    def base_of(name: str) -> str:
        """Strip every trailing lane chunk and return the bare base.

        :param name: Name to strip.
        :returns: ``name`` with every recognised trailing chunk removed
            (or ``name`` itself if no chunk is present).
        """
        parsed = LaneIdScheme.parse_chunks(name)
        return parsed[0] if parsed is not None else name

    @staticmethod
    def peel_dim(name: str, dim: int) -> Optional[str]:
        """Drop the chunk for tile dim ``dim`` from ``name``.

        Reassembles remaining chunks in source order (canonical form). Used when a downstream pass
        collapses one tile dim (partial collapse, mask projection) and leftover lane chunks must keep
        agreeing with the survived dim indices.

        :param name: Lane-encoded name.
        :param dim: Tile-dim index to drop.
        :returns: Peeled name (canonical form), or ``None`` when ``name`` carries no chunk for
            ``dim``.
        """
        parsed = LaneIdScheme.parse_chunks(name)
        if parsed is None:
            return None
        base, chunks = parsed
        kept = [(d, n) for d, n in chunks if d != dim]
        if len(kept) == len(chunks):
            return None
        return LaneIdScheme.make_multi(base, kept)

    @staticmethod
    def contains_lane_chunk(text: str) -> bool:
        """Whether ``text`` contains any lane-chunk substring.

        Substring-only scanner for audit passes classifying memlet subset strings / interstate
        expressions without parsing the whole symbol. Matches canonical + legacy forms.

        :param text: Free-form text (e.g., ``str(memlet.subset)``).
        :returns: True iff at least one lane-chunk present in ``text``.
        """
        return LaneIdScheme.LANE_INFIX_RE.search(text) is not None

    @staticmethod
    def varies_with_dim(name: str, dim: int) -> bool:
        """Whether ``name`` carries a chunk for tile dim ``dim``.

        :param name: Lane-encoded name.
        :param dim: Tile-dim index to test.
        :returns: True iff at least one ``_lane<dim>id_<...>`` chunk is
            present (the legacy 1D form counts as dim 0).
        """
        parsed = LaneIdScheme.parse_chunks(name)
        if parsed is None:
            return False
        return any(d == dim for d, _ in parsed[1])


class TileNameScheme:
    """Owner of the v2 tile-transient names.

    The K-dim path mints multi-dim arrays (register tiles, mask tiles, gather/scatter index tiles),
    not per-lane scalars; names reserved here so the audit pass distinguishes v2 transients from
    leftover 1D-path ``_iter_mask`` / ``_vec`` / ``_packed`` names.

    :cvar TILE_SUFFIX: Suffix for a register-tile transient.
    :cvar IDX_SUFFIX: Suffix for a gather / scatter index tile.
    :cvar ITER_MASK: Reserved name for the iteration-mask transient produced by ``TileMaskGen``
        inside a masked-remainder body.
    :cvar COND_MASK_SUFFIX: Suffix for branch-condition mask transients produced by post-MVP
        ``TileMaskGen`` (paired with merge).
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
    :yields: ``sdfg`` first, then each inner SDFG body depth-first. Covers SDFGs nested inside
        NestedSDFG nodes (``all_sdfgs_recursive`` only descends control-flow regions, not NSDFGs).
    """
    yield sdfg
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                yield from _walk_sdfgs(node.sdfg)


def _iter_all_symbol_strings(sdfg: SDFG) -> Iterator[str]:
    """Yield every name the K-dim path might leak a per-lane scalar through: array names, SDFG
    symbols, tasklet connector names, interstate-edge assignment targets. Recurses every nested SDFG.

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

    The v2 multi-dim track relies on lib-node emission carrying lane offsets implicitly; a leaked
    ``<base>_lane<d>id_<n>`` (canonical) or ``<base>_laneid_<n>`` (legacy) means some prep pass
    accidentally fanned out to per-lane scalars. Loud failure keeps the invariant honest. Runs at
    orchestrator exit.

    :param sdfg: SDFG to audit (walked recursively, including nested SDFGs and interstate-edge
        assignments).
    :raises AssertionError: If any classified name is lane-fanned per
        :meth:`LaneIdScheme.is_lane_fanned`.
    """
    leaks = sorted({n for n in _iter_all_symbol_strings(sdfg) if LaneIdScheme.is_lane_fanned(n)})
    if leaks:
        raise AssertionError(f"K-dim tile path leaked {len(leaks)} per-lane scalar(s): {leaks!r}. "
                             f"Lib-node emission must carry lane offsets implicitly — check the prep passes.")
