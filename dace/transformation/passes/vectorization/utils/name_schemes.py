# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Centralised synthetic-name schemes for the vectorization pipeline.

A single owner for each class of minted identifier avoids the silent
mismatch where one site composes a name and another parses it differently:

- ``LaneIdScheme`` — per-lane symbols ``<base>_laneid_<i>``.
- ``VecNameScheme`` — vector buffers ``<base>_vec`` / ``<base>_vec_<i>`` /
  ``<base>_vec_k``.
- ``PackedNameScheme`` — scatter / gather buffers ``<base>_packed``.

``_packed`` is reserved for scatter/gather; ``_vec`` is used for everything
else. Inout connectors must use the same suffix on both directions, so
callers classify by access kind, not by direction.
"""
import re
from typing import Optional, Tuple


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
