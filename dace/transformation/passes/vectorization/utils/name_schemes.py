# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Centralised name schemes for the vectorization pipeline.

The vectorization pipeline mints a few classes of synthetic identifiers
(per-lane symbols, vector-suffixed arrays, packed scatter/gather
buffers). Keeping the rules for each class in one place avoids the
silent-mismatch class of bug where one site composes ``<name>_laneid_3``
by string concatenation while another site parses it with a different
regex.

Three classes live here:

- ``LaneIdScheme`` — per-lane symbol names ``<base>_laneid_<i>``.
- ``VecNameScheme`` — contiguous vector buffers (``<base>_vec``,
  per-subset variants ``<base>_vec_<i>``, and the per-NSDFG-connector
  shape ``<base>_vec_k``).
- ``PackedNameScheme`` — scatter / gather packed buffers
  (``<base>_packed``).

Per the locked naming directive (``project_yakup_dev_naming_scheme``):

- ``_packed`` is reserved for **scatter/gather** access patterns.
- ``_vec`` is the suffix for **everything else** (contiguous vector
  loads/stores, copy-in/out vector arrays, vector accumulators).
- **Inout connectors** must use the *same* suffix on both directions.
  Callers should classify by access kind (which determines the scheme),
  not by direction (which would silently split inout names).
"""
import re
from typing import Optional, Tuple


class LaneIdScheme:
    """Centralised lane-id naming for the vectorization passes.

    The vectorization pipeline expands a single symbol used inside a vector tile into
    one symbol per lane, named ``<base>_laneid_<i>``. This class is the single owner
    of that scheme — every place in the codebase that constructs or inspects such a
    name must go through ``LaneIdScheme.make`` / ``LaneIdScheme.parse`` /
    ``LaneIdScheme.is_laneid`` instead of raw string concatenation or regex.

    Centralising the scheme is what makes the lane-expansion passes idempotent: a
    symbol that already encodes its lane in its name (parses non-trivially) is
    treated as fixed, never re-expanded into ``<base>_laneid_<i>_laneid_<j>``.
    """

    SUFFIX = "_laneid_"
    _PARSE_RE = re.compile(r"^(.*)_laneid_(\d+)$")

    @staticmethod
    def make(base: str, lane: int) -> str:
        """Build the lane-encoded name ``<base>_laneid_<lane>``."""
        return f"{base}{LaneIdScheme.SUFFIX}{lane}"

    @staticmethod
    def parse(name: str) -> Optional[Tuple[str, int]]:
        """Return ``(base, lane)`` if ``name`` ends with ``_laneid_<digits>``, else ``None``.

        For nested forms like ``foo_laneid_3_laneid_0`` the *trailing* lane is peeled
        once: the result is ``("foo_laneid_3", 0)``. Callers that want the original
        un-encoded base must call ``parse`` repeatedly until it returns ``None``.
        """
        m = LaneIdScheme._PARSE_RE.match(name)
        if m is None:
            return None
        return m.group(1), int(m.group(2))

    @staticmethod
    def is_laneid(name: str) -> bool:
        """True iff ``name`` matches the ``<base>_laneid_<digits>`` pattern."""
        return LaneIdScheme.parse(name) is not None


class VecNameScheme:
    """Centralised vector-buffer naming for the vectorization passes.

    The pipeline mints three shapes of vector buffer name:

    - ``<base>_vec``         — the default contiguous vector buffer
                                attached to a connector or array. Used by
                                ``add_copies_before_and_after_nsdfg`` for
                                the inout-connector rename and by the
                                tasklet helpers that allocate per-call
                                vector temporaries.
    - ``<base>_vec_<i>``     — per-subset variants when a single base
                                array sees multiple distinct subsets
                                inside a NestedSDFG (each subset gets its
                                own vector buffer; the index disambiguates).
    - ``<base>_vec_k``       — the per-NSDFG ``(vector_width,)`` shaped
                                transient minted by
                                ``prepare_vectorized_array``. The ``k``
                                marks the vector-width-bound shape (kept
                                distinct from plain ``_vec`` because the
                                allocator picks a different storage path).

    All three share the ``_vec`` family root; the helpers below are the
    one canonical owner. Inout connectors route through ``make`` on
    both directions (in and out) so the names agree.
    """

    SUFFIX = "_vec"
    K_SUFFIX = "_vec_k"
    _INDEXED_RE = re.compile(r"^(.*)_vec_(\d+)$")

    @staticmethod
    def make(base: str) -> str:
        """Build the plain vector name ``<base>_vec``."""
        return f"{base}{VecNameScheme.SUFFIX}"

    @staticmethod
    def make_indexed(base: str, index: int) -> str:
        """Build the per-subset variant ``<base>_vec_<index>``."""
        return f"{base}_vec_{index}"

    @staticmethod
    def make_k(base: str) -> str:
        """Build the per-NSDFG ``(vector_width,)`` name ``<base>_vec_k``."""
        return f"{base}{VecNameScheme.K_SUFFIX}"

    @staticmethod
    def is_vec(name: str) -> bool:
        """True iff ``name`` ends with any of the ``_vec`` family suffixes.

        Matches plain ``_vec``, indexed ``_vec_<digits>``, and the
        ``_vec_k`` variant. Does NOT match ``_packed`` (a sibling
        scheme — checked via ``PackedNameScheme.is_packed``).
        """
        if name.endswith(VecNameScheme.SUFFIX):
            return True
        if name.endswith(VecNameScheme.K_SUFFIX):
            return True
        return VecNameScheme._INDEXED_RE.match(name) is not None


class PackedNameScheme:
    """Centralised packed-buffer naming for scatter / gather paths.

    ``<base>_packed`` is reserved for **scatter / gather** access
    patterns only. The ``detect_{gather,scatter,strided_*}`` passes
    classify access nodes via this suffix; the producer
    (``_generate_loads_to_packed_storage`` and friends in ``vectorize.py``)
    must mint names through ``make`` so the classifier always agrees.
    """

    SUFFIX = "_packed"

    @staticmethod
    def make(base: str) -> str:
        """Build the packed name ``<base>_packed``."""
        return f"{base}{PackedNameScheme.SUFFIX}"

    @staticmethod
    def is_packed(name: str) -> bool:
        """True iff ``name`` ends with the ``_packed`` suffix."""
        return name.endswith(PackedNameScheme.SUFFIX)
