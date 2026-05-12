# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Centralised name schemes for the vectorization pipeline.

The vectorization pipeline mints a few classes of synthetic identifiers
(per-lane symbols, vector-suffixed arrays, packed scatter/gather
buffers). Keeping the rules for each class in one place avoids the
silent-mismatch class of bug where one site composes ``<name>_laneid_3``
by string concatenation while another site parses it with a different
regex.

For now this module owns ``LaneIdScheme``. The companion
``VecNameScheme`` (the contiguous ``_vec`` / ``_vec_k`` suffix) and
``PackedNameScheme`` (scatter/gather ``_packed`` suffix) are still
inlined as constants at their use sites; centralising them is tracked
as the NameScheme follow-up alongside the
``project_yakup_dev_naming_scheme`` memo (``_packed`` = scatter/gather
only; ``_vec`` = everything else; inout connectors share the same
suffix across both directions).
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
