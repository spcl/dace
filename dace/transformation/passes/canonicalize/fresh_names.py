# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fresh ``<prefix><N>`` name suffixes shared by the canonicalization passes."""
from typing import Dict, Tuple

from dace import SDFG
from dace.sdfg.state import LoopRegion


def lowest_free_suffix(sdfg: SDFG, prefixes: Tuple[str, ...], with_free_symbols: bool = False) -> int:
    """Lowest ``N`` such that no ``<prefix><N>`` for any of ``prefixes`` is used in the SDFG tree.

    :param sdfg: The root SDFG; nested SDFGs are scanned too.
    :param prefixes: The name prefixes that share one suffix counter.
    :param with_free_symbols: Also scan each SDFG's free symbols, not only its declared symbols.
    :return: The first unused suffix.
    """
    used: Dict[int, None] = {}
    for sd in sdfg.all_sdfgs_recursive():
        names = list(sd.symbols.keys())
        if with_free_symbols:
            names += list(sd.free_symbols)
        names += [cfg.loop_variable for cfg in sd.all_control_flow_regions() if isinstance(cfg, LoopRegion)]
        for name in names:
            for pre in prefixes:
                if name and name.startswith(pre) and name[len(pre):].isdigit():
                    used[int(name[len(pre):])] = None
    n = 0
    while n in used:
        n += 1
    return n
