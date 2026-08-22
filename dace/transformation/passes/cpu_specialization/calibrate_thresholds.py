# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Calibrate the fork/join thresholds to the host CPU, at the start of the specialization band."""
from typing import Any, Dict, Optional

from dace import SDFG, properties
from dace.config import Config
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.cpu_specialization import machine

#: Config keys this pass derives, paired with the function that derives each.
CALIBRATED = (
    (('compiler', 'cpu', 'parallel_min_work_per_region'), machine.min_work_per_region),
    (('compiler', 'cpu', 'parallel_transfer_min_elements'), machine.transfer_min_elements),
)


@properties.make_properties
@xf.explicit_cf_compatible
class CalibrateCpuThresholds(ppl.Pass):
    """Point the CPU fork/join thresholds at the machine that will RUN the code.

    Canonical form is parallel; this band is where a scope is allowed to become sequential again.
    That verdict is a cost comparison, and its two thresholds shipped as constants measured on one
    development box -- so a 72-core server inherited an 8-core machine's break-even points. Both are
    now derived from the host's own core count (see
    :mod:`~dace.transformation.passes.cpu_specialization.machine`).

    Only a key still sitting at its SCHEMA DEFAULT is touched. A value the user put in a config file
    or an environment variable is a deliberate choice and is left exactly as it is -- which also
    means a sweep that pins these to compare two machines keeps comparing what it pinned.

    Reads nothing from the SDFG and writes nothing to it: the host is a property of the process, so
    this is idempotent and its verdict is the same for every SDFG compiled in it.
    """

    CATEGORY: str = 'CPU Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """:returns: the keys this pass set, mapped to the derived value; ``None`` when it set none."""
        applied = {}
        for key, derive in CALIBRATED:
            current = Config.get(*key)
            # A user-set value wins. ``Config.set`` would lose to a ``DACE_*`` environment variable
            # anyway, but a config FILE would be overwritten, so the comparison is what protects it.
            if int(current) != int(Config.get_default(*key)):
                continue
            value = derive()
            if int(value) == int(current):
                continue
            Config.set(*key, value=value)
            applied['.'.join(key)] = value
        return applied or None
