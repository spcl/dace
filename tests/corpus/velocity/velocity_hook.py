# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CPU velocity-tendencies pipeline, as a dace-fortran e2e harness hook (``OCEAN_E2E_SDFG_HOOK``).

The harness runs it on the freshly built SDFG in its child process: bake the configuration in ``VT_CONFIG``, drop
the branches it decides, parallelize, and re-snapshot the Fortran signature the binding is generated from.
"""

import json
import os

from dace.sdfg import SDFG
from dace.transformation.passes.config_propagation import specialize_variant


def run(sdfg: SDFG) -> None:
    from dace_fortran.bindings.frozen_signature import refreeze
    from dace_fortran.pipelines import num_loops, num_maps, optimize

    values = json.loads(os.environ["VT_CONFIG"])
    specialize_variant(sdfg, values)
    optimize(sdfg)
    print(f"VT_PIPELINE maps={num_maps(sdfg)} loops={num_loops(sdfg)}", flush=True)
    refreeze(sdfg)
