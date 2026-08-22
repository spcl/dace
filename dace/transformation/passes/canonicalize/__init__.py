# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""SDFG canonicalization pipeline subpackage.

See ``DESIGN.md`` in this directory for the living design document.
"""
from dace.transformation.passes.canonicalize.pipeline import (CanonicalizationPipeline, canonicalize,
                                                              CANONICALIZE_STAGES)
from dace.transformation.passes.canonicalize.debug import (canonicalize_with_stage_checks, first_failing_stage,
                                                           StageCheckResult)
