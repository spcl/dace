# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Vectorization helper utilities.

This package is the destination of the planned split of the legacy
``vectorization_utils.py`` junk-drawer module. Modules are organised by
concern and added one slice at a time. The legacy
``vectorization_utils.py`` keeps re-exporting from here during the
migration; once every consumer is migrated the legacy file is deleted
(see plan slice S7).
"""
from .layout import assert_strides_are_packed_C_or_packed_Fortran  # noqa: F401
