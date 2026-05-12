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
from .queries import (  # noqa: F401
    collect_accesses_to_array_name,
    collect_all_memlets_to_dataname,
    collect_non_unit_stride_accesses_in_map,
    parse_int_or_default,
    to_ints,
)
from .code_rewrite import (  # noqa: F401
    drop_dims,
    drop_dims_from_str,
    extract_bracket_contents,
    offset_symbol_in_expression,
    use_laneid_symbol_in_expression,
)
from .iteration import walk_memlets_of  # noqa: F401
