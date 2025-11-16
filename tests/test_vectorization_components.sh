# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
#!/bin/sh
pytest -n $(nproc) tests/transformations/interstate/branch_elimination_test.py tests/passes/split_tasklets_test.py tests/utils/array_dimension_utils_test.py tests/utils/is_contiguous_subset_test.py tests/utils/symbol_in_code_test.py tests/utils/specialize_scalar_test.py tests/utils/classify_tasklet_test.py tests/utils/duplicate_memlets_from_single_in_connector_test.py tests/utils/generate_assignment_as_tasklet_instate_test.py tests/passes/vectorization_test.py
