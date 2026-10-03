# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Frame code emits environment headers in the order ``get_environments_and_dependencies`` returns."""
import pytest

import dace.library
from dace.libraries.sort.environments.cub import (CUB, BlockCollectives, DetectScratch, ReduceScratch, ScanScratch,
                                                  SortScratch)
from dace.libraries.standard.environments.cuda import CUDA
from dace.ordered import OrderedSet

CUB_USERS = [SortScratch, ScanScratch, BlockCollectives, ReduceScratch, DetectScratch]


@pytest.mark.parametrize('requested', [CUB_USERS, CUB_USERS[::-1]], ids=['forward', 'reversed'])
def test_sibling_environments_come_back_in_request_order_after_their_shared_dependencies(requested: list[type]):
    names = OrderedSet(env.full_class_path() for env in requested)

    resolved = dace.library.get_environments_and_dependencies(names)

    assert resolved == [CUDA, CUB, *requested]
