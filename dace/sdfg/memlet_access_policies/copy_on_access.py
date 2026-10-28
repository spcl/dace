# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The default memlet access policy."""

from dataclasses import dataclass
from typing import ClassVar

import dace.serialize
from dace.sdfg.memlet_access_policy import MemletAccessPolicy


@dace.serialize.serializable
@dataclass
class CopyOnAccess(MemletAccessPolicy):
    """The default policy and the behavior every SDFG had before access policies existed: the full offset is
    evaluated at every access and data moves at the access (or through the copy library node that
    :class:`~dace.transformation.passes.insert_explicit_copies.InsertExplicitCopies` materializes for
    access-node-to-access-node copies). Lowering is a no-op."""

    is_default: ClassVar[bool] = True
