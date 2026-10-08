# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lazy fallback to ``dace.libraries.onnx`` pure expansions for operators without a native lowering."""

from typing import Callable, Optional


def lookup(target) -> Optional[Callable]:
    """Returns an ONNX-library-based lowering for ``target`` if one is available, else ``None``."""
    # Not implemented yet. Importing dace.libraries.onnx requires the ``onnx`` package, so this module must
    # import it lazily and return None when unavailable.
    return None
