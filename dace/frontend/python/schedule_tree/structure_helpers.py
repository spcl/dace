# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Compatibility shim for the older structure helper module name."""

from dace.frontend.python.schedule_tree.structure_support import bind_target_structure, clone_descriptor, \
    descriptor_from_structure

__all__ = ['bind_target_structure', 'clone_descriptor', 'descriptor_from_structure']
