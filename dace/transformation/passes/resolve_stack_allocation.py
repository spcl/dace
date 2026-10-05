# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Resolves ``Data.stack_vla == Auto`` on register arrays to ``Stack`` or ``Heap``. """

from dataclasses import dataclass
import sympy

from dace import SDFG, Config, data, dtypes, properties, symbolic
from dace.transformation import pass_pipeline as ppl, transformation

#: Constant-sized register arrays below this many elements default to the stack.
STACK_ARRAY_MAX_ELEMENTS = 2048


def resolve_stack_allocation(desc: data.Data, constants: dict[str, object]) -> dtypes.StackAllocation:
    """ The placement of a register array: an explicit choice is kept; ``Auto`` puts a constant size
        below ``STACK_ARRAY_MAX_ELEMENTS`` elements and ``compiler.max_stack_array_size`` bytes on the
        stack, and a symbolic size on the heap.
    """
    if desc.stack_vla is not dtypes.StackAllocation.Auto:
        return desc.stack_vla
    size = desc.total_size
    if symbolic.issymbolic(size, constants):
        return dtypes.StackAllocation.Heap
    if isinstance(size, sympy.Basic):
        # By name: the constant's symbol may have another dtype than the one in the shape.
        size = size.subs({sym: constants[sym.name] for sym in size.free_symbols})
    size = int(size)
    size_bytes = size * desc.dtype.bytes if not isinstance(desc.dtype, dtypes.opaque) else 0
    if size < STACK_ARRAY_MAX_ELEMENTS and size_bytes <= Config.get('compiler', 'max_stack_array_size'):
        return dtypes.StackAllocation.Stack
    return dtypes.StackAllocation.Heap


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class ResolveStackAllocation(ppl.Pass):
    """ Replaces ``Auto`` stack placement on every transient register array with the decision of
        ``resolve_stack_allocation``, so the generated code follows a placement visible in the SDFG.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Descriptors)

    def apply_pass(self, sdfg: SDFG, _: dict) -> set[tuple[int, str]] | None:
        """ :return: The ``(cfg_id, name)`` of every resolved array, or None if none was resolved. """
        resolved = set()
        for nsdfg in sdfg.all_sdfgs_recursive():
            for name, desc in nsdfg.arrays.items():
                if (isinstance(desc, data.Array) and not isinstance(desc, data.View) and desc.transient
                        and desc.storage == dtypes.StorageType.Register
                        and desc.stack_vla is dtypes.StackAllocation.Auto):
                    desc.stack_vla = resolve_stack_allocation(desc, nsdfg.constants)
                    resolved.add((nsdfg.cfg_id, name))
        return resolved or None
