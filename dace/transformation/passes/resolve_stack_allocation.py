# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Resolves the stack placement of register arrays that leave it to the code generator. """

from dataclasses import dataclass
import sympy

from dace import SDFG, Config, data, dtypes, properties, symbolic
from dace.transformation import pass_pipeline as ppl, transformation

#: Constant-sized register arrays below this many elements default to the stack.
STACK_ARRAY_MAX_ELEMENTS = 2048


def resolve_stack_allocation(desc: data.Data, constants: dict[str, object]) -> bool:
    """ Whether a register array is placed on the stack: an explicit ``StorageType.Register(stack=...)`` is kept;
        otherwise a constant size below ``STACK_ARRAY_MAX_ELEMENTS`` elements and ``compiler.max_stack_array_size``
        bytes is on the stack, and a symbolic size on the heap.
    """
    stack = dtypes.is_stack_register(desc.storage)
    if stack is not None:
        return stack
    size = desc.total_size
    if symbolic.issymbolic(size, constants):
        return False
    if isinstance(size, sympy.Basic):
        # By name: the constant's symbol may have another dtype than the one in the shape.
        size = size.subs({sym: constants[sym.name] for sym in size.free_symbols})
    size = int(size)
    size_bytes = size * desc.dtype.bytes if not isinstance(desc.dtype, dtypes.opaque) else 0
    return size < STACK_ARRAY_MAX_ELEMENTS and size_bytes <= Config.get('compiler', 'max_stack_array_size')


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class ResolveStackAllocation(ppl.Pass):
    """ Replaces the undecided ``StorageType.Register`` of every transient register array with
        ``StorageType.Register(stack=...)`` as decided by ``resolve_stack_allocation``, so the generated code follows
        a placement visible in the SDFG.
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
                        and dtypes.is_stack_register(desc.storage) is None):
                    desc.storage = dtypes.StorageType.Register(stack=resolve_stack_allocation(desc, nsdfg.constants))
                    resolved.add((nsdfg.cfg_id, name))
        return resolved or None
