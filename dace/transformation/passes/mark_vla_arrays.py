# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Marks the register arrays that belong on the stack as variable-length arrays. """

from dataclasses import dataclass
from typing import Optional, Set, Tuple

from dace import SDFG, data, dtypes, properties, symbolic
from dace.transformation import pass_pipeline as ppl, transformation

BLOCK_SCOPED_LIFETIMES = (dtypes.AllocationLifetime.Scope, dtypes.AllocationLifetime.State,
                          dtypes.AllocationLifetime.SDFG)


def array_takes_vla(sdfg: SDFG, desc: data.Data, defined_symbols: Set[str]) -> bool:
    """
    Whether a register array can live as a stack variable-length array, which GCC, Clang and NVHPC
    all accept. A VLA is defined at its declaration and dies with its block, so only a lifetime
    that ends with a block qualifies, and only when every symbol in the size is already defined
    where the array is declared: a size that an interstate edge assigns splits the declaration from
    the allocation (see ``framecode.determine_allocation_lifetime``), and a VLA cannot bridge that
    split.

    :param sdfg: The SDFG that owns the descriptor.
    :param desc: The data descriptor to decide on.
    :param defined_symbols: Names of the symbols defined where ``sdfg`` is entered.
    :return: True if the descriptor can be declared as a variable-length array.
    """
    if not isinstance(desc, data.Array) or isinstance(desc, data.View):
        return False
    if not desc.transient or desc.storage != dtypes.StorageType.Register:
        return False
    if desc.lifetime not in BLOCK_SCOPED_LIFETIMES:
        return False
    if not symbolic.issymbolic(desc.total_size, sdfg.constants):
        return False
    return all(str(s) in defined_symbols for s in desc.free_symbols)


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class MarkVLAArrays(ppl.Pass):
    """
    Sets ``Data.vla`` on the symbolically-sized register arrays that belong on the stack. Code
    generation only renders the declaration; the safety analysis lives here.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Descriptors | ppl.Modifies.Symbols | ppl.Modifies.InterstateEdges)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[Tuple[int, str]]]:
        """
        Marks every qualifying array in ``sdfg`` and its nested SDFGs, and clears the mark on the rest.

        :param sdfg: The SDFG to modify.
        :param _: Pipeline results, unused.
        :return: The marked arrays as ``(CFG id, data name)``, or ``None`` if none qualified.
        """
        marked: Set[Tuple[int, str]] = set()
        for nsdfg in sdfg.all_sdfgs_recursive():
            # A subset of what code generation calls "symbols and constants": the parent's constants
            # are left out, so an array is only ever kept off the stack, never put on it wrongly.
            defined_symbols = {str(s) for s in nsdfg.free_symbols} | set(nsdfg.constants_prop.keys())
            for name, desc in nsdfg.arrays.items():
                desc.vla = array_takes_vla(nsdfg, desc, defined_symbols)
                if desc.vla:
                    marked.add((nsdfg.cfg_id, name))
        return marked or None
