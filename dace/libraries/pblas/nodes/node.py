# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Communicator and BLACS-context resolution shared by the PBLAS library nodes.

Follows :mod:`dace.libraries.mpi.nodes.node`: the connector NAME is the semantic
discriminator, the connector is wired dynamically instead of being declared in
the node's fixed inputs, and a node with nothing wired falls back to the process
default. PBLAS reads two of them:

  * ``_context`` -- an integer BLACS context, as produced by
    :class:`~dace.libraries.pblas.nodes.gridinit.BlacsGridInit`. Used directly.
  * ``_comm`` -- a raw ``opaque(MPI_Comm)``, the same value ``CommF2c`` produces
    and :func:`dace.libraries.mpi.nodes.node.resolve_comm` hands the MPI nodes.
    The ``Py`` x ``Px`` grid is built on it. Falls back to ``MPI_COMM_WORLD``,
    which is what the environment's ``__dace_init_`` used to build it on.

``_context`` takes priority. Neither the communicator nor the grid shape reaches
``__dace_init_``: a symbol an initializer consumes is frozen at the first call,
so a program whose grid came from there could only ever have one grid.
"""
from typing import Any

from dace.libraries.mpi.nodes.node import input_descriptor_name

#: (integer type, state-field prefix) of each environment family.
MKL_SPELLING = ('MKL_INT', '__mkl_scalapack')
REF_SPELLING = ('int', '__scalapack')


def resolve_comm(node: Any, state: Any) -> str:
    """The communicator a PBLAS node builds its BLACS grid on."""
    if input_descriptor_name(node, state, '_comm'):
        return '_comm'
    return 'MPI_COMM_WORLD'


def blacs_grid_code(mkl: bool, comm: str, prows: str, pcols: str) -> str:
    """C++ defining ``__ctxt``, the context of the ``prows`` x ``pcols`` grid over
    ``comm``. The grid is built on first use and memoized in the program state,
    which is also what keeps ``Cblacs_gridexit`` reachable at finalize time."""
    itype, prefix = MKL_SPELLING if mkl else REF_SPELLING
    if mkl:
        gridinit = 'blacs_gridinit(&__ctxt, "C", &__prows, &__pcols);'
    else:
        gridinit = "char __order = 'C';\n                Cblacs_gridinit(&__ctxt, &__order, __prows, __pcols);"
    return f"""
        {itype} __ctxt;
        {{
            {itype} __prows = {prows}, __pcols = {pcols};
            int __fcomm = (int)MPI_Comm_c2f({comm});
            if (!dace_blacs_grid_find(__state->{prefix}_grids, __fcomm, __prows, __pcols, &__ctxt)) {{
                __ctxt = Csys2blacs_handle({comm});
                {gridinit}
                __state->{prefix}_grids.push_back({{__fcomm, __prows, __pcols, __ctxt}});
            }}
        }}"""


def scalapack_grid_code(node: Any, state: Any, mkl: bool) -> str:
    """C++ defining the BLACS context ``__ctxt`` a PBLAS node operates on, the
    ``__nprow`` x ``__npcol`` shape of that grid and the node's ``__myprow`` /
    ``__mypcol`` position in it."""
    itype, _ = MKL_SPELLING if mkl else REF_SPELLING
    if input_descriptor_name(node, state, '_context'):
        head = f'{itype} __ctxt = _context;'
    else:
        head = blacs_grid_code(mkl, resolve_comm(node, state), 'Py', 'Px')
    if mkl:
        gridinfo = 'blacs_gridinfo(&__ctxt, &__nprow, &__npcol, &__myprow, &__mypcol);'
    else:
        gridinfo = 'Cblacs_gridinfo(__ctxt, &__nprow, &__npcol, &__myprow, &__mypcol);'
    return f"""
        {head}
        {itype} __nprow, __npcol, __myprow, __mypcol;
        {gridinfo}"""
