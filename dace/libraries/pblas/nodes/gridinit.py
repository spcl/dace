# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Build a BLACS process grid inside an SDFG from an MPI communicator, as an
explicit dataflow node.

The ScaLAPACK analogue of :class:`~dace.libraries.mpi.nodes.comm_f2c.CommF2c`:
the communicator arrives on ``_comm`` (``opaque(MPI_Comm)``, produced by
``CommF2c`` or ``CommSplit``), the grid shape arrives on the ``_prows`` /
``_pcols`` integer scalars, and the BLACS context leaves on ``_context`` as an
ordinary integer value that flows into the ``_context`` input of ``Pgemm`` /
``Pgemv`` / the block-cyclic redistribution nodes.

This is what makes more than one process grid expressible: the context is a value
in the graph, so nothing about the grid reaches ``__dace_init_``, where a symbol
is frozen at the first call and every grid after the first would silently have
been the first one.
"""
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.mpi.nodes.node import expanded_input_connectors, validate_integer_descriptor
from dace.libraries.pblas import environments
from dace.libraries.pblas.nodes.node import blacs_grid_code, resolve_comm
from dace.ordered import OrderedSet


def grid_tasklet(node, parent_state, parent_sdfg, mkl: bool):
    node.validate(parent_sdfg, parent_state)
    code = blacs_grid_code(mkl, resolve_comm(node, parent_state), '_prows', '_pcols') + '\n        _context = __ctxt;'
    return dace.sdfg.nodes.Tasklet(node.name,
                                   expanded_input_connectors(node, parent_state),
                                   node.out_connectors,
                                   code,
                                   language=dtypes.Language.CPP,
                                   side_effects=True)


@dace.library.expansion
class ExpandBlacsGridInitMKLMPICH(ExpandTransformation):
    environments = [environments.intel_mkl_mpich.IntelMKLScaLAPACKMPICH]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return grid_tasklet(node, parent_state, parent_sdfg, True)


@dace.library.expansion
class ExpandBlacsGridInitMKLOpenMPI(ExpandTransformation):
    environments = [environments.intel_mkl_openmpi.IntelMKLScaLAPACKOpenMPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return grid_tasklet(node, parent_state, parent_sdfg, True)


@dace.library.expansion
class ExpandBlacsGridInitReferenceMPICH(ExpandTransformation):
    environments = [environments.ref_mpich.ScaLAPACKMPICH]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return grid_tasklet(node, parent_state, parent_sdfg, False)


@dace.library.expansion
class ExpandBlacsGridInitReferenceOpenMPI(ExpandTransformation):
    environments = [environments.ref_openmpi.ScaLAPACKOpenMPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return grid_tasklet(node, parent_state, parent_sdfg, False)


@dace.library.node
class BlacsGridInit(dace.sdfg.nodes.LibraryNode):
    """Collective ``Cblacs_gridinit`` over the communicator on ``_comm``, producing
    the ``_prows`` x ``_pcols`` grid's BLACS context on ``_context``."""

    # Global properties
    implementations = {
        "MKLMPICH": ExpandBlacsGridInitMKLMPICH,
        "MKLOpenMPI": ExpandBlacsGridInitMKLOpenMPI,
        "ReferenceMPICH": ExpandBlacsGridInitReferenceMPICH,
        "ReferenceOpenMPI": ExpandBlacsGridInitReferenceOpenMPI
    }
    default_implementation = None

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=OrderedSet(('_prows', '_pcols')), outputs={"_context"}, **kwargs)

    def has_side_effects(self, sdfg) -> bool:
        return True

    def validate(self, sdfg, state):
        """
        :return: a two-tuple (prows, pcols) of the input data descriptors in the
                 parent SDFG.
        """
        prows, pcols = None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_prows":
                prows = sdfg.arrays[e.data.data]
            if e.dst_conn == "_pcols":
                pcols = sdfg.arrays[e.data.data]
        validate_integer_descriptor(prows, "BlacsGridInit _prows")
        validate_integer_descriptor(pcols, "BlacsGridInit _pcols")
        return prows, pcols
