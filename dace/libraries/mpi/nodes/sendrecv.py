# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import (MPINode, resolve_comm, validate_integer_descriptor,
                                           expanded_input_connectors)


@dace.library.expansion
class ExpandSendrecvMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inbuffer, scount), (outbuffer, rcount), dest, src, sendtag, recvtag = node.validate(parent_sdfg, parent_state)
        send_dtype = dace.libraries.mpi.utils.MPI_DDT(inbuffer.dtype.base_type)
        recv_dtype = dace.libraries.mpi.utils.MPI_DDT(outbuffer.dtype.base_type)

        if inbuffer.dtype.veclen > 1 or outbuffer.dtype.veclen > 1:
            raise NotImplementedError

        comm = resolve_comm(node, parent_state)
        code = f"""
            MPI_Sendrecv(&(_inbuffer[0]), {scount}, {send_dtype}, int(_dest), int(_sendtag),
                         _outbuffer, {rcount}, {recv_dtype}, int(_src), int(_recvtag), {comm}, MPI_STATUS_IGNORE);
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)
        return tasklet


@dace.library.node
class Sendrecv(MPINode):
    """Combined MPI_Sendrecv: send ``_inbuffer`` to ``_dest`` and receive ``_outbuffer`` from ``_src`` in one
    call (deadlock-free ring exchange). Buffers must be contiguous -- run ``MpiPackUnpack`` first for a
    strided (permuted/blocked) buffer, so MPI never sees a derived datatype."""

    # Global properties
    implementations = {
        "MPI": ExpandSendrecvMPI,
    }
    default_implementation = "MPI"

    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inbuffer", "_dest", "_src", "_sendtag", "_recvtag"},
                         outputs={"_outbuffer"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: ((inbuffer, send_count), (outbuffer, recv_count), dest, src, sendtag, recvtag)
        """
        inbuffer, outbuffer = None, None
        dest, src, sendtag, recvtag = None, None, None, None
        send_memlet, recv_memlet = None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer, send_memlet = sdfg.arrays[e.data.data], e.data
            elif e.dst_conn == "_dest":
                dest = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_src":
                src = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_sendtag":
                sendtag = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_recvtag":
                recvtag = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer, recv_memlet = sdfg.arrays[e.data.data], e.data

        validate_integer_descriptor(dest, 'Destination')
        validate_integer_descriptor(src, 'Source')
        validate_integer_descriptor(sendtag, 'Send tag')
        validate_integer_descriptor(recvtag, 'Recv tag')
        if send_memlet is None or recv_memlet is None:
            raise ValueError("Sendrecv requires both an _inbuffer input and an _outbuffer output.")

        # Contiguous buffers only: the layout MPI pack/unpack pass makes them so, keeping MPI stride-free.
        if not dace.libraries.mpi.utils.is_access_contiguous(send_memlet, sdfg.arrays[send_memlet.data]):
            raise NotImplementedError("Sendrecv: non-contiguous send buffer; run MpiPackUnpack first.")
        if not dace.libraries.mpi.utils.is_access_contiguous(recv_memlet, sdfg.arrays[recv_memlet.data]):
            raise NotImplementedError("Sendrecv: non-contiguous recv buffer; run MpiPackUnpack first.")

        send_count = "*".join(str(e) for e in send_memlet.subset.size_exact())
        recv_count = "*".join(str(e) for e in recv_memlet.subset.size_exact())
        return (inbuffer, send_count), (outbuffer, recv_count), dest, src, sendtag, recvtag
