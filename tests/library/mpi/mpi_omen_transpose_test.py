# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Distributed OMEN self-energy transpose: the cross-rank sibling of the local k10 witness.

The SC19 Gordon Bell OMEN reformulation (Ziogas, Ben-Nun, Indalecio Fernandez, Schulthess, Hoefler,
"A Data-Centric Approach to Extreme-Scale Ab initio Dissipative Quantum Transport Simulations", SC'19,
arXiv:1912.10024) runs two phases over a tensor of tiny ``M x M`` Green's-function blocks that want
DIFFERENT distributed axes, and switches between them with an all-to-all transpose:

    phase 1 (RGF-like, ENERGY-distributed):  each rank owns an E slab, all atoms.
                                             G_loc[el, a] = H[a, el] @ X[a, el]   (producer/E-outer order)
    transpose (MPI_Alltoall):                energy-distributed  ->  atom-distributed
    phase 2 (SSE window, ATOM-distributed):  each rank owns an atom slab, all energies.
                                             Sigma[al, eo] = sum_{w<W} G[al, eo+W-1-w] @ D[w]

The redistribution is a P x P block transpose: rank r packs its ``G_loc`` into P contiguous chunks
(chunk d = the atoms owned by rank d, gathered STRIDED out of the energy-outer producer layout -- this
strided gather IS the transpose), ``MPI_Alltoall`` swaps the chunks, and the received chunks unpack into
the atom-outer consumer layout. Every rank shares the same global partition, so the wire order matches by
construction. This mirrors the paper's four ``MPI_Alltoallv`` on G/D/Sigma/Pi that cut the exchanged
Green's functions ~2.58 PiB -> ~1.8 TiB.

``MpiPackUnpack`` automates this contiguity for point-to-point buffers only; collectives are packed
manually here (mapped tasklets), matching the paper's hand-built Alltoallv pack. The local (single-process)
form of the same layout choice is ``tests/transformations/layout/kernels/k10_omen_windowed_contraction.py``.
"""
import numpy
import pytest

import dace
import dace.libraries.mpi as mpi

# small static toy dims; P must divide NA and NE
P, NA, NE, M, W = 2, 4, 4, 2, 2
NAl, NEl, NEO = NA // P, NE // P, NE - W
# not decorative: an indivisible NA/NE leaves the tail of every rank's local block untouched by the pack,
# so the numpy.empty buffers carry uninitialized values into the comparison and the test passes or fails
# on garbage instead of on the transpose
assert NA % P == 0 and NE % P == 0, f"P={P} must divide NA={NA} and NE={NE}"


@dace.program
def phase1_pack(Hl: dace.complex128[NA, NEl, M, M], Xl: dace.complex128[NA, NEl, M, M],
                send: dace.complex128[P, NAl, NEl, M, M]):
    """Energy-distributed producer + pack into per-destination-rank chunks (the strided transpose gather)."""
    Gl = numpy.empty((NEl, NA, M, M), dace.complex128)  # producer/E-outer order, single-writer
    for el, a, i, j in dace.map[0:NEl, 0:NA, 0:M, 0:M] @ dace.ScheduleType.Sequential:
        s = dace.complex128(0)
        for k in range(M):
            s = s + Hl[a, el, i, k] * Xl[a, el, k, j]
        Gl[el, a, i, j] = s
    for d, al, el, i, j in dace.map[0:P, 0:NAl, 0:NEl, 0:M, 0:M] @ dace.ScheduleType.Sequential:
        send[d, al, el, i, j] = Gl[el, d * NAl + al, i, j]  # atom d*NAl+al -> chunk d


@dace.program
def unpack_phase2(recv: dace.complex128[P, NAl, NEl, M, M], D: dace.complex128[W, M, M],
                  Sigma: dace.complex128[NAl, NEO, M, M]):
    """Unpack the received chunks into atom-outer consumer order + the SSE energy-window contraction."""
    Ga = numpy.empty((NAl, NE, M, M), dace.complex128)  # consumer/atom-outer order
    for s, al, el, i, j in dace.map[0:P, 0:NAl, 0:NEl, 0:M, 0:M] @ dace.ScheduleType.Sequential:
        Ga[al, s * NEl + el, i, j] = recv[s, al, el, i, j]  # energy s*NEl+el arrived from rank s
    for al, eo, i, j, w, k in dace.map[0:NAl, 0:NEO, 0:M, 0:M, 0:W, 0:M] @ dace.ScheduleType.Sequential:
        Sigma[al, eo, i, j] += Ga[al, eo + W - 1 - w, i, k] * D[w, k, j]


def build_distributed_sdfg():
    """phase1_pack -> Alltoall -> unpack_phase2, wired as one dataflow state (Alltoall in the middle)."""
    sdfg = dace.SDFG("omen_transpose_dist")
    sdfg.add_array("Hl", [NA, NEl, M, M], dace.complex128)
    sdfg.add_array("Xl", [NA, NEl, M, M], dace.complex128)
    sdfg.add_array("D", [W, M, M], dace.complex128)
    sdfg.add_array("Sigma", [NAl, NEO, M, M], dace.complex128)
    sdfg.add_transient("send", [P, NAl, NEl, M, M], dace.complex128)
    sdfg.add_transient("recv", [P, NAl, NEl, M, M], dace.complex128)
    st = sdfg.add_state("transpose")

    n1 = st.add_nested_sdfg(phase1_pack.to_sdfg(simplify=True), {"Hl", "Xl"}, {"send"})
    st.add_edge(st.add_read("Hl"), None, n1, "Hl", dace.Memlet.from_array("Hl", sdfg.arrays["Hl"]))
    st.add_edge(st.add_read("Xl"), None, n1, "Xl", dace.Memlet.from_array("Xl", sdfg.arrays["Xl"]))
    send = st.add_access("send")
    st.add_edge(n1, "send", send, None, dace.Memlet.from_array("send", sdfg.arrays["send"]))

    a2a = mpi.nodes.alltoall.Alltoall("transpose_a2a")
    recv = st.add_access("recv")
    st.add_edge(send, None, a2a, "_inbuffer", dace.Memlet.from_array("send", sdfg.arrays["send"]))
    st.add_edge(a2a, "_outbuffer", recv, None, dace.Memlet.from_array("recv", sdfg.arrays["recv"]))

    n2 = st.add_nested_sdfg(unpack_phase2.to_sdfg(simplify=True), {"recv", "D"}, {"Sigma"})
    st.add_edge(recv, None, n2, "recv", dace.Memlet.from_array("recv", sdfg.arrays["recv"]))
    st.add_edge(st.add_read("D"), None, n2, "D", dace.Memlet.from_array("D", sdfg.arrays["D"]))
    st.add_edge(n2, "Sigma", st.add_write("Sigma"), None, dace.Memlet.from_array("Sigma", sdfg.arrays["Sigma"]))
    return sdfg


# --------------------------------------------------------------------------- #
#  oracle + inputs (single-process k10 self-energy)
# --------------------------------------------------------------------------- #
def make_inputs(seed=0):
    rng = numpy.random.default_rng(seed)
    c = lambda *s: (rng.random(s) + 1j * rng.random(s)).astype(numpy.complex128)
    return c(NA, NE, M, M), c(NA, NE, M, M), c(W, M, M)


def oracle(H, X, D):
    G = numpy.matmul(H, X)
    Sig = numpy.zeros((NA, NEO, M, M), numpy.complex128)
    for a in range(NA):
        for eo in range(NEO):
            acc = numpy.zeros((M, M), numpy.complex128)
            for w in range(W):
                acc = acc + G[a, eo + W - 1 - w] @ D[w]
            Sig[a, eo] = acc
    return Sig


# --------------------------------------------------------------------------- #
#  Offline: the DaCe compute + pack/unpack kernels with the Alltoall simulated
#  in numpy (a P x P chunk swap). Exercises the real reindexing without MPI.
# --------------------------------------------------------------------------- #
def test_omen_transpose_kernels_offline():
    H, X, D = make_inputs()
    ref = oracle(H, X, D)

    p1 = phase1_pack.to_sdfg(simplify=True)
    p2 = unpack_phase2.to_sdfg(simplify=True)

    sends = []
    for r in range(P):
        send = numpy.zeros((P, NAl, NEl, M, M), numpy.complex128)
        p1(Hl=H[:, r * NEl:(r + 1) * NEl].copy(), Xl=X[:, r * NEl:(r + 1) * NEl].copy(), send=send)
        sends.append(send)

    Sig = numpy.zeros((NA, NEO, M, M), numpy.complex128)
    for r in range(P):
        recv = numpy.stack([sends[s][r] for s in range(P)])  # MPI_Alltoall: recv_r[s] = send_s[r]
        sl = numpy.zeros((NAl, NEO, M, M), numpy.complex128)
        p2(recv=recv.copy(), D=D.copy(), Sigma=sl)
        Sig[r * NAl:(r + 1) * NAl] = sl
    assert numpy.allclose(Sig, ref), numpy.abs(Sig - ref).max()


# --------------------------------------------------------------------------- #
#  Distributed: real 2-rank MPI_Alltoall transpose. Run under `mpirun -n 2`.
# --------------------------------------------------------------------------- #
@pytest.mark.mpi
def test_omen_transpose_mpi():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size != P:
        raise ValueError(f"run with exactly {P} processes")

    compiled = None
    for r in range(size):
        if r == rank:
            compiled = build_distributed_sdfg().compile()
        comm.Barrier()

    H, X, D = make_inputs()  # every rank builds the same global inputs, slices its own
    ref = oracle(H, X, D)
    Sigma = numpy.zeros((NAl, NEO, M, M), numpy.complex128)
    compiled(Hl=H[:, rank * NEl:(rank + 1) * NEl].copy(),
             Xl=X[:, rank * NEl:(rank + 1) * NEl].copy(),
             D=D.copy(),
             Sigma=Sigma)
    mine = ref[rank * NAl:(rank + 1) * NAl]
    if not numpy.allclose(Sigma, mine):
        raise ValueError(f"rank {rank}: max abs diff {numpy.abs(Sigma - mine).max()}")


if __name__ == "__main__":
    test_omen_transpose_kernels_offline()
    test_omen_transpose_mpi()
