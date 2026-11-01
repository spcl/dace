# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-source shortest paths with the Consume scope: a worklist of vertices to relax.

Every task is a vertex ``u`` whose distance went down. It relaxes the edges leaving ``u``: when the path through
``u`` is shorter than the best known distance of a neighbor ``v``, it lowers ``dist[v]`` and pushes ``v``, whose
own edges must then be relaxed again. The computation is done once the worklist is empty and no task is running.

The algorithm is *asynchronous* (asynchronous Bellman-Ford), not bulk-synchronous: unlike the rounds of classic
Bellman-Ford or of a level-by-level breadth-first search, nothing waits for all vertices to finish a step. Processing
elements pop vertices in any order, and a distance that one task lowers is used by the next task that reads it.

Distances are lowered with ``dace.atomic.min``, which returns the old distance: concurrent improvements of one vertex
are merged rather than lost, and only a task that lowered ``dist[v]`` pushes ``v``. Tasks also *read* the distance of
``u``, which other tasks may lower at the same time. That read may see an older, larger value, which is harmless: a
task that lowered ``dist[u]`` pushes ``u`` again, so ``u`` is relaxed with its final distance eventually.
"""

import argparse
import heapq

import numpy as np

import dace

V = dace.symbol("V")  # Vertices
E = dace.symbol("E")  # Edges


@dace.program
def sssp(
    row: dace.int32[V + 1], col: dace.int32[E], weight: dace.float64[E], dist: dace.float64[V], source: dace.int32
):
    # The vertices whose distance went down and whose edges need relaxing (on a GPU, a queue of this capacity)
    S = dace.define_stream(dace.int32, 1 << 20)

    for _ in dace.map[0:1]:  # A map rather than a lone push, so that it also runs on a GPU
        S.append(source)

    for u, _ in dace.consume(S, 8):
        for e in range(row[u], row[u + 1]):
            v = col[e]
            candidate = dist[u] + weight[e]
            # Lowers dist[v] to the candidate if it is shorter, and returns the distance before
            if dace.atomic.min(dist, v, candidate) > candidate:
                S.append(v)  # Lowered: the edges of v must be relaxed again


def random_graph(num_vertices: int, degree: int, seed: int = 0):
    """A random directed graph in CSR form with ``degree`` edges per vertex and positive weights."""
    rng = np.random.default_rng(seed)
    row = np.arange(0, num_vertices * degree + 1, degree, dtype=np.int32)
    col = rng.integers(0, num_vertices, num_vertices * degree, dtype=np.int32)
    weight = rng.random(num_vertices * degree) + 0.01
    return row, col, weight


def dijkstra(row: np.ndarray, col: np.ndarray, weight: np.ndarray, source: int) -> np.ndarray:
    dist = np.full(len(row) - 1, np.inf)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for e in range(row[u], row[u + 1]):
            v, nd = col[e], d + weight[e]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def to_gpu(sdfg: dace.SDFG) -> None:
    """Runs the same program on a GPU: one GPU thread per processing element."""
    for desc in sdfg.arrays.values():
        if not isinstance(desc, dace.data.Scalar):
            desc.storage = dace.StorageType.GPU_Global
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            node.map.schedule = dace.ScheduleType.GPU_Device
        elif isinstance(node, dace.nodes.ConsumeEntry):
            node.consume.schedule = dace.ScheduleType.GPU_Device
            node.consume.num_pes = 4096


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vertices", type=int, nargs="?", default=10000)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--gpu", action="store_true", help="run on a GPU (requires CuPy)")
    args = parser.parse_args()

    row, col, weight = random_graph(args.vertices, args.degree)
    dist = np.full(args.vertices, np.inf)
    dist[0] = 0.0
    sdfg = sssp.to_sdfg(simplify=True)
    if args.gpu:
        import cupy

        to_gpu(sdfg)
        gpu_dist = cupy.asarray(dist)
        sdfg(row=cupy.asarray(row), col=cupy.asarray(col), weight=cupy.asarray(weight), dist=gpu_dist, source=0)
        dist = gpu_dist.get()
    else:
        sdfg(row=row, col=col, weight=weight, dist=dist, source=0)

    reference = dijkstra(row, col, weight, 0)
    reached = np.isfinite(reference)
    print(f"Reached {reached.sum()} of {args.vertices} vertices")
    correct = np.array_equal(np.isfinite(dist), reached) and np.allclose(dist[reached], reference[reached])
    print("Distances match Dijkstra's" if correct else "Distances differ from Dijkstra's")
    exit(0 if correct else 1)
