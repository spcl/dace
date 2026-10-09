.. _tile-calls:

Tile Calls
==========

The tile library nodes of :mod:`dace.libraries.tileops` can be called from a ``@dace.program`` as ``dace.tile``
functions, without the vectorizer. A *window* is an array or a slice of one, such as ``A[i:i + 8]``. Its extents other
than 1 are the lanes of the tile it makes, of which there are one to three dims of a constant extent. A *tile* is a
register array, ``dace.define_local([8], dace.float64, storage=dace.StorageType.Register)``. A call takes tiles or
windows, copying a window into a new tile first, and returns a new tile.

Tile nodes are supported by the new code generators: ``compiler.cpu.implementation: experimental_readable`` and
``compiler.cuda.implementation: experimental``.

Who runs a tile
---------------

Every call makes a node of the ``BLOCK`` group (:class:`~dace.libraries.tileops.dispatch.TileGroup`), the only group
the frontend offers: a call takes no group argument. The program says what a whole thread block computes, and the
compiler decides which thread computes which element:

* On a CPU one core runs the tile, as a loop over its lanes that the compiler vectorizes. A ``BLOCK`` node lowers
  exactly as a ``THREAD`` node does.
* In a GPU kernel the threads of the block share the tile. Each node becomes a thread-block map whose thread ``t``
  computes every ``threads``-th element from ``t``, so consecutive threads touch consecutive elements; a node with a
  target ISA takes ``lanes_per_thread`` consecutive elements at a time, two fp16 lanes for one half2 instruction. The
  tiles move to shared memory. A node whose lanes depend on each other (``sum``) runs on one thread of the block.

The block has ``32 * num_warps`` threads, four warps unless a node's ``num_warps`` says otherwise. The vectorizer emits
``THREAD`` nodes, one thread's register tile, which lower to the instructions of a target ISA.

Barriers
--------

:class:`~dace.transformation.passes.tile_synchronization.InsertTileSync` runs before the tile nodes expand. Every
write of a tile leaves a token for the subset it wrote and the threads that wrote each element. Walking the kernel in
program order, the first node that reads the subset on other threads -- an ``mma`` reading its operands, a ``sum``,
plain code -- gets a :class:`~dace.libraries.standard.nodes.barrier.Barrier` before it, the latest point the write can
be waited for; one barrier settles every token. A node that overwrites a subset other threads still read waits too, and
a loop body is walked again from what its end leaves, which places the barrier the next iteration needs. Accesses that
keep every element on its thread (an elementwise chain over one tile) need none. Tokens are per subset, so the slots of
a circular buffer wait only for themselves.

A ``Barrier`` waits for a warp, a thread block or the grid (:class:`~dace.libraries.standard.nodes.barrier.SyncScope`)
and lowers to ``__syncwarp()``, ``__syncthreads()`` or, on AMD, the wavefront barrier; on a CPU it is nothing.

Calls
-----

``dace.tile.load(window)``
    A new tile holding the lanes of ``window``.

``dace.tile.store(destination, source)``
    Copies the tile or window ``source`` into ``destination``.

``dace.tile.fill(tile, value)``
    Sets every lane of ``tile`` to the constant ``value``.

``dace.tile.add``, ``sub``, ``mul``, ``div``, ``minimum``, ``maximum`` ``(a, b)``
    The elementwise result for two windows or tiles of the same lanes and type.

``dace.tile.abs``, ``exp``, ``log``, ``sqrt``, ``neg``, ``tanh`` ``(a)``
    The elementwise result for one window or tile.

``dace.tile.fma(a, b, c)``
    ``a * b + c``, elementwise.

``dace.tile.where(mask, then, otherwise)``
    ``then`` where the ``bool`` tile or window ``mask`` is set, ``otherwise`` elsewhere.

``dace.tile.sum(a)``
    The sum of all lanes of ``a``, as a tile of one lane.

``dace.tile.mma(a, b, c)``
    Accumulates the matrix product of the ``(M, K)`` tile ``a`` and the ``(K, N)`` tile ``b`` into the ``(M, N)`` tile
    ``c``: ``c += a @ b``.

``dace.tile.masked_copy(destination, source, mask)``
    Copies ``source`` to ``destination`` where ``mask``, a window or tile of ``bool`` of the same lanes, is set. The
    lanes of a window destination that the mask switches off keep their value, and those of a tile destination are
    zeroed. A lane the mask switches off is neither read from the source nor written to the destination. The mask
    itself is read in every lane. Copying a window to a window goes through a tile.

.. code-block:: python

    @dace.program
    def masked_sum(A: dace.float64[N], B: dace.float64[N], M: dace.bool_[N], C: dace.float64[N]):
        for i in dace.map[0:N:8]:
            dace.tile.masked_copy(C[i:i + 8], dace.tile.add(A[i:i + 8], B[i:i + 8]), M[i:i + 8])

A blocked GEMM
--------------

Each ``(BM, BN)`` block of ``C`` accumulates the products of the ``(BM, BK)`` and ``(BK, BN)`` blocks of ``A`` and
``B`` along ``K``:

.. code-block:: python

    M, N, K = (dace.symbol(name) for name in "MNK")
    BM, BN, BK = 16, 16, 8

    @dace.program
    def blocked_gemm(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
        for i, j in dace.map[0:M:BM, 0:N:BN]:
            acc = dace.define_local([BM, BN], dace.float64, storage=dace.StorageType.Register)
            dace.tile.fill(acc, 0.0)
            for k in range(0, K, BK):
                dace.tile.mma(A[i:i + BM, k:k + BK], B[k:k + BK, j:j + BN], acc)
            dace.tile.store(C[i:i + BM, j:j + BN], acc)

On a CPU the map runs one block per core and each ``mma`` is a loop nest over the ``(BM, BN)`` outputs. After
``sdfg.apply_gpu_transformations()`` the map is the grid, one thread block per block of ``C``, and the kernel reads:

.. code-block:: c++

    __shared__ double acc[256];
    fill_acc(...);                          // each thread two of the 256 elements
    for (k = 0; k < K; k += 8) {
        __syncthreads();                    // the last mma still read the tiles this iteration overwrites
        copy_A_tile(...);                   // 128 threads, one element each
        copy_B_tile(...);
        __syncthreads();                    // the mma reads rows and columns other threads loaded
        mma(...);                           // each thread the dot products of its two elements of acc
    }
    copy_C(...);                            // the same two elements of acc per thread: no barrier

The same program runs on both; the block decides nothing about threads.
