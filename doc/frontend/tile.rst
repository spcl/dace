.. _tile-calls:

Tile Calls
==========

The tile library nodes of :mod:`dace.libraries.tileops` can be called from a ``@dace.program`` as ``dace.tile``
functions, without the vectorizer. A *window* is an array or a slice of one, such as ``A[i:i + 8]``. Its extents other
than 1 are the lanes of the tile it makes, of which there are one to three dims of a constant extent. A *tile* is a
register array, ``dace.define_local([8], dace.float64, storage=dace.StorageType.Register)``. A call takes tiles or
windows, copying a window into a new tile first, and returns a new tile.

Who runs a tile
---------------

Every call makes a node of the ``BLOCK`` group (:class:`~dace.libraries.tileops.dispatch.TileGroup`). The program
says what a whole block computes, and the compiler decides which thread computes which element:

* On a CPU one core runs the tile, as a loop over its lanes that the compiler vectorizes. A ``BLOCK`` node lowers
  exactly as a ``THREAD`` node does.
* In a GPU kernel the threads of the block share the tile. Each node becomes a thread-block map whose thread ``t``
  computes every ``threads``-th element from ``t``, so consecutive threads touch consecutive elements. The tiles move
  to shared memory, and the code generator puts a ``__syncthreads()`` wherever a tile written by some threads is read
  by others. A node whose lanes depend on each other (``sum``) runs on one thread of the block.

The block has ``32 * num_warps`` threads, four warps unless a node's ``num_warps`` says otherwise. The vectorizer emits
``THREAD`` nodes, one thread's register tile, which lower to the instructions of a target ISA.

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
    for (k = 0; k < K; k += 8) {
        __syncthreads();                    // the last mma read the tiles this iteration overwrites
        __shared__ double A_tile[128];
        __shared__ double B_tile[128];
        copy_A_tile(...);                   // 128 threads, one element each
        copy_B_tile(...);
        mma(...);                           // __syncthreads(), then each thread two of the 256 dot products
    }
    __syncthreads();
    copy_C(...);

The same program runs on both; the block decides nothing about threads.
