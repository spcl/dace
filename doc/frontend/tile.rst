.. _tile-calls:

Tile Calls
==========

The tile library nodes of :mod:`dace.libraries.tileops` can be called from a ``@dace.program`` as ``dace.tile``
functions, without the vectorizer. A *window* is an array or a slice of one, such as ``A[i:i + 8]``. Its extents other
than 1 are the lanes of the tile it makes, of which there are one to three dims of a constant extent. A *tile* is a
register array, ``dace.define_local([8], dace.float64, storage=dace.StorageType.Register)``.

``dace.tile.add(a, b)``
    The sum of two windows or tiles of the same lanes and type, as a new tile.

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

The calls expand to a loop over the lanes. The vectorizer lowers its tiles to the instructions of a target ISA instead.
