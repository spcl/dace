# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The tile library nodes."""

from dace.libraries.tileops.nodes.tile_mask_gen import TileMaskGen
from dace.libraries.tileops.nodes.tile_gather import TileGather
from dace.libraries.tileops.nodes.tile_scatter import TileScatter
from dace.libraries.tileops.nodes.tile_binop import TileBinop
from dace.libraries.tileops.nodes.tile_fma import TileFMA
from dace.libraries.tileops.nodes.tile_unop import TileUnop
from dace.libraries.tileops.nodes.tile_ite import TileITE
from dace.libraries.tileops.nodes.tile_reduce import TileReduce
from dace.libraries.tileops.nodes.tile_iota import TileIota
from dace.libraries.tileops.nodes.tile_mma import TileMMA
from dace.libraries.tileops.nodes.masked_copy import MaskedCopyLibraryNode

#: The nodes that move a tile between an array and registers.
TILE_TRANSFER_NODES = (MaskedCopyLibraryNode, TileGather, TileScatter)

#: Every tile library node.
TILE_NODES = (*TILE_TRANSFER_NODES, TileBinop, TileFMA, TileIota, TileITE, TileMaskGen, TileMMA, TileReduce, TileUnop)
