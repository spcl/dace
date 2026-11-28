# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""How a tile op reads one operand.

``Tile`` reads a tile-shaped array through the operand's connector, one element per lane. ``Scalar`` broadcasts the
single element a connector carries to every lane (a tile-shaped array an upstream tile op widened is still read per
lane). ``Symbol`` embeds an expression of the symbols in scope inline and has no connector.
"""

TILE = "Tile"
SYMBOL = "Symbol"
SCALAR = "Scalar"
VALID_KINDS = (TILE, SYMBOL, SCALAR)
