# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering mechanisms: the *how* of turning computations into schedule tree
nodes, organized by mechanism (elementwise maps, compile-time value
materialization, callbacks) — never by source library. Syntax rules and the
dispatch seam decide *which* mechanism applies based on operand types.
"""
