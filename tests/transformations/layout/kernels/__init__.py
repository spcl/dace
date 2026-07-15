# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe ports of the SC26 layout reference kernels (numpy specs in ~/Downloads/layout/k01-k15).

Each module defines a DaCe program, a numpy oracle, and the global layout candidates it exercises,
so the brute-force sweep (dace.transformation.layout.brute_force) can enumerate layouts, verify each
against the oracle, and rank them. These are correctness ports (the invariant is that every
transparent candidate reproduces the oracle); authoritative CPU/GPU timing is a separate measurement.
"""
