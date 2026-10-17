-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.Multiset.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finsupp.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Bitwise
import Mathlib.Data.Rat.Floor
import Mathlib.Logic.Function.Basic
import Mathlib.Order.Basic
import Mathlib.Tactic

/-!
# `Dace.Prelude`

The single place where Mathlib is imported.  Every other module of the `Dace`
library imports `Dace.Prelude` (directly or transitively) and **must not**
import `Mathlib.*` itself; CI enforces this with a grep.  Keeping the Mathlib
surface in one file makes it possible to swap in a Batteries-only prelude later.
-/

namespace Dace

/-- Names of data containers (arrays, scalars, streams, views, ...). -/
abbrev Name := String

/-- Symbol names (integer/boolean/float symbols of an SDFG, map parameters, ...). -/
abbrev Sym := String

/-- Connector names on nodes. -/
abbrev Conn := String

end Dace
