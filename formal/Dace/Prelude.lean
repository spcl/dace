-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
module

public import Mathlib.Data.List.Basic
public import Mathlib.Data.List.Perm.Basic
public import Mathlib.Data.Multiset.Basic
public import Mathlib.Data.Finset.Basic
public import Mathlib.Data.Finsupp.Basic
public import Mathlib.Data.Int.Basic
public import Mathlib.Data.Int.Bitwise
public import Mathlib.Data.Rat.Floor
public import Mathlib.Logic.Function.Basic
public import Mathlib.Order.Basic
public import Mathlib.Tactic

/-!
# `Dace.Prelude`

The single place where Mathlib is imported.  Every other module of the `Dace`
library imports `Dace.Prelude` (directly or transitively) and **must not**
import `Mathlib.*` itself; CI enforces this with a grep.  Keeping the Mathlib
surface in one file makes it possible to swap in a Batteries-only prelude later.

Mathlib is built for the module system, so this file is a `module` that
re-exports its Mathlib imports (`public import`) and exposes its definitions;
the rest of the library stays outside the module system.
-/

@[expose] public section

namespace Dace

/-- Names of data containers (arrays, scalars, streams, views, ...). -/
abbrev Name := String

/-- Symbol names (integer/boolean/float symbols of an SDFG, map parameters, ...). -/
abbrev Sym := String

/-- Connector names on nodes. -/
abbrev Conn := String

end Dace
