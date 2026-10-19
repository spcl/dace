-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Prelude
import Dace.Core.Expr
import Dace.Core.Subset
import Dace.Core.Parse
import Dace.Core.Data
import Dace.Core.Memlet
import Dace.Core.Memory
import Dace.Core.Wcr
import Dace.Core.Merge
import Dace.Core.MergeLemmas
import Dace.Core.Tasklet
import Dace.Core.Interp
import Dace.Core.PyTasklet
import Dace.SDFG.Syntax
import Dace.SDFG.Scope
import Dace.SDFG.ScopeSpec
import Dace.SDFG.WF
import Dace.SDFG.Semantics
import Dace.SDFG.Interp
import Dace.SDFG.Soundness
import Dace.SDFG.Json
import Dace.STree.Syntax
import Dace.STree.Json
import Dace.STree.WF
import Dace.STree.Semantics.Basic
import Dace.STree.Semantics.Leaf
import Dace.STree.Semantics.Exec
import Dace.STree.Semantics
import Dace.STree.Interp
import Dace.STree.Soundness

/-!
# DaCe formal semantics

Root module of the `Dace` library: a Lean 4 formalization of the SDFG and
schedule-tree intermediate representations of DaCe, their semantics, and a
verified equivalence checker.  See `README.md` and `Dace/Decisions.md`.
-/
