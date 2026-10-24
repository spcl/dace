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
import Dace.Core.Facts
import Dace.Core.SymArith
import Dace.Core.SymCert
import Dace.Core.SymSearch
import Dace.Core.SymSubset
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
import Dace.Equiv.Observable
import Dace.Equiv.Equiv
import Dace.Equiv.Frag
import Dace.SDFG.LogShift
import Dace.STree.LogShift
import Dace.SDFG.Footprint
import Dace.Equiv.Footprint
import Dace.Equiv.Congr
import Dace.SDFG.Footprinted
import Dace.SDFG.Frame
import Dace.Equiv.Order
import Dace.SDFG.Clean
import Dace.SDFG.Names
import Dace.SDFG.Sim
import Dace.Equiv.RaceFree
import Dace.Equiv.ScopeBody
import Dace.Equiv.Det
import Dace.Equiv.DetProg
import Dace.Equiv.CorrLeaf
import Dace.Equiv.CorrMatch
import Dace.Equiv.EvalCongr
import Dace.Equiv.Corr
import Dace.Equiv.CorrCtl
import Dace.Equiv.CorrCheck

/-!
# DaCe formal semantics

Root module of the `Dace` library: a Lean 4 formalization of the SDFG and
schedule-tree intermediate representations of DaCe, their semantics, and a
verified equivalence checker.  See `README.md` and `Dace/Decisions.md`.
-/
