-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Prelude
-- Symbolic expressions and verified integer arithmetic
import Dace.Sym.Expr
import Dace.Sym.Facts
import Dace.Sym.Parse
import Dace.Sym.Poly
import Dace.Sym.Cert
import Dace.Sym.Search
-- What a program manipulates
import Dace.Model.Subset
import Dace.Model.Data
import Dace.Model.Memlet
import Dace.Model.Memory
import Dace.Model.Tasklet
import Dace.Model.SubsetCheck
import Dace.Model.Wcr
import Dace.Model.Interp
import Dace.Model.Log
import Dace.Model.LogAlgebra
import Dace.Model.PyTasklet
-- SDFGs
import Dace.SDFG.Syntax
import Dace.SDFG.BEq
import Dace.SDFG.Scope
import Dace.SDFG.ScopeSpec
import Dace.SDFG.WF
import Dace.SDFG.Semantics.Basic
import Dace.SDFG.Json
import Dace.SDFG.Semantics.Leaf
import Dace.SDFG.Semantics.Exec
import Dace.SDFG.Semantics
import Dace.SDFG.Interp
import Dace.SDFG.Names
import Dace.SDFG.Soundness
-- Schedule trees
import Dace.STree.Syntax
import Dace.STree.WF
import Dace.STree.Json
import Dace.STree.Semantics.Basic
import Dace.STree.Semantics.Leaf
import Dace.STree.Semantics.Exec
import Dace.STree.Semantics
import Dace.STree.Interp
import Dace.STree.Soundness
-- Refinement and equivalence
import Dace.Equiv.Observable
import Dace.Equiv.Refines
import Dace.Equiv.Matched
-- Runs: memory and logs
import Dace.Runs.Log.Clean
import Dace.Runs.Log.WriteOnly
import Dace.Runs.Log.Footprinted
import Dace.Runs.Log.WriteOnlyTree
import Dace.Runs.Log.Frame
import Dace.Runs.Log.Coherence
-- Runs: static footprints
import Dace.Runs.Footprint.Static
import Dace.Runs.Footprint.Iter
-- Runs: order independence and determinism
import Dace.Runs.Order.Order
import Dace.Runs.Order.Det
import Dace.Runs.Order.DetProg
-- Runs: renaming, relabeling, simulation
import Dace.Runs.Invariance.Retarget
import Dace.Runs.Invariance.EvalCongr
import Dace.Runs.Invariance.Sim
import Dace.Runs.Invariance.Relabel
import Dace.Runs.Invariance.TreeRename
import Dace.Runs.Invariance.Rename
-- Rewrite rules
import Dace.Rules.Block
import Dace.Rules.RelabelNodes
import Dace.Rules.ScopeBody
import Dace.Rules.Region
import Dace.Rules.StateLocal
import Dace.Rules.RenameContainers
-- Verified checkers
import Dace.Check.TreePair.Names
import Dace.Check.RaceFree
import Dace.Check.TreePair.Leaf
import Dace.Check.TreePair.Segment
import Dace.Check.TreePair.State
import Dace.Check.TreePair.Control
import Dace.Check.TreePair.Checker
import Dace.Check.SdfgPair

/-!
# DaCe formal semantics

Root module of the `Dace` library: a Lean 4 formalization of the SDFG and
schedule-tree intermediate representations of DaCe, their semantics, and a
verified equivalence checker.  See `README.md` and `Dace/Decisions.md`.
-/
