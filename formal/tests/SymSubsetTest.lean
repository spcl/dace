-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Checks for the subset procedures (`Core/SymSubset`): signs, disjointness,
inclusion and disjointness of distinct map iterations.  Run with
`lake env lean tests/SymSubsetTest.lean`; every `#guard` must pass. -/

open Dace

def E (s : String) : Expr := (Expr.parse? s).getD (.sym "<parse error>")

/-- `lo:hi:step` with inclusive `hi`. -/
def D (lo hi : String) (step : String := "1") : Dim := { lo := E lo, hi := E hi, step := E step }

/-- The index `[e]`. -/
def Ix (e : String) : Dim := Dim.index (E e)

def R (ds : List Dim) : Subset := .range ⟨ds⟩

/-- Facts: integer symbols `N`, `M`, `T`, `K`. -/
def C0 : Constraints := { intSyms := ["N", "M", "T", "K"] }

/-- With `T > 0` and `N ≥ 1`. -/
def C1 : Constraints := { C0 with sign := [("T", .pos), ("N", .pos)] }

/-! ## Signs -/

#guard (E "N*T").pos? C1
-- incomplete: squares are not searched for
#guard !(E "N*N").nonneg? C0
#guard !(E "N").nonneg? C0
#guard (E "N").nonneg? C1
#guard (E "N + T - 1").pos? C1
#guard !(E "N - T").pos? C1
#guard (E "Min(N, T)").pos? C1
#guard (E "N // T").nonneg? C1
-- symbols that are not integer facts symbols are rejected
#guard !(E "x*x").nonneg? C0

/-! ## Disjointness and inclusion -/

#guard (R [D "0" "K - 1"]).disjoint? C0 (R [D "K" "N - 1"])
#guard !(R [D "0" "K"]).disjoint? C0 (R [D "K" "N - 1"])
#guard (R [Ix "0", D "0" "M - 1"]).disjoint? C0 (R [Ix "1", D "0" "M - 1"])
-- tiles `[t*T : t*T + T - 1]` and `[(t + 1)*T : …]`
#guard (R [D "K*T" "K*T + T - 1"]).disjoint? C0 (R [D "(K + 1)*T" "(K + 1)*T + T - 1"])
#guard (R [D "1" "N - 2"]).subsetOf? C0 (R [D "0" "N - 1"])
#guard !(R [D "0" "N"]).subsetOf? C0 (R [D "0" "N - 1"])
#guard (R [Ix "K", D "0" "M - 1"]).subsetOf? C0 (R [D "0" "N - 1", D "0" "M - 1"]) == false
#guard (R [Ix "K", D "0" "M - 1"]).subsetOf? { C0 with rels := [⟨.lt, E "K", E "N"⟩, ⟨.le, E "0", E "K"⟩] }
  (R [D "0" "N - 1", D "0" "M - 1"])
#guard (R [D "0" "Min(N, T) - 1"]).subsetOf? C0 (R [D "0" "N - 1"])
-- a union is included range by range
#guard (Subset.union [⟨[D "0" "K - 1"]⟩, ⟨[D "K" "N - 1"]⟩]).subsetOf?
  { C0 with rels := [⟨.le, E "K", E "N"⟩, ⟨.le, E "0", E "K"⟩] }
  (R [D "0" "N - 1"])
#guard !(Subset.union [⟨[D "0" "K - 1"]⟩, ⟨[D "K" "N - 1"]⟩]).subsetOf? C0 (R [D "0" "N - 1"])
-- a strided superset is not supported (needs divisibility)
#guard !(R [D "0" "N - 1" "2"]).subsetOf? C0 (R [D "0" "N - 1" "2"])

/-! ## Distinct map iterations (`iterDisjoint?`)

Corpus-like accesses; `A` in one iteration against `B` in another.  Accesses
to different containers (axpy `x[i]` vs `y[i]`) never conflict and are not
asked: the caller compares container names (`raceFree?`). -/

def vec (n : String) : DataDesc := DataDesc.array (.float 64) [E n]

def mat (n m : String) : DataDesc := DataDesc.array (.float 64) [E n, E m]

def it1 (A B : Subset) (C : Constraints := C0) (rng : List Dim := [D "0" "N - 1"]) (d : DataDesc := vec "N") : Bool :=
  iterDisjoint? C ["i"] ⟨rng⟩ d A d B

-- axpy-like `y[i] = a*x[i] + y[i]`: the write `y[i]` against itself
#guard it1 (R [Ix "i"]) (R [Ix "i"])
-- `A[0] = i`: a race
#guard !it1 (R [Ix "0"]) (R [Ix "0"])
-- `C[i // 2]`: iterations `2k` and `2k + 1` share an element
#guard !it1 (R [Ix "i // 2"]) (R [Ix "i // 2"])
-- strided `A[2*i]` (and against a read `A[2*i + 1]`)
#guard it1 (R [Ix "2*i"]) (R [Ix "2*i"]) (d := vec "2*N")
#guard it1 (R [Ix "2*i"]) (R [Ix "2*i + 1"]) (d := vec "2*N")
-- a stencil `A[i] = A[i + 1]` races
#guard !it1 (R [Ix "i"]) (R [Ix "i + 1"])
-- strided map `i ∈ 0:N-1:2` writing `A[i]` and `A[i + 1]`
#guard it1 (R [Ix "i", ]) (R [Ix "i + 1"]) (rng := [D "0" "N - 1" "2"])
-- row `A[i, 0:M-1]` of an `N×M` matrix
#guard it1 (R [Ix "i", D "0" "M - 1"]) (R [Ix "i", D "0" "M - 1"]) (d := mat "N" "M")
-- column `A[0:N-1, i]` (needs the stride `M` and the bound `i < M`)
#guard iterDisjoint? C0 ["i"] ⟨[D "0" "M - 1"]⟩ (mat "N" "M") (R [D "0" "N - 1", Ix "i"]) (mat "N" "M")
  (R [D "0" "N - 1", Ix "i"])
-- `A[i, 0]` against `A[0, i]`: only iteration `0` touches `A[0, 0]` (index separation)
#guard iterDisjoint? C0 ["i"] ⟨[D "0" "N - 1"]⟩ (mat "N" "N") (R [Ix "i", Ix "0"]) (mat "N" "N") (R [Ix "0", Ix "i"])
-- `A[i, 0]` against `A[1, i]`: iterations `1` and `0` both touch `A[1, 0]`
#guard !iterDisjoint? C0 ["i"] ⟨[D "0" "N - 1"]⟩ (mat "N" "N") (R [Ix "i", Ix "0"]) (mat "N" "N") (R [Ix "1", Ix "i"])
-- out-of-bounds accesses lose index separation: `A[i, j]` with a `j` range wider than the row
#guard !iterDisjoint? C0 ["i"] ⟨[D "0" "N - 1"]⟩ (mat "N" "M") (R [Ix "i", D "0" "M"]) (mat "N" "M")
  (R [Ix "i", D "0" "M"])

-- 2D map `A[i, j]` over `i ∈ 0:N-1, j ∈ 0:M-1` with symbolic row-major strides
def it2 (A B : Subset) : Bool :=
  iterDisjoint? C0 ["i", "j"] ⟨[D "0" "N - 1", D "0" "M - 1"]⟩ (mat "N" "M") A (mat "N" "M") B

#guard it2 (R [Ix "i", Ix "j"]) (R [Ix "i", Ix "j"])
#guard !it2 (R [Ix "i", Ix "0"]) (R [Ix "i", Ix "0"])
#guard !it2 (R [Ix "j", Ix "i"]) (R [Ix "j", Ix "i"])

-- tiled map as produced by MapTiling: `ti ∈ 0:N-1:T`, footprint `A[ti : Min(N - 1, ti + T - 1)]`
#guard iterDisjoint? C0 ["ti"] ⟨[D "0" "N - 1" "T"]⟩ (vec "N") (R [D "ti" "Min(N - 1, ti + T - 1)"]) (vec "N")
  (R [D "ti" "Min(N - 1, ti + T - 1)"])
-- tile index form: `ti ∈ 0:int_ceil(N, T) - 1`, footprint `A[ti*T : Min(ti*T + T, N) - 1]`
#guard iterDisjoint? C0 ["ti"] ⟨[D "0" "int_ceil(N, T) - 1"]⟩ (vec "N") (R [D "ti*T" "Min(ti*T + T, N) - 1"])
  (vec "N") (R [D "ti*T" "Min(ti*T + T, N) - 1"])
-- 2D tiles `A[ti : ti + T - 1, 0:M-1]`, `ti ∈ 0:N-1:T` (rows of tiles)
#guard iterDisjoint? C0 ["ti"] ⟨[D "0" "N - 1" "T"]⟩ (mat "N" "M") (R [D "ti" "Min(N - 1, ti + T - 1)", D "0" "M - 1"])
  (mat "N" "M") (R [D "ti" "Min(N - 1, ti + T - 1)", D "0" "M - 1"])
-- a parameter that shadows a facts symbol is renamed consistently; unsupported shapes are rejected
#guard !iterDisjoint? C0 ["i"] ⟨[D "0" "N - 1"]⟩ (vec "N") (R [Ix "x"]) (vec "N") (R [Ix "x"])
#guard !iterDisjoint? C0 ["i"] ⟨[D "0" "i"]⟩ (vec "N") (R [Ix "i"]) (vec "N") (R [Ix "i"])

/-! ## Assumed-queries mode (kernel fallback) -/

-- decided cases produce no obligations
#guard (iterObligations? C0 ["i"] ⟨[D "0" "N - 1"]⟩ (vec "N") (R [Ix "i"]) (vec "N") (R [Ix "i"])).map
  List.length == some 0
-- `C[i // 2]` leaves one obligation per undecided case (they are invalid here)
#guard ((iterObligations? C0 ["i"] ⟨[D "0" "N - 1"]⟩ (vec "N") (R [Ix "i // 2"]) (vec "N")
  (R [Ix "i // 2"])).map List.length).getD 0 > 0
-- outside the fragment: no queries at all
#guard (iterObligations? C0 ["i"] ⟨[D "0" "N - 1"]⟩ (vec "N") (R [Ix "x"]) (vec "N") (R [Ix "x"])).isNone

#eval IO.println (Query.toLeanFile (((iterObligations? C0 ["i"] ⟨[D "0" "N - 1"]⟩ (vec "N") (R [Ix "i // 2"])
  (vec "N") (R [Ix "i // 2"])).getD []).zipIdx.map fun (q, k) => (s!"half_index_{k}", q)))
