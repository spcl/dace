-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Import of the symbol repository (`attributes.symbol_repo`, `Core/Facts`)
and use of the facts as query hypotheses.  Run with
`lake env lean tests/FactsTest.lean`; every `#guard` must pass. -/

open Dace

/-- An empty SDFG in the format of DaCe's `symbol-registry` branch: no
`symbols` attribute, the symbol types live in `symbol_repo.params.types`. -/
def withRepo : String := "
{\"type\": \"SDFG\", \"attributes\": {\"name\": \"f\", \"_arrays\": {},
  \"symbol_repo\": {
    \"params\": {
      \"types\": {\"K\": \"int64\", \"N\": \"int64\", \"U\": \"uint32\", \"x\": \"float64\"},
      \"predicates\": {\"N\": [\"POSITIVE\"], \"K\": [\"NONNEGATIVE\", \"NONZERO\"]},
      \"relations\": [{\"kind\": \"LT\", \"lhs\": \"K\", \"rhs\": \"N\"},
                      {\"kind\": \"DIVIDES\", \"lhs\": \"4\", \"rhs\": \"N\"},
                      {\"kind\": \"MUCH_LT\", \"lhs\": \"K\", \"rhs\": \"N\"},
                      {\"kind\": \"LE\", \"lhs\": \"2*K\", \"rhs\": \"N - 1\"}]},
    \"small\": [\"K\"], \"runtime_assumptions\": [], \"shape_declared\": [],
    \"scopes\": [{\"owner\": {\"kind\": \"LOOP\", \"graph\": [0], \"position\": 1},
                  \"types\": {\"i\": \"int32\"},
                  \"relations\": [{\"kind\": \"LT\", \"lhs\": \"i\", \"rhs\": \"N\"}]}]}},
 \"nodes\": [], \"edges\": [], \"start_block\": 0}"

def withoutRepo : String :=
  "{\"type\": \"SDFG\", \"attributes\": {\"name\": \"g\", \"symbols\": {\"N\": \"int32\"}}, \"nodes\": [], \"edges\": []}"

def facts (s : String) : Option Constraints := (SDFG.Json.parseSDFG s).toOption.map (·.facts)
def syms (s : String) : Option (List (Sym × Dtype)) := (SDFG.Json.parseSDFG s).toOption.map (·.symbols)

#eval facts withRepo

def expected : Constraints where
  intSyms := ["K", "N", "U"]
  sign := [("K", .nonneg), ("K", .nonzero), ("N", .pos), ("U", .nonneg)]
  rels := [{ kind := .lt, lhs := .sym "K", rhs := .sym "N" },
           { kind := .dvd, lhs := 4, rhs := .sym "N" },
           { kind := .le, lhs := .bin .mul 2 (.sym "K"), rhs := .bin .sub (.sym "N") 1 }]

#guard facts withRepo == some expected
-- `SDFG.symbols` is the view of `params.types`
#guard (syms withRepo).map (·.map (·.1)) == some ["K", "N", "U", "x"]
-- no repository: the declared types are the facts (`int32` ⇒ integer), symbols unchanged
#guard facts withoutRepo == some { intSyms := ["N"] }
#guard (syms withoutRepo).map (·.map (·.1)) == some ["N"]

/-! SymPy assumptions of typed symbols `symbol($N, k=True, …)` in expressions
(`SDFG.Json.inlineFacts`). -/

#guard Parse.symbolAssumptions "0:symbol($N, dtype=dace.uint32, positive=True) - 1:1" ==
  [("N", [("dtype", .attr (.sym "dace") "uint32"), ("positive", .lit (.bool true))])]
#guard Parse.symbolAssumptions "symbol($i, even=True)*symbol($j, nonzero=True)" ==
  [("i", [("even", .lit (.bool true))]), ("j", [("nonzero", .lit (.bool true))])]
#guard Parse.symbolAssumptions "N + 1" == []

/-- An array whose shape uses typed symbols with assumptions; `x` is a float,
so its assumption is no fact, and the nested SDFG's `Q` is its own symbol. -/
def withAssumptions : String := "
{\"type\": \"SDFG\", \"attributes\": {\"name\": \"h\",
  \"symbols\": {\"N\": \"int64\", \"E\": \"int32\", \"O\": \"int32\", \"P\": \"int32\",
                \"U\": \"uint32\", \"Z\": \"int32\", \"Q\": \"int32\", \"x\": \"float64\"},
  \"_arrays\": {\"A\": {\"type\": \"Array\", \"attributes\": {\"dtype\": \"float64\",
    \"shape\": [\"symbol($N, dtype=dace.int64, nonnegative=True, nonzero=True, positive=True)\",
               \"symbol($E, even=True) + symbol($O, odd=True)\", \"symbol($P, prime=True)\",
               \"symbol($U, dtype=dace.uint32)\", \"symbol($Z, zero=True) + 1\",
               \"symbol($x, dtype=dace.float64, positive=True)\"],
    \"strides\": [\"1\", \"1\", \"1\", \"1\", \"1\", \"1\"], \"total_size\": \"1\", \"offset\": [\"0\", \"0\", \"0\", \"0\", \"0\", \"0\"]}}},
  \"unused\": {\"type\": \"SDFG\", \"attributes\": {\"note\": \"symbol($Q, positive=True)\"}}},
 \"nodes\": [], \"edges\": []}"

#eval facts withAssumptions

#guard (facts withAssumptions).map (·.sign) ==
  some [("U", .nonneg), ("N", .nonneg), ("N", .nonzero), ("N", .pos)]
#guard (facts withAssumptions).map (·.rels) == some
  [{ kind := .dvd, lhs := 2, rhs := .sym "E" },
   { kind := .dvd, lhs := 2, rhs := .bin .sub (.sym "O") 1 },
   { kind := .le, lhs := 2, rhs := .sym "P" },
   { kind := .eq, lhs := .sym "Z", rhs := 0 }]

/-! Facts as hypotheses (`Constraints.toHyps`, sound by `Constraints.toHyps_sound`). -/

def factQuery (goal : Poly) : Query :=
  { hyps := expected.toHyps.1, disj := expected.toHyps.2, goal }

def N : Poly := Poly.var (.sym "N")
def K : Poly := Poly.var (.sym "K")

-- `K < N` and `2K ≤ N − 1` give `K ≤ N − 1` and `N ≥ 1`
#guard (factQuery (Poly.sub (Poly.sub N K) (Poly.const 1))).decide
#guard (factQuery (Poly.sub N (Poly.const 1))).decide
-- `K ≠ 0 ∧ K ≥ 0` gives `K ≥ 1` (case split on the `NONZERO` disjunction)
#guard (factQuery (Poly.sub K (Poly.const 1))).decide
-- `2K ≤ N` follows from `2K ≤ N − 1`
#guard (factQuery (Poly.sub (Poly.sub N K) K)).decide
-- not entailed
#guard !(factQuery (Poly.sub K (Poly.const 2))).decide
