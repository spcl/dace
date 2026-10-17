-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-!
Axiom audit: every trusted theorem of the development is listed here so that
`scripts/check_axioms.sh` can verify none depends on `sorryAx`.
-/

-- Theorems are appended here as they are proven, e.g.
-- #print axioms Dace.Equiv.corr_sound

-- M3: soundness of the SDFG interpreter with respect to the relational semantics.
#print axioms Dace.sound
#print axioms Dace.execSDFG_sound
#print axioms Dace.SDFG.run_sound
#print axioms Dace.SDFG.run_ok_sound
