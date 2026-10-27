-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Syntax

/-!
# Boolean equality of SDFGs

`deriving DecidableEq` does not apply to the mutual family of SDFG syntax
(nested lists of regions and branches), and values (`Val`) contain floats,
which have no decidable equality.  `SDFG.beq` is a structural equality test
that is **sound** (`SDFG.beq_sound`: `beq a b = true → a = b`); it answers
`false` on equal constants holding floats, so it is not complete.  The
SDFG–SDFG checker (`Dace.Equiv.SdfgCorr`) uses it for the final comparison.
-/

set_option autoImplicit false

namespace Dace

/-- Sound equality of values: floats are never equal. -/
def Val.beqSound : Val → Val → Bool
  | .int a, .int b => a == b
  | .rat a, .rat b => a == b
  | .bool a, .bool b => a == b
  | .undef, .undef => true
  | _, _ => false

theorem Val.beqSound_sound : ∀ {a b : Val}, Val.beqSound a b = true → a = b
  | .int a, .int b, h => by simp only [Val.beqSound, beq_iff_eq] at h; rw [h]
  | .rat a, .rat b, h => by simp only [Val.beqSound, beq_iff_eq] at h; rw [h]
  | .bool a, .bool b, h => by simp only [Val.beqSound, beq_iff_eq] at h; rw [h]
  | .undef, .undef, _ => rfl
  | .int _, .rat _, h | .int _, .float _, h | .int _, .bool _, h | .int _, .undef, h => by cases h
  | .rat _, .int _, h | .rat _, .float _, h | .rat _, .bool _, h | .rat _, .undef, h => by cases h
  | .float _, _, h => by cases h
  | .bool _, .int _, h | .bool _, .rat _, h | .bool _, .float _, h | .bool _, .undef, h => by cases h
  | .undef, .int _, h | .undef, .rat _, h | .undef, .float _, h | .undef, .bool _, h => by cases h

/-- Sound equality of the constant table. -/
def constsBEq : List (Name × DataDesc × Option Val) → List (Name × DataDesc × Option Val) → Bool
  | [], [] => true
  | (n, d, v) :: xs, (n', d', v') :: ys =>
    n == n' && d == d' &&
      (match v, v' with
       | none, none => true
       | some a, some b => Val.beqSound a b
       | _, _ => false) && constsBEq xs ys
  | _, _ => false

theorem constsBEq_sound : ∀ {xs ys : List (Name × DataDesc × Option Val)}, constsBEq xs ys = true → xs = ys
  | [], [], _ => rfl
  | (n, d, v) :: xs, (n', d', v') :: ys, h => by
    simp only [constsBEq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨rfl, rfl⟩, hv⟩, hr⟩ := h
    rw [constsBEq_sound hr]
    congr
    cases v <;> cases v' <;> simp_all
    exact Val.beqSound_sound hv
  | [], _ :: _, h | _ :: _, [], h => by cases h

mutual

def NodeKind.beq : NodeKind → NodeKind → Bool
  | .access a, .access b => a == b
  | .tasklet a, .tasklet b => a == b
  | .mapEntry a, .mapEntry b => a == b
  | .mapExit, .mapExit => true
  | .consumeEntry a, .consumeEntry b => a == b
  | .consumeExit, .consumeExit => true
  | .nested g m ni, .nested g' m' ni' => SDFG.beq g g' && m == m' && ni == ni'
  | .library a, .library b => a == b
  | _, _ => false

def DNode.beq : DNode → DNode → Bool
  | ⟨i, l, ic, oc, se, k⟩, ⟨i', l', ic', oc', se', k'⟩ =>
    i == i' && l == l' && ic == ic' && oc == oc' && se == se' && NodeKind.beq k k'

def DNodes.beq : List DNode → List DNode → Bool
  | [], [] => true
  | a :: as, b :: bs => DNode.beq a b && DNodes.beq as bs
  | _, _ => false

def State.beq : State → State → Bool
  | ⟨l, ns, es, ny⟩, ⟨l', ns', es', ny'⟩ => l == l' && DNodes.beq ns ns' && es == es' && ny == ny'

def LoopRegion.beq : LoopRegion → LoopRegion → Bool
  | ⟨l, v, i, c, u, inv, ubc, body⟩, ⟨l', v', i', c', u', inv', ubc', body'⟩ =>
    l == l' && v == v' && i == i' && c == c' && u == u' && inv == inv' && ubc == ubc' && Region.beq body body'

def Block.beq : Block → Block → Bool
  | .state s, .state s' => State.beq s s'
  | .region r, .region r' => Region.beq r r'
  | .loop l, .loop l' => LoopRegion.beq l l'
  | .cond lb brs, .cond lb' brs' => lb == lb' && Branches.beq brs brs'
  | .breakBlock, .breakBlock => true
  | .continueBlock, .continueBlock => true
  | .returnBlock, .returnBlock => true
  | _, _ => false

def Blocks.beq : List Block → List Block → Bool
  | [], [] => true
  | a :: as, b :: bs => Block.beq a b && Blocks.beq as bs
  | _, _ => false

def Branches.beq : List (Option Expr × Region) → List (Option Expr × Region) → Bool
  | [], [] => true
  | (c, r) :: as, (c', r') :: bs => c == c' && Region.beq r r' && Branches.beq as bs
  | _, _ => false

def Region.beq : Region → Region → Bool
  | ⟨l, k, bs, st, es⟩, ⟨l', k', bs', st', es'⟩ =>
    l == l' && k == k' && Blocks.beq bs bs' && st == st' && es == es'

def SDFG.beq : SDFG → SDFG → Bool
  | ⟨n, a, sy, c, an, top, f⟩, ⟨n', a', sy', c', an', top', f'⟩ =>
    n == n' && a == a' && sy == sy' && constsBEq c c' && an == an' && Region.beq top top' && f == f'

end

mutual

theorem NodeKind.beq_sound : ∀ {a b : NodeKind}, NodeKind.beq a b = true → a = b
  | .access _, .access _, h | .tasklet _, .tasklet _, h | .mapEntry _, .mapEntry _, h
  | .consumeEntry _, .consumeEntry _, h | .library _, .library _, h => by
    simp only [NodeKind.beq, beq_iff_eq] at h; rw [h]
  | .mapExit, .mapExit, _ | .consumeExit, .consumeExit, _ => rfl
  | .nested g m ni, .nested g' m' ni', h => by
    simp only [NodeKind.beq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨hg, rfl⟩, rfl⟩ := h
    rw [SDFG.beq_sound hg]
  | .access _, .tasklet _, h | .access _, .mapEntry _, h | .access _, .mapExit, h
  | .access _, .consumeEntry _, h | .access _, .consumeExit, h | .access _, .nested _ _ _, h
  | .access _, .library _, h => by simp [NodeKind.beq] at h
  | .tasklet _, .access _, h | .tasklet _, .mapEntry _, h | .tasklet _, .mapExit, h
  | .tasklet _, .consumeEntry _, h | .tasklet _, .consumeExit, h | .tasklet _, .nested _ _ _, h
  | .tasklet _, .library _, h => by simp [NodeKind.beq] at h
  | .mapEntry _, .access _, h | .mapEntry _, .tasklet _, h | .mapEntry _, .mapExit, h
  | .mapEntry _, .consumeEntry _, h | .mapEntry _, .consumeExit, h | .mapEntry _, .nested _ _ _, h
  | .mapEntry _, .library _, h => by simp [NodeKind.beq] at h
  | .mapExit, .access _, h | .mapExit, .tasklet _, h | .mapExit, .mapEntry _, h
  | .mapExit, .consumeEntry _, h | .mapExit, .consumeExit, h | .mapExit, .nested _ _ _, h
  | .mapExit, .library _, h => by simp [NodeKind.beq] at h
  | .consumeEntry _, .access _, h | .consumeEntry _, .tasklet _, h | .consumeEntry _, .mapEntry _, h
  | .consumeEntry _, .mapExit, h | .consumeEntry _, .consumeExit, h | .consumeEntry _, .nested _ _ _, h
  | .consumeEntry _, .library _, h => by simp [NodeKind.beq] at h
  | .consumeExit, .access _, h | .consumeExit, .tasklet _, h | .consumeExit, .mapEntry _, h
  | .consumeExit, .mapExit, h | .consumeExit, .consumeEntry _, h | .consumeExit, .nested _ _ _, h
  | .consumeExit, .library _, h => by simp [NodeKind.beq] at h
  | .nested _ _ _, .access _, h | .nested _ _ _, .tasklet _, h | .nested _ _ _, .mapEntry _, h
  | .nested _ _ _, .mapExit, h | .nested _ _ _, .consumeEntry _, h | .nested _ _ _, .consumeExit, h
  | .nested _ _ _, .library _, h => by simp [NodeKind.beq] at h
  | .library _, .access _, h | .library _, .tasklet _, h | .library _, .mapEntry _, h
  | .library _, .mapExit, h | .library _, .consumeEntry _, h | .library _, .consumeExit, h
  | .library _, .nested _ _ _, h => by simp [NodeKind.beq] at h

theorem DNode.beq_sound : ∀ {a b : DNode}, DNode.beq a b = true → a = b
  | ⟨i, l, ic, oc, se, k⟩, ⟨i', l', ic', oc', se', k'⟩, h => by
    simp only [DNode.beq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩, rfl⟩, hk⟩ := h
    rw [NodeKind.beq_sound hk]

theorem DNodes.beq_sound : ∀ {a b : List DNode}, DNodes.beq a b = true → a = b
  | [], [], _ => rfl
  | a :: as, b :: bs, h => by
    simp only [DNodes.beq, Bool.and_eq_true] at h
    rw [DNode.beq_sound h.1, DNodes.beq_sound h.2]
  | [], _ :: _, h | _ :: _, [], h => by simp [DNodes.beq] at h

theorem State.beq_sound : ∀ {a b : State}, State.beq a b = true → a = b
  | ⟨l, ns, es, ny⟩, ⟨l', ns', es', ny'⟩, h => by
    simp only [State.beq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨rfl, hn⟩, rfl⟩, rfl⟩ := h
    rw [DNodes.beq_sound hn]

theorem LoopRegion.beq_sound : ∀ {a b : LoopRegion}, LoopRegion.beq a b = true → a = b
  | ⟨l, v, i, c, u, inv, ubc, body⟩, ⟨l', v', i', c', u', inv', ubc', body'⟩, h => by
    simp only [LoopRegion.beq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, hb⟩ := h
    rw [Region.beq_sound hb]

theorem Block.beq_sound : ∀ {a b : Block}, Block.beq a b = true → a = b
  | .state _, .state _, h => by simp only [Block.beq] at h; rw [State.beq_sound h]
  | .region _, .region _, h => by simp only [Block.beq] at h; rw [Region.beq_sound h]
  | .loop _, .loop _, h => by simp only [Block.beq] at h; rw [LoopRegion.beq_sound h]
  | .cond _ _, .cond _ _, h => by
    simp only [Block.beq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, hb⟩ := h
    rw [Branches.beq_sound hb]
  | .breakBlock, .breakBlock, _ | .continueBlock, .continueBlock, _ | .returnBlock, .returnBlock, _ => rfl
  | .state _, .region _, h | .state _, .loop _, h | .state _, .cond _ _, h | .state _, .breakBlock, h
  | .state _, .continueBlock, h | .state _, .returnBlock, h => by simp [Block.beq] at h
  | .region _, .state _, h | .region _, .loop _, h | .region _, .cond _ _, h | .region _, .breakBlock, h
  | .region _, .continueBlock, h | .region _, .returnBlock, h => by simp [Block.beq] at h
  | .loop _, .state _, h | .loop _, .region _, h | .loop _, .cond _ _, h | .loop _, .breakBlock, h
  | .loop _, .continueBlock, h | .loop _, .returnBlock, h => by simp [Block.beq] at h
  | .cond _ _, .state _, h | .cond _ _, .region _, h | .cond _ _, .loop _, h | .cond _ _, .breakBlock, h
  | .cond _ _, .continueBlock, h | .cond _ _, .returnBlock, h => by simp [Block.beq] at h
  | .breakBlock, .state _, h | .breakBlock, .region _, h | .breakBlock, .loop _, h | .breakBlock, .cond _ _, h
  | .breakBlock, .continueBlock, h | .breakBlock, .returnBlock, h => by simp [Block.beq] at h
  | .continueBlock, .state _, h | .continueBlock, .region _, h | .continueBlock, .loop _, h
  | .continueBlock, .cond _ _, h | .continueBlock, .breakBlock, h | .continueBlock, .returnBlock, h => by simp [Block.beq] at h
  | .returnBlock, .state _, h | .returnBlock, .region _, h | .returnBlock, .loop _, h
  | .returnBlock, .cond _ _, h | .returnBlock, .breakBlock, h | .returnBlock, .continueBlock, h => by simp [Block.beq] at h

theorem Blocks.beq_sound : ∀ {a b : List Block}, Blocks.beq a b = true → a = b
  | [], [], _ => rfl
  | a :: as, b :: bs, h => by
    simp only [Blocks.beq, Bool.and_eq_true] at h
    rw [Block.beq_sound h.1, Blocks.beq_sound h.2]
  | [], _ :: _, h | _ :: _, [], h => by simp [Blocks.beq] at h

theorem Branches.beq_sound : ∀ {a b : List (Option Expr × Region)}, Branches.beq a b = true → a = b
  | [], [], _ => rfl
  | (c, r) :: as, (c', r') :: bs, h => by
    simp only [Branches.beq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨rfl, hr⟩, hs⟩ := h
    rw [Region.beq_sound hr, Branches.beq_sound hs]
  | [], _ :: _, h | _ :: _, [], h => by simp [Branches.beq] at h

theorem Region.beq_sound : ∀ {a b : Region}, Region.beq a b = true → a = b
  | ⟨l, k, bs, st, es⟩, ⟨l', k', bs', st', es'⟩, h => by
    simp only [Region.beq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨rfl, rfl⟩, hb⟩, rfl⟩, rfl⟩ := h
    rw [Blocks.beq_sound hb]

theorem SDFG.beq_sound : ∀ {a b : SDFG}, SDFG.beq a b = true → a = b
  | ⟨n, a, sy, c, an, top, f⟩, ⟨n', a', sy', c', an', top', f'⟩, h => by
    simp only [SDFG.beq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, hc⟩, rfl⟩, ht⟩, rfl⟩ := h
    rw [constsBEq_sound hc, Region.beq_sound ht]

end

end Dace
