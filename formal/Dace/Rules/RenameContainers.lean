-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Runs.Invariance.Rename
import Dace.Runs.Invariance.Sim

/-!
# Renaming containers preserves meaning

`SDFG.ren σ` renames containers by a permutation `σ` (`Runs/Invariance/Rename`).  The
renamed SDFG is equivalent to the original when `σ` moves only names that are
neither observed nor defined at the start and that no expression reads
(`SDFG.RenOk`).  Two steps relate a run of `g` to a run of `g.ren σ`:

1. **Memory names** (`SDFGExec.sim`, `Runs/Invariance/Sim`): running `g` with every
   container `n` stored under memory name `σ n` (`renCtx`) simulates running
   it with the identity names; the start memories agree because `σ` moves no
   name defined at the start.
2. **Container names** (`RenCtx.sdfgExec`, `Runs/Invariance/Rename`): running `g` with
   memory names `σ` is running `g.ren σ` with the identity names, up to the
   wording of error messages (`Res.er`; messages name containers).

The converse direction is the same argument for `σ⁻¹` on `g.ren σ`
(`SDFG.RenOk.symm`, `SDFG.ren_symm`).
-/

set_option autoImplicit false
set_option linter.unusedSectionVars false

namespace Dace

/-- Conditions under which renaming the containers of `g` by `σ` preserves
its meaning. -/
structure SDFG.RenOk (σ : _root_.Equiv.Perm Name) (g : SDFG) : Prop where
  /-- No expression reads a moved name, and no consumed stream is moved. -/
  top : Region.RenOk σ g.top
  /-- No shape reads a moved name. -/
  shapes : ∀ n d, g.array? n = some d → ∀ x ∈ d.fullRange.names, σ x = x
  /-- Memory names of nested calls (`Ctx.enter`) are fixed. -/
  slash : ∀ p x : Name, σ (p ++ "/" ++ x) = p ++ "/" ++ x
  /-- Moved names are uninitialized at the start. -/
  fresh : ∀ x, σ x ≠ x → g.keepAtStart x = false
  /-- Observable containers are fixed. -/
  iface : ∀ x ∈ g.interface, σ x = x

namespace SDFG.RenOk

variable {σ : _root_.Equiv.Perm Name} {g : SDFG}

theorem interface_ren (σ : _root_.Equiv.Perm Name) (g : SDFG) : (g.ren σ).interface = g.interface.map σ := by
  simp only [SDFG.interface, interfaceOf, SDFG.ren, List.filter_map, List.map_map]
  rfl

theorem interface (h : g.RenOk σ) : (g.ren σ).interface = g.interface := by
  rw [interface_ren]
  conv => rhs; rw [← List.map_id g.interface]
  exact List.map_congr_left fun x hx => h.iface x hx

theorem keepAtStart (h : g.RenOk σ) : (g.ren σ).keepAtStart = g.keepAtStart := by
  funext n
  rw [SDFG.keepAtStart_ren]
  by_cases hn : σ.symm n = n
  · rw [hn]
  · have h₁ : σ (σ.symm n) ≠ σ.symm n := by rw [_root_.Equiv.apply_symm_apply]; exact fun h => hn h.symm
    have h₂ : σ n ≠ n := fun h => hn (perm_symm_fix h)
    rw [h.fresh _ h₁, h.fresh _ h₂]

theorem startMem (h : g.RenOk σ) (m₀ : Mem) : (g.ren σ).startMem m₀ = g.startMem m₀ := by
  simp only [SDFG.startMem, h.keepAtStart]

/-- The conditions hold for the inverse renaming of the renamed SDFG. -/
theorem symm (h : g.RenOk σ) : (g.ren σ).RenOk σ.symm where
  top := Region.RenOk.symm σ g.top h.top
  shapes n d hd x hx := by
    rw [SDFG.array?_ren'] at hd
    exact perm_symm_fix (h.shapes _ d hd x hx)
  slash p x := perm_symm_fix (h.slash p x)
  fresh x hx := by
    rw [SDFG.keepAtStart_ren]
    refine h.fresh _ fun h' => hx ?_
    rw [_root_.Equiv.apply_symm_apply] at h'
    exact h'.symm
  iface x hx := by
    rw [h.interface] at hx
    exact perm_symm_fix (h.iface x hx)

end SDFG.RenOk

/-- The context running `g` with every container `n` stored under memory
name `σ n`. -/
def renCtx (σ : _root_.Equiv.Perm Name) (I : Interp) (g : SDFG) : Ctx := { I, g, cname := σ, scope := σ }

section Exec

variable {σ : _root_.Equiv.Perm Name} {g : SDFG} {I : Interp} {ρ₀ : Valuation} {m₀ : Mem}

theorem SDFG.RenOk.renCtx (h : g.RenOk σ) (I : Interp) : RenCtx σ { I, g := g.ren σ } (Dace.renCtx σ I g) where
  I := rfl
  desc n := SDFG.array?_ren σ g n
  cname _ := rfl
  scope p x := (h.slash p x).symm
  shapes := h.shapes

theorem SDFG.RenOk.renG (h : g.RenOk σ) (I : Interp) : RenG σ { I, g := g.ren σ } (Dace.renCtx σ I g) where
  names := SDFG.containerNames_ren σ g
  count t := Region.countMentions_ren σ g.top h.top t

/-- **Runs of the renamed SDFG**: every outcome of `g` is matched by one of
`g.ren σ`: undefined behavior, an error of the same class, or a memory equal
on the observable containers. -/
theorem SDFG.RenOk.exec (h : g.RenOk σ) (hw : ∀ op v, I.wcr op .undef v = .undef)
    (hna : ∀ n, m₀.aliases n = none) {r : Res Mem} (hx : SDFG.Exec I g ρ₀ m₀ r) :
    (∃ e, e.isUB = true ∧ SDFG.Exec I (g.ren σ) ρ₀ m₀ (.err e)) ∨
      (∃ e e', r = .err e ∧ e'.isUB = e.isUB ∧ SDFG.Exec I (g.ren σ) ρ₀ m₀ (.err e')) ∨
      ∃ m, r = .ok m ∧ ∃ m', SDFG.Exec I (g.ren σ) ρ₀ m₀ (.ok m') ∧ obsEq g.interface m m' := by
  obtain ⟨res, hx, rfl⟩ := hx
  have hcs : CtxSim σ (fun _ => True) (Dace.renCtx σ I g) { I, g } := ⟨rfl, rfl⟩
  have hall : ∀ x, Ctx.Strong σ (fun _ => True) (Dace.renCtx σ I g) { I, g } x :=
    fun _ => ⟨rfl, rfl, trivial, fun _ _ _ _ => ⟨rfl, fun _ => ⟨rfl, trivial⟩⟩⟩
  have hsc : ScopeSim σ (fun _ => True) (Dace.renCtx σ I g) { I, g } := fun _ => ⟨rfl, trivial⟩
  obtain ⟨r₂, hx₂, hr⟩ := SDFGExec.sim (L := fun _ => False) hcs hw (Region.Names.ofAll hall hsc g.top) hx _
    (Corr.startMem_sim h.fresh hna)
  obtain ⟨r', hx', her⟩ := RenCtx.sdfgExec σ (h.renCtx I) (h.renG I) h.top hx₂
  rw [← h.startMem] at hx'
  cases res with
  | err e =>
    obtain ⟨e₂, rfl, hu⟩ := ResSim.err_inv hr
    obtain ⟨e', rfl, he'⟩ := Res.er_err_inv her
    rcases hu with hu | rfl
    · exact .inl ⟨e', he'.trans hu, _, hx', rfl⟩
    · exact .inr (.inl ⟨e₂, e', rfl, he', _, hx', rfl⟩)
  | ok x₁ =>
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨x₂, rfl, -, hm, -, -⟩
    · obtain ⟨e', rfl, he'⟩ := Res.er_err_inv her
      exact .inl ⟨e', he'.trans hu, _, hx', rfl⟩
    · cases Res.er_ok_inv her
      exact .inr (.inr ⟨_, rfl, x₂.2.1, ⟨_, hx', rfl⟩, Corr.obsEq_of_sim h.iface hm⟩)

/-- The converse: every outcome of `g.ren σ` is matched by one of `g`. -/
theorem SDFG.RenOk.exec_symm (h : g.RenOk σ) (hw : ∀ op v, I.wcr op .undef v = .undef)
    (hna : ∀ n, m₀.aliases n = none) {r : Res Mem} (hx : SDFG.Exec I (g.ren σ) ρ₀ m₀ r) :
    (∃ e, e.isUB = true ∧ SDFG.Exec I g ρ₀ m₀ (.err e)) ∨
      (∃ e e', r = .err e ∧ e'.isUB = e.isUB ∧ SDFG.Exec I g ρ₀ m₀ (.err e')) ∨
      ∃ m, r = .ok m ∧ ∃ m', SDFG.Exec I g ρ₀ m₀ (.ok m') ∧ obsEq g.interface m m' := by
  have := h.symm.exec hw hna hx
  rwa [SDFG.ren_symm, h.interface] at this

end Exec

/-- **Renaming containers** gives an equivalent SDFG (`Decisions.md#D69`). -/
theorem Congr.rename {σ : _root_.Equiv.Perm Name} {g : SDFG} (h : g.RenOk σ) (hs : Supported g.prog) :
    Equiv (g.ren σ).prog g.prog := by
  -- Every error of either program makes the other one possibly undefined.
  have hsQ : Supported (g.ren σ).prog := by
    intro I ρ₀ m₀ hp
    by_cases hu : mayUB (g.ren σ).prog I ρ₀ m₀
    · exact .inl hu
    refine .inr fun e he => hu ?_
    have hw := hp.1.undef_absorb
    have hgerr : ∃ e₁, SDFG.Exec I g ρ₀ m₀ (.err e₁) := by
      rcases h.exec_symm hw hp.2.2 he with ⟨e₁, -, hy⟩ | ⟨_, e₁, -, -, hy⟩ | ⟨m, hm, -⟩
      · exact ⟨e₁, hy⟩
      · exact ⟨e₁, hy⟩
      · cases hm
    obtain ⟨e₁, hy⟩ := hgerr
    obtain ⟨e₂, hu₂, hy₂⟩ : mayUB g.prog I ρ₀ m₀ := by
      rcases hs I ρ₀ m₀ hp with h' | h'
      · exact h'
      · exact absurd hy (h' e₁)
    rcases h.exec hw hp.2.2 hy₂ with ⟨e₃, hu₃, hy₃⟩ | ⟨_, e₃, he₃, hu₃, hy₃⟩ | ⟨m, hm, -⟩
    · exact ⟨e₃, hu₃, hy₃⟩
    · cases he₃; exact ⟨e₃, hu₃.trans hu₂, hy₃⟩
    · cases hm
  refine ⟨?_, ?_⟩
  · intro I ρ₀ m₀ hp
    by_cases hmu : mayUB g.prog I ρ₀ m₀
    · exact .inl hmu
    refine .inr fun r hr => ?_
    rcases h.exec_symm hp.1.undef_absorb hp.2.2 hr with ⟨e, hu, hy⟩ | ⟨_, e', -, -, hy⟩ | ⟨m, rfl, m', hy, ho⟩
    · exact absurd ⟨e, hu, hy⟩ hmu
    · rcases hs I ρ₀ m₀ hp with h' | h'
      · exact absurd h' hmu
      · exact absurd hy (h' e')
    · exact ⟨m, rfl, m', hy, ho⟩
  · intro I ρ₀ m₀ hp
    by_cases hmu : mayUB (g.ren σ).prog I ρ₀ m₀
    · exact .inl hmu
    refine .inr fun r hr => ?_
    rcases h.exec hp.1.undef_absorb hp.2.2 hr with ⟨e, hu, hy⟩ | ⟨_, e', -, -, hy⟩ | ⟨m, rfl, m', hy, ho⟩
    · exact absurd ⟨e, hu, hy⟩ hmu
    · rcases hsQ I ρ₀ m₀ hp with h' | h'
      · exact absurd h' hmu
      · exact absurd hy (h' e')
    · refine ⟨m, rfl, m', hy, ?_⟩
      show obsEq (g.ren σ).interface m m'
      rw [h.interface]; exact ho

/-! ## Deciding the conditions for swapping two names -/

section Check

variable (ok : Name → Bool)

/-- The expression names and consumed streams of a state pass `ok`. -/
def stateRenOk? (s : State) : Bool :=
  s.exprNames.all ok && s.edges.all fun e => (s.consumedStream? e.dst).all fun d => ok d && ok (consumeChunkName d)

mutual

/-- The expression names and consumed streams of a region pass `ok`. -/
def Region.renOk? : Region → Bool
  | ⟨_, _, blocks, _, edges⟩ => Blocks.renOk? blocks && edges.all fun e => e.exprNamesR.all ok

def Blocks.renOk? : List Block → Bool
  | [] => true
  | b :: bs => Block.renOk? b && Blocks.renOk? bs

def Block.renOk? : Block → Bool
  | .state s => stateRenOk? ok s
  | .region r => Region.renOk? r
  | .loop ⟨_, _, init, cond, update, _, _, body⟩ =>
    init.all (fun a => a.2.names.all ok) && cond.names.all ok && update.all (fun a => a.2.names.all ok) &&
      Region.renOk? body
  | .cond _ brs => Branches.renOk? brs
  | .breakBlock | .continueBlock | .returnBlock => true

def Branches.renOk? : List (Option Expr × Region) → Bool
  | [] => true
  | (c, r) :: rest => c.all (fun e => e.names.all ok) && Region.renOk? r && Branches.renOk? rest

end

end Check

section Sound

variable {σ : _root_.Equiv.Perm Name} {ok : Name → Bool} (hok : ∀ x, ok x = true → σ x = x)
include hok

theorem allOk {xs : List Name} (h : xs.all ok = true) : ∀ x ∈ xs, σ x = x :=
  fun x hx => hok x (List.all_eq_true.1 h x hx)

theorem optAllOk {α : Type} {o : Option α} {f : α → List Name} (h : o.all (fun a => (f a).all ok) = true) :
    ∀ a ∈ o, ∀ x ∈ f a, σ x = x := by
  intro a ha
  cases ha
  exact allOk hok (by simpa using h)

theorem stateRenOk?_sound {s : State} (h : stateRenOk? ok s = true) : RenState σ s := by
  simp only [stateRenOk?, Bool.and_eq_true] at h
  refine ⟨allOk hok h.1, fun entry d hd => ?_⟩
  have hd' := hd
  unfold State.consumedStream? at hd
  split at hd
  · rename_i e he
    have hmem := List.mem_of_find?_eq_some he
    simp only [State.inEdges, List.mem_filter, decide_eq_true_eq] at hmem
    have := List.all_eq_true.1 h.2 e hmem.1
    rw [hmem.2, hd'] at this
    simp only [Option.all_some, Bool.and_eq_true] at this
    exact ⟨hok _ this.1, hok _ this.2⟩
  · cases hd

mutual

theorem Region.renOk?_sound : (rg : Region) → Region.renOk? ok rg = true → Region.RenOk σ rg
  | ⟨_, _, blocks, _, edges⟩, h => by
    simp only [Region.renOk?, Bool.and_eq_true, List.all_eq_true] at h
    exact ⟨Blocks.renOk?_sound blocks h.1, fun e he x hx => hok x (h.2 e he x hx)⟩

theorem Blocks.renOk?_sound : (bs : List Block) → Blocks.renOk? ok bs = true → Blocks.RenOk σ bs
  | [], _ => trivial
  | b :: bs, h => by
    simp only [Blocks.renOk?, Bool.and_eq_true] at h
    exact ⟨Block.renOk?_sound b h.1, Blocks.renOk?_sound bs h.2⟩

theorem Block.renOk?_sound : (b : Block) → Block.renOk? ok b = true → Block.RenOk σ b
  | .state s, h => stateRenOk?_sound hok h
  | .region r, h => Region.renOk?_sound r h
  | .loop ⟨_, _, init, cond, update, _, _, body⟩, h => by
    simp only [Block.renOk?, Bool.and_eq_true] at h
    exact ⟨optAllOk hok h.1.1.1, allOk hok h.1.1.2, optAllOk hok h.1.2, Region.renOk?_sound body h.2⟩
  | .cond _ brs, h => Branches.renOk?_sound brs h
  | .breakBlock, _ => trivial
  | .continueBlock, _ => trivial
  | .returnBlock, _ => trivial

theorem Branches.renOk?_sound : (brs : List (Option Expr × Region)) → Branches.renOk? ok brs = true →
    Branches.RenOk σ brs
  | [], _ => trivial
  | (c, r) :: rest, h => by
    simp only [Branches.renOk?, Bool.and_eq_true] at h
    exact ⟨optAllOk hok h.1.1, Region.renOk?_sound r h.1.2, Branches.renOk?_sound rest h.2⟩

end

end Sound

/-- Swapping the container names `t` and `u` preserves meaning: no expression
or shape reads them, they are no consumed streams or chunk containers, they
contain no `/`, and they are neither defined at the start nor observable. -/
def renameOk? (g : SDFG) (t u : Name) : Bool :=
  let ok := fun x => x != t && x != u
  Region.renOk? ok g.top && g.arrays.all (fun a => a.2.fullRange.names.all ok) &&
    !t.toList.contains '/' && !u.toList.contains '/' &&
    !g.keepAtStart t && !g.keepAtStart u && !g.interface.contains t && !g.interface.contains u

theorem renameOk?_sound {g : SDFG} {t u : Name} (h : renameOk? g t u = true) : g.RenOk (_root_.Equiv.swap t u) := by
  simp only [renameOk?, Bool.and_eq_true, Bool.not_eq_true'] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨htop, hsh⟩, hst⟩, hsu⟩, hkt⟩, hku⟩, hit⟩, hiu⟩ := h
  have hok : ∀ x, (x != t && x != u) = true → _root_.Equiv.swap t u x = x := by
    intro x hx
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hx
    exact _root_.Equiv.swap_apply_of_ne_of_ne hx.1 hx.2
  have hmoved : ∀ x, _root_.Equiv.swap t u x ≠ x → x = t ∨ x = u := by
    intro x hx
    by_contra hc
    simp only [not_or] at hc
    exact hx (_root_.Equiv.swap_apply_of_ne_of_ne hc.1 hc.2)
  refine ⟨Region.renOk?_sound hok g.top htop, fun n d hd x hx => ?_, fun p x => ?_, fun x hx => ?_, fun x hx => ?_⟩
  · obtain ⟨a, ha, rfl⟩ : ∃ a ∈ g.arrays, a.2 = d := by
      simp only [SDFG.array?, Option.map_eq_some_iff] at hd
      obtain ⟨a, ha, rfl⟩ := hd
      exact ⟨a, List.mem_of_find?_eq_some ha, rfl⟩
    exact allOk hok (List.all_eq_true.1 hsh a ha) x hx
  · refine _root_.Equiv.swap_apply_of_ne_of_ne ?_ ?_
    · rintro rfl; simp [String.toList_append] at hst
    · rintro rfl; simp [String.toList_append] at hsu
  · rcases hmoved x hx with rfl | rfl
    · exact hkt
    · exact hku
  · refine _root_.Equiv.swap_apply_of_ne_of_ne ?_ ?_
    · rintro rfl; simp [hx] at hit
    · rintro rfl; simp [hx] at hiu

/-- Renaming by a checked swap gives an equivalent SDFG. -/
theorem Congr.swapNames {g : SDFG} {t u : Name} (h : renameOk? g t u = true) (hs : Supported g.prog) :
    Equiv (g.ren (_root_.Equiv.swap t u)).prog g.prog :=
  Congr.rename (renameOk?_sound h) hs

end Dace
