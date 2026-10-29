-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Memlet

/-!
# Memory

The memory model shared by the SDFG and schedule-tree semantics.

* Every data container is addressed by **linear element offsets**, exactly as
  DaCe's generated code does (`cpp_offset_expr`): an index vector is mapped
  through the container's strides and start offset.  This makes views (pointer
  + offset reinterpretations with their own strides) exact.
* Structure members and container-array elements are addressed by *paths* of
  accessors (`Decisions.md#D21`): `s.m[i]` is `(s, [member m, lin i])`,
  `c[i].m[j]` is `(c, [lin i, member m, lin j])`.
* Streams are FIFO queues, one per stream-array element (`Decisions.md#D5`).
* Views and references are **aliases**: a container name bound to a base
  container, a path prefix and a linear offset.  Reading or writing an aliased
  name resolves the alias first (`Decisions.md#D11`).
-/

namespace Dace

/-- One step of a location path. -/
inductive Accessor where
  | lin (o : Int)
  | member (m : String)
  deriving DecidableEq, Repr, Inhabited

/-- A memory location. -/
structure Loc where
  name : Name
  path : List Accessor
  deriving DecidableEq, Repr, Inhabited

namespace Loc

/-- The element at linear offset `o` of a plain container. -/
def elem (name : Name) (o : Int) : Loc := ⟨name, [.lin o]⟩

/-- The whole-container location (used for scalars with no index). -/
def whole (name : Name) : Loc := ⟨name, []⟩

end Loc

/-- Binding of an alias (view or reference) to its base. -/
structure AliasBinding where
  base : Name
  prefix_ : List Accessor := []
  offset : Int := 0
  deriving DecidableEq, Repr, Inhabited

/-- The memory. -/
structure Mem where
  cells : Loc → Val
  queues : Loc → List Val
  aliases : Name → Option AliasBinding

namespace Mem

/-- Memory with all cells at a default value, no queue contents, no aliases. -/
def empty (dflt : Val := default) : Mem :=
  { cells := fun _ => dflt, queues := fun _ => [], aliases := fun _ => none }

/-- Apply one alias binding to a location: `v[path]` becomes
`base[prefix ++ shifted path]`, where the first linear accessor is shifted by
the binding's offset. -/
def applyAlias (b : AliasBinding) (l : Loc) : Loc :=
  let path := match l.path with
    | .lin o :: rest => .lin (o + b.offset) :: rest
    | p => p
  ⟨b.base, b.prefix_ ++ path⟩

/-- Resolve alias chains (bounded by `fuel`; alias chains are acyclic in
well-formed SDFGs, `fuel` only guarantees termination). -/
def resolveFuel (m : Mem) : Nat → Loc → Loc
  | 0, l => l
  | fuel + 1, l =>
    match m.aliases l.name with
    | none => l
    | some b => m.resolveFuel fuel (applyAlias b l)

/-- Resolve a location through at most 64 alias hops. -/
def resolve (m : Mem) (l : Loc) : Loc := m.resolveFuel 64 l

/-- Is `name` currently bound as an alias? -/
def isAliased (m : Mem) (name : Name) : Bool := (m.aliases name).isSome

def read (m : Mem) (l : Loc) : Val := m.cells (m.resolve l)

def write (m : Mem) (l : Loc) (v : Val) : Mem :=
  { m with cells := Function.update m.cells (m.resolve l) v }

/-- Bind an alias (view definition or reference `set`). -/
def setAlias (m : Mem) (name : Name) (b : AliasBinding) : Mem :=
  { m with aliases := Function.update m.aliases name (some b) }

/-- Remove an alias binding. -/
def clearAlias (m : Mem) (name : Name) : Mem :=
  { m with aliases := Function.update m.aliases name none }

/-! ### Streams -/

/-- Push one value to the FIFO at `l` (never blocks, `Decisions.md#D5`). -/
def push (m : Mem) (l : Loc) (v : Val) : Mem :=
  let r := m.resolve l
  { m with queues := Function.update m.queues r (m.queues r ++ [v]) }

def pushMany (m : Mem) (l : Loc) (vs : List Val) : Mem :=
  vs.foldl (fun acc v => acc.push l v) m

/-- Pop one value; `none` on an empty queue (the generated code would block
forever). -/
def pop (m : Mem) (l : Loc) : Option (Val × Mem) :=
  let r := m.resolve l
  match m.queues r with
  | [] => none
  | v :: rest => some (v, { m with queues := Function.update m.queues r rest })

/-- Pop up to `n` values (bulk pop returns what is available, at least one). -/
def popUpTo (m : Mem) (l : Loc) (n : Nat) : Option (List Val × Mem) :=
  let r := m.resolve l
  match m.queues r with
  | [] => none
  | q => some (q.take n, { m with queues := Function.update m.queues r (q.drop n) })

def queueLength (m : Mem) (l : Loc) : Nat := (m.queues (m.resolve l)).length

/-! ### Lemmas -/

/-- Alias resolution only depends on the alias table. -/
theorem resolveFuel_set_cells (m : Mem) (c : Loc → Val) (n : Nat) (l : Loc) :
    ({ m with cells := c } : Mem).resolveFuel n l = m.resolveFuel n l := by
  induction n generalizing l with
  | zero => rfl
  | succ n ih =>
    simp only [resolveFuel]
    cases m.aliases l.name
    · rfl
    · simp [ih]

theorem resolveFuel_set_queues (m : Mem) (q : Loc → List Val) (n : Nat) (l : Loc) :
    ({ m with queues := q } : Mem).resolveFuel n l = m.resolveFuel n l := by
  induction n generalizing l with
  | zero => rfl
  | succ n ih =>
    simp only [resolveFuel]
    cases m.aliases l.name
    · rfl
    · simp [ih]

@[simp] theorem resolve_write (m : Mem) (l l' : Loc) (v : Val) :
    (m.write l v).resolve l' = m.resolve l' := by
  simp [write, resolve, resolveFuel_set_cells]

@[simp] theorem read_write_same (m : Mem) (l : Loc) (v : Val) :
    (m.write l v).read l = v := by
  simp [read, write, resolve, resolveFuel_set_cells]

theorem read_write_other (m : Mem) (l l' : Loc) (v : Val) (h : m.resolve l' ≠ m.resolve l) :
    (m.write l v).read l' = m.read l' := by
  simp [read, write, resolve, resolveFuel_set_cells] at *
  exact Function.update_of_ne h _ _

/-! ### Allocation (`Decisions.md#D68`) -/

/-- Allocate the container stored under memory name `name`: all its cells
become uninitialized and its queues empty.  Alias bindings are unchanged. -/
def alloc (m : Mem) (name : Name) : Mem :=
  { m with cells := fun l => if l.name = name then .undef else m.cells l,
           queues := fun l => if l.name = name then [] else m.queues l }

/-- Allocate a container that is not a stream: its cells become
uninitialized; queues and alias bindings are unchanged. -/
def allocCells (m : Mem) (name : Name) : Mem :=
  { m with cells := fun l => if l.name = name then .undef else m.cells l }

/-- Allocate several containers. -/
def allocAll (m : Mem) (names : List Name) : Mem := names.foldl alloc m

/-- Keep only the containers whose memory names satisfy `keep`; every other
location is uninitialized with an empty queue (program start). -/
def restrict (m : Mem) (keep : Name → Bool) : Mem :=
  { m with cells := fun l => if keep l.name then m.cells l else .undef,
           queues := fun l => if keep l.name then m.queues l else [] }

@[simp] theorem alloc_aliases (m : Mem) (name : Name) : (m.alloc name).aliases = m.aliases := rfl

@[simp] theorem allocAll_aliases (m : Mem) (names : List Name) : (m.allocAll names).aliases = m.aliases := by
  induction names generalizing m with
  | nil => rfl
  | cons n ns ih => exact (ih (m.alloc n)).trans rfl

@[simp] theorem allocAll_nil (m : Mem) : m.allocAll [] = m := rfl

theorem resolveFuel_aliases {m m' : Mem} (h : m.aliases = m'.aliases) (n : Nat) (l : Loc) :
    m.resolveFuel n l = m'.resolveFuel n l := by
  induction n generalizing l with
  | zero => rfl
  | succ n ih => simp only [resolveFuel, h, ih]

@[simp] theorem resolve_allocAll (m : Mem) (names : List Name) (l : Loc) :
    (m.allocAll names).resolve l = m.resolve l :=
  resolveFuel_aliases (allocAll_aliases m names) 64 l

end Mem

end Dace
