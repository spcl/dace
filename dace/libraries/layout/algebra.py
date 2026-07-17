# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Layout algebra: mixed-radix digit-tuple DSL (Permute/Block/Unblock/Pad/Shuffle/Zip/Unzip) and its optimizer."""
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import sympy


def _sym(x):
    """Coerce an int / str / sympy value into a sympy expression (for structural equality)."""
    if isinstance(x, sympy.Basic):
        return x
    return sympy.sympify(x)


def _ceil_div(e, b):
    """Ceiling division ``ceil(e / b)`` for int or symbolic ``e`` and integer ``b``."""
    e = _sym(e)
    if e.is_Integer:
        return sympy.Integer(-(-int(e) // int(b)))
    return sympy.ceiling(e / b)


@dataclass(frozen=True)
class Digit:
    """One mixed-radix digit: value ``(index[dim] // stride) % extent``."""
    dim: int
    stride: object
    extent: object

    def __post_init__(self):
        object.__setattr__(self, 'stride', _sym(self.stride))
        object.__setattr__(self, 'extent', _sym(self.extent))


@dataclass(frozen=True)
class LayoutMap:
    """A materialized layout: ordered digit tuple plus per-dim annotations (dim_sizes, shuffles, element)."""
    dim_sizes: Dict[int, object]
    digits: Tuple[Digit, ...]
    shuffles: Tuple[Tuple[int, Tuple[Tuple[str, bool], ...]], ...] = ()
    element: Optional[Tuple[str, ...]] = None

    def shape(self) -> Tuple[object, ...]:
        return tuple(d.extent for d in self.digits)


def identity_map(shape, dims: Optional[List[int]] = None) -> LayoutMap:
    """The packed-C identity layout for an array of the given ``shape``."""
    if dims is None:
        dims = list(range(len(shape)))
    dim_sizes = {d: _sym(s) for d, s in zip(dims, shape)}
    digits = tuple(Digit(d, 1, _sym(s)) for d, s in zip(dims, shape))
    return LayoutMap(dim_sizes=dim_sizes, digits=digits)


# Ops
@dataclass(frozen=True)
class Permute:
    """Reorder digit-tuple positions: ``new[i] = old[perm[i]]``."""
    perm: Tuple[int, ...]

    def apply(self, m: LayoutMap) -> LayoutMap:
        assert len(self.perm) == len(m.digits), \
            f"Permute {self.perm} does not match digit count {len(m.digits)}"
        return replace(m, digits=tuple(m.digits[self.perm[i]] for i in range(len(self.perm))))

    def inverse(self) -> 'Permute':
        inv = [0] * len(self.perm)
        for i, p in enumerate(self.perm):
            inv[p] = i
        return Permute(tuple(inv))


@dataclass(frozen=True)
class Block:
    """Split ``dim``'s finest digit ``(d,s,e)`` into outer ``(d,s*b,ceil(e/b))`` + inner ``(d,s,b)`` (appended)."""
    dim: int
    factor: int

    def _finest_pos(self, m: LayoutMap) -> int:
        candidates = [i for i, dg in enumerate(m.digits) if dg.dim == self.dim]
        if not candidates:
            raise ValueError(f"Block: dim {self.dim} not present in layout")
        return min(candidates, key=lambda i: m.digits[i].stride)

    def apply(self, m: LayoutMap) -> LayoutMap:
        p = self._finest_pos(m)
        old = m.digits[p]
        outer = Digit(self.dim, old.stride * self.factor, _ceil_div(old.extent, self.factor))
        inner = Digit(self.dim, old.stride, self.factor)
        digits = list(m.digits)
        digits[p] = outer
        digits.append(inner)
        return replace(m, digits=tuple(digits))

    def inverse(self) -> 'Unblock':
        return Unblock(self.dim, self.factor)


@dataclass(frozen=True)
class Unblock:
    """Merge ``dim``'s inner digit ``(d,s,b)`` with partner ``(d,s*b,eo)`` into ``(d,s,eo*b)``; inverse of Block."""
    dim: int
    factor: int

    def apply(self, m: LayoutMap) -> LayoutMap:
        inners = [i for i, dg in enumerate(m.digits) if dg.dim == self.dim and dg.extent == _sym(self.factor)]
        if not inners:
            raise ValueError(f"Unblock: no inner digit of dim {self.dim} with extent {self.factor}")
        # Finest inner first (smallest stride).
        for ip in sorted(inners, key=lambda i: m.digits[i].stride):
            inner = m.digits[ip]
            target_outer_stride = inner.stride * self.factor
            outers = [i for i, dg in enumerate(m.digits) if dg.dim == self.dim and dg.stride == target_outer_stride]
            if outers:
                op = outers[0]
                outer = m.digits[op]
                merged = Digit(self.dim, inner.stride, outer.extent * self.factor)
                digits = [dg for i, dg in enumerate(m.digits) if i != ip]
                op2 = op if op < ip else op - 1  # index shift after removing inner
                digits[op2] = merged
                return replace(m, digits=tuple(digits))
        raise ValueError(f"Unblock: no outer partner for dim {self.dim} factor {self.factor}")

    def inverse(self) -> 'Block':
        return Block(self.dim, self.factor)


@dataclass(frozen=True)
class Pad:
    """Grow the padded extent of ``dim``'s coarsest digit by ``amount``."""
    dim: int
    amount: object

    def _coarsest_pos(self, m: LayoutMap) -> int:
        candidates = [i for i, dg in enumerate(m.digits) if dg.dim == self.dim]
        if not candidates:
            raise ValueError(f"Pad: dim {self.dim} not present in layout")
        return max(candidates, key=lambda i: m.digits[i].stride)

    def apply(self, m: LayoutMap) -> LayoutMap:
        p = self._coarsest_pos(m)
        old = m.digits[p]
        digits = list(m.digits)
        digits[p] = Digit(self.dim, old.stride, old.extent + _sym(self.amount))
        sizes = dict(m.dim_sizes)
        sizes[self.dim] = _sym(sizes.get(self.dim, 0)) + _sym(self.amount)
        return replace(m, dim_sizes=sizes, digits=tuple(digits))

    def inverse(self) -> 'Pad':
        return Pad(self.dim, -_sym(self.amount))


@dataclass(frozen=True)
class Shuffle:
    """Attach a value-permutation token to ``dim`` (opaque; reads use its inverse)."""
    dim: int
    name: str
    inverted: bool = False

    def apply(self, m: LayoutMap) -> LayoutMap:
        chain = dict(m.shuffles)
        prev = chain.get(self.dim, ())
        chain[self.dim] = prev + ((self.name, self.inverted), )
        return replace(m, shuffles=tuple(sorted((d, c) for d, c in chain.items())))

    def inverse(self) -> 'Shuffle':
        return Shuffle(self.dim, self.name, not self.inverted)


@dataclass(frozen=True)
class Zip:
    """Fuse the current arrays into a struct element with the given fields (boundary op)."""
    fields: Tuple[str, ...]

    def apply(self, m: LayoutMap) -> LayoutMap:
        if m.element is not None:
            raise ValueError("Zip: element is already a struct")
        return replace(m, element=tuple(self.fields))

    def inverse(self) -> 'Unzip':
        return Unzip(self.fields)


@dataclass(frozen=True)
class Unzip:
    """Project a struct element back to separate arrays (boundary op)."""
    fields: Tuple[str, ...]

    def apply(self, m: LayoutMap) -> LayoutMap:
        if m.element != tuple(self.fields):
            raise ValueError(f"Unzip: element {m.element} != {self.fields}")
        return replace(m, element=None)

    def inverse(self) -> 'Zip':
        return Zip(self.fields)


# Optimizer: compose + simplify
def compose_ops(ops: List, base: Optional[LayoutMap] = None, shape=None) -> LayoutMap:
    """Apply ``ops`` in order to ``base`` (or ``identity_map(shape)``), returning one LayoutMap."""
    if base is None:
        if shape is None:
            raise ValueError("compose_ops needs either base or shape")
        base = identity_map(shape)
    m = base
    for op in ops:
        m = op.apply(m)
    return m


def _fuse_pair(a, b):
    """Fuse two adjacent ops: ``[]`` cancels both, ``[op]`` fuses to one, ``None`` leaves as-is."""
    # Inverse pair -> cancel.
    if a.inverse() == b:
        return []
    # Same-type fusion.
    if isinstance(a, Permute) and isinstance(b, Permute):
        fused = Permute(tuple(a.perm[b.perm[i]] for i in range(len(a.perm))))
        if fused.perm == tuple(range(len(fused.perm))):
            return []
        return [fused]
    if isinstance(a, Pad) and isinstance(b, Pad) and a.dim == b.dim:
        total = _sym(a.amount) + _sym(b.amount)
        if total == 0:
            return []
        return [Pad(a.dim, total)]
    if isinstance(a, Shuffle) and isinstance(b, Shuffle) and a.dim == b.dim and a.name == b.name \
            and a.inverted != b.inverted:
        return []
    return None


def simplify_ops(ops: List) -> List:
    """Normalize an op sequence to a minimal canonical form (peephole to a fixpoint)."""
    ops = list(ops)
    changed = True
    while changed:
        changed = False
        out: List = []
        i = 0
        while i < len(ops):
            if i + 1 < len(ops):
                fused = _fuse_pair(ops[i], ops[i + 1])
                if fused is not None:
                    out.extend(fused)
                    i += 2
                    changed = True
                    # Re-examine the tail against the newly appended op next round.
                    out.extend(ops[i:])
                    ops = out
                    break
            out.append(ops[i])
            i += 1
        else:
            ops = out
    return ops


def is_identity(ops: List) -> bool:
    """True iff ``ops`` reduces to a no-op under ``simplify_ops``."""
    return len(simplify_ops(ops)) == 0


def physical_index_exprs(m: LayoutMap) -> List:
    """Per-digit sympy index expression ``(idx[dim] // stride) % extent`` (for lowering)."""
    exprs = []
    for dg in m.digits:
        idx = sympy.Symbol(f"__i{dg.dim}", nonnegative=True, integer=True)
        exprs.append((idx // dg.stride) % dg.extent)
    return exprs


# Serialization: op sequence <-> JSON-safe list of dicts.
def op_to_dict(op) -> Dict:
    """Encode one op as a JSON-safe dict ``{'op': <name>, ...fields}``."""
    if isinstance(op, Permute):
        return {'op': 'Permute', 'perm': list(op.perm)}
    if isinstance(op, Block):
        return {'op': 'Block', 'dim': op.dim, 'factor': op.factor}
    if isinstance(op, Unblock):
        return {'op': 'Unblock', 'dim': op.dim, 'factor': op.factor}
    if isinstance(op, Pad):
        return {'op': 'Pad', 'dim': op.dim, 'amount': str(op.amount)}
    if isinstance(op, Shuffle):
        return {'op': 'Shuffle', 'dim': op.dim, 'name': op.name, 'inverted': op.inverted}
    if isinstance(op, Zip):
        return {'op': 'Zip', 'fields': list(op.fields)}
    if isinstance(op, Unzip):
        return {'op': 'Unzip', 'fields': list(op.fields)}
    raise TypeError(f"op_to_dict: unknown op {op!r}")


def op_from_dict(d: Dict):
    """Decode an op previously encoded by :func:`op_to_dict`."""
    kind = d['op']
    if kind == 'Permute':
        return Permute(tuple(d['perm']))
    if kind == 'Block':
        return Block(d['dim'], d['factor'])
    if kind == 'Unblock':
        return Unblock(d['dim'], d['factor'])
    if kind == 'Pad':
        return Pad(d['dim'], _sym(d['amount']))
    if kind == 'Shuffle':
        return Shuffle(d['dim'], d['name'], bool(d['inverted']))
    if kind == 'Zip':
        return Zip(tuple(d['fields']))
    if kind == 'Unzip':
        return Unzip(tuple(d['fields']))
    raise ValueError(f"op_from_dict: unknown op kind {kind!r}")


def ops_to_list(ops: List) -> List[Dict]:
    """Encode an op sequence as a JSON-safe list of dicts (for a Property)."""
    return [op_to_dict(op) for op in ops]


def ops_from_list(items: List[Dict]) -> List:
    """Decode an op sequence encoded by :func:`ops_to_list`."""
    return [op_from_dict(d) for d in items]
