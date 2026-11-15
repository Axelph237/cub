"""Lower IR to CNF. Every node is an MSB-first tuple of DIMACS literals.

A Struct's bits are its leaves, element 0 first, so every Struct operation is rewiring.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from typing import overload

from cub import encode, ir
from cub._millisat import Solver
from cub.cnf import CNF, FALSE, TRUE, Bits, Lit
from cub.dsl import Bool, Int, Struct, UInt

type Value = bool | tuple[Value, ...]
"""A Bool's or a Struct's value: nested tuples that mirror the Struct's shape."""

# (swap operands, strict) for encode.less_than, which computes a < b or a <= b.
_ORDERINGS: dict[ir.CompareOp, tuple[bool, bool]] = {
    "lt": (False, True),
    "le": (False, False),
    "gt": (True, True),
    "ge": (True, False),
}

# k op c is c mirrored(op) k.
_MIRRORED: dict[ir.CompareOp, ir.CompareOp] = {
    "eq": "eq",
    "ne": "ne",
    "lt": "gt",
    "le": "ge",
    "gt": "lt",
    "ge": "le",
}


class Lowering:
    def __init__(self) -> None:
        self.cnf = CNF()
        self.vars: dict[str, ir.Var] = {}
        self.memo: dict[ir.Node, Bits] = {}
        self.divisions: dict[tuple[ir.Node, ir.Node], tuple[Bits, Bits]] = {}
        self.counters: dict[ir.Count, dict[tuple[int, int], Lit]] = {}

    def bits(self, node: ir.Node) -> Bits:
        if node not in self.memo:
            self.memo[node] = self._lower(node)
        return self.memo[node]

    def lit(self, node: ir.Node) -> Lit:
        (bit,) = self.bits(node)
        return bit

    def _lower(self, node: ir.Node) -> Bits:
        match node:
            case ir.Var(name=name, type=type_):
                seen = self.vars.setdefault(name, node)
                if seen != node:
                    raise TypeError(f"variable {name!r} is used as both {seen.type} and {type_}")
                return tuple(self.cnf.new_var() for _ in range(type_.width))
            case ir.Const(value=value, type=ir.BV(width=width)):
                return _constant(value, width)
            case ir.Unary(op=op, x=x):
                bits = self.bits(x)
                match op:
                    case "bvnot":
                        return tuple(-bit for bit in bits)
                    case "neg":
                        return self._neg(bits)
            case ir.Binary(op=op, a=a, b=b):
                left, right = self.bits(a), self.bits(b)
                match op:
                    case "add":
                        return self._add(left, right)
                    case "sub":
                        return self._add(left, self._neg(right))
                    case "mul":
                        return encode.mul(self.cnf, left, right)[-len(left) :]
                    case "div":
                        return self._divide(a, b)[0]
                    case "rem":
                        return self._divide(a, b)[1]
                    case "shl":
                        return self._shift(left, right, FALSE, toward_msb=True)
                    case "shr":
                        fill = left[0] if node.type.signed else FALSE
                        return self._shift(left, right, fill, toward_msb=False)
            case ir.Compare(op=op, a=ir.Count() as count, b=ir.Const(value=k)):
                return (self._count_compare(op, count, k),)
            case ir.Compare(op=op, a=ir.Const(value=k), b=ir.Count() as count):
                return (self._count_compare(_MIRRORED[op], count, k),)
            case ir.Compare(op=op, a=a, b=b):
                match op:
                    case "eq" | "ne":
                        pairs = zip(self.bits(a), self.bits(b))
                        same = encode.and_(self.cnf, [encode.iff(self.cnf, x, y) for x, y in pairs])
                        return (same if op == "eq" else -same,)
                    case "lt" | "le" | "gt" | "ge":
                        swap, strict = _ORDERINGS[op]
                        left, right = self._ordered(a), self._ordered(b)
                        if swap:
                            left, right = right, left
                        return (encode.less_than(self.cnf, left, right, strict),)
            case ir.Logic(op=op, args=args):
                inputs = [self.lit(arg) for arg in args]
                match op:
                    case "and":
                        return (encode.and_(self.cnf, inputs),)
                    case "or":
                        return (encode.or_(self.cnf, inputs),)
            case ir.Not(x=x):
                return (-self.lit(x),)
            case ir.Ite(cond=cond, then=then, else_=else_):
                return self._mux(self.lit(cond), self.bits(then), self.bits(else_))
            case ir.Resize(op=op, x=x, width=width):
                bits = self.bits(x)
                match op:
                    case "sext":
                        return (bits[0],) * (width - len(bits)) + bits
                    case "zext":
                        return _constant(0, width - len(bits)) + bits
                    case "trunc":
                        return bits[-width:]
                    case "reinterpret":
                        return bits
            case ir.Pack(elements=elements):
                return tuple(bit for element in elements for bit in self.bits(element))
            case ir.Index(x=x, offset=offset, type=type_):
                return self.bits(x)[offset : offset + type_.width]
            case ir.Flatten(x=x) | ir.View(x=x) | ir.Bits(x=x):
                return self.bits(x)
            case ir.Count(x=x, type=ir.BV(width=width)):
                # The sum is at most the leaf count, which fits in width bits, so trunc drops zeros.
                return encode.add_all(self.cnf, [(bit,) for bit in self.bits(x)])[-width:]

    def _count_compare(self, op: ir.CompareOp, count: ir.Count, k: int) -> Lit:
        """count op k. Every op is a literal of at_least, so the result is a function of the
        leaves and stays sound under | and ~."""
        match op:
            case "ge":
                return self._at_least(count, k)
            case "gt":
                return self._at_least(count, k + 1)
            case "le":
                return -self._at_least(count, k + 1)
            case "lt":
                return -self._at_least(count, k)
            case "eq" | "ne":
                at_least, more = self._at_least(count, k), self._at_least(count, k + 1)
                exactly = encode.and_(self.cnf, (at_least, -more))
                return exactly if op == "eq" else -exactly

    def _at_least(self, count: ir.Count, j: int) -> Lit:
        """count >= j from whichever of the sequential counter and the adder predicts fewer clauses.

        The counter reaches cell (n, j) only through cells (i, c) with c <= j and
        i - c <= n - j, a j by (n - j + 1) rectangle. Thresholds of one count share the cells
        where their rectangles overlap.
        """
        leaves = self.bits(count.x)
        n = len(leaves)
        if j <= 0:
            return TRUE
        if j > n:
            return FALSE
        if _counter_clauses(n, j) > _adder_clauses(n, j):
            return -self._below_by_adder(count, j)
        cells = self.counters.setdefault(count, {})

        def cell(i: int, c: int) -> Lit:
            return TRUE if c == 0 else FALSE if c > i else cells[i, c]

        for i in range(1, n + 1):
            for c in range(max(1, j - (n - i)), min(i, j) + 1):
                if (i, c) not in cells:
                    above, diagonal = cell(i - 1, c), cell(i - 1, c - 1)
                    cells[i, c] = encode.at_least_cell(self.cnf, above, leaves[i - 1], diagonal)
        return cells[n, j]

    def _below_by_adder(self, count: ir.Count, j: int) -> Lit:
        bound = _constant(j, count.type.width)
        return encode.less_than(self.cnf, self.bits(count), bound, strict=True)

    def _ordered(self, node: ir.Node) -> Bits:
        bits = self.bits(node)
        if isinstance(node.type, ir.BV) and node.type.signed:
            # Flipping the sign bit maps two's complement [-2^(w-1), 2^(w-1)) onto
            # unsigned [0, 2^w) in order, so unsigned comparison becomes signed.
            return (-bits[0], *bits[1:])
        return bits

    def _divide(self, a: ir.Node, b: ir.Node) -> tuple[Bits, Bits]:
        """Quotient and remainder, shared by div and rem on the same operands."""
        if (a, b) not in self.divisions:
            dividend, divisor = self.bits(a), self.bits(b)
            if isinstance(a.type, ir.BV) and a.type.signed:
                # SMT-LIB bvsdiv/bvsrem: divide magnitudes, then restore signs. The quotient
                # is negative iff the signs differ and the remainder takes the dividend's sign.
                quotient, remainder = self._udivrem(self._abs(dividend), self._abs(divisor))
                negate = encode.xor(self.cnf, dividend[0], divisor[0])
                quotient = self._mux(negate, self._neg(quotient), quotient)
                remainder = self._mux(dividend[0], self._neg(remainder), remainder)
                self.divisions[a, b] = quotient, remainder
            else:
                self.divisions[a, b] = self._udivrem(dividend, divisor)
        return self.divisions[a, b]

    def _udivrem(self, a: Bits, b: Bits) -> tuple[Bits, Bits]:
        """bvudiv/bvurem as fresh bits that every (a, b) determines uniquely.

        Each branch is guarded by b == 0 or its negation, and exactly one (q, r) satisfies
        them for every (a, b). The clauses only define q and r, so the encoding stays
        sound under | and ~, unlike an encoding that constrains b globally.
        """
        q = tuple(self.cnf.new_var() for _ in a)
        r = tuple(self.cnf.new_var() for _ in a)
        b_zero = encode.and_(self.cnf, [-bit for bit in b])
        self._implies_equal(b_zero, q, _constant(-1, len(q)))
        self._implies_equal(b_zero, r, a)
        # b * q + r in full width cannot wrap, so a == b * q + r with r < b pins q and r.
        total = encode.add(self.cnf, encode.mul(self.cnf, b, q), r)
        self._implies_equal(-b_zero, total, _constant(0, len(total) - len(a)) + a)
        self.cnf.add_clause(b_zero, encode.less_than(self.cnf, r, b, strict=True))
        return q, r

    def _implies_equal(self, guard: Lit, xs: Bits, ys: Bits) -> None:
        for x, y in zip(xs, ys, strict=True):
            self.cnf.add_clause(-guard, -x, y)
            self.cnf.add_clause(-guard, x, -y)

    def _shift(self, bits: Bits, amount: Bits, fill: Lit, toward_msb: bool) -> Bits:
        """Barrel shifter: stage k shifts by 2^k when amount bit k is set."""
        width = len(bits)
        stages = (width - 1).bit_length()  # the k with 2^k < width
        for k in range(stages):
            step, pad = 1 << k, (fill,) * (1 << k)
            shifted = bits[step:] + pad if toward_msb else pad + bits[:-step]
            bits = self._mux(amount[-1 - k], shifted, bits)
        overflow = self._any(amount[: width - stages])
        return self._mux(overflow, (fill,) * width, bits)

    def _any(self, bits: Bits) -> Lit:
        if TRUE in bits:
            return TRUE
        unknown = [bit for bit in bits if bit != FALSE]
        if len(unknown) > 1:
            return encode.or_(self.cnf, unknown)
        return unknown[0] if unknown else FALSE

    def _abs(self, bits: Bits) -> Bits:
        """Magnitude as an unsigned pattern. INT_MIN maps to itself, which is 2^(w-1) unsigned."""
        return self._mux(bits[0], self._neg(bits), bits)

    def _add(self, a: Bits, b: Bits) -> Bits:
        return encode.add(self.cnf, a, b)[1:]

    def _neg(self, bits: Bits) -> Bits:
        return self._add(tuple(-bit for bit in bits), _constant(1, len(bits)))

    def _mux(self, cond: Lit, then: Bits, else_: Bits) -> Bits:
        if cond == TRUE:
            return then
        if cond == FALSE:
            return else_
        out = tuple(self.cnf.new_var() for _ in then)
        for o, t, e in zip(out, then, else_):
            self.cnf.add_clause(-cond, -t, o)
            self.cnf.add_clause(-cond, t, -o)
            self.cnf.add_clause(cond, -e, o)
            self.cnf.add_clause(cond, e, -o)
        return out


class Model:
    """A satisfying assignment, read through the handles that were solved."""

    def __init__(self, assignment: Iterable[Lit], bits: Mapping[ir.Node, Bits]) -> None:
        self._true_vars = {TRUE, *(lit for lit in assignment if lit > 0)}
        self._bits = bits

    @overload
    def __getitem__(self, handle: Bool) -> bool: ...
    @overload
    def __getitem__(self, handle: Int | UInt) -> int: ...
    @overload
    def __getitem__(self, handle: Struct) -> tuple[Value, ...]: ...
    def __getitem__(self, handle: Bool | Int | UInt | Struct) -> int | Value:
        values = self._values(handle.node)
        match handle.node.type:
            case ir.BV(width=width, signed=signed):
                unsigned = sum(value << i for i, value in enumerate(reversed(values)))
                return unsigned - (1 << width) if signed and values[0] else unsigned
            case ir.BoolType() | ir.StructType() as type_:
                return _decode(type_, values)

    def _values(self, node: ir.Node) -> list[bool]:
        if node in self._bits:
            return [(abs(bit) in self._true_vars) == (bit > 0) for bit in self._bits[node]]
        if isinstance(node, ir.Count) and node.x in self._bits:
            # A count compared only with constants may lower to counter cells, not to bits.
            total = sum(self._values(node.x))
            return [total >> i & 1 == 1 for i in reversed(range(node.type.width))]
        raise KeyError(f"{node} is not part of the solved constraints")


def _decode(type_: ir.Element, values: Sequence[bool]) -> Value:
    match type_:
        case ir.BoolType():
            return values[0]
        case ir.StructType(elements=elements, offsets=offsets):
            return tuple(
                _decode(element, values[offset : offset + element.width])
                for element, offset in zip(elements, offsets)
            )


def compile(*constraints: Bool) -> CNF:
    return _lower_all(constraints).cnf


def solve(*constraints: Bool) -> Model | None:
    lowering = _lower_all(constraints)
    assignment = Solver().solve(lowering.cnf.num_vars, lowering.cnf.clauses)
    return None if assignment is False else Model(assignment, lowering.memo)


def _lower_all(constraints: Iterable[Bool]) -> Lowering:
    lowering = Lowering()
    for constraint in constraints:
        lowering.cnf.add_clause(lowering.lit(constraint.node))
    return lowering


def _counter_clauses(n: int, j: int) -> int:
    """Clauses for count >= j through the counter over n variable leaves. Cell (1, 1) is the
    leaf itself, the other n - j cells of column 1 and j - 1 cells of the diagonal fold to 3
    clauses each, and the rest of the rectangle takes 4."""
    return 4 * j * (n - j + 1) - n - 3


@functools.cache
def _adder_clauses(n: int, j: int) -> int:
    """Clauses for count >= j through the adder over n variable leaves, counted by building it."""
    lowering = Lowering()
    lowering._below_by_adder(ir.Count(Struct.var("s", n).node), j)
    return len(lowering.cnf.clauses)


def _constant(value: int, width: int) -> Bits:
    return tuple(TRUE if value >> i & 1 else FALSE for i in reversed(range(width)))
