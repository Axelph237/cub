"""Lower IR to CNF. Every bitvector is an MSB-first tuple of DIMACS literals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import overload

from cub import encode, ir
from cub._millisat import Solver
from cub.cnf import CNF, FALSE, TRUE, Bits, Lit
from cub.dsl import Bool, Int, UInt

# (swap operands, strict) for encode.less_than, which computes a < b or a <= b.
_ORDERINGS: dict[ir.CompareOp, tuple[bool, bool]] = {
    "lt": (False, True),
    "le": (False, False),
    "gt": (True, True),
    "ge": (True, False),
}


class Lowering:
    def __init__(self) -> None:
        self.cnf = CNF()
        self.vars: dict[str, ir.Var] = {}
        self.memo: dict[ir.Node, Bits] = {}
        self.divisions: dict[tuple[ir.Node, ir.Node], tuple[Bits, Bits]] = {}

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
                width = type_.width if isinstance(type_, ir.BV) else 1
                return tuple(self.cnf.new_var() for _ in range(width))
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
    def __getitem__(self, handle: Bool | Int | UInt) -> int | bool:
        if handle.node not in self._bits:
            raise KeyError(f"{handle.node} is not part of the solved constraints")
        unsigned = 0
        for bit in self._bits[handle.node]:
            unsigned = unsigned << 1 | ((abs(bit) in self._true_vars) == (bit > 0))
        match handle.node.type:
            case ir.BoolType():
                return bool(unsigned)
            case ir.BV(width=width, signed=True) if unsigned >> (width - 1):
                return unsigned - (1 << width)
            case ir.BV():
                return unsigned


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


def _constant(value: int, width: int) -> Bits:
    return tuple(TRUE if value >> i & 1 else FALSE for i in reversed(range(width)))
