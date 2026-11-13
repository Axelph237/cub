"""Lower IR to a cnfc Formula. Every bitvector is an MSB-first tuple of literals."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Mapping
from typing import overload

import cnfc
from cnfc import tseytin, tuples

from cub import ir
from cub.dsl import Bool, Int, UInt

type Lit = cnfc.Var | cnfc.Literal | cnfc.BooleanLiteral
type Bits = tuple[Lit, ...]
type Clauses[R] = Generator[tuple[Lit, ...], None, R]
type Gate = Callable[[tuple[Lit, ...], Lit], Clauses[None]]

# (swap operands, strict) for tuple_less_than, which computes a < b or a <= b.
_ORDERINGS: dict[ir.CompareOp, tuple[bool, bool]] = {
    "lt": (False, True),
    "le": (False, False),
    "gt": (True, True),
    "ge": (True, False),
}


class Lowering:
    def __init__(self) -> None:
        self.formula = cnfc.Formula()
        self.memo: dict[ir.Node, Bits] = {}

    def bits(self, node: ir.Node) -> Bits:
        if node not in self.memo:
            self.memo[node] = self._lower(node)
        return self.memo[node]

    def lit(self, node: ir.Node) -> Lit:
        (bit,) = self.bits(node)
        return bit

    def _lower(self, node: ir.Node) -> Bits:
        match node:
            case ir.Var(name=name, type=ir.BV(width=width)):
                return tuple(self.formula.AddVar(f"{name}:{i}") for i in range(width))
            case ir.Var(name=name):
                return (self.formula.AddVar(name),)
            case ir.Const(value=value, type=ir.BV(width=width)):
                return _constant(value, width)
            case ir.Unary(op=op, x=x):
                bits = self.bits(x)
                match op:
                    case "bvnot":
                        return tuple(~bit for bit in bits)
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
                        product = self._emit(tuples.tuple_mul(self.formula, [*left], [*right]))
                        return tuple(product[-len(left) :])
            case ir.Compare(op=op, a=a, b=b):
                match op:
                    case "eq" | "ne":
                        pairs = zip(self.bits(a), self.bits(b))
                        bitwise = [self._gate(tseytin.gen_eq, pair) for pair in pairs]
                        same = self._gate(tseytin.gen_and, bitwise)
                        return (same if op == "eq" else ~same,)
                    case "lt" | "le" | "gt" | "ge":
                        swap, strict = _ORDERINGS[op]
                        left, right = self._ordered(a), self._ordered(b)
                        if swap:
                            left, right = right, left
                        return (self._emit(tuples.tuple_less_than(self.formula, left, right, strict)),)
            case ir.Logic(op=op, args=args):
                inputs = [self.lit(arg) for arg in args]
                match op:
                    case "and":
                        return (self._gate(tseytin.gen_and, inputs),)
                    case "or":
                        return (self._gate(tseytin.gen_or, inputs),)
            case ir.Not(x=x):
                return (~self.lit(x),)
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

    def _ordered(self, node: ir.Node) -> list[Lit]:
        bits = [*self.bits(node)]
        if isinstance(node.type, ir.BV) and node.type.signed:
            # Flipping the sign bit maps two's complement [-2^(w-1), 2^(w-1)) onto
            # unsigned [0, 2^w) in order, so unsigned comparison becomes signed.
            bits[0] = ~bits[0]
        return bits

    def _add(self, a: Bits, b: Bits) -> Bits:
        total = self._emit(tuples.tuple_add(self.formula, [*a], [*b]))
        return tuple(total[1:])

    def _neg(self, bits: Bits) -> Bits:
        return self._add(tuple(~bit for bit in bits), _constant(1, len(bits)))

    def _mux(self, cond: Lit, then: Bits, else_: Bits) -> Bits:
        out = tuple(self.formula.AddVar() for _ in then)
        for o, t, e in zip(out, then, else_):
            self.formula.AddClause(~cond, ~t, o)
            self.formula.AddClause(~cond, t, ~o)
            self.formula.AddClause(cond, ~e, o)
            self.formula.AddClause(cond, e, ~o)
        return out

    def _gate(self, gen: Gate, inputs: Iterable[Lit]) -> Lit:
        out = self.formula.AddVar()
        self._emit(gen(tuple(inputs), out))
        return out

    def _emit[R](self, clauses: Clauses[R]) -> R:
        while True:
            try:
                clause = next(clauses)
            except StopIteration as done:
                return done.value
            self.formula.AddClause(*clause)


class Model:
    """A satisfying assignment, read through the handles that were solved."""

    def __init__(self, solution: cnfc.Solution, bits: Mapping[ir.Node, Bits]) -> None:
        self._solution = solution
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
            unsigned = unsigned << 1 | self._value(bit)
        match handle.node.type:
            case ir.BoolType():
                return bool(unsigned)
            case ir.BV(width=width, signed=True) if unsigned >> (width - 1):
                return unsigned - (1 << width)
            case ir.BV():
                return unsigned

    def _value(self, bit: Lit) -> int:
        match bit:
            case cnfc.Var(name=name):
                return int(self._solution[name])
            case cnfc.Literal(var=var, sign=sign):
                return int(self._solution[var.name] == (sign > 0))
            case cnfc.BooleanLiteral(val=val):
                return int(val)


def compile(*constraints: Bool) -> cnfc.Formula:
    return _lower_all(constraints).formula


def solve(*constraints: Bool) -> Model | None:
    lowering = _lower_all(constraints)
    solution = lowering.formula.Solve()
    return None if solution is None else Model(solution, lowering.memo)


def _lower_all(constraints: Iterable[Bool]) -> Lowering:
    lowering = Lowering()
    for constraint in constraints:
        lowering.formula.AddClause(lowering.lit(constraint.node))
    return lowering


def _constant(value: int, width: int) -> Bits:
    return tuple(cnfc.BooleanLiteral(bool(value >> i & 1)) for i in reversed(range(width)))
