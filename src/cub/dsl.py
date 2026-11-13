"""User-facing handles. Each wraps one IR node; the IR enforces the type rules."""

from __future__ import annotations

from typing import ClassVar, NoReturn, Self

from cub import ir


class _BitVec:
    signed: ClassVar[bool]
    node: ir.Node
    type: ir.BV

    def __init__(self, node: ir.Node) -> None:
        if not (isinstance(node.type, ir.BV) and node.type.signed == self.signed):
            raise TypeError(f"{type(self).__name__} cannot wrap a node of type {node.type}")
        self.node, self.type = node, node.type

    @classmethod
    def var(cls, name: str, width: int) -> Self:
        return cls(ir.Var(name, ir.BV(width, cls.signed)))

    @classmethod
    def const(cls, value: int, width: int) -> Self:
        return cls(ir.Const(value, ir.BV(width, cls.signed)))

    @property
    def width(self) -> int:
        return self.type.width

    def _operand(self, other: _BitVec | int, int_type: ir.BV | None = None) -> ir.Node:
        if isinstance(other, int):
            return ir.Const(other, int_type or self.type)
        if isinstance(other, _BitVec):
            return other.node
        raise TypeError(f"cannot combine {type(self).__name__} with {type(other).__name__}")

    def _binary(self, op: ir.BinaryOp, a: ir.Node, b: ir.Node) -> Self:
        return type(self)(ir.Binary(op, a, b))

    def _shift(self, op: ir.BinaryOp, amount: UInt | int) -> Self:
        return self._binary(op, self.node, self._operand(amount, ir.BV(self.width, signed=False)))

    def _compare(self, op: ir.CompareOp, other: Self | int) -> Bool:
        return Bool(ir.Compare(op, self.node, self._operand(other)))

    def __add__(self, other: Self | int) -> Self:
        return self._binary("add", self.node, self._operand(other))

    def __radd__(self, other: int) -> Self:
        return self._binary("add", self._operand(other), self.node)

    def __sub__(self, other: Self | int) -> Self:
        return self._binary("sub", self.node, self._operand(other))

    def __rsub__(self, other: int) -> Self:
        return self._binary("sub", self._operand(other), self.node)

    def __mul__(self, other: Self | int) -> Self:
        return self._binary("mul", self.node, self._operand(other))

    def __rmul__(self, other: int) -> Self:
        return self._binary("mul", self._operand(other), self.node)

    def __floordiv__(self, other: Self | int) -> Self:
        return self._binary("div", self.node, self._operand(other))

    def __rfloordiv__(self, other: int) -> Self:
        return self._binary("div", self._operand(other), self.node)

    def __mod__(self, other: Self | int) -> Self:
        return self._binary("rem", self.node, self._operand(other))

    def __rmod__(self, other: int) -> Self:
        return self._binary("rem", self._operand(other), self.node)

    def __lshift__(self, amount: UInt | int) -> Self:
        return self._shift("shl", amount)

    def __rshift__(self, amount: UInt | int) -> Self:
        return self._shift("shr", amount)

    def __neg__(self) -> Self:
        return type(self)(ir.Unary("neg", self.node))

    def __invert__(self) -> Self:
        return type(self)(ir.Unary("bvnot", self.node))

    # == and != return a constraint, which breaks object's -> bool signature.
    def __eq__(self, other: Self | int) -> Bool:  # pyrefly: ignore[bad-override]
        return self._compare("eq", other)

    def __ne__(self, other: Self | int) -> Bool:  # pyrefly: ignore[bad-override]
        return self._compare("ne", other)

    def __lt__(self, other: Self | int) -> Bool:
        return self._compare("lt", other)

    def __le__(self, other: Self | int) -> Bool:
        return self._compare("le", other)

    def __gt__(self, other: Self | int) -> Bool:
        return self._compare("gt", other)

    def __ge__(self, other: Self | int) -> Bool:
        return self._compare("ge", other)

    def sext(self, width: int) -> Self:
        return type(self)(ir.Resize("sext", self.node, width))

    def zext(self, width: int) -> Self:
        return type(self)(ir.Resize("zext", self.node, width))

    def trunc(self, width: int) -> Self:
        return type(self)(ir.Resize("trunc", self.node, width))

    def as_signed(self) -> Int:
        return Int(self.node if self.signed else ir.Resize("reinterpret", self.node, self.width))

    def as_unsigned(self) -> UInt:
        return UInt(ir.Resize("reinterpret", self.node, self.width) if self.signed else self.node)


class Int(_BitVec):
    """Signed two's-complement bitvector. Arithmetic wraps modulo 2**width.

    // and % truncate toward zero (SMT-LIB bvsdiv/bvsrem), unlike Python ints. >> is arithmetic.
    """

    signed = True


class UInt(_BitVec):
    """Unsigned bitvector. Arithmetic wraps modulo 2**width. >> is logical."""

    signed = False


class Bool:
    node: ir.Node

    def __init__(self, node: ir.Node) -> None:
        if node.type != ir.BOOL:
            raise TypeError(f"Bool cannot wrap a node of type {node.type}")
        self.node = node

    @classmethod
    def var(cls, name: str) -> Bool:
        return cls(ir.Var(name, ir.BOOL))

    def __and__(self, other: Bool) -> Bool:
        return Bool(ir.Logic("and", (self.node, other.node)))

    def __or__(self, other: Bool) -> Bool:
        return Bool(ir.Logic("or", (self.node, other.node)))

    def __invert__(self) -> Bool:
        return Bool(ir.Not(self.node))

    # Same reason as _BitVec: identity equality here would silently answer False.
    def __eq__(self, other: Bool) -> Bool:  # pyrefly: ignore[bad-override]
        return (self & other) | (~self & ~other)

    def __ne__(self, other: Bool) -> Bool:  # pyrefly: ignore[bad-override]
        return ~(self == other)

    def ite[H: (Int, UInt, Bool)](self, then: H, else_: H) -> H:
        return then.__class__(ir.Ite(self.node, then.node, else_.node))

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "a cub Bool is a symbolic constraint, not a Python bool: use & | ~ instead of "
            "and/or/not/if, and write a < b < c as (a < b) & (b < c)"
        )
