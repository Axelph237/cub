"""User-facing handles. Each wraps one IR node; the IR enforces the type rules."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar, NoReturn, Protocol, Self, overload

from cub import ir

type Of = ir.Element | int
"""A Struct element type: BOOL, a struct type, or n for a flat Struct of n Bools."""


class _BitVec:
    signed: ClassVar[bool]
    node: ir.Node
    type: ir.BV

    def __init__(self, node: ir.Node) -> None:
        if not (isinstance(node.type, ir.BV) and node.type.signed == self.signed):
            raise TypeError(f"{type(self).__name__} cannot wrap a node of type {node.type}")
        self.node, self.type = node, node.type

    def _like(self, node: ir.Node) -> Self:
        return type(self)(node)

    @classmethod
    def var(cls, name: str, width: int) -> Self:
        return cls(ir.View(ir.Var(name, ir.flat_struct(width)), cls.signed))

    @classmethod
    def const(cls, value: int, width: int) -> Self:
        return cls(ir.Const(value, ir.BV(width, cls.signed)))

    @property
    def width(self) -> int:
        return self.type.width

    @property
    def struct(self) -> Struct:
        """The bits as a flat Struct of Bools, most significant bit first."""
        return Struct._wrap(ir.Bits(self.node))

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
    type: ir.BoolType

    def __init__(self, node: ir.Node) -> None:
        if node.type != ir.BOOL:
            raise TypeError(f"Bool cannot wrap a node of type {node.type}")
        self.node, self.type = node, ir.BOOL

    def _like(self, node: ir.Node) -> Self:
        return type(self)(node)

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

    def ite[H: (Int, UInt, Bool, Struct)](self, then: H, else_: H) -> H:
        return then._like(ir.Ite(self.node, then.node, else_.node))

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "a cub Bool is a symbolic constraint, not a Python bool: use & | ~ instead of "
            "and/or/not/if, and write a < b < c as (a < b) & (b < c)"
        )


class _TypeBuilder(Protocol):
    def __call__(self, length: int, of: Of = ...) -> ir.StructType: ...


def _struct_type(length: int, of: Of = ir.BOOL) -> ir.StructType:
    element = ir.flat_struct(of) if isinstance(of, int) else of
    return ir.StructType((element,) * length)


class _TypeAttribute:
    """`Struct.type(n, of=...)` builds a struct type, and `s.type` is the type of handle `s`."""

    @overload
    def __get__(self, handle: None, owner: type[Struct]) -> _TypeBuilder: ...
    @overload
    def __get__(self, handle: Struct, owner: type[Struct]) -> ir.StructType: ...
    def __get__(self, handle: Struct | None, owner: type[Struct]) -> _TypeBuilder | ir.StructType:
        return _struct_type if handle is None else handle._type


class Struct:
    """A fixed-length tuple whose elements are Bools and Structs.

    Struct(a, b, ...) packs handles in order. Its bits are its leaves, element 0 first.
    """

    node: ir.Node
    _type: ir.StructType
    type = _TypeAttribute()

    def __init__(self, *elements: Bool | Struct) -> None:
        self._bind(ir.Pack(tuple(element.node for element in elements)))

    @classmethod
    def _wrap(cls, node: ir.Node) -> Self:
        handle = cls.__new__(cls)
        handle._bind(node)
        return handle

    def _like(self, node: ir.Node) -> Self:
        return type(self)._wrap(node)

    def _bind(self, node: ir.Node) -> None:
        if not isinstance(node.type, ir.StructType):
            raise TypeError(f"Struct cannot wrap a node of type {node.type}")
        self.node, self._type = node, node.type

    @classmethod
    def var(cls, name: str, length: int, of: Of = ir.BOOL) -> Self:
        return cls._wrap(ir.Var(name, _struct_type(length, of)))

    def __len__(self) -> int:
        return len(self._type.elements)

    def __getitem__(self, i: int) -> Bool | Struct:
        if not isinstance(i, int):
            raise TypeError(f"Struct indices must be integers, got {type(i).__name__}")
        node = ir.Index(self.node, i + len(self) if -len(self) <= i < 0 else i)
        return Bool(node) if node.type == ir.BOOL else Struct._wrap(node)

    def __iter__(self) -> Iterator[Bool | Struct]:
        return (self[i] for i in range(len(self)))

    def sub(self, i: int) -> Struct:
        element = self[i]
        if not isinstance(element, Struct):
            raise TypeError(f"element {i} is a Bool, not a Struct; use .bit({i})")
        return element

    def bit(self, i: int) -> Bool:
        element = self[i]
        if not isinstance(element, Bool):
            raise TypeError(f"element {i} is a Struct, not a Bool; use .sub({i})")
        return element

    def flatten(self) -> Struct:
        """The leaves as a flat Struct of Bools, element 0 first."""
        return Struct._wrap(ir.Flatten(self.node))

    def as_int(self) -> Int:
        """Read a flat Struct of Bools as a signed bitvector, element 0 as the sign bit."""
        return Int(ir.View(self.node, signed=True))

    def as_uint(self) -> UInt:
        """Read a flat Struct of Bools as an unsigned bitvector, element 0 as the top bit."""
        return UInt(ir.View(self.node, signed=False))

    def count(self) -> UInt:
        """How many leaves are true, as a UInt of width n.bit_length() for n leaves."""
        return UInt(ir.Count(self.node))

    def _compare(self, op: ir.CompareOp, other: Struct) -> Bool:
        if not isinstance(other, Struct):
            raise TypeError(f"cannot compare Struct with {type(other).__name__}")
        return Bool(ir.Compare(op, self.node, other.node))

    # Same reason as _BitVec: identity equality here would silently answer False.
    def __eq__(self, other: Struct) -> Bool:  # pyrefly: ignore[bad-override]
        return self._compare("eq", other)

    def __ne__(self, other: Struct) -> Bool:  # pyrefly: ignore[bad-override]
        return self._compare("ne", other)
