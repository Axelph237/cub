"""Typed bitvector IR. Nodes are immutable values; type rules live in __post_init__."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class BV:
    width: int
    signed: bool

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError(f"bitvector width must be at least 1, got {self.width}")

    @property
    def min(self) -> int:
        return -(1 << (self.width - 1)) if self.signed else 0

    @property
    def max(self) -> int:
        return (1 << (self.width - 1 if self.signed else self.width)) - 1


@dataclass(frozen=True)
class BoolType:
    pass


BOOL = BoolType()

type Type = BV | BoolType
type UnaryOp = Literal["neg", "bvnot"]
type BinaryOp = Literal["add", "sub", "mul"]
type CompareOp = Literal["eq", "ne", "lt", "le", "gt", "ge"]
type LogicOp = Literal["and", "or"]
type ResizeOp = Literal["sext", "zext", "trunc", "reinterpret"]
type Node = Var | Const | Unary | Binary | Compare | Logic | Not | Ite | Resize


def _bv(node: Node, role: str) -> BV:
    if not isinstance(node.type, BV):
        raise TypeError(f"{role} must be a bitvector, got {node.type}")
    return node.type


def _bool(node: Node, role: str) -> None:
    if node.type != BOOL:
        raise TypeError(f"{role} must be a Bool, got {node.type}")


def _same_bv(a: Node, b: Node) -> BV:
    left, right = _bv(a, "left operand"), _bv(b, "right operand")
    if left != right:
        raise TypeError(f"operands must have identical types, got {left} and {right}")
    return left


@dataclass(frozen=True)
class Var:
    name: str
    type: Type


@dataclass(frozen=True)
class Const:
    value: int
    type: BV

    def __post_init__(self) -> None:
        if not self.type.min <= self.value <= self.type.max:
            raise ValueError(f"{self.value} does not fit in {self.type}")


@dataclass(frozen=True)
class Unary:
    op: UnaryOp
    x: Node
    type: BV = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _bv(self.x, self.op))


@dataclass(frozen=True)
class Binary:
    op: BinaryOp
    a: Node
    b: Node
    type: BV = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _same_bv(self.a, self.b))


@dataclass(frozen=True)
class Compare:
    op: CompareOp
    a: Node
    b: Node
    type: BoolType = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _same_bv(self.a, self.b)
        object.__setattr__(self, "type", BOOL)


@dataclass(frozen=True)
class Logic:
    op: LogicOp
    args: tuple[Node, ...]
    type: BoolType = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for arg in self.args:
            _bool(arg, f"{self.op} argument")
        object.__setattr__(self, "type", BOOL)


@dataclass(frozen=True)
class Not:
    x: Node
    type: BoolType = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _bool(self.x, "not argument")
        object.__setattr__(self, "type", BOOL)


@dataclass(frozen=True)
class Ite:
    cond: Node
    then: Node
    else_: Node
    type: Type = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _bool(self.cond, "ite condition")
        if self.then.type != self.else_.type:
            raise TypeError(
                f"ite branches must have identical types, got {self.then.type} and {self.else_.type}"
            )
        object.__setattr__(self, "type", self.then.type)


@dataclass(frozen=True)
class Resize:
    """Change width (sext, zext, trunc) or flip signedness at the same width (reinterpret)."""

    op: ResizeOp
    x: Node
    width: int
    type: BV = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        source = _bv(self.x, self.op)
        valid = {
            "sext": self.width >= source.width,
            "zext": self.width >= source.width,
            "trunc": 1 <= self.width <= source.width,
            "reinterpret": self.width == source.width,
        }[self.op]
        if not valid:
            raise TypeError(f"cannot {self.op} {source} to width {self.width}")
        signed = not source.signed if self.op == "reinterpret" else source.signed
        object.__setattr__(self, "type", BV(self.width, signed))
