"""Typed IR over Bools and Structs, with bitvector views. Nodes are immutable values; type
rules live in __post_init__."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cached_property
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


@dataclass(frozen=True, repr=False)
class BoolType:
    @property
    def width(self) -> int:
        return 1

    def __repr__(self) -> str:
        return "Bool"


BOOL = BoolType()


@dataclass(frozen=True, repr=False)
class StructType:
    """A fixed-length tuple of Bools and Structs. Its bits are its leaves, element 0 first."""

    elements: tuple[Element, ...]

    def __post_init__(self) -> None:
        if not self.elements:
            raise ValueError("a struct needs at least one element")

    @cached_property
    def width(self) -> int:
        return sum(element.width for element in self.elements)

    @cached_property
    def offsets(self) -> tuple[int, ...]:
        """The index of each element's first bit."""
        return tuple(itertools.accumulate((e.width for e in self.elements[:-1]), initial=0))

    @property
    def flat(self) -> bool:
        return all(element == BOOL for element in self.elements)

    def __repr__(self) -> str:
        runs = [(element, len(list(run))) for element, run in itertools.groupby(self.elements)]
        return f"Struct({', '.join(f'{n} x {e}' if n > 1 else repr(e) for e, n in runs)})"


def flat_struct(width: int) -> StructType:
    return StructType((BOOL,) * width)


type Element = BoolType | StructType
type Type = Element | BV
type UnaryOp = Literal["neg", "bvnot"]
type BinaryOp = Literal["add", "sub", "mul", "div", "rem", "shl", "shr"]
type CompareOp = Literal["eq", "ne", "lt", "le", "gt", "ge"]
type LogicOp = Literal["and", "or"]
type ResizeOp = Literal["sext", "zext", "trunc", "reinterpret"]
type Node = (
    Var | Const | Unary | Binary | Compare | Logic | Not | Ite | Resize
    | Pack | Index | Flatten | View | Bits | Count
)


def _bv(node: Node, role: str) -> BV:
    if not isinstance(node.type, BV):
        raise TypeError(f"{role} must be a bitvector, got {node.type}")
    return node.type


def _bool(node: Node, role: str) -> None:
    if node.type != BOOL:
        raise TypeError(f"{role} must be a Bool, got {node.type}")


def _struct(node: Node, role: str) -> StructType:
    if not isinstance(node.type, StructType):
        raise TypeError(f"{role} must be a Struct, got {node.type}")
    return node.type


def _element(node: Node, role: str) -> Element:
    if isinstance(node.type, BV):
        raise TypeError(f"{role} must be a Bool or a Struct, got {node.type}; use .struct for its bits")
    return node.type


def _same_bv(a: Node, b: Node) -> BV:
    left, right = _bv(a, "left operand"), _bv(b, "right operand")
    if left != right:
        raise TypeError(f"operands must have identical types, got {left} and {right}")
    return left


def _shift_operands(value: Node, amount: Node) -> BV:
    shifted, actual = _bv(value, "shifted operand"), _bv(amount, "shift amount")
    expected = BV(shifted.width, signed=False)
    if actual != expected:
        hint = "; convert a signed amount with .as_unsigned()" if actual.signed else ""
        raise TypeError(f"shift amount must be {expected}, got {actual}{hint}")
    return shifted


@dataclass(frozen=True)
class Var:
    name: str
    type: Element

    def __post_init__(self) -> None:
        if not isinstance(self.type, (BoolType, StructType)):
            raise TypeError(f"a variable must be a Bool or a Struct, got {self.type}")


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
        if self.op in ("shl", "shr"):
            object.__setattr__(self, "type", _shift_operands(self.a, self.b))
        else:
            object.__setattr__(self, "type", _same_bv(self.a, self.b))


@dataclass(frozen=True)
class Compare:
    op: CompareOp
    a: Node
    b: Node
    type: BoolType = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.op in ("eq", "ne") and isinstance(self.a.type, StructType):
            if self.a.type != self.b.type:
                raise TypeError(
                    f"operands must have identical types, got {self.a.type} and {self.b.type}"
                )
        else:
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


@dataclass(frozen=True)
class Pack:
    elements: tuple[Node, ...]
    type: StructType = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        types = tuple(_element(element, "packed element") for element in self.elements)
        object.__setattr__(self, "type", StructType(types))


@dataclass(frozen=True)
class Index:
    x: Node
    i: int
    type: Element = field(init=False, repr=False, compare=False)
    offset: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        struct = _struct(self.x, "indexed operand")
        if not 0 <= self.i < len(struct.elements):
            raise IndexError(f"index {self.i} is out of range for {struct}")
        object.__setattr__(self, "type", struct.elements[self.i])
        object.__setattr__(self, "offset", struct.offsets[self.i])


@dataclass(frozen=True)
class Flatten:
    """The same bits as a flat Struct of Bools."""

    x: Node
    type: StructType = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", flat_struct(_struct(self.x, "flatten").width))


@dataclass(frozen=True)
class View:
    """Read a flat Struct of Bools as a bitvector, element 0 as the most significant bit."""

    x: Node
    signed: bool
    type: BV = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        struct = _struct(self.x, "view operand")
        if not struct.flat:
            raise TypeError(
                f"an integer view needs a flat Struct of Bools, got {struct}; call .flatten() first"
            )
        object.__setattr__(self, "type", BV(struct.width, self.signed))


@dataclass(frozen=True)
class Bits:
    """A bitvector's bits as a flat Struct of Bools, most significant bit first."""

    x: Node
    type: StructType = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", flat_struct(_bv(self.x, "bits").width))


@dataclass(frozen=True)
class Count:
    """How many of a Struct's leaves are true, as an unsigned bitvector just wide enough."""

    x: Node
    type: BV = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        width = _struct(self.x, "count").width.bit_length()
        object.__setattr__(self, "type", BV(width, signed=False))
