from collections.abc import Callable

import pytest

from cub.ir import (
    BOOL,
    BV,
    Binary,
    Compare,
    Const,
    Ite,
    Logic,
    Node,
    Not,
    Resize,
    Unary,
    Var,
)

I8 = BV(8, True)
U8 = BV(8, False)
x8, y8, u8 = Var("x", I8), Var("y", I8), Var("u", U8)
x16 = Var("w", BV(16, True))
p = Var("p", BOOL)

ILL_TYPED: dict[str, Callable[[], Node]] = {
    "add mixed widths": lambda: Binary("add", x8, x16),
    "add mixed signedness": lambda: Binary("add", x8, u8),
    "compare mixed signedness": lambda: Compare("lt", x8, u8),
    "compare bools": lambda: Compare("eq", p, p),
    "neg of bool": lambda: Unary("neg", p),
    "and of bitvector": lambda: Logic("and", (p, x8)),
    "not of bitvector": lambda: Not(x8),
    "ite on bitvector condition": lambda: Ite(x8, y8, y8),
    "ite branches differ": lambda: Ite(p, x8, u8),
    "sext narrower": lambda: Resize("sext", x16, 8),
    "zext narrower": lambda: Resize("zext", x16, 8),
    "trunc wider": lambda: Resize("trunc", x8, 16),
    "trunc to zero": lambda: Resize("trunc", x8, 0),
    "reinterpret changes width": lambda: Resize("reinterpret", x8, 16),
    "div mixed signedness": lambda: Binary("div", x8, u8),
    "shift by signed amount": lambda: Binary("shr", u8, x8),
    "shift amount width differs": lambda: Binary("shl", x16, u8),
    "shift of bool": lambda: Binary("shl", p, u8),
}


@pytest.mark.parametrize("build", ILL_TYPED.values(), ids=ILL_TYPED.keys())
def test_ill_typed_node_raises_type_error(build: Callable[[], Node]) -> None:
    with pytest.raises(TypeError):
        build()


@pytest.mark.parametrize(
    ("value", "type_"),
    [(128, I8), (-129, I8), (256, U8), (-1, U8)],
)
def test_const_out_of_range_raises_value_error(value: int, type_: BV) -> None:
    with pytest.raises(ValueError):
        Const(value, type_)


@pytest.mark.parametrize(("value", "type_"), [(127, I8), (-128, I8), (255, U8), (0, U8)])
def test_const_at_range_edge_is_accepted(value: int, type_: BV) -> None:
    assert Const(value, type_).value == value


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (Binary("mul", x8, y8), I8),
        (Binary("shr", x8, u8), I8),
        (Compare("le", u8, u8), BOOL),
        (Resize("sext", x8, 16), BV(16, True)),
        (Resize("zext", u8, 12), BV(12, False)),
        (Resize("trunc", x16, 3), BV(3, True)),
        (Resize("reinterpret", x8, 8), U8),
        (Resize("reinterpret", u8, 8), I8),
        (Ite(p, x8, y8), I8),
    ],
)
def test_node_type_follows_operands(node: Node, expected: object) -> None:
    assert node.type == expected


def test_structurally_equal_nodes_are_equal_and_hash_alike() -> None:
    first, second = Binary("add", x8, y8), Binary("add", Var("x", I8), Var("y", I8))
    assert first == second
    assert hash(first) == hash(second)
