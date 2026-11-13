import itertools
import operator
from collections.abc import Callable
from typing import Any

import cnfc
import pytest

from cub import Bool, Int, UInt, compile, solve
from cub.ir import Var

type BitVec = Int | UInt

WIDTHS = [3, 4]
CLASSES = [Int, UInt]
type Op = Callable[..., Any]

BINARY: dict[str, Op] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
}
UNARY: dict[str, Op] = {"neg": operator.neg, "bvnot": operator.invert}
COMPARE: dict[str, Op] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def wrap(value: int, width: int, signed: bool) -> int:
    value &= (1 << width) - 1
    return value - (1 << width) if signed and value >> (width - 1) else value


def domain(cls: type[BitVec], width: int) -> range:
    return range(-(1 << (width - 1)), 1 << (width - 1)) if cls is Int else range(1 << width)


def feasible_values[V: (Int, UInt)](expr: V, *given: Bool) -> list[int]:
    result = expr.__class__(Var("result", expr.type))
    seen: list[int] = []
    while (model := solve(*given, result == expr, *(result != v for v in seen))) is not None:
        seen.append(model[result])
    return seen


def satisfiable(*constraints: Bool) -> bool:
    return solve(*constraints) is not None


def ids(param: object) -> str:
    return getattr(param, "__name__", str(param))


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("cls", CLASSES, ids=ids)
@pytest.mark.parametrize("op", BINARY)
def test_binary_op_takes_only_the_wrapped_value[V: (Int, UInt)](op: str, cls: type[V], width: int) -> None:
    signed = cls is Int
    mismatches = [
        (a, b, got, expected)
        for a, b in itertools.product(domain(cls, width), repeat=2)
        if (got := feasible_values(BINARY[op](cls.const(a, width), cls.const(b, width))))
        != [expected := wrap(BINARY[op](a, b), width, signed)]
    ]
    assert mismatches == []


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("cls", CLASSES, ids=ids)
@pytest.mark.parametrize("op", UNARY)
def test_unary_op_takes_only_the_wrapped_value[V: (Int, UInt)](op: str, cls: type[V], width: int) -> None:
    signed = cls is Int
    mismatches = [
        (a, got, expected)
        for a in domain(cls, width)
        if (got := feasible_values(UNARY[op](cls.const(a, width))))
        != [expected := wrap(UNARY[op](a), width, signed)]
    ]
    assert mismatches == []


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("cls", CLASSES, ids=ids)
@pytest.mark.parametrize("op", COMPARE)
def test_comparison_is_sat_iff_python_agrees[V: (Int, UInt)](op: str, cls: type[V], width: int) -> None:
    mismatches = [
        (a, b, expected)
        for a, b in itertools.product(domain(cls, width), repeat=2)
        for constraint in [COMPARE[op](cls.const(a, width), cls.const(b, width))]
        for expected in [COMPARE[op](a, b)]
        if (satisfiable(constraint), satisfiable(~constraint)) != (expected, not expected)
    ]
    assert mismatches == []


def resize[V: (Int, UInt)](op: str, x: V) -> V:
    match op:
        case "sext":
            return x.sext(x.width + 2)
        case "zext":
            return x.zext(x.width + 2)
        case _:
            return x.trunc(x.width - 1)


def resize_oracle(op: str, value: int, width: int, signed: bool) -> int:
    match op:
        case "sext":
            return wrap(wrap(value, width, True), width + 2, signed)
        case "zext":
            return wrap(wrap(value, width, False), width + 2, signed)
        case _:
            return wrap(value, width - 1, signed)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("cls", CLASSES, ids=ids)
@pytest.mark.parametrize("op", ["sext", "zext", "trunc"])
def test_resize_takes_only_the_expected_value[V: (Int, UInt)](op: str, cls: type[V], width: int) -> None:
    signed = cls is Int
    mismatches = [
        (a, got, expected)
        for a in domain(cls, width)
        if (got := feasible_values(resize(op, cls.const(a, width))))
        != [expected := resize_oracle(op, a, width, signed)]
    ]
    assert mismatches == []


@pytest.mark.parametrize("width", WIDTHS)
def test_reinterpret_reads_the_same_bits_with_flipped_signedness(width: int) -> None:
    mismatches = [
        (a, got)
        for a in domain(Int, width)
        if (got := feasible_values(Int.const(a, width).as_unsigned())) != [wrap(a, width, False)]
    ] + [
        (a, got)
        for a in domain(UInt, width)
        if (got := feasible_values(UInt.const(a, width).as_signed())) != [wrap(a, width, True)]
    ]
    assert mismatches == []


def pinned(var: Bool, value: bool) -> Bool:
    return var if value else ~var


@pytest.mark.parametrize("cls", CLASSES, ids=ids)
def test_ite_takes_only_the_selected_branch[V: (Int, UInt)](cls: type[V]) -> None:
    cond = Bool.var("cond")
    mismatches = [
        (c, a, b, got)
        for c in (False, True)
        for a, b in itertools.product(domain(cls, 3), repeat=2)
        if (got := feasible_values(cond.ite(cls.const(a, 3), cls.const(b, 3)), pinned(cond, c)))
        != [a if c else b]
    ]
    assert mismatches == []


LOGIC: dict[str, tuple[Callable[[Bool, Bool], Bool], Callable[[bool, bool], bool]]] = {
    "and": (operator.and_, operator.and_),
    "or": (operator.or_, operator.or_),
    "not-left": (lambda p, _: ~p, lambda p, _: not p),
    "ite-bool": (lambda p, q: p.ite(q, ~q), lambda p, q: q if p else not q),
}


@pytest.mark.parametrize("op", LOGIC)
def test_logic_is_sat_iff_truth_table_agrees(op: str) -> None:
    build, oracle = LOGIC[op]
    p, q = Bool.var("p"), Bool.var("q")
    mismatches = [
        (pv, qv)
        for pv, qv in itertools.product((False, True), repeat=2)
        if satisfiable(pinned(p, pv), pinned(q, qv), build(p, q)) != oracle(pv, qv)
    ]
    assert mismatches == []


def test_subtraction_inside_or_stays_satisfiable_when_x_is_below_y() -> None:
    x, y, z = UInt.var("x", 4), UInt.var("y", 4), Bool.var("z")
    model = solve(x < y, z, (x - y == 0) | z)
    assert model is not None, "cnfc 1.0 returns UNSAT here; wraparound keeps it SAT"
    assert (model[x] < model[y], model[z]) == (True, True)


def test_model_decodes_signedness_from_the_handle() -> None:
    x = Int.var("x", 4)
    u = x.as_unsigned()
    model = solve(x == -3, u == 13)
    assert model is not None, "-3 and 13 share the bit pattern 1101"
    assert (model[x], model[u]) == (-3, 13)


def test_model_rejects_a_handle_outside_the_solved_constraints() -> None:
    x = Int.var("x", 4)
    model = solve(x == 1)
    assert model is not None
    with pytest.raises(KeyError):
        model[x + 1]


def test_compile_returns_a_cnfc_formula_with_named_bits() -> None:
    formula = compile(Int.var("x", 4) == -3)
    assert isinstance(formula, cnfc.Formula)
    solution = formula.Solve()
    assert solution is not None
    assert solution.integer("x", 4) == 0b1101, "bits are named x:0 (MSB) through x:3"
