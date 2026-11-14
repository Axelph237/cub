import itertools
import operator
from collections.abc import Callable
from typing import Any

import pytest

from cub import CNF, Bool, Int, UInt, compile, solve
from cub._millisat import Solver
from cub.cnf import FALSE, TRUE
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


def truncating_divmod(a: int, b: int) -> tuple[int, int]:
    magnitude = abs(a) // abs(b)
    quotient = -magnitude if (a < 0) != (b < 0) else magnitude
    return quotient, a - b * quotient


def div_oracle(a: int, b: int, width: int, signed: bool) -> int:
    if b != 0:
        return wrap(truncating_divmod(a, b)[0], width, signed)
    if signed:
        return -1 if a >= 0 else 1
    return (1 << width) - 1


def rem_oracle(a: int, b: int, width: int, signed: bool) -> int:
    return wrap(truncating_divmod(a, b)[1], width, signed) if b != 0 else a


def shl_oracle(a: int, amount: int, width: int, signed: bool) -> int:
    return 0 if amount >= width else wrap(a << amount, width, signed)


def shr_oracle(a: int, amount: int, width: int, signed: bool) -> int:
    if amount >= width:
        return -1 if signed and a < 0 else 0
    return wrap(a >> amount, width, signed)


type Oracle = Callable[[int, int, int, bool], int]

DIVISION: dict[str, tuple[Op, Oracle]] = {
    "div": (operator.floordiv, div_oracle),
    "rem": (operator.mod, rem_oracle),
}
SHIFT: dict[str, tuple[Op, Oracle]] = {
    "shl": (operator.lshift, shl_oracle),
    "shr": (operator.rshift, shr_oracle),
}


def operands[V: (Int, UInt), W: (Int, UInt)](
    mode: str, width: int, a: tuple[type[V], int], b: tuple[type[W], int]
) -> tuple[V, W, list[Bool]]:
    """The same values as constants, or as variables pinned by equality, so both the
    constant-folded and the symbolic encodings are exercised."""
    (a_cls, a_value), (b_cls, b_value) = a, b
    if mode == "const":
        return a_cls.const(a_value, width), b_cls.const(b_value, width), []
    x, y = a_cls.var("x", width), b_cls.var("y", width)
    return x, y, [x == a_value, y == b_value]


@pytest.mark.parametrize("mode", ["const", "var"])
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("cls", CLASSES, ids=ids)
@pytest.mark.parametrize("op", DIVISION)
def test_division_takes_only_the_smtlib_value[V: (Int, UInt)](
    op: str, cls: type[V], width: int, mode: str
) -> None:
    build, oracle = DIVISION[op]
    mismatches = [
        (a, b, got, expected)
        for a, b in itertools.product(domain(cls, width), repeat=2)
        for x, y, given in [operands(mode, width, (cls, a), (cls, b))]
        if (got := feasible_values(build(x, y), *given))
        != [expected := oracle(a, b, width, cls is Int)]
    ]
    assert mismatches == []


@pytest.mark.parametrize("mode", ["const", "var"])
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("cls", CLASSES, ids=ids)
@pytest.mark.parametrize("op", SHIFT)
def test_shift_takes_only_the_smtlib_value[V: (Int, UInt)](
    op: str, cls: type[V], width: int, mode: str
) -> None:
    build, oracle = SHIFT[op]
    mismatches = [
        (a, amount, got, expected)
        for a, amount in itertools.product(domain(cls, width), domain(UInt, width))
        for x, n, given in [operands(mode, width, (cls, a), (UInt, amount))]
        if (got := feasible_values(build(x, n), *given))
        != [expected := oracle(a, amount, width, cls is Int)]
    ]
    assert mismatches == []


def test_division_by_zero_inside_or_stays_satisfiable() -> None:
    x, y, z = UInt.var("x", 4), UInt.var("y", 4), Bool.var("z")
    assert satisfiable(y == 0, z, (x // y == 1) | z), "an encoding that constrains y globally returns UNSAT here"


def test_negated_remainder_by_zero_leaves_operands_free() -> None:
    x, y = UInt.var("x", 4), UInt.var("y", 4)
    assert satisfiable(y == 0, x == 5, ~(x % y == 3))


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
    assert model is not None, "wraparound keeps this SAT"
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


def test_reusing_a_variable_name_with_another_type_raises() -> None:
    with pytest.raises(TypeError, match="variable 'x' is used as both"):
        compile(Int.var("x", 4) == 0, UInt.var("x", 4) == 0)


def test_reusing_a_variable_name_with_the_same_type_is_the_same_variable() -> None:
    assert not satisfiable(Int.var("x", 4) == 1, Int.var("x", 4) != 1)


def test_compile_returns_a_cnf_that_holds_the_constraints() -> None:
    x = Int.var("x", 4)
    sat, unsat = compile(x == -3), compile(x == -3, x == 2)
    assert Solver().solve(sat.num_vars, sat.clauses) is not False
    assert Solver().solve(unsat.num_vars, unsat.clauses) is False


def test_add_clause_folds_constants_and_keeps_the_empty_clause() -> None:
    cnf = CNF()
    x = cnf.new_var()
    cnf.add_clause(x, TRUE)
    cnf.add_clause(x, FALSE)
    cnf.add_clause(FALSE)
    assert cnf.clauses == [(x,), ()]


def test_dimacs_header_counts_every_allocated_variable() -> None:
    cnf = CNF()
    x, y = cnf.new_var(), cnf.new_var()
    cnf.add_clause(x, -y)
    cnf.add_clause()
    assert cnf.to_dimacs() == "p cnf 3 2\n2 -3 0\n0\n", "variable 1 is the reserved TRUE"
