import itertools
import operator
import random
from collections.abc import Callable
from typing import Any, assert_type

import pytest

from cub import Bool, Struct, UInt, compile, encode, solve
from cub import ir
from cub._millisat import Solver
from cub.cnf import CNF, Lit
from cub.lower import Lowering

type Leaves = tuple[bool, ...]

COMPARE: dict[ir.CompareOp, Callable[[Any, Any], Any]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def assignments(n: int) -> set[Leaves]:
    return set(itertools.product((False, True), repeat=n))


def pins(s: Struct, value: Leaves) -> list[Bool]:
    flat = s.flatten()
    return [flat.bit(i) if v else ~flat.bit(i) for i, v in enumerate(value)]


def feasible_assignments(s: Struct, constraint: Bool) -> set[Leaves]:
    return lowered_assignments(s, lambda lowering: lowering.lit(constraint.node))


def lowered_assignments(s: Struct, build: Callable[[Lowering], Lit]) -> set[Leaves]:
    """Every assignment of s's leaves under which build's literal holds, by blocking models."""
    lowering = Lowering()
    lowering.cnf.add_clause(build(lowering))
    leaves = lowering.bits(s.node)
    found: set[Leaves] = set()
    while (model := Solver().solve(lowering.cnf.num_vars, lowering.cnf.clauses)) is not False:
        true = set(model)
        value = tuple(leaf in true for leaf in leaves)
        found.add(value)
        lowering.cnf.add_clause(*(-leaf if v else leaf for leaf, v in zip(leaves, value)))
    return found


def feasible_counts(s: Struct, *given: Bool) -> list[int]:
    y = UInt.var("y", s.type.width.bit_length())
    seen: list[int] = []
    while (model := solve(*given, s.count() == y, *(y != v for v in seen))) is not None:
        seen.append(model[y])
    return seen


@pytest.mark.parametrize("n, width", [(1, 1), (2, 2), (3, 2), (4, 3), (7, 3), (8, 4)])
def test_count_width_is_the_bit_length_of_the_leaf_count(n: int, width: int) -> None:
    assert Struct.var("s", n).count().width == width


MIRRORED: dict[ir.CompareOp, ir.CompareOp] = {
    "eq": "eq", "ne": "ne", "lt": "gt", "le": "ge", "gt": "lt", "ge": "le"
}


def or_false(cnf: CNF, lit: Lit) -> Lit:
    z = cnf.new_var()
    cnf.add_clause(-z)
    return encode.or_(cnf, (lit, z))


# Each wrap of a comparison literal, and whether it should admit the complement of the oracle.
WRAPS: dict[str, tuple[Callable[[CNF, Lit], Lit], bool]] = {
    "plain": (lambda cnf, lit: lit, False),
    "negated": (lambda cnf, lit: -lit, True),
    "or false z": (or_false, False),
    "negated or false z": (lambda cnf, lit: or_false(cnf, -lit), True),
}


@pytest.mark.parametrize("orientation", ["count-left", "const-left"])
@pytest.mark.parametrize("n", range(1, 7))
def test_constant_comparison_admits_exactly_the_python_assignments(n: int, orientation: str) -> None:
    """Every k in -1..n+1 and every k of the count's width. A k the width cannot hold has no
    Const, so it goes straight to the count comparison the IR rewrite would call."""
    s = Struct.var("s", n)
    count = s.count()
    count_node = count.node
    assert isinstance(count_node, ir.Count)
    mismatches = []
    for k, op in itertools.product(range(-1, max(n + 2, 1 << count.width)), COMPARE):
        fits = 0 <= k < 1 << count.width
        if orientation == "count-left":
            node = COMPARE[op](count, k).node if fits else None
            direct_op = op
            expected = {v for v in assignments(n) if COMPARE[op](sum(v), k)}
        else:
            node = COMPARE[op](UInt.const(k, count.width), count).node if fits else None
            direct_op = MIRRORED[op]
            expected = {v for v in assignments(n) if COMPARE[op](k, sum(v))}

        def compared(lowering: Lowering) -> Lit:
            if node is not None:
                return lowering.lit(node)
            return lowering._count_compare(direct_op, count_node, k)

        for wrap, (apply, complemented) in WRAPS.items():
            got = lowered_assignments(s, lambda lowering: apply(lowering.cnf, compared(lowering)))
            want = assignments(n) - expected if complemented else expected
            if got != want:
                mismatches.append((k, op, wrap, sorted(got ^ want)))
    assert mismatches == []


JOINT: dict[str, tuple[Callable[[UInt, int, int], Bool], Callable[[int, int, int], bool]]] = {
    "ge and le": (lambda c, a, b: (c >= a) & (c <= b), lambda t, a, b: a <= t <= b),
    "eq or eq": (lambda c, a, b: (c == a) | (c == b), lambda t, a, b: t in (a, b)),
    "lt or not gt": (lambda c, a, b: (c < a) | ~(c > b), lambda t, a, b: t < a or t <= b),
}


@pytest.mark.parametrize("joint", JOINT)
def test_comparisons_of_one_count_are_jointly_correct(joint: str) -> None:
    """The second comparison of each pair reads counter cells that the first one built."""
    build, oracle = JOINT[joint]
    n = 6
    s = Struct.var("s", n)
    mismatches = [
        (a, b, sorted(got ^ want))
        for a, b in itertools.product(range(n + 1), repeat=2)
        if (got := feasible_assignments(s, build(s.count(), a, b)))
        != (want := {v for v in assignments(n) if oracle(sum(v), a, b)})
    ]
    assert mismatches == []


def test_large_count_comparison_matches_the_popcount_around_each_threshold() -> None:
    """At 40 leaves the middle thresholds go through the adder and the outer ones through the
    counter, so pinned assignments next to every threshold check both."""
    n = 40
    s = Struct.var("s", n)
    count = s.count()
    rng = random.Random(0)
    mismatches = []

    def check(name: str, constraint: Bool, oracle: Callable[[int], bool], near: tuple[int, ...]) -> None:
        for total in sorted({t for k in near for t in (k - 1, k, k + 1) if 0 <= t <= n}):
            true = set(rng.sample(range(n), total))
            value = tuple(i in true for i in range(n))
            holds = solve(*pins(s, value), constraint) is not None
            fails = solve(*pins(s, value), ~constraint) is not None
            if (holds, fails) != (oracle(total), not oracle(total)):
                mismatches.append((name, total, holds, fails))

    ops: tuple[ir.CompareOp, ...] = ("ge", "le", "eq")
    for k, op in itertools.product(range(n + 1), ops):
        check(f"{op} {k}", COMPARE[op](count, k), lambda t: COMPARE[op](t, k), (k,))
    check("ge 2 and le 20", (count >= 2) & (count <= 20), lambda t: 2 <= t <= 20, (2, 20))
    assert mismatches == []


@pytest.mark.parametrize("n", range(1, 6))
def test_count_equals_only_the_popcount(n: int) -> None:
    s = Struct.var("s", n)
    mismatches = [
        (value, got)
        for value in sorted(assignments(n))
        if (got := feasible_counts(s, *pins(s, value))) != [sum(value)]
    ]
    assert mismatches == []


COMPOSITIONS: dict[str, tuple[Callable[[UInt, UInt], Bool], Callable[[int, int], bool]]] = {
    "wrapping add": (lambda a, b: a + b == 3, lambda a, b: (a + b) % 4 == 3),
    "zext add": (lambda a, b: a.zext(3) + b.zext(3) == 5, lambda a, b: a + b == 5),
}


@pytest.mark.parametrize("composition", COMPOSITIONS)
def test_counts_compose_as_uints(composition: str) -> None:
    build, oracle = COMPOSITIONS[composition]
    s, t = Struct.var("s", 2), Struct.var("t", 3)
    both = Struct(s, t)
    expected = {v for v in assignments(5) if oracle(sum(v[:2]), sum(v[2:]))}
    assert s.count().width == t.count().width == 2
    assert feasible_assignments(both, build(s.count(), t.count())) == expected


def test_nested_struct_counts_every_leaf() -> None:
    g = Struct(Bool.var("p"), Struct.var("q", 2, of=2))
    assert g.count().width == 3
    mismatches = [
        (value, got)
        for value in sorted(assignments(5))
        if (got := feasible_counts(g, *pins(g, value))) != [sum(value)]
    ]
    assert mismatches == []
    assert feasible_assignments(g, g.count() >= 3) == {v for v in assignments(5) if sum(v) >= 3}


def test_exactly_one_idiom() -> None:
    a, b, c = Bool.var("a"), Bool.var("b"), Bool.var("c")
    s = Struct(a, b, c)
    assert feasible_assignments(s, Struct(a, b, c).count() == 1) == {
        (True, False, False),
        (False, True, False),
        (False, False, True),
    }


# + 0 keeps the Count out of the constant-comparison rewrite, so the adder path is exercised.
PATHS: dict[str, Callable[[UInt], UInt]] = {"constant": lambda c: c, "adder": lambda c: c + 0}


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("op", COMPARE)
def test_count_comparison_inside_or_and_not_leaves_the_leaves_free(
    op: ir.CompareOp, path: str
) -> None:
    s, z = Struct.var("s", 4), Bool.var("z")
    count = PATHS[path](s.count())
    constrained = [
        (k, negated)
        for k in range(8)
        for negated, c in [(False, COMPARE[op](count, k)), (True, ~COMPARE[op](count, k))]
        if feasible_assignments(s, c | z) != assignments(4)
    ]
    assert constrained == [], "an asserting encoding restricts s even when z holds"


def test_model_reads_a_count_compared_only_with_constants() -> None:
    s = Struct.var("s", 4)
    mismatches = [
        (value, got)
        for value in sorted(assignments(4))
        for model in [solve(*pins(s, value), s.count() <= 4)]
        if model is None or (got := model[s.count()]) != sum(value)
    ]
    assert mismatches == []


def test_comparisons_of_one_count_share_counter_cells() -> None:
    c = Struct.var("s", 8).count()
    one = compile(c >= 3)
    assert len(compile(c >= 3, c <= 2).clauses) == len(one.clauses) + 1, "c <= 2 adds only its unit"


@pytest.mark.parametrize("n", [8, 32, 64])
def test_constant_comparison_costs_no_more_than_the_counter_or_the_adder(n: int) -> None:
    """The counter spends at most 4 clauses on each of its (k + 1)(n - k) cells for c <= k, and a
    constant comparison never costs more than comparing the adder's sum."""
    c = Struct.var("s", n).count()
    adder = len(compile(c <= UInt.var("y", c.width)).clauses)
    over = [
        (k, cost)
        for k in range(n)
        if (cost := len(compile(c <= k).clauses)) > min(adder, 4 * (k + 1) * (n - k) + 1)
    ]
    assert over == []


@pytest.mark.parametrize("k", [8, -1])
def test_constant_that_does_not_fit_the_count_raises_value_error(k: int) -> None:
    with pytest.raises(ValueError):
        Struct.var("s", 5).count() == k


def test_count_against_another_width_raises_type_error() -> None:
    s = Struct.var("s", 5)
    with pytest.raises(TypeError):
        s.count() == UInt.var("y", 4)
    with pytest.raises(TypeError):
        s.count() + Struct.var("t", 2).count()


def test_count_of_a_non_struct_raises_type_error() -> None:
    with pytest.raises(TypeError):
        ir.Count(ir.Var("p", ir.BOOL))


def test_count_comparison_is_a_bool_statically() -> None:
    s = Struct.var("s", 5)
    assert_type(s.count(), UInt)
    assert_type(s.count() < 3, Bool)
