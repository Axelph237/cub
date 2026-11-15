import itertools
import random
from collections.abc import Callable
from typing import Any

import pytest

from cub import Bool, Int, Struct, UInt, solve
from cub.ir import StructType, flat_struct
from cub.lower import Value

GRID = Struct.type(9, of=Struct.type(9, of=4))


def pins(handle: Bool | Struct, value: Value) -> list[Bool]:
    """Element-wise constraints that fix handle to the nested-tuple value."""
    if isinstance(handle, Bool):
        return [handle if value else ~handle]
    assert isinstance(value, tuple)
    return [pin for element, part in zip(handle, value, strict=True) for pin in pins(element, part)]


def random_value(type_: StructType, rng: random.Random) -> tuple[Value, ...]:
    return tuple(
        random_value(element, rng) if isinstance(element, StructType) else rng.random() < 0.5
        for element in type_.elements
    )


def flatten_value(value: Value) -> tuple[bool, ...]:
    if isinstance(value, bool):
        return (value,)
    return tuple(leaf for part in value for leaf in flatten_value(part))


def test_nested_struct_has_the_declared_shape() -> None:
    grid = Struct.var("g", 9, of=Struct.type(9, of=4))
    assert len(grid) == 9
    assert isinstance(grid[2], Struct) and len(grid.sub(2)) == 9
    assert isinstance(grid.sub(2)[5], Struct) and len(grid.sub(2).sub(5)) == 4
    assert isinstance(grid.sub(2).sub(5)[0], Bool)
    assert [element.node for element in grid] == [grid[i].node for i in range(9)]
    assert grid.type == GRID == StructType((StructType((flat_struct(4),) * 9),) * 9)
    assert Struct.var("h", 9, of=grid.type).sub(0).type == grid.type
    assert Struct.type(4) == Struct.type(4, of=Bool.var("p").type) == flat_struct(4)
    assert (GRID.width, len(grid.flatten())) == (324, 324)


def test_negative_index_counts_from_the_end() -> None:
    grid = Struct.var("g", 9, of=Struct.type(9, of=4))
    assert grid[-1].node == grid[8].node
    assert grid.sub(-9).sub(-9).bit(-4).node == grid.sub(0).sub(0).bit(0).node


@pytest.mark.parametrize("index", [9, -10, 100])
def test_out_of_range_index_raises_index_error(index: int) -> None:
    with pytest.raises(IndexError):
        Struct.var("g", 9, of=4)[index]


def test_slice_and_wrong_kind_accessors_raise_type_error() -> None:
    cell, grid = Struct.var("c", 4), Struct.var("g", 9, of=4)
    with pytest.raises(TypeError):
        cell[0:2]  # pyrefly: ignore[bad-index]
    with pytest.raises(TypeError, match=r"\.bit\(0\)"):
        cell.sub(0)
    with pytest.raises(TypeError, match=r"\.sub\(0\)"):
        grid.bit(0)


@pytest.mark.parametrize(
    "build",
    [lambda: Struct.var("z", 0), lambda: Struct.type(3, of=0), lambda: Struct()],
    ids=["var", "type", "pack"],
)
def test_zero_length_struct_raises_value_error(build: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        build()


def test_nested_struct_round_trips_through_the_solver() -> None:
    rng = random.Random(0)
    shape = Struct.type(3, of=Struct.type(2, of=3))
    for trial in range(5):
        value = random_value(shape, rng)
        s = Struct.var("s", 3, of=Struct.type(2, of=3))
        model = solve(*pins(s, value))
        assert model is not None
        assert model[s] == value, f"trial {trial}"


def test_flatten_puts_element_0_first_msb_first() -> None:
    s, flat = Struct.var("s", 3, of=2), Struct.var("flat", 6)
    value = ((True, False), (False, False), (True, True))
    model = solve(*pins(s, value), s.flatten() == flat)
    assert model is not None
    assert model[flat] == flatten_value(value) == (True, False, False, False, True, True)


def test_packing_preserves_element_order() -> None:
    flag, cell, grid = Bool.var("f"), Struct.var("c", 4), Struct.var("g", 2, of=3)
    rec = Struct(flag, cell, grid[0])
    assert (len(rec), rec.type.width) == (3, 8)
    value = (False, (True, True, False, True), (True, False, False))
    model = solve(*pins(rec, value))
    assert model is not None
    assert (model[flag], model[cell], model[grid]) == (False, value[1], (value[2], model[grid][1]))
    assert model[rec] == value


def test_int_and_struct_views_round_trip() -> None:
    x, cell = Int.var("x", 4), Struct.var("c", 4)
    assert solve(x.struct.as_int() != x) is None, "x.struct.as_int() must equal x"
    assert solve(cell.as_int().struct != cell) is None, "cell.as_int().struct must equal cell"
    model = solve(x.struct.as_int() == -3)
    assert model is not None and model[x] == -3


def test_int_and_uint_views_read_twos_complement_and_unsigned() -> None:
    cell = Struct.var("c", 4)
    mismatches = []
    for bits in itertools.product((False, True), repeat=4):
        unsigned = sum(bit << (3 - i) for i, bit in enumerate(bits))
        signed = unsigned - 16 if bits[0] else unsigned
        model = solve(*pins(cell, bits), cell.as_int() == Int.var("i", 4), cell.as_uint() == UInt.var("u", 4))
        assert model is not None
        if (model[cell.as_int()], model[cell.as_uint()]) != (signed, unsigned):
            mismatches.append((bits, model[cell.as_int()], model[cell.as_uint()]))
    assert mismatches == []


def test_int_var_is_sugar_for_a_struct_view() -> None:
    assert Int.var("x", 8).node == Struct.var("x", 8).as_int().node
    assert UInt.var("x", 8).node == Struct.var("x", 8).as_uint().node


@pytest.mark.parametrize("view", ["as_int", "as_uint"])
def test_view_of_a_nested_struct_points_at_flatten(view: str) -> None:
    grid = Struct.var("g", 2, of=2)
    with pytest.raises(TypeError, match=r"\.flatten\(\)"):
        getattr(grid, view)()
    assert getattr(grid.flatten(), view)().width == 4


def mixed(name: str) -> Struct:
    return Struct(Bool.var(f"{name}0"), Struct.var(f"{name}1", 2))


def test_structural_equality_is_sat_iff_python_tuples_agree() -> None:
    a, b = mixed("a"), mixed("b")
    values = [(p, (q, r)) for p, q, r in itertools.product((False, True), repeat=3)]
    mismatches = [
        (av, bv, op)
        for av, bv in itertools.product(values, repeat=2)
        for op, relation, expected in [("eq", a == b, av == bv), ("ne", a != b, av != bv)]
        if (solve(*pins(a, av), *pins(b, bv), relation) is not None) != expected
    ]
    assert mismatches == []


@pytest.mark.parametrize(
    "other",
    [lambda: Struct.var("q", 2, of=2), lambda: Struct.var("q", 3), lambda: Struct.var("q", 4).as_uint()],
    ids=["same width other shape", "other length", "uint"],
)
def test_equality_with_another_type_raises(other: Callable[[], Any]) -> None:
    with pytest.raises(TypeError):
        Struct.var("p", 4) == other()


def test_struct_ordering_raises() -> None:
    with pytest.raises(TypeError):
        Struct.var("p", 4) < Struct.var("q", 4)  # pyrefly: ignore[unsupported-operation]


def test_ite_on_structs_selects_the_branch() -> None:
    cond, a, b = Bool.var("cond"), Struct.var("a", 2, of=2), Struct.var("b", 2, of=2)
    av, bv = ((True, False), (False, True)), ((False, False), (True, True))
    for c in (False, True):
        out = Struct.var("out", 2, of=2)
        model = solve(cond if c else ~cond, *pins(a, av), *pins(b, bv), cond.ite(a, b) == out)
        assert model is not None
        assert model[out] == (av if c else bv)
    with pytest.raises(TypeError):
        cond.ite(a, Struct.var("c", 4))
