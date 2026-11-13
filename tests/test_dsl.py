import itertools
import operator
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from cub import Bool, Int, UInt, solve
from cub.ir import BV, Binary, Const, Resize, Var

PACKAGE = Path(__file__).resolve().parents[1]

x8, y8 = Int.var("x", 8), Int.var("y", 8)
w16 = Int.var("w", 16)
u8 = UInt.var("u", 8)

MISUSES: dict[str, tuple[Callable[[], object], type[Exception]]] = {
    "Int[8] + Int[16]": (lambda: x8 + w16, TypeError),
    "Int + UInt": (lambda: x8 + u8, TypeError),  # pyrefly: ignore[unsupported-operation]
    "Int < UInt": (lambda: x8 < u8, TypeError),  # pyrefly: ignore[unsupported-operation]
    "Int + Bool": (lambda: x8 + Bool.var("p"), TypeError),  # pyrefly: ignore[unsupported-operation]
    "Int.const(128, 8)": (lambda: Int.const(128, 8), ValueError),
    "UInt.const(-1, 8)": (lambda: UInt.const(-1, 8), ValueError),
    "Int[8] + 200": (lambda: x8 + 200, ValueError),
    "200 - Int[8]": (lambda: 200 - x8, ValueError),
    "Int[8].sext(4)": (lambda: x8.sext(4), TypeError),
    "Int[8].trunc(16)": (lambda: x8.trunc(16), TypeError),
    "ite with mixed widths": (lambda: Bool.var("p").ite(x8, w16), TypeError),
    "Int wrapping a UInt node": (lambda: Int(u8.node), TypeError),
    "Int // UInt": (lambda: x8 // u8, TypeError),  # pyrefly: ignore[unsupported-operation]
    "Int >> Int": (lambda: x8 >> y8, TypeError),  # pyrefly: ignore[unsupported-operation]
    "UInt[4] << UInt[8]": (lambda: UInt.var("v", 4) << u8, TypeError),
    "Int[8] >> -1": (lambda: x8 >> -1, ValueError),
    "Int[8] << 256": (lambda: x8 << 256, ValueError),
}


@pytest.mark.parametrize(("misuse", "error"), MISUSES.values(), ids=MISUSES.keys())
def test_misuse_raises(misuse: Callable[[], object], error: type[Exception]) -> None:
    with pytest.raises(error):
        misuse()


def test_chained_comparison_raises() -> None:
    z8 = Int.var("z", 8)
    with pytest.raises(TypeError, match="symbolic constraint"):
        x8 < y8 < z8


def test_comparison_in_if_raises() -> None:
    with pytest.raises(TypeError, match="symbolic constraint"):
        if x8 < y8:
            pass


def test_int_operand_is_coerced_to_the_other_operands_type() -> None:
    assert (x8 + 1).node == Binary("add", x8.node, Const(1, BV(8, True)))
    assert (1 - u8).node == Binary("sub", Const(1, BV(8, False)), u8.node)


def test_signed_shift_amount_error_points_at_as_unsigned() -> None:
    with pytest.raises(TypeError, match=r"\.as_unsigned\(\)"):
        x8 >> y8  # pyrefly: ignore[unsupported-operation]


def test_int_shift_amount_is_coerced_to_unsigned_of_the_same_width() -> None:
    assert (x8 >> 3).node == Binary("shr", x8.node, Const(3, BV(8, False)))
    assert (x8 << u8).type == BV(8, True), "a shift keeps the shifted operand's type"


def test_reinterpret_flips_signedness_and_keeps_width() -> None:
    assert u8.as_signed().node == Resize("reinterpret", u8.node, 8)
    assert x8.as_unsigned().node == Resize("reinterpret", x8.node, 8)
    assert x8.as_signed().node == Var("x", BV(8, True))
    assert u8.as_unsigned().node == Var("u", BV(8, False))


def _pyrefly_errors(tmp_path: Path, source: str) -> str:
    probe = tmp_path / "probe.py"
    probe.write_text(source)
    result = subprocess.run(
        [Path(sys.executable).parent / "pyrefly", "check", probe, "-c", PACKAGE / "pyrefly.toml"],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def test_pyrefly_rejects_mixed_signedness(tmp_path: Path) -> None:
    prelude = 'from cub import Int, UInt\nx, u = Int.var("x", 8), UInt.var("u", 8)\n'
    assert "0 errors" in _pyrefly_errors(tmp_path, prelude + "x + 1\nx + x\n"), (
        "well-typed control file must check clean"
    )
    report = _pyrefly_errors(tmp_path, prelude + "x + u\n")
    assert "`+` is not supported between `Int` and `UInt`" in report, report


def test_pyrefly_rejects_a_signed_shift_amount(tmp_path: Path) -> None:
    prelude = 'from cub import Int, UInt\nx, u = Int.var("x", 8), UInt.var("u", 8)\n'
    assert "0 errors" in _pyrefly_errors(tmp_path, prelude + "x >> u\nx << 1\nu >> u\n"), (
        "well-typed control file must check clean"
    )
    report = _pyrefly_errors(tmp_path, prelude + "x >> x\n")
    assert "`>>` is not supported between `Int` and `Int`" in report, report


@pytest.mark.parametrize("op", [operator.eq, operator.ne], ids=["eq", "ne"])
def test_bool_equality_is_sat_iff_python_agrees(op: Callable[[object, object], object]) -> None:
    p, q = Bool.var("p"), Bool.var("q")
    relation = op(p, q)
    assert isinstance(relation, Bool), "Bool == Bool must build a constraint, not a Python bool"
    mismatches = [
        (pv, qv)
        for pv, qv in itertools.product((False, True), repeat=2)
        if (solve(p if pv else ~p, q if qv else ~q, relation) is not None) != op(pv, qv)
    ]
    assert mismatches == []
