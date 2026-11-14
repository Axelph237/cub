# cub

cub is a typed bitvector DSL. It compiles constraints to CNF, solves them
with a bundled CDCL solver, and exports them to JSON. It has no runtime
dependencies.

```python
from cub import Bool, Int, UInt, compile, solve, to_json

x, y = Int.var("x", 8), Int.var("y", 8)
model = solve(x + y == 3, x < -100)
model[x], model[y]          # Python ints, read with the handle's signedness
cnf = compile(x < y)        # a cub.CNF
cnf.to_dimacs()             # DIMACS text for an external solver
to_json(x + y == 3)         # a JSON DAG, one entry per distinct node
```

## Modules

| Module | Role |
|---|---|
| `ir.py` | Frozen IR nodes. The type rules live in their constructors. |
| `dsl.py` | The `Int`, `UInt`, and `Bool` handles, which overload operators. |
| `lower.py` | `compile`, `solve`, and `Model`, which lower the IR to CNF. |
| `cnf.py` | `CNF`, the variable allocator and clause list. |
| `encode.py` | Tseytin gates and the adder, multiplier, and comparator circuits. |
| `_millisat.py` | The CDCL solver behind `solve`. |
| `export.py` | `to_json`. |

## Type rules

- `Int` is signed two's complement and `UInt` is unsigned. Each has a fixed
  width that you set at construction, as in `Int.var("x", 8)`.
- `+ - * // % == != < <= > >=` need two operands of the same class and width.
  A mismatch raises `TypeError`. pyrefly also rejects `Int + UInt`
  statically. Python types cannot express widths, so widths are checked at
  runtime only.
- A Python `int` operand takes the type of the other operand. An int that
  does not fit raises `ValueError`, so `Int.const(128, 8)` and `x8 + 200`
  both fail.
- The shift amount in `<<` and `>>` must be a `UInt` with the same width as
  the shifted operand, for both `Int` and `UInt` operands. An `Int` amount
  raises `TypeError`, and pyrefly also rejects it statically. Convert it with
  `.as_unsigned()`. A Python `int` amount becomes that `UInt` type, so
  `x >> -1` raises `ValueError`.
- `.sext(w)` and `.zext(w)` widen, and `.trunc(w)` narrows.
  `.as_signed()` and `.as_unsigned()` read the same bits with the other
  signedness.
- `Bool` supports `& | ~` and `cond.ite(then, else_)`. Calling `bool()` on a
  `Bool` raises `TypeError`. As a result, `a < b < c` and `if x < y:` fail
  loudly and do not drop a constraint. Write `(a < b) & (b < c)` instead.

## Semantics

Arithmetic wraps modulo $2^{w}$, as in SMT-LIB `QF_BV`. For example, a
3-bit `UInt` satisfies `x + 1 == 0` with `x = 7`. Every operation is total
and adds no side constraint. As a result, an operation stays sound inside
`|` and `~`.
Signed comparisons flip the sign bit and then compare as unsigned.

Division follows SMT-LIB, not Python. Read `//` and `%` as `bvudiv` and
`bvurem` on `UInt`, and as `bvsdiv` and `bvsrem` on `Int`.

- Signed division truncates toward zero, and the remainder takes the sign of
  the dividend. For example, `-7 // 2 == -3` and `-7 % 2 == -1`. Python gives
  `-4` and `1`.
- `INT_MIN // -1` wraps to `INT_MIN`.
- Division by zero is defined. For `UInt`, `x // 0` is $2^{w} - 1$ and
  `x % 0` is `x`. For `Int`, `x // 0` is $-1$ when $x \ge 0$ and $1$ when
  $x < 0$, and `x % 0` is `x`.

`<<` is a logical shift left. `>>` is an arithmetic shift on `Int` and a
logical shift on `UInt`. An amount of $w$ or more gives $0$, except that an
arithmetic `>>` gives copies of the sign bit, so a negative `x` gives $-1$.

Division and shifts are also total. Division adds fresh quotient and
remainder bits whose clauses hold for exactly one value per pair of operands,
so `(x // y == 1) | z` stays satisfiable when $y = 0$. A shift by a constant amount adds no clauses.

`model[h]` works for any handle that appears in the solved constraints. It
raises `KeyError` for any other handle. A variable name refers to one
variable, so two `Int.var("x", 8)` handles are the same variable. Reusing a
name with a different type, as in `Int.var("x", 8)` and `UInt.var("x", 8)`,
raises `TypeError`.

## CNF

`compile(*constraints)` returns a `CNF`. Its `clauses` are tuples of DIMACS
literals, where the int $v$ is variable $v$ and $-v$ is its negation.
`num_vars` counts every allocated variable, and `to_dimacs()` writes the
`p cnf` header and one line per clause. Variable 1 is reserved as the
constant true and never appears in a clause. Variable numbers carry no names,
so read values back through `solve` and `Model`.

`encode.py` and `_millisat.py` are adapted from
[cnfc](https://pypi.org/project/cnfc/) 1.0.0, which is released under the
Unlicense.

## Development

```bash
uv sync
uv run pytest -q
uv run pyrefly check
```
