# cub

cub is a typed DSL for Boolean constraints over Bools and Structs, with
integer views. It compiles constraints to CNF, solves them with a bundled
CDCL solver, and exports them to JSON. It has no runtime dependencies.

```python
from cub import Bool, Int, Struct, UInt, compile, solve, to_json

x, y = Int.var("x", 8), Int.var("y", 8)
model = solve(x + y == 3, x < -100)
model[x], model[y]          # Python ints, read with the handle's signedness
cnf = compile(x < y)        # a cub.CNF
cnf.to_dimacs()             # DIMACS text for an external solver
to_json(x + y == 3)         # a JSON DAG, one entry per distinct node

grid = Struct.var("g", 9, of=Struct.type(9, of=4))   # 9 x 9 cells of 4 Bools
cell = grid.sub(2).sub(5)                            # a Struct of 4 Bools
model = solve(cell.as_uint() == 6, grid.sub(0).sub(0).bit(0))
model[grid]                 # nested tuples of bools, 9 x 9 x 4
```

## Modules

| Module | Role |
|---|---|
| `ir.py` | Frozen IR nodes and types. The type rules live in their constructors. |
| `dsl.py` | The `Bool`, `Struct`, `Int`, and `UInt` handles, which overload operators. |
| `lower.py` | `compile`, `solve`, and `Model`, which lower the IR to CNF. |
| `cnf.py` | `CNF`, the variable allocator and clause list. |
| `encode.py` | Tseytin gates, the adder, multiplier, and comparator circuits, and the sequential-counter cell. |
| `_millisat.py` | The CDCL solver behind `solve`. |
| `export.py` | `to_json`. |

## Data model

cub has two root datatypes, `Bool` and `Struct`. Integers are views over
Structs.

- A `Bool` is one symbolic truth value. `Bool.var("f")` creates one.
- A `Struct` is a fixed-length tuple of elements. Each element is a `Bool`
  or a `Struct`, so nested Structs form a tree. A Struct has at least one
  element.
- `Int` and `UInt` read a flat Struct of Bools as a signed or an unsigned
  bitvector. Element 0 is the most significant bit. `Int.var("x", 8)` is
  short for `Struct.var("x", 8).as_int()`, and `UInt.var` is short for
  `Struct.var(...).as_uint()`.

`Struct.var(name, n, of=...)` creates a Struct variable with `n` elements of
one type. `of` is `BOOL` by default. An int `m` means a flat Struct of `m`
Bools, and a struct type means elements of that type.
`Struct.type(n, of=...)` builds a struct type with the same arguments, so
you can nest it. Every handle has a `.type`, so
`Struct.var("h", 9, of=grid.type)` creates 9 elements shaped like `grid`.

| Expression | Result |
|---|---|
| `len(s)`, `list(s)` | The element count and the element handles. |
| `s[i]` | Element `i`, typed `Bool \| Struct`. Negative `i` counts from the end. |
| `s.sub(i)`, `s.bit(i)` | Element `i`, typed `Struct` or `Bool`. Use these to chain indexing under pyrefly. |
| `Struct(a, b, ...)` | A Struct that packs Bools and Structs in order. |
| `s.flatten()` | A flat Struct of every leaf, element 0 first. |
| `s.as_int()`, `s.as_uint()` | The `Int` or `UInt` view of a flat Struct. |
| `x.struct` | The bits of an `Int` or `UInt` as a flat Struct. |
| `s.count()` | How many leaves are true, as a `UInt`. See [Cardinality](#cardinality). |
| `s == t`, `s != t` | Structural equality, as a `Bool`. |
| `cond.ite(a, b)` | `a` if `cond` holds, else `b`, for any handle kind. |

A Struct's bits are its leaves in order, element 0 first. As a result,
indexing, packing, flattening, and the integer views add no clauses.

## Type rules

- `s[i]` with an `i` outside the Struct raises `IndexError`. A slice raises
  `TypeError`. `s.sub(i)` on a Bool element and `s.bit(i)` on a Struct
  element raise `TypeError`.
- `as_int()` and `as_uint()` need a flat Struct of Bools. On a nested Struct
  they raise `TypeError`, and the message points at `.flatten()`.
- `==` and `!=` on Structs need two Structs of identical type, including
  their nesting. A Struct has no ordering, so `<` raises `TypeError` and
  pyrefly rejects it statically.
- `Int` is signed two's complement and `UInt` is unsigned. Each has a fixed
  width, which is the length of the Struct it views.
- `+ - * // % == != < <= > >=` need two operands of the same class and width.
  A mismatch raises `TypeError`. pyrefly also rejects `Int + UInt` and
  `Int + Struct` statically. Python types cannot express widths, so widths
  are checked at runtime only.
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
- `Bool` supports `& | ~ == !=`. `cond.ite(then, else_)` needs two branches
  of identical type. Calling `bool()` on a `Bool` raises `TypeError`. As a
  result, `a < b < c` and `if x < y:` fail loudly and do not drop a
  constraint. Write `(a < b) & (b < c)` instead.

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
raises `KeyError` for any other handle. It returns a `bool` for a `Bool`, an
`int` for an `Int` or a `UInt`, and nested tuples of bools that mirror the
shape for a `Struct`.

A variable name refers to one variable, so two `Struct.var("s", 4)` handles
are the same variable. `Int.var("x", 8)` and `UInt.var("x", 8)` are two views
of the same 8-bit Struct variable. Reusing a name with a different type, as
in `Int.var("x", 8)` and `Int.var("x", 4)`, raises `TypeError`.

## Cardinality

`s.count()` counts the true leaves of a Struct and returns a `UInt`. It
counts every leaf, so a nested Struct counts all of its Bools. For a Struct
with $n$ leaves, the count has width `n.bit_length()`, the smallest width
that holds $n$. For example, 2 or 3 leaves give width 2, 4 to 7 leaves give
width 3, and 8 leaves give width 4.

```python
a, b, c = Bool.var("a"), Bool.var("b"), Bool.var("c")
exactly_one = Struct(a, b, c).count() == 1
at_most_three = Struct.var("s", 8).count() <= 3
```

A count is an ordinary `UInt`, so the type rules above apply to it.

- A Python `int` takes the width of the count. An int that does not fit
  raises `ValueError`, so `Struct.var("s", 5).count() == 8` fails.
- An operand of another width raises `TypeError`. Widen counts with `.zext`
  before you combine them.
- Arithmetic on counts wraps. Two 3-leaf counts have width 2, so their sum
  wraps modulo $4$. Write `s.count().zext(3) + t.count().zext(3)` to keep
  the full sum.

A comparison between a count and a constant, such as `c <= 3` or
`UInt.const(2, 4) < c`, reads one or two thresholds "at least $j$ leaves are
true". For example, `c <= 3` is the negation of "at least 4". Each threshold
comes from a sequential counter. Cell $(i, j)$ holds "at least $j$ of the
first $i$ leaves are true" and is defined from two cells of row $i - 1$ in 4
clauses. Threshold $j$ needs only a $j \times (n - j + 1)$ rectangle of
cells, so `c <= k` costs about $4(k + 1)(n - k)$ clauses, which is
$O(n \cdot \min(k, n - k))$. All thresholds of one count share their cells.
When the rectangle would cost more clauses than summing the leaves and
comparing the sum, the threshold uses the sum instead. For $n \le 30$ that
never happens, and for $n = 64$ it happens for $10 \le j \le 55$. Any other
use of a count, such as `c == y` or `c + d`, sums the leaves with a balanced
tree of adders. Both encodings define their outputs as functions of the
leaves, so a count comparison stays sound inside `|` and `~`.

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
