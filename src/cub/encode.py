"""Tseytin gates and bitvector circuits that emit clauses into a CNF.

Adapted from cnfc 1.0.0 (Unlicense): cnfc.tseytin and cnfc.tuples.
"""

from __future__ import annotations

from collections.abc import Sequence

from cub.cnf import CNF, FALSE, TRUE, Bits, Lit


def and_(cnf: CNF, xs: Sequence[Lit]) -> Lit:
    out = cnf.new_var()
    cnf.add_clause(out, *(-x for x in xs))
    for x in xs:
        cnf.add_clause(-out, x)
    return out


def or_(cnf: CNF, xs: Sequence[Lit]) -> Lit:
    out = cnf.new_var()
    cnf.add_clause(-out, *xs)
    for x in xs:
        cnf.add_clause(out, -x)
    return out


def xor(cnf: CNF, a: Lit, b: Lit) -> Lit:
    out = cnf.new_var()
    cnf.add_clause(-a, -b, -out)
    cnf.add_clause(a, b, -out)
    cnf.add_clause(a, -b, out)
    cnf.add_clause(-a, b, out)
    return out


def iff(cnf: CNF, a: Lit, b: Lit) -> Lit:
    out = cnf.new_var()
    cnf.add_clause(-a, -b, out)
    cnf.add_clause(a, b, out)
    cnf.add_clause(a, -b, -out)
    cnf.add_clause(-a, b, -out)
    return out


def less_than(cnf: CNF, a: Bits, b: Bits, strict: bool) -> Lit:
    """a < b if strict, else a <= b, unsigned, by Harvey's lexicographic encoding.

    See https://www.curtisbright.com/bln/2024/12/24/harveys-sat-encoding-for-lexicographic-ordering
    """
    chain = [cnf.new_var() for _ in a] + [FALSE if strict else TRUE]
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        here, rest = chain[i], chain[i + 1]
        cnf.add_clause(rest, y, -here)
        cnf.add_clause(rest, -x, -here)
        cnf.add_clause(y, -x, -here)
        cnf.add_clause(-rest, -y, here)
        cnf.add_clause(-rest, x, here)
        cnf.add_clause(-y, x, here)
    return chain[0]


def add(cnf: CNF, a: Bits, b: Bits) -> Bits:
    """a + b with the carry as an extra leading bit. The shorter operand is zero-extended."""
    width = max(len(a), len(b))
    a_lsb = ((FALSE,) * (width - len(a)) + a)[::-1]
    b_lsb = ((FALSE,) * (width - len(b)) + b)[::-1]
    # Generate/propagate pairs, accumulated linearly so carries[i] is the carry out of bit i.
    gps = [(and_(cnf, (x, y)), xor(cnf, x, y)) for x, y in zip(a_lsb, b_lsb)]
    for i in range(1, width):
        (g_hi, p_hi), (g_lo, p_lo) = gps[i], gps[i - 1]
        g_through = and_(cnf, (p_hi, g_lo))
        gps[i] = or_(cnf, (g_hi, g_through)), and_(cnf, (p_hi, p_lo))
    carries = [g for g, _ in gps]
    total = [xor(cnf, a_lsb[0], b_lsb[0])]
    for i in range(1, width):
        total.append(xor(cnf, xor(cnf, a_lsb[i], b_lsb[i]), carries[i - 1]))
    total.append(carries[-1])
    return tuple(reversed(total))


def mul(cnf: CNF, a: Bits, b: Bits) -> Bits:
    """The full product of a and b, wide enough that it never wraps.

    Shift-and-add: one partial product per bit of b, summed pairwise.
    """
    partials = [
        tuple(and_(cnf, (x, bit)) for x in a) + (FALSE,) * shift
        for shift, bit in enumerate(reversed(b))
    ]
    while len(partials) > 1:
        reduced = [add(cnf, x, y) for x, y in zip(partials[:-1:2], partials[1::2])]
        if len(partials) % 2 == 1:
            reduced.append(partials[-1])
        partials = reduced
    return partials[0]
