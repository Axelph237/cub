"""A CNF formula over DIMACS literals: variable v is the int v, and -v is its negation."""

from __future__ import annotations

type Lit = int
type Bits = tuple[Lit, ...]
"""A bitvector, most significant bit first."""

TRUE: Lit = 1
FALSE: Lit = -TRUE


class CNF:
    """Variable 1 is reserved as the constant TRUE, so it never appears in a clause."""

    def __init__(self) -> None:
        self.num_vars = TRUE
        self.clauses: list[tuple[Lit, ...]] = []

    def new_var(self) -> Lit:
        self.num_vars += 1
        return self.num_vars

    def add_clause(self, *lits: Lit) -> None:
        """Add a clause with constants folded. An all-FALSE clause stays as the empty clause."""
        if TRUE in lits:
            return
        self.clauses.append(tuple(lit for lit in lits if lit != FALSE))

    def to_dimacs(self) -> str:
        lines = [f"p cnf {self.num_vars} {len(self.clauses)}"]
        lines += [" ".join(map(str, (*clause, 0))) for clause in self.clauses]
        return "\n".join(lines) + "\n"
