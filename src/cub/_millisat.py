"""A small but complete CDCL SAT solver.

Adapted from cnfc 1.0.0 (Unlicense), which vendors https://github.com/aaw/millisat.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Literal


class Clause:
    def __init__(self, lits: list[int], watch1: int, watch2: int) -> None:
        self.lits, self.watches = lits, set((watch1, watch2))

    def other_watch(self, watch: int) -> int:
        assert watch in self.watches; return (self.watches - {watch}).pop()


class Solver:
    assign: dict[int, bool]
    free: dict[int, bool]
    level: dict[int, int]
    agility: float
    saved: dict[int, bool]
    units: set[int]
    watch: dict[int, set[Clause]]
    clauses: list[Clause]
    max_clauses: int
    first_learned: int
    trail: list[tuple[int, Clause | None]]
    tp: int

    # A literal can either agree with its variable assignment, disagree, or not
    # have an assignment at all.
    def _lit_satisfied(self, lit: int) -> bool: return self.assign.get(abs(lit)) == (lit > 0)
    def _lit_falsified(self, lit: int) -> bool: return self.assign.get(abs(lit)) == (lit < 0)
    def _lit_unassigned(self, lit: int) -> bool: return abs(lit) not in self.assign
    def _assign_lit_true(self, lit: int) -> None: self.assign[abs(lit)] = (lit > 0); del self.free[abs(lit)]

    def _assign_and_add_to_trail(self, lit: int, reason: Clause | None, level: int) -> None:
        self.agility *= 0.9999
        if lit > 0 != self.saved[abs(lit)]: self.agility += 1 - 0.9999
        self._assign_lit_true(lit)
        self.trail.append((lit, reason))
        self.level[abs(lit)] = level

    def _add_clause(self, c: Iterable[int], watch1: int | None = None, watch2: int | None = None) -> Clause | None:
        c = list(set(c))
        if len(c) == 1:
            self.units.add(c[0])
            return None
        x = watch1 or c[0]
        y = (watch2 or c[0]) if (watch1 and x != c[0]) else (watch2 or c[1])
        clause = Clause(c, x, y)
        self.clauses.append(clause)
        self.watch[x].add(clause)
        self.watch[y].add(clause)
        return clause

    def _prune_lemmas(self) -> None:
        self.max_clauses += 300
        trail_clauses = set(reason for lit, reason in self.trail if reason is not None)
        num_to_keep = ((len(self.clauses) - self.first_learned) // 2) - len(trail_clauses)
        best_clauses = sorted(((c, len(c.lits)) for c in self.clauses[self.first_learned:] if c not in trail_clauses), key=lambda x: x[-1])[:num_to_keep]
        for clause in self.clauses[self.first_learned:]:
            for w in clause.watches:
                self.watch[w].remove(clause)
        self.clauses = self.clauses[:self.first_learned]
        for clause in itertools.chain(trail_clauses, (x[0] for x in best_clauses)):
            w1, w2 = clause.watches
            self._add_clause(clause.lits, w1, w2)

    def _backjump(self, backjump_level: int) -> None:
        while self.trail:
            lit, reason = self.trail[-1]
            if self.level[abs(lit)] <= backjump_level: return
            self.saved[abs(lit)] = lit > 0
            del self.assign[abs(lit)]
            del self.level[abs(lit)]
            self.free[abs(lit)] = True
            self.trail.pop()

    def solve(self, nvars: int, clauses: Iterable[Sequence[int]]) -> list[int] | Literal[False]:
        """A satisfying assignment as one literal per variable 1..nvars, or False if UNSAT."""
        # Every variable should be in either assign (mapped to its current assignment) or in free,
        # but never in both. free is a dict so that we can test membership efficiently and use
        # popitem as a rough proxy for VSIDS, since popitem always returns the most recently added.
        self.assign = {} # var -> True/False
        self.free = {i: True for i in range(1, nvars+1)}
        self.level = {} # var -> level in the trail
        self.agility = 1.0  # When this gets too low, we should restart.
        self.saved = dict((v, False) for v in range(1,nvars+1))  # Saved recent values of vars.
        self.units = set()  # All unit clauses, stored as literals.
        self.watch = dict(((x, set()) for x in range(-nvars,nvars+1) if x != 0))  # Watchlists
        self.clauses = []  # The clause database.
        for clause in clauses:
            if not clause: return False  # empty clause, UNSAT
            assert max(abs(x) for x in clause) <= nvars
            self._add_clause(clause)
        self.max_clauses = len(self.clauses) * 2  # The max clauses we'll learn before pruning.
        self.first_learned = len(self.clauses)  # Index of the first learned clause, for pruning.
        self.trail = [(l, None) for l in self.units]  # Tuples of (lit, reason).
        self.tp = 0  # Next unprocessed trail item.
        for unit in self.units:
            if self._lit_falsified(unit): return False  # Conflicting units
            self._assign_lit_true(unit)

        current_level = 0
        while len(self.assign) < nvars or self.tp < len(self.trail):
            if len(self.clauses) > self.max_clauses: self._prune_lemmas()
            while self.tp < len(self.trail):  # Propagate pending implications
                lit, reason = self.trail[self.tp]
                self.level[abs(lit)] = current_level
                for clause in self.watch[-lit].copy():
                    if self._lit_satisfied(clause.other_watch(-lit)): continue
                    for clause_lit in clause.lits:  # Try to find another watch.
                        if clause_lit in clause.watches: continue
                        if self._lit_unassigned(clause_lit) or self._lit_satisfied(clause_lit):
                            clause.watches.remove(-lit)
                            clause.watches.add(clause_lit)
                            self.watch[-lit].remove(clause)
                            self.watch[clause_lit].add(clause)
                            break
                    if -lit in clause.watches:  # Did we fail in finding another watch?
                        forced = clause.other_watch(-lit)
                        if self._lit_falsified(forced):
                            if current_level == 0: return False  # UNSAT
                            # Resolve a conflict, producing a new clause. Then backjump.
                            stamp = dict((-l, True) for l in clause.lits)
                            resolved = set(clause.lits)
                            current_level_lits = set(l for l in resolved if self.level[abs(l)] == current_level)
                            backjump_level, backjump_lit = current_level, None
                            for trail_lit, trail_clause in reversed(self.trail):
                                if trail_clause is None: continue  # Decision
                                if stamp.get(trail_lit):
                                    for l in trail_clause.lits:
                                        if l != trail_lit: stamp[-l] = True
                                        if -l in resolved:
                                            if self.level[abs(l)] == current_level: current_level_lits.remove(-l)
                                            resolved.remove(-l)
                                        else:
                                            if self.level[abs(l)] == current_level: current_level_lits.add(l)
                                            resolved.add(l)
                                    if len(current_level_lits) == 1:
                                        backjump_level, backjump_lit = 0, None
                                        for l in resolved:
                                            if backjump_level < self.level[abs(l)] < current_level:
                                                backjump_level, backjump_lit = self.level[abs(l)], l
                                        break
                            decision_lit, = current_level_lits
                            self._backjump(backjump_level)
                            self.tp = len(self.trail) - 1  # We'll inc again at end of this loop
                            resolved_clause = self._add_clause(resolved, decision_lit, backjump_lit)
                            self._assign_and_add_to_trail(decision_lit, resolved_clause, backjump_level)
                            current_level = backjump_level
                            break
                        elif not self._lit_satisfied(forced):
                            self._assign_and_add_to_trail(forced, clause, current_level)
                self.tp += 1  # Move on to the next unprocessed trail entry.

            if len(self.assign) == nvars: break  # SAT

            # If we haven't been selecting many new values for variables, we're probably stuck in a
            # rut and should restart, wiping out variable assignments but keeping learned clauses.
            if self.agility < 0.20:
                self._backjump(0)
                self.tp = len(self.trail)
                self.agility = 1.0

            # Nothing left to propagate. Make a decision and assignment then start a new level.
            v = self.free.popitem()[0]
            self.assign[v] = self.saved[v]
            current_level += 1
            self.level[v] = current_level
            self.trail.append(((1 if self.assign[v] else -1) * v, None))

        return [l * (1 if v else -1) for l, v in sorted(self.assign.items())]
