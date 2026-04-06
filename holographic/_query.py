"""
_query.py — Chainable query builder for HoloDB.

    results = (db.query()
        .where(role="engineer")
        .where(level="senior")
        .or_where(level="staff")
        .exclude(lang="figma")
        .range("age", lo=25, hi=45)
        .limit(10)
        .run())

Composite semantics:
  .where(f=v)      AND constraint — adds bind(role(f), val(v)) to probe
  .or_where(f=v)   OR branch — executes as separate probe, results unioned
  .exclude(f=v)    NOT — subtracts bind(role(f), val(v)) from probe
  .range(f, lo,hi) numeric range filter — applied post-search as exact filter
  .similar_to(id)  seed probe from a known record vector
  .limit(n)        cap results
  .offset(n)       skip first n results
  .order_by(field) sort by raw field value (post-hoc, string or numeric)
  .run()           execute and return list[QueryResult]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ._db import HoloDB


@dataclass
class QueryResult:
    """A single result from a query execution."""
    id: str
    score: float
    record: Any  # HoloRecord

    def __repr__(self) -> str:
        return f"QueryResult(id={self.id!r}, score={self.score:.3f})"

    def get(self, field_name: str) -> Any:
        return self.record.raw_values.get(field_name)

    def to_dict(self) -> dict:
        return {"id": self.id, "score": self.score, **self.record.raw_values}


class QueryBuilder:
    """
    Chainable query builder. Instantiated by db.query().
    Holds instructions; executes lazily on .run().
    """

    def __init__(self, db: "HoloDB"):
        self._db = db
        self._and_constraints: list[tuple[str, Any]] = []   # [(field, value), ...]
        self._not_constraints: list[tuple[str, Any]] = []
        self._or_branches: list[list[tuple[str, Any]]] = []  # each = AND group
        self._range_filters: list[tuple[str, float, float]] = []  # (field, lo, hi)
        self._seed_id: str | None = None
        self._limit: int | None = None
        self._offset: int = 0
        self._order_field: str | None = None
        self._order_asc: bool = True

    # ── Constraint builders ───────────────────────────────────────────────────

    def where(self, **kwargs: Any) -> "QueryBuilder":
        """AND constraint. Multiple calls are chained with AND."""
        for k, v in kwargs.items():
            self._and_constraints.append((k, v))
        return self

    def or_where(self, **kwargs: Any) -> "QueryBuilder":
        """
        Open an OR branch with these constraints.
        Results from all OR branches are unioned with the main AND probe.
        """
        self._or_branches.append([(k, v) for k, v in kwargs.items()])
        return self

    def exclude(self, **kwargs: Any) -> "QueryBuilder":
        """NOT constraint. Subtracts these bindings from the probe vector."""
        for k, v in kwargs.items():
            self._not_constraints.append((k, v))
        return self

    def range(self, field_name: str, lo: float, hi: float) -> "QueryBuilder":
        """Numeric range filter. Applied post-search as an exact value check."""
        self._range_filters.append((field_name, lo, hi))
        return self

    def similar_to(self, record_id: str) -> "QueryBuilder":
        """Seed the probe from an existing record's vector."""
        self._seed_id = record_id
        return self

    # ── Pagination / ordering ────────────────────────────────────────────────

    def limit(self, n: int) -> "QueryBuilder":
        self._limit = n
        return self

    def offset(self, n: int) -> "QueryBuilder":
        self._offset = n
        return self

    def order_by(self, field_name: str, ascending: bool = True) -> "QueryBuilder":
        self._order_field = field_name
        self._order_asc = ascending
        return self

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, top_k: int = 20) -> list[QueryResult]:
        """
        Execute the query. Returns QueryResult list sorted by score.

        Execution plan:
          1. Build the main AND probe vector from .where() constraints
          2. Execute OR branches as separate probes
          3. Union all scored results, take max score per record
          4. Post-filter: exact categorical match for .where() constraints
          5. Post-filter: exact categorical exclusion for .exclude() constraints
          6. Apply range filters (exact, on raw values)
          7. Sort, offset, limit
        """
        import numpy as np
        db = self._db

        # ── 1. Main AND probe ─────────────────────────────────────────────────
        probe = self._build_probe(self._and_constraints, [])

        if self._seed_id:
            seed_rec = db._records.get(self._seed_id)
            if seed_rec is not None:
                if probe is None:
                    probe = seed_rec.vector.copy()
                else:
                    from ._algebra import normalize_phase
                    probe = normalize_phase(probe + seed_rec.vector)

        # ── 2. Score records ──────────────────────────────────────────────────
        if probe is not None:
            scored = dict(db._batch_search(probe, top_k=len(db._records)))
        else:
            scored = {rid: 1.0 for rid in db._records}

        # ── 3a. Exact post-filter for the main AND constraints ────────────────
        # Applied NOW before OR union, so OR branches can add records back.
        main_scored = dict(scored)
        for fname, raw in self._and_constraints:
            if not self._is_numeric_field(db, fname, raw):
                raw_str = str(raw)
                main_scored = {
                    rid: s for rid, s in main_scored.items()
                    if str(db._records[rid].raw_values.get(fname)) == raw_str
                }

        # ── 3b. OR branches — each gets its own exact filter then unioned ─────
        or_scored: dict[str, float] = {}
        for branch in self._or_branches:
            branch_probe = self._build_probe(branch, [])
            if branch_probe is not None:
                branch_raw = dict(db._batch_search(branch_probe, top_k=len(db._records)))
            else:
                branch_raw = {rid: 1.0 for rid in db._records}

            # Exact filter for this branch's constraints
            for fname, raw in branch:
                if not self._is_numeric_field(db, fname, raw):
                    raw_str = str(raw)
                    branch_raw = {
                        rid: s for rid, s in branch_raw.items()
                        if str(db._records[rid].raw_values.get(fname)) == raw_str
                    }
            for rid, s in branch_raw.items():
                or_scored[rid] = max(or_scored.get(rid, -1.0), s)

        # Union: start from main AND results, add OR results
        scored = dict(main_scored)
        for rid, s in or_scored.items():
            scored[rid] = max(scored.get(rid, -1.0), s)

        # ── 5. Exact exclusion ────────────────────────────────────────────────
        for fname, raw in self._not_constraints:
            raw_str = str(raw)
            scored = {
                rid: s for rid, s in scored.items()
                if str(db._records[rid].raw_values.get(fname)) != raw_str
            }

        # ── 6. Range filters ──────────────────────────────────────────────────
        if self._range_filters:
            scored = {
                rid: s for rid, s in scored.items()
                if all(self._passes_range(db._records[rid], fn, lo, hi)
                       for fn, lo, hi in self._range_filters)
            }

        # ── 7. Build results ──────────────────────────────────────────────────
        results = [
            QueryResult(id=rid, score=s, record=db._records[rid])
            for rid, s in scored.items()
        ]

        # ── 8. Order ──────────────────────────────────────────────────────────
        if self._order_field:
            results.sort(
                key=lambda r: r.get(self._order_field) or 0,
                reverse=not self._order_asc,
            )
        else:
            results.sort(key=lambda r: r.score, reverse=True)

        # ── 9. Offset + limit ─────────────────────────────────────────────────
        results = results[self._offset:]
        if self._limit is not None:
            results = results[:self._limit]

        return results

    # ── Internals ─────────────────────────────────────────────────────────────

    def _is_numeric_field(self, db, fname: str, raw) -> bool:
        """True if this field should be treated as numeric (no exact post-filter)."""
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return True
        if db.schema_def:
            from ._schema import NumericField
            return isinstance(db.schema_def.get(fname), NumericField)
        return False

    def _build_probe(
        self,
        and_constraints: list[tuple[str, Any]],
        not_constraints: list[tuple[str, Any]],
    ):
        """Assemble a probe vector from AND and NOT constraints."""
        import numpy as np
        db = self._db

        bindings = []
        weight_sum = 0.0

        for fname, raw in and_constraints:
            role = db._role_vector(fname)
            val_vec, _ = db._encode_value(fname, raw)
            w = db._weight_for(fname)
            bindings.append(w * role * val_vec)   # bind = multiply
            weight_sum += w

        if not bindings:
            probe = None
        else:
            from ._algebra import normalize_phase
            probe = normalize_phase(np.sum(bindings, axis=0) / weight_sum)

        # Subtract NOT bindings
        if not_constraints and probe is not None:
            from ._algebra import normalize_phase
            for fname, raw in not_constraints:
                role = db._role_vector(fname)
                val_vec, _ = db._encode_value(fname, raw)
                w = db._weight_for(fname)
                probe = probe - 0.5 * w * (role * val_vec)
            probe = normalize_phase(probe)

        return probe

    def _passes_range(self, rec, field_name: str, lo: float, hi: float) -> bool:
        """Check if a record's raw value for field_name lies in [lo, hi]."""
        v = rec.raw_values.get(field_name)
        if v is None:
            return False
        try:
            return lo <= float(v) <= hi
        except (TypeError, ValueError):
            return False

    # ── Sugar ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return len(self.run(top_k=len(self._db._records)))

    def first(self) -> QueryResult | None:
        results = self.limit(1).run()
        return results[0] if results else None

    def ids(self) -> list[str]:
        return [r.id for r in self.run()]

    def to_dicts(self) -> list[dict]:
        return [r.to_dict() for r in self.run()]
