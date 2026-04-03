"""
_db.py — HoloDB: the main database class.

Features:
  Full CRUD          insert / get / update / delete
  Vectorized search  record matrix (N×D matmul) — no Python loop
  Query builder      db.query().where(...).range(...).limit(n).run()
  Transactions       atomic multi-record operations with rollback
  Analogy            a:b :: c:? with structured SymbolSchema support
  Aggregation        centroid / superposition of record sets
  Relational edit    surgical field swap preserving all other bindings
  Export             to_records() / to_json()
  Persistence        save(path) / HoloDB.load(path)
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from ._algebra import (
    bind, unbind, superpose, normalize_phase,
    random_phase_vector, similarity, batch_similarity,
)
from ._registry import NumericEncoder, SymbolRegistry, SymbolSchema
from ._schema import NumericField, Schema
from ._query import QueryBuilder, QueryResult


# ─────────────────────────────────────────────────────────────────────────────
# HoloRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HoloRecord:
    id: str
    vector: np.ndarray
    fields: dict[str, str]        # field → symbol name
    raw_values: dict[str, Any]    # field → original Python value
    metadata: dict = field(default_factory=dict)

    def get(self, field_name: str) -> Any:
        return self.raw_values.get(field_name)

    def to_dict(self) -> dict:
        return {"id": self.id, **self.raw_values}

    def __repr__(self) -> str:
        return f"HoloRecord(id={self.id!r}, fields={list(self.fields.keys())})"


# ─────────────────────────────────────────────────────────────────────────────
# Default field weights
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_WEIGHTS: dict[str, float] = {
    "id":       3.0,
    "name":     2.5,
    "type":     2.0,
    "role":     2.0,
    "category": 1.8,
    "lang":     1.5,
    "level":    1.2,
}


# ─────────────────────────────────────────────────────────────────────────────
# HoloDB
# ─────────────────────────────────────────────────────────────────────────────

class HoloDB:
    """
    Holographic Vector Database.

    Quick start:
        db = HoloDB(dim=1024)
        db.insert("alice", role="engineer", lang="python", level="senior")
        db.insert("bob",   role="designer", lang="figma",  level="mid")

        results = db.query().where(role="engineer").run()
        db.update("alice", level="staff")
        db.delete("bob")
        db.save("mydb")
        db2 = HoloDB.load("mydb")
    """

    def __init__(
        self,
        dim: int = 1024,
        seed: Optional[int] = None,
        num_shards: int = 16,
        schema: Optional[Schema] = None,
        field_weights: dict[str, float] | None = None,
        _skip_init: bool = False,
    ):
        if _skip_init:
            # Used by load() — caller fills all state manually
            self.dim = dim
            self._num_shards = num_shards
            self.schema_def = schema
            self.symbols = SymbolRegistry(dim=dim)
            self.numeric = NumericEncoder(dim=dim, registry=self.symbols)
            self.symbols_schema = SymbolSchema(self.symbols)
            self._records: dict[str, HoloRecord] = {}
            self._shards: list[np.ndarray] = [np.zeros(dim, dtype=complex)] * num_shards
            self._field_weights: dict[str, float] = {}
            self._field_vocab: dict[str, set[str]] = {}
            self._role_cache: dict[str, np.ndarray] = {}
            self._rec_matrix: Optional[np.ndarray] = None
            self._rec_ids_ordered: list[str] = []
            self._rec_matrix_dirty = True
            return

        self.dim = dim
        self._num_shards = num_shards
        self.schema_def = schema

        self.symbols = SymbolRegistry(dim=dim, seed=seed)
        self.numeric = NumericEncoder(dim=dim, registry=self.symbols)
        self.symbols_schema = SymbolSchema(self.symbols)

        self._records: dict[str, HoloRecord] = {}
        self._shards: list[np.ndarray] = [
            np.zeros(dim, dtype=complex) for _ in range(num_shards)
        ]

        # Merge schema-declared weights with caller overrides
        self._field_weights: dict[str, float] = dict(_DEFAULT_WEIGHTS)
        if schema:
            for fname in schema.declared_fields():
                self._field_weights[fname] = schema.weight_for(fname)
        if field_weights:
            self._field_weights.update(field_weights)

        # Configure numeric encoder from schema
        if schema:
            for fname in schema.numeric_fields():
                fdesc = schema.get(fname)
                self.numeric.configure(fname, fdesc.scale)

        self._field_vocab: dict[str, set[str]] = {}
        self._role_cache: dict[str, np.ndarray] = {}

        # Vectorized record matrix (rebuilt lazily)
        self._rec_matrix: Optional[np.ndarray] = None
        self._rec_ids_ordered: list[str] = []
        self._rec_matrix_dirty = True

    # ─────────────────────────────────────────────────────────────────────────
    # INSERT
    # ─────────────────────────────────────────────────────────────────────────

    def insert(self, record_id: str, _meta: dict | None = None, **fields: Any) -> HoloRecord:
        """
        Insert a record. Fields passed as keyword arguments.

            db.insert("alice", role="engineer", lang="python", age=32)
        """
        if not fields:
            raise ValueError("insert() requires at least one field")
        if record_id in self._records:
            raise KeyError(f"Record {record_id!r} already exists. Use update() to modify.")

        # Validate against schema if declared
        if self.schema_def:
            fields = self.schema_def.validate(fields)

        return self._encode_and_store(record_id, fields, metadata=_meta or {})

    def insert_many(self, records: list[tuple[str, dict]]) -> list[HoloRecord]:
        """Bulk insert. Each item is (record_id, fields_dict)."""
        return [self.insert(rid, **flds) for rid, flds in records]

    # ─────────────────────────────────────────────────────────────────────────
    # READ
    # ─────────────────────────────────────────────────────────────────────────

    def get(self, record_id: str) -> Optional[HoloRecord]:
        return self._records.get(record_id)

    def get_field(self, record_id: str, field_name: str) -> Any:
        """
        Holographic field probe: unbind the record with the role vector,
        clean up via typed vocab, return nearest symbol.
        """
        rec = self._records.get(record_id)
        if rec is None:
            raise KeyError(f"Record {record_id!r} not found")

        role_vec = self._role_vector(field_name)
        probed   = unbind(rec.vector, role_vec)

        vocab = list(self._field_vocab.get(field_name) or [])
        if vocab:
            hits = self.symbols.nearest(probed, top_k=1, include=vocab)
        else:
            hits = self.symbols.nearest(probed, top_k=1)

        return hits[0][0] if hits else None

    def probe_field(self, record_id: str, field_name: str,
                    top_k: int = 5) -> list[tuple[str, float]]:
        """Like get_field but returns ranked (symbol, similarity) list."""
        rec = self._records.get(record_id)
        if rec is None:
            raise KeyError(f"Record {record_id!r} not found")

        role_vec = self._role_vector(field_name)
        probed   = unbind(rec.vector, role_vec)
        vocab    = list(self._field_vocab.get(field_name) or [])

        return self.symbols.nearest(probed, top_k=top_k, include=vocab or None)

    def query(self) -> QueryBuilder:
        """Return a chainable query builder."""
        return QueryBuilder(self)

    def search(self, query: np.ndarray | str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Raw vector similarity search.
        query can be a symbol name or an arbitrary vector.
        """
        if isinstance(query, str):
            query = self.symbols.vector(query)
        return self._batch_search(query, top_k=top_k)

    def all_ids(self) -> list[str]:
        return list(self._records.keys())

    def all(self) -> list[HoloRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, record_id: str) -> bool:
        return record_id in self._records

    def __iter__(self) -> Iterator[HoloRecord]:
        return iter(self._records.values())

    # ─────────────────────────────────────────────────────────────────────────
    # UPDATE
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, record_id: str, **fields: Any) -> HoloRecord:
        """
        Update specific fields of an existing record.
        Unmodified fields are preserved exactly.
        """
        rec = self._records.get(record_id)
        if rec is None:
            raise KeyError(f"Record {record_id!r} not found")

        if self.schema_def:
            fields = self.schema_def.validate(fields)

        # Merge: start from existing raw values, overlay updates
        merged = {**rec.raw_values, **fields}

        # Remove old shard contribution
        shard_idx = self._shard_idx(record_id)
        self._shards[shard_idx] -= rec.vector

        # Re-encode and store
        new_rec = self._encode_and_store(
            record_id, merged,
            metadata=rec.metadata,
            _overwrite=True,
        )

        return new_rec

    def upsert(self, record_id: str, **fields: Any) -> HoloRecord:
        """Insert if not exists, update if exists."""
        if record_id in self._records:
            return self.update(record_id, **fields)
        return self.insert(record_id, **fields)

    # ─────────────────────────────────────────────────────────────────────────
    # DELETE
    # ─────────────────────────────────────────────────────────────────────────

    def delete(self, record_id: str) -> None:
        rec = self._records.get(record_id)
        if rec is None:
            raise KeyError(f"Record {record_id!r} not found")

        # Remove from shard memory
        shard_idx = self._shard_idx(record_id)
        self._shards[shard_idx] -= rec.vector

        del self._records[record_id]
        self._rec_matrix_dirty = True

    def delete_many(self, record_ids: list[str]) -> int:
        """Delete multiple records. Returns count deleted."""
        count = 0
        for rid in record_ids:
            if rid in self._records:
                self.delete(rid)
                count += 1
        return count

    def clear(self) -> None:
        """Remove all records (preserves schema and symbol vocab)."""
        self._records.clear()
        self._shards = [np.zeros(self.dim, dtype=complex) for _ in range(self._num_shards)]
        self._rec_matrix = None
        self._rec_ids_ordered = []
        self._rec_matrix_dirty = True
        self._field_vocab.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # TRANSACTIONS
    # ─────────────────────────────────────────────────────────────────────────

    @contextmanager
    def transaction(self):
        """
        Atomic batch operations. On exception, rolls back all changes made
        inside the context.

            with db.transaction() as tx:
                tx.insert("u1", role="engineer")
                tx.insert("u2", role="designer")
                # committed only if no exception

        The transaction object is the db itself (all methods available).
        Rollback restores the record dict and shard state from a snapshot.
        """
        snapshot_records = {rid: HoloRecord(
            id=r.id, vector=r.vector.copy(), fields=dict(r.fields),
            raw_values=dict(r.raw_values), metadata=dict(r.metadata),
        ) for rid, r in self._records.items()}
        snapshot_shards = [s.copy() for s in self._shards]
        snapshot_vocab  = {k: set(v) for k, v in self._field_vocab.items()}

        try:
            yield self
        except Exception:
            # Rollback
            self._records = snapshot_records
            self._shards  = snapshot_shards
            self._field_vocab = snapshot_vocab
            self._rec_matrix_dirty = True
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # ANALOGY
    # ─────────────────────────────────────────────────────────────────────────

    def analogy(self, a: str, b: str, c: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Solve a:b :: c:? via Phase HRR arithmetic.
        (b ⊙ a) ⊛ c ≈ d

        Works best when a/b/c are defined via db.symbols_schema.define_symbol().
        Falls back gracefully to raw symbol search otherwise.
        """
        va = self.symbols.vector(a)
        vb = self.symbols.vector(b)
        vc = self.symbols.vector(c)
        query = normalize_phase(bind(unbind(vb, va), vc))
        return self.symbols.nearest(query, top_k=top_k, exclude=[a, b, c])

    # ─────────────────────────────────────────────────────────────────────────
    # AGGREGATION
    # ─────────────────────────────────────────────────────────────────────────

    def aggregate(self, record_ids: list[str]) -> np.ndarray:
        """Holographic centroid of a set of records via superposition."""
        vecs = [self._records[rid].vector for rid in record_ids if rid in self._records]
        return superpose(*vecs) if vecs else np.zeros(self.dim, dtype=complex)

    def relational_edit(
        self, record_id: str, field_name: str,
        old_value: Any, new_value: Any,
    ) -> HoloRecord:
        """
        Return an ephemeral record that is identical to record_id except
        field_name is changed from old_value to new_value. Not stored.

        H' = H - w*(role ⊛ old) + w*(role ⊛ new)
        """
        rec = self._records.get(record_id)
        if rec is None:
            raise KeyError(f"Record {record_id!r} not found")

        role_vec = self._role_vector(field_name)
        old_vec, _ = self._encode_value(field_name, old_value)
        new_vec, _ = self._encode_value(field_name, new_value)
        w = self._weight_for(field_name)

        modified = normalize_phase(
            rec.vector - w * bind(role_vec, old_vec) + w * bind(role_vec, new_vec)
        )
        return HoloRecord(
            id=f"{record_id}[{field_name}={new_value}]",
            vector=modified,
            fields={**rec.fields, field_name: str(new_value)},
            raw_values={**rec.raw_values, field_name: new_value},
            metadata={"derived_from": record_id},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def to_records(self) -> list[dict]:
        """Export all records as plain dicts."""
        return [r.to_dict() for r in self._records.values()]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_records(), indent=indent)

    def stats(self) -> dict:
        n = len(self._records)
        return {
            "records":            n,
            "symbols":            len(self.symbols),
            "dim":                self.dim,
            "shards":             self._num_shards,
            "fields":             list(self._field_vocab.keys()),
            "theoretical_cap":    int(self.dim * 0.15),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        from ._persist import save as _save
        _save(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "HoloDB":
        from ._persist import load as _load
        return _load(path)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL — encoding
    # ─────────────────────────────────────────────────────────────────────────

    def _encode_and_store(
        self, record_id: str, fields: dict[str, Any],
        metadata: dict, _overwrite: bool = False,
    ) -> HoloRecord:
        binding_vecs = []
        field_map: dict[str, str] = {}
        weight_sum = 0.0

        for fname, raw in fields.items():
            role_vec  = self._role_vector(fname)
            val_vec, sym_name = self._encode_value(fname, raw)
            field_map[fname] = sym_name

            if not sym_name.startswith("_numeric:"):
                self._field_vocab.setdefault(fname, set()).add(sym_name)

            w = self._weight_for(fname)
            binding_vecs.append(w * bind(role_vec, val_vec))
            weight_sum += w

        combined = np.sum(binding_vecs, axis=0) / (weight_sum or 1.0)
        rec_vec  = normalize_phase(combined)

        rec = HoloRecord(
            id=record_id, vector=rec_vec,
            fields=field_map, raw_values=dict(fields), metadata=metadata,
        )
        self._records[record_id] = rec

        shard_idx = self._shard_idx(record_id)
        self._shards[shard_idx] += rec_vec
        self._rec_matrix_dirty = True

        return rec

    def _encode_value(self, fname: str, raw: Any) -> tuple[np.ndarray, str]:
        if isinstance(raw, bool):
            sym = f"{fname}:{'true' if raw else 'false'}"
            return self.symbols.vector(sym), sym
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            # Check if it's declared as numeric in schema
            if self.schema_def and isinstance(self.schema_def.get(fname), NumericField):
                vec = self.numeric.encode(fname, float(raw))
                return vec, f"_numeric:{fname}:{raw}"
            # Auto-detect: if field has no string vocab yet, treat as numeric
            if fname not in self._field_vocab:
                vec = self.numeric.encode(fname, float(raw))
                return vec, f"_numeric:{fname}:{raw}"
        sym = str(raw)
        return self.symbols.vector(sym), sym

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL — role vectors (frozen, orthogonalized)
    # ─────────────────────────────────────────────────────────────────────────

    def _role_vector(self, field_name: str) -> np.ndarray:
        if field_name in self._role_cache:
            return self._role_cache[field_name]

        v = self.symbols.vector(f"_role:{field_name}").copy()

        # Gram-Schmidt against existing frozen role vectors
        alpha = 0.15
        for other_vec in self._role_cache.values():
            proj = np.vdot(other_vec, v)
            v = v - alpha * proj * other_vec

        v = normalize_phase(v)
        self._role_cache[field_name] = v
        return v

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL — vectorized record search
    # ─────────────────────────────────────────────────────────────────────────

    def _rebuild_record_matrix(self) -> None:
        if not self._records:
            self._rec_matrix = np.zeros((0, self.dim), dtype=complex)
            self._rec_ids_ordered = []
        else:
            ids  = list(self._records.keys())
            vecs = np.stack([self._records[rid].vector for rid in ids])
            self._rec_matrix = vecs
            self._rec_ids_ordered = ids
        self._rec_matrix_dirty = False

    def _batch_search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Single matmul over all record vectors. O(N·D) not O(N) loop."""
        if self._rec_matrix_dirty:
            self._rebuild_record_matrix()
        if self._rec_matrix is None or self._rec_matrix.shape[0] == 0:
            return []

        sims = batch_similarity(self._rec_matrix, query)
        k    = min(top_k, len(self._rec_ids_ordered))
        idx  = np.argpartition(-sims, k - 1)[:k] if k > 0 else []
        idx  = idx[np.argsort(-sims[idx])]
        return [(self._rec_ids_ordered[i], float(sims[i])) for i in idx]

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL — helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _shard_idx(self, record_id: str) -> int:
        return int(hashlib.md5(record_id.encode()).hexdigest(), 16) % self._num_shards

    def _weight_for(self, field_name: str) -> float:
        return self._field_weights.get(field_name, 1.0)

    def __repr__(self) -> str:
        return (
            f"HoloDB(dim={self.dim}, records={len(self._records)}, "
            f"symbols={len(self.symbols)}, shards={self._num_shards})"
        )
