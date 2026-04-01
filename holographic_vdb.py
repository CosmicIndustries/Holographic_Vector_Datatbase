"""
holographic_vdb.py
==================
Holographic Vector Database using Holographic Reduced Representations (HRRs).

Core math:
  - Vectors: unit-normalized Gaussian random vectors in R^D
  - Bind:    circular convolution  A ⊛ B  (associates two concepts)
  - Unbind:  circular correlation  A ⊙ B  (retrieves bound partner)
  - Store:   superposition  M = Σ (A_i ⊛ B_i)  (composite memory trace)
  - Query:   M ⊙ A ≈ B  (probe with key, recover approximate value)
  - Sim:     cosine similarity for nearest-neighbor retrieval

References:
  Plate (1995) "Holographic Reduced Representations"
  Kanerva (2009) "Hyperdimensional Computing"
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.fft import fft, ifft


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def random_vector(dim: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate a unit-norm Gaussian random vector (holographic atom)."""
    rng = rng or np.random.default_rng()
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Circular convolution: a ⊛ b
    Associates two vectors. Result is approximately orthogonal to both inputs.
    Commutative, approximately associative.
    """
    return np.real(ifft(fft(a) * fft(b)))


def unbind(composite: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Circular correlation: composite ⊙ key
    Approximately inverts binding: (a ⊛ b) ⊙ a ≈ b
    Uses the approximate inverse: a* = flip(a)
    """
    key_inv = np.roll(np.flip(key), 1)  # approximate inverse
    result = np.real(ifft(fft(composite) * fft(key_inv)))
    norm = np.linalg.norm(result)
    return result / norm if norm > 1e-10 else result


def superpose(*vectors: np.ndarray) -> np.ndarray:
    """
    Superposition (addition): stores multiple concepts in one vector.
    Information is distributed holographically — each part encodes the whole.
    """
    combined = np.sum(vectors, axis=0)
    norm = np.linalg.norm(combined)
    return combined / norm if norm > 1e-10 else combined


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity ∈ [-1, 1]. ~0 = orthogonal (unrelated)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


# ---------------------------------------------------------------------------
# Symbol Registry
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    name: str
    vector: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"


class SymbolRegistry:
    """
    Manages named atomic symbols — the vocabulary of the database.
    Each symbol gets a random orthogonal-ish unit vector.
    """

    def __init__(self, dim: int = 1024, seed: Optional[int] = None):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._symbols: dict[str, Symbol] = {}

    def get_or_create(self, name: str, metadata: dict | None = None) -> Symbol:
        if name not in self._symbols:
            v = random_vector(self.dim, self._rng)
            self._symbols[name] = Symbol(name=name, vector=v, metadata=metadata or {})
        return self._symbols[name]

    def get(self, name: str) -> Optional[Symbol]:
        return self._symbols.get(name)

    def vector(self, name: str) -> np.ndarray:
        return self.get_or_create(name).vector

    def nearest(self, query: np.ndarray, top_k: int = 5, exclude: list[str] | None = None) -> list[tuple[str, float]]:
        """Find closest symbols to a query vector by cosine similarity."""
        exclude = set(exclude or [])
        scores = [
            (sym.name, cosine_similarity(query, sym.vector))
            for sym in self._symbols.values()
            if sym.name not in exclude
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def __len__(self) -> int:
        return len(self._symbols)

    def __contains__(self, name: str) -> bool:
        return name in self._symbols

    def __repr__(self) -> str:
        return f"SymbolRegistry(dim={self.dim}, symbols={len(self)})"


# ---------------------------------------------------------------------------
# HoloRecord: a structured entity stored as a bound superposition
# ---------------------------------------------------------------------------

@dataclass
class HoloRecord:
    """
    A structured record encoded as a single holographic vector.

    Encoding: H = (role_1 ⊛ value_1) + (role_2 ⊛ value_2) + ...
    Retrieval: H ⊙ role_i ≈ value_i
    """
    id: str
    vector: np.ndarray
    fields: dict[str, str]       # field_name -> symbol_name
    raw_values: dict[str, Any]   # field_name -> original Python value
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"HoloRecord(id={self.id!r}, fields={list(self.fields.keys())})"


# ---------------------------------------------------------------------------
# HolographicVDB — main database class
# ---------------------------------------------------------------------------

class HolographicVDB:
    """
    Holographic Vector Database.

    Stores structured records as distributed holographic vectors.
    Supports:
      - insert(id, fields)          — encode and store a record
      - query_field(id, field)      — probe a record's field
      - search(query_vector)        — cosine similarity search
      - analogy(a, b, c)            — solve a:b :: c:? via HRR arithmetic
      - aggregate(ids)              — merge records into a superposition
      - cluster_probe(vector)       — find nearest known records
    """

    def __init__(self, dim: int = 1024, seed: Optional[int] = None):
        self.dim = dim
        self.registry = SymbolRegistry(dim=dim, seed=seed)
        self._records: dict[str, HoloRecord] = {}
        # Flat memory: superposition of all records (for mass search)
        self._memory: np.ndarray = np.zeros(dim)
        self._memory_ids: list[str] = []

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, record_id: str, fields: dict[str, Any], metadata: dict | None = None) -> HoloRecord:
        """
        Encode a dict of field→value pairs into a holographic vector and store it.

        Each value is coerced to a symbol name (str). Numeric values are
        discretized into named bins for clean HRR encoding.
        """
        bindings = []
        field_map: dict[str, str] = {}

        for field_name, raw_value in fields.items():
            sym_name = self._coerce_to_symbol(field_name, raw_value)
            field_map[field_name] = sym_name

            role_vec  = self.registry.vector(f"_role:{field_name}")
            value_vec = self.registry.vector(sym_name)
            bindings.append(bind(role_vec, value_vec))

        # Holographic record = superposition of all field bindings
        record_vec = superpose(*bindings) if bindings else random_vector(self.dim)

        rec = HoloRecord(
            id=record_id,
            vector=record_vec,
            fields=field_map,
            raw_values=dict(fields),
            metadata=metadata or {},
        )
        self._records[record_id] = rec

        # Update flat memory superposition
        self._memory = normalize(self._memory + record_vec)
        self._memory_ids.append(record_id)

        return rec

    # ------------------------------------------------------------------
    # Field retrieval
    # ------------------------------------------------------------------

    def query_field(self, record_id: str, field_name: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        Probe a stored record for the value of a given field.
        Returns ranked (symbol_name, similarity) pairs.
        """
        rec = self._records.get(record_id)
        if rec is None:
            raise KeyError(f"Record {record_id!r} not found")

        role_vec = self.registry.vector(f"_role:{field_name}")
        probed   = unbind(rec.vector, role_vec)

        # Search all non-role symbols for nearest match
        candidates = self.registry.nearest(
            probed, top_k=top_k,
            exclude=[k for k in self.registry._symbols if k.startswith("_role:")]
        )
        return candidates

    def get_field(self, record_id: str, field_name: str) -> str:
        """Convenience: return best-guess symbol name for a field."""
        results = self.query_field(record_id, field_name, top_k=1)
        return results[0][0] if results else ""

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def search(self, query: np.ndarray | str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Find stored records most similar to a query vector (or symbol name).
        Returns [(record_id, similarity), ...].
        """
        if isinstance(query, str):
            query = self.registry.vector(query)

        scores = [
            (rec_id, cosine_similarity(query, rec.vector))
            for rec_id, rec in self._records.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search_by_fields(self, fields: dict[str, Any], top_k: int = 5) -> list[tuple[str, float]]:
        """
        Build a probe vector from partial field specs, then search.
        Useful for partial-match / fuzzy queries.
        """
        bindings = []
        for field_name, raw_value in fields.items():
            sym_name  = self._coerce_to_symbol(field_name, raw_value)
            role_vec  = self.registry.vector(f"_role:{field_name}")
            value_vec = self.registry.vector(sym_name)
            bindings.append(bind(role_vec, value_vec))

        probe = superpose(*bindings) if bindings else np.zeros(self.dim)
        return self.search(probe, top_k=top_k)

    # ------------------------------------------------------------------
    # Analogy: a:b :: c:?
    # ------------------------------------------------------------------

    def analogy(self, a: str, b: str, c: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Holographic analogy: a:b :: c:?
        Uses HRR arithmetic: (b ⊙ a) ⊛ c ≈ d

        Example: king:queen :: man:? → woman
        """
        va = self.registry.vector(a)
        vb = self.registry.vector(b)
        vc = self.registry.vector(c)

        # Transform: extract relationship, apply to c
        relationship = unbind(vb, va)         # b ⊙ a ≈ "relationship"
        query = normalize(bind(relationship, vc))  # apply relationship to c

        return self.registry.nearest(query, top_k=top_k, exclude=[a, b, c])

    # ------------------------------------------------------------------
    # Aggregate / superposition of records
    # ------------------------------------------------------------------

    def aggregate(self, record_ids: list[str]) -> np.ndarray:
        """
        Superpose multiple records into one vector.
        The aggregate vector will be similar to all constituent records.
        """
        vecs = [self._records[rid].vector for rid in record_ids if rid in self._records]
        return superpose(*vecs) if vecs else np.zeros(self.dim)

    # ------------------------------------------------------------------
    # Cluster probe: probe flat memory
    # ------------------------------------------------------------------

    def cluster_probe(self, vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Find records nearest to an arbitrary vector."""
        return self.search(vector, top_k=top_k)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get(self, record_id: str) -> Optional[HoloRecord]:
        return self._records.get(record_id)

    def all_ids(self) -> list[str]:
        return list(self._records.keys())

    def similarity_matrix(self, ids: list[str] | None = None) -> tuple[list[str], np.ndarray]:
        """
        Compute pairwise cosine similarity matrix for a set of records.
        Returns (ids, matrix).
        """
        ids = ids or list(self._records.keys())
        n = len(ids)
        mat = np.zeros((n, n))
        vecs = [self._records[i].vector for i in ids]
        for i in range(n):
            for j in range(n):
                mat[i, j] = cosine_similarity(vecs[i], vecs[j])
        return ids, mat

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved to {path}")

    @staticmethod
    def load(path: str | Path) -> "HolographicVDB":
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _coerce_to_symbol(self, field_name: str, value: Any) -> str:
        """Convert a Python value to a clean symbol name."""
        if isinstance(value, bool):
            return f"{field_name}:{'true' if value else 'false'}"
        if isinstance(value, float):
            # Log-scale bucketing for floats
            bucket = _float_bucket(value)
            return f"{field_name}:{bucket}"
        if isinstance(value, int):
            bucket = _int_bucket(value)
            return f"{field_name}:{bucket}"
        return str(value)

    def __repr__(self) -> str:
        return (
            f"HolographicVDB(dim={self.dim}, "
            f"records={len(self._records)}, "
            f"symbols={len(self.registry)})"
        )

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Bucketing helpers for numeric values
# ---------------------------------------------------------------------------

def _int_bucket(n: int) -> str:
    """Discretize integers into named bins."""
    if n < 0:    return f"neg_{_int_bucket(-n)}"
    if n == 0:   return "zero"
    if n < 5:    return f"tiny"
    if n < 10:   return f"small"
    if n < 50:   return f"mid"
    if n < 200:  return f"large"
    if n < 1000: return f"xlarge"
    return "huge"


def _float_bucket(x: float) -> str:
    if math.isnan(x): return "nan"
    if math.isinf(x): return "inf" if x > 0 else "neginf"
    if x == 0.0:      return "zero"
    sign = "neg_" if x < 0 else ""
    x = abs(x)
    if x < 0.01:  return f"{sign}tiny"
    if x < 0.1:   return f"{sign}small"
    if x < 1.0:   return f"{sign}lt1"
    if x < 10.0:  return f"{sign}units"
    if x < 100.0: return f"{sign}tens"
    if x < 1e4:   return f"{sign}hundreds"
    return f"{sign}large"
