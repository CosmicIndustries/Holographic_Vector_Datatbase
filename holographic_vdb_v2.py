"""
holographic_vdb_v2.py
======================
Holographic Vector Database — v2, research-grade.

Upgrades from v1:
  ① Phase HRR backend   — complex unit vectors, bind = a*b, unbind = a*conj(b)
                           Exact inverse. No FFT overhead. Lower noise accumulation.
  ② Cleanup memory      — vectorized matrix multiply for fast denoising.
                           Snap noisy probes back to valid symbols.
  ③ Memory normalization fix — additive (no destructive normalize on insert).
  ④ Memory sharding     — hash-route records to 16 independent shards.
                           Capacity scales ~linearly.
  ⑤ Weighted binding    — per-field importance weights.
  ⑥ Continuous numeric  — base + scalar*direction encoding.
                           Preserves ordinal relationships; 42 ≠ 43.

Math:
  Vectors: unit complex, v_i = exp(i*θ), θ ~ Uniform[0,2π]
  Bind:    a ⊛ b  = a * b        (elementwise complex multiply)
  Unbind:  m ⊙ k  = m * conj(k)  (exact inverse, no flip hack)
  Sim:     Re(v†·w) / (|v||w|)
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core algebra (Phase HRR)
# ─────────────────────────────────────────────────────────────────────────────

def random_phase_vector(dim: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Unit complex vector: v_i = exp(i*θ_i), θ_i ~ U[0,2π]."""
    rng = rng or np.random.default_rng()
    phases = rng.uniform(0.0, 2.0 * np.pi, dim)
    return np.exp(1j * phases)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Phase binding: a ⊛ b = a * b  (elementwise complex multiply)."""
    return a * b


def unbind(composite: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Phase unbinding: composite ⊙ key = composite * conj(key).
    Exact inverse — no approximation error from the flip hack."""
    return composite * np.conj(key)


def superpose(*vectors: np.ndarray) -> np.ndarray:
    """Superposition: sum + normalize to unit complex."""
    combined = np.sum(vectors, axis=0)
    mag = np.abs(combined)
    mag[mag < 1e-12] = 1.0
    return combined / mag


def normalize_phase(v: np.ndarray) -> np.ndarray:
    """Project each component back onto the unit circle."""
    mag = np.abs(v)
    mag[mag < 1e-12] = 1.0
    return v / mag


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Real part of normalized Hermitian inner product."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.real(np.vdot(a, b)) / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────
# Numeric encoding (continuous)
# ─────────────────────────────────────────────────────────────────────────────

class NumericEncoder:
    """
    Encode scalar values via phase rotation.

    v = base * exp(i * theta)   where theta = (value / scale) * (π/4)

    Similarity:  cos(Δθ) = cos((v1 - v2) / scale * π/4)
    → small Δ → sim near 1.0
    → large Δ → sim decays toward 0 or negative

    The normalize_phase trap: adding a direction vector then normalizing
    projects every component back onto the unit circle, destroying magnitude
    information entirely. Phase rotation avoids this — the rotation IS the
    encoding, so normalize_phase is never needed.

    Multi-axis mode (n_axes > 1): encodes at coarse + fine resolutions,
    giving better interpolation and less aliasing at large ranges.
    """

    def __init__(self, dim: int, registry: "SymbolRegistry", n_axes: int = 3):
        self.dim = dim
        self._registry = registry
        self._scale: dict[str, float] = {}
        self._n_axes = n_axes

    def register_field(self, field_name: str, scale: float = 1.0) -> None:
        """Set the expected value range for a field.
        scale ≈ typical magnitude of values (e.g. 100 for age 0-100).
        """
        self._scale[field_name] = scale

    def encode(self, field_name: str, value: float) -> np.ndarray:
        """
        Multi-axis phase rotation encoding.

        Each axis i encodes at resolution scale * 2^i, so:
          axis 0 → coarse (full range)
          axis 1 → medium
          axis 2 → fine   (sub-unit precision)

        Combined vector = base ⊛ rot_0 ⊛ rot_1 ⊛ rot_2
        (binding preserves the rotation structure in superposition)
        """
        scale = self._scale.get(field_name, 1.0)
        v = self._registry.vector(f"_num_base:{field_name}").copy()

        for i in range(self._n_axes):
            # Finer axis = smaller scale divisor = more sensitive rotation
            axis_scale = scale * (2 ** i)
            theta = (value / axis_scale) * (np.pi / 4.0)
            dir_vec = self._registry.vector(f"_num_dir:{field_name}:{i}")
            # Rotate dir_vec by theta and bind into v
            rotated = dir_vec * np.exp(1j * theta)
            v = v * rotated  # bind (phase multiply)

        # Final normalize to unit phase — safe here because the rotation
        # IS fully encoded in the phase angles, not in magnitude.
        return normalize_phase(v)


# ─────────────────────────────────────────────────────────────────────────────
# Symbol Registry with vectorized cleanup memory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Symbol:
    name: str
    vector: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"


class SymbolRegistry:
    """
    Vocabulary of atomic symbols.

    Features:
      - Vectorized nearest-neighbor via matrix multiply (cleanup memory)
      - Separate fast-path matrix excluding role vectors
    """

    def __init__(self, dim: int = 512, seed: Optional[int] = None):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._symbols: dict[str, Symbol] = {}
        self._matrix: Optional[np.ndarray] = None   # N×D complex matrix
        self._matrix_names: list[str] = []
        self._dirty = True

    # ── symbol management ─────────────────────────────────────────────────

    def get_or_create(self, name: str, metadata: dict | None = None) -> Symbol:
        if name not in self._symbols:
            v = random_phase_vector(self.dim, self._rng)
            self._symbols[name] = Symbol(name=name, vector=v, metadata=metadata or {})
            self._dirty = True
        return self._symbols[name]

    def get(self, name: str) -> Optional[Symbol]:
        return self._symbols.get(name)

    def vector(self, name: str) -> np.ndarray:
        return self.get_or_create(name).vector

    # ── cleanup memory (vectorized) ────────────────────────────────────────

    def _rebuild_matrix(self) -> None:
        """Precompute cleanup matrix from all non-role symbols."""
        names, vecs = [], []
        for sym in self._symbols.values():
            if not sym.name.startswith("_role:") and not sym.name.startswith("_num_"):
                names.append(sym.name)
                vecs.append(sym.vector)
        self._matrix_names = names
        if vecs:
            self._matrix = np.stack(vecs)          # shape (N, D) complex
        else:
            self._matrix = np.zeros((0, self.dim), dtype=complex)
        self._dirty = False

    def nearest_fast(self, query: np.ndarray, top_k: int = 5,
                     exclude: list[str] | None = None) -> list[tuple[str, float]]:
        """
        Vectorized cleanup-memory nearest search.
        Returns top_k (name, similarity) pairs.
        """
        if self._dirty:
            self._rebuild_matrix()
        if self._matrix.shape[0] == 0:
            return []

        # Re(M @ conj(q)) — batch Hermitian inner products
        sims = np.real(self._matrix @ np.conj(query))
        # Proper cosine normalization: ||row|| * ||query||
        row_norms = np.linalg.norm(self._matrix, axis=1) + 1e-12
        q_norm    = float(np.linalg.norm(query)) or 1.0
        sims /= row_norms * q_norm

        exclude_set = set(exclude or [])
        results = []
        for idx in np.argsort(-sims):
            name = self._matrix_names[idx]
            if name not in exclude_set:
                results.append((name, float(sims[idx])))
            if len(results) >= top_k:
                break
        return results

    def cleanup(self, v: np.ndarray, steps: int = 2) -> np.ndarray:
        """
        Iterative cleanup: snap noisy vector to nearest clean symbol.
        Each step reduces noise by projecting onto the winner.
        """
        for _ in range(steps):
            hits = self.nearest_fast(v, top_k=1)
            if not hits:
                break
            v = self.vector(hits[0][0])
        return v

    # ── legacy nearest (still available) ──────────────────────────────────

    def nearest(self, query: np.ndarray, top_k: int = 5,
                exclude: list[str] | None = None) -> list[tuple[str, float]]:
        return self.nearest_fast(query, top_k=top_k, exclude=exclude)

    def __len__(self) -> int:
        return len(self._symbols)

    def __contains__(self, name: str) -> bool:
        return name in self._symbols

    def __repr__(self) -> str:
        return f"SymbolRegistry(dim={self.dim}, symbols={len(self)})"


# ─────────────────────────────────────────────────────────────────────────────
# HoloRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HoloRecord:
    id: str
    vector: np.ndarray
    fields: dict[str, str]
    raw_values: dict[str, Any]
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"HoloRecord(id={self.id!r}, fields={list(self.fields.keys())})"


# ─────────────────────────────────────────────────────────────────────────────
# Default field weights
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "id":       3.0,
    "name":     2.5,
    "type":     2.0,
    "role":     2.0,
    "category": 1.8,
    "lang":     1.5,
    "level":    1.2,
}


# ─────────────────────────────────────────────────────────────────────────────
# HolographicVDB v2
# ─────────────────────────────────────────────────────────────────────────────

class HolographicVDB:
    """
    Holographic Vector Database — v2.

    Core operations:
      insert(id, fields)            — encode + store
      query_field(id, field)        — probe → cleanup → nearest symbol
      search(query)                 — cosine similarity over records
      search_by_fields(partial)     — partial-match fuzzy query
      analogy(a, b, c)              — a:b :: c:?
      aggregate(ids)                — cluster centroid
      relational_edit(id, field, old_val, new_val) — surgical field swap
    """

    def __init__(
        self,
        dim: int = 512,
        seed: Optional[int] = None,
        num_shards: int = 16,
        field_weights: dict[str, float] | None = None,
    ):
        self.dim = dim
        self.registry = SymbolRegistry(dim=dim, seed=seed)
        self.numeric  = NumericEncoder(dim=dim, registry=self.registry, n_axes=3)
        self._records: dict[str, HoloRecord] = {}

        # Sharded memory: hash(record_id) → shard index
        self._num_shards = num_shards
        self._shards: list[np.ndarray] = [
            np.zeros(dim, dtype=complex) for _ in range(num_shards)
        ]

        self._field_weights = {**DEFAULT_FIELD_WEIGHTS, **(field_weights or {})}

    # ── insert ─────────────────────────────────────────────────────────────

    def insert(self, record_id: str, fields: dict[str, Any],
               metadata: dict | None = None) -> HoloRecord:
        """Encode a structured record into a holographic vector and store it."""
        binding_vecs = []
        field_map: dict[str, str] = {}
        weight_sum = 0.0

        for fname, raw in fields.items():
            role_vec  = self.registry.vector(f"_role:{fname}")
            value_vec, sym_name = self._encode_value(fname, raw)
            field_map[fname] = sym_name

            w = self._field_weights.get(fname, 1.0)
            binding_vecs.append(w * bind(role_vec, value_vec))
            weight_sum += w

        if binding_vecs:
            combined = np.sum(binding_vecs, axis=0) / weight_sum
            record_vec = normalize_phase(combined)
        else:
            record_vec = random_phase_vector(self.dim)

        rec = HoloRecord(
            id=record_id,
            vector=record_vec,
            fields=field_map,
            raw_values=dict(fields),
            metadata=metadata or {},
        )
        self._records[record_id] = rec

        # Route to shard (additive — no destructive normalize)
        shard_idx = self._shard_idx(record_id)
        self._shards[shard_idx] += record_vec

        return rec

    # ── field retrieval ─────────────────────────────────────────────────────

    def query_field(self, record_id: str, field_name: str,
                    top_k: int = 3, cleanup_steps: int = 2) -> list[tuple[str, float]]:
        """
        Probe a record for a field value.
        Applies cleanup memory to denoise the retrieved vector.
        """
        rec = self._records.get(record_id)
        if rec is None:
            raise KeyError(f"Record {record_id!r} not found")

        role_vec = self.registry.vector(f"_role:{field_name}")
        probed   = unbind(rec.vector, role_vec)

        # Cleanup: snap to nearest valid symbol (reduces drift)
        if cleanup_steps > 0:
            probed = self.registry.cleanup(probed, steps=cleanup_steps)

        candidates = self.registry.nearest_fast(
            probed, top_k=top_k,
            exclude=[k for k in self.registry._symbols if k.startswith("_")]
        )
        return candidates

    def get_field(self, record_id: str, field_name: str) -> str:
        results = self.query_field(record_id, field_name, top_k=1)
        return results[0][0] if results else ""

    # ── similarity search ────────────────────────────────────────────────────

    def search(self, query: np.ndarray | str, top_k: int = 5) -> list[tuple[str, float]]:
        if isinstance(query, str):
            query = self.registry.vector(query)
        scores = [
            (rid, cosine_similarity(query, rec.vector))
            for rid, rec in self._records.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search_by_fields(self, fields: dict[str, Any],
                         top_k: int = 5) -> list[tuple[str, float]]:
        """Build a probe from partial field specs, then search."""
        binding_vecs = []
        weight_sum = 0.0
        for fname, raw in fields.items():
            role_vec  = self.registry.vector(f"_role:{fname}")
            value_vec, _ = self._encode_value(fname, raw)
            w = self._field_weights.get(fname, 1.0)
            binding_vecs.append(w * bind(role_vec, value_vec))
            weight_sum += w

        if not binding_vecs:
            return []
        probe = normalize_phase(np.sum(binding_vecs, axis=0) / weight_sum)
        return self.search(probe, top_k=top_k)

    # ── analogy ──────────────────────────────────────────────────────────────

    def analogy(self, a: str, b: str, c: str,
                top_k: int = 5) -> list[tuple[str, float]]:
        """a:b :: c:?  via Phase HRR: (b ⊙ a) ⊛ c"""
        va = self.registry.vector(a)
        vb = self.registry.vector(b)
        vc = self.registry.vector(c)
        relationship = unbind(vb, va)          # extract transform a→b
        query = normalize_phase(bind(relationship, vc))  # apply to c
        return self.registry.nearest_fast(query, top_k=top_k, exclude=[a, b, c])

    # ── relational edit (killer feature) ─────────────────────────────────────

    def relational_edit(
        self, record_id: str, field_name: str,
        old_value: Any, new_value: Any
    ) -> HoloRecord:
        """
        Surgically swap one field value, preserving all others.
        e.g. "find alice but with rust instead of python"

        H' = H - w*(role ⊛ old_val) + w*(role ⊛ new_val)
        """
        rec = self._records.get(record_id)
        if rec is None:
            raise KeyError(f"Record {record_id!r} not found")

        role_vec  = self.registry.vector(f"_role:{field_name}")
        old_vec, _ = self._encode_value(field_name, old_value)
        new_vec, _ = self._encode_value(field_name, new_value)
        w = self._field_weights.get(field_name, 1.0)

        modified = rec.vector - w * bind(role_vec, old_vec) + w * bind(role_vec, new_vec)
        modified = normalize_phase(modified)

        # Store as a new ephemeral record
        edited = HoloRecord(
            id=f"{record_id}+{field_name}={new_value}",
            vector=modified,
            fields={**rec.fields, field_name: str(new_value)},
            raw_values={**rec.raw_values, field_name: new_value},
            metadata={"derived_from": record_id},
        )
        return edited

    # ── aggregate ─────────────────────────────────────────────────────────────

    def aggregate(self, record_ids: list[str]) -> np.ndarray:
        vecs = [self._records[rid].vector for rid in record_ids if rid in self._records]
        return superpose(*vecs) if vecs else np.zeros(self.dim, dtype=complex)

    def cluster_probe(self, vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        return self.search(vector, top_k=top_k)

    # ── shard memory query ────────────────────────────────────────────────────

    def shard_search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Multi-stage: find candidates via shard similarity, then re-rank.
        Scales to large databases without linear scan over all records.
        """
        if isinstance(query, str):
            query = self.registry.vector(query)

        # Phase 1: score shards
        shard_sims = [
            cosine_similarity(query, s) if np.any(s != 0) else 0.0
            for s in self._shards
        ]
        hot_shards = sorted(range(self._num_shards),
                            key=lambda i: shard_sims[i], reverse=True)[:4]

        # Phase 2: full scan only within hot shards
        candidates = []
        for rid, rec in self._records.items():
            if self._shard_idx(rid) in hot_shards:
                candidates.append((rid, cosine_similarity(query, rec.vector)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # ── similarity matrix ─────────────────────────────────────────────────────

    def similarity_matrix(self, ids: list[str] | None = None) -> tuple[list[str], np.ndarray]:
        ids = ids or list(self._records.keys())
        n = len(ids)
        mat = np.zeros((n, n))
        vecs = [self._records[i].vector for i in ids]
        for i in range(n):
            for j in range(n):
                mat[i, j] = cosine_similarity(vecs[i], vecs[j])
        return ids, mat

    # ── introspection ──────────────────────────────────────────────────────────

    def get(self, record_id: str) -> Optional[HoloRecord]:
        return self._records.get(record_id)

    def all_ids(self) -> list[str]:
        return list(self._records.keys())

    def capacity_stats(self) -> dict:
        n = len(self._records)
        return {
            "records": n,
            "symbols": len(self.registry),
            "dim": self.dim,
            "shards": self._num_shards,
            "records_per_shard": n / self._num_shards,
            "theoretical_capacity": int(self.dim * 0.15),  # ~15% of dim for reliable recall
        }

    # ── persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        with open(Path(path), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "HolographicVDB":
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    # ── internals ──────────────────────────────────────────────────────────────

    def _shard_idx(self, record_id: str) -> int:
        return int(hashlib.md5(record_id.encode()).hexdigest(), 16) % self._num_shards

    def _encode_value(self, field_name: str, raw: Any) -> tuple[np.ndarray, str]:
        """Return (vector, symbol_name) for any Python value."""
        if isinstance(raw, bool):
            sym = f"{field_name}:{'true' if raw else 'false'}"
            return self.registry.vector(sym), sym
        if isinstance(raw, (int, float)):
            vec = self.numeric.encode(field_name, float(raw))
            sym = f"_numeric:{field_name}:{raw}"
            return vec, sym
        sym = str(raw)
        return self.registry.vector(sym), sym

    def __repr__(self) -> str:
        return (
            f"HolographicVDB_v2(dim={self.dim}, records={len(self._records)}, "
            f"symbols={len(self.registry)}, shards={self._num_shards})"
        )

    def __len__(self) -> int:
        return len(self._records)
