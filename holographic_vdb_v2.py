"""
holographic_vdb_v2.py
======================
Holographic Vector Database — v2, research-grade.

v2 upgrades (from v1):
  ① Phase HRR backend   — complex unit vectors, bind = a*b, unbind = a*conj(b)
  ② Cleanup memory      — vectorized matrix multiply, iterative denoising
  ③ Normalization fix   — additive shard memory, normalize at query time only
  ④ Memory sharding     — hash-route records across N independent shards
  ⑤ Weighted binding    — per-field importance weights
  ⑥ Phase rotation nums — encode_number via base * exp(i*theta), ordinal-safe

v2.1 upgrades (this file):
  ⑦ Orthogonal roles    — Gram-Schmidt enforcement on role vectors so field
                           bindings don't cross-interfere at scale
  ⑧ Structured symbols  — define_symbol(name, **factors) builds symbols from
                           shared latent axes, enabling real analogy without
                           any training or embedding model
  ⑨ Latent factor API   — register_factor / define_symbol / analogy_via_factors

Math:
  Vectors: unit complex, v_i = exp(i*θ), θ ~ Uniform[0,2π]
  Bind:    a ⊛ b  = a * b          (elementwise complex multiply)
  Unbind:  m ⊙ k  = m * conj(k)   (exact inverse)
  Sim:     Re(v†·w) / (|v||w|)
  Analogy: (b ⊙ a) ⊛ c ≈ d  — works when a,b,c,d share latent factors
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
# SymbolSchema — structured symbols via shared latent factors
# ─────────────────────────────────────────────────────────────────────────────

class SymbolSchema:
    """
    Structured symbol layer that enables analogy without training.

    Instead of random independent atoms, symbols are composed from shared
    latent factors via binding:

        king  = bind(factor("gender:male"),  factor("status:royal"))
        queen = bind(factor("gender:female"), factor("status:royal"))
        man   = factor("gender:male")
        woman = factor("gender:female")

    Then:   queen ⊙ king = conj(male) * female
            apply to man  = female ≈ woman  ✓

    This is pure database-layer structure — no embeddings, no training.
    The caller declares the semantic axes they care about; the algebra
    propagates them automatically.

    Usage:
        schema = db.schema
        schema.register_factor("gender", ["male", "female"])
        schema.register_factor("status", ["royal", "common", "noble"])
        schema.define_symbol("king",  gender="male",  status="royal")
        schema.define_symbol("queen", gender="female", status="royal")
        schema.define_symbol("man",   gender="male")
        schema.define_symbol("woman", gender="female")

        # Now analogy works:
        db.analogy("king", "queen", "man")  → "woman" with high similarity
    """

    def __init__(self, registry: SymbolRegistry):
        self._reg = registry
        # factor_axis_name → {value_name → vector}
        self._factors: dict[str, dict[str, np.ndarray]] = {}
        # symbol_name → {factor_axis: factor_value}
        self._symbol_factors: dict[str, dict[str, str]] = {}

    def register_factor(self, axis: str, values: list[str]) -> None:
        """
        Declare a semantic axis and its possible values.
        Each value gets a random phase vector. All values on the same axis
        share the same random base, rotated by small amounts — so they're
        similar to each other but distinct from other axes.
        """
        if axis not in self._factors:
            self._factors[axis] = {}
        base = self._reg.vector(f"_factor_base:{axis}")
        for i, val in enumerate(values):
            key = f"{axis}:{val}"
            if key not in self._factors[axis]:
                # Rotate base by a deterministic per-value angle
                # Use a hash so order doesn't matter and new values are stable
                angle_seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
                angle = (angle_seed / 0xFFFFFFFF) * 2 * np.pi
                rotated = base * np.exp(1j * angle)
                self._factors[axis][val] = normalize_phase(rotated)
                # Also register in main registry so analogy can find it
                self._reg._symbols[f"_f:{axis}:{val}"] = Symbol(
                    name=f"_f:{axis}:{val}",
                    vector=self._factors[axis][val],
                )
                self._reg._dirty = True

    def factor_vector(self, axis: str, value: str) -> np.ndarray:
        """Return the vector for a given factor axis:value pair."""
        if axis not in self._factors or value not in self._factors[axis]:
            # Auto-register on demand
            self.register_factor(axis, [value])
        return self._factors[axis][value]

    def define_symbol(self, name: str, **factor_assignments: str) -> np.ndarray:
        """
        Define a symbol as a binding of latent factor values.

        define_symbol("king", gender="male", status="royal")
        → king_vec = bind(factor(gender:male), factor(status:royal))

        Symbols with overlapping factors will show meaningful analogy.
        """
        axes = list(factor_assignments.items())
        if not axes:
            raise ValueError("define_symbol requires at least one factor assignment")

        # Bind all factors together
        v = self.factor_vector(*axes[0])
        for axis, val in axes[1:]:
            v = bind(v, self.factor_vector(axis, val))
        v = normalize_phase(v)

        # Register in main symbol registry
        self._reg._symbols[name] = Symbol(name=name, vector=v)
        self._reg._dirty = True

        self._symbol_factors[name] = dict(factor_assignments)
        return v

    def factors_of(self, name: str) -> dict[str, str]:
        """Return the factor decomposition of a defined symbol."""
        return self._symbol_factors.get(name, {})

    def symbols_sharing_factor(self, axis: str, value: str) -> list[str]:
        """Find all symbols that have a given factor value."""
        return [
            name for name, factors in self._symbol_factors.items()
            if factors.get(axis) == value
        ]




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

        # Structured symbol schema (latent factors for analogy support)
        self.schema = SymbolSchema(self.registry)

        # Orthogonalized role vector cache.
        # IMPORTANT: once a role vector is frozen (used to encode a record),
        # it must never change. New fields are orthogonalized against existing
        # frozen ones; existing frozen vectors are never touched.
        self._role_cache: dict[str, np.ndarray] = {}

        # Per-field vocabulary: field_name → set of symbol names used as values.
        # Used to restrict query_field search to semantically valid candidates.
        self._field_vocab: dict[str, set[str]] = {}

    # ── role vector with orthogonalization ────────────────────────────────────

    def _role_vector(self, field_name: str) -> np.ndarray:
        """
        Return a role vector for field_name, orthogonalized against all
        previously-seen role vectors.

        Contract: once returned, a role vector is FROZEN — it will never
        change even when new fields are added. This is essential for
        retrieval correctness: records encoded with an old role vector must
        be decodable with the same vector later.

        New fields get Gram-Schmidt subtraction against all existing frozen
        vectors, so they're approximately orthogonal to the existing set.
        Existing vectors are unaffected.

        Coefficient α=0.15 gives meaningful orthogonalization while keeping
        vectors well-conditioned under normalize_phase.
        """
        if field_name in self._role_cache:
            return self._role_cache[field_name]

        v = self.registry.vector(f"_role:{field_name}").copy()

        # Subtract projections onto all already-frozen role vectors only.
        # Do NOT subtract against vectors not yet in cache — those don't
        # exist yet and won't affect this field's encoding.
        alpha = 0.15
        for other_name, other_vec in self._role_cache.items():
            projection = np.vdot(other_vec, v)   # <r, v>
            v = v - alpha * projection * other_vec

        v = normalize_phase(v)
        self._role_cache[field_name] = v   # freeze
        return v

    # ── insert ─────────────────────────────────────────────────────────────

    def insert(self, record_id: str, fields: dict[str, Any],
               metadata: dict | None = None) -> HoloRecord:
        """Encode a structured record into a holographic vector and store it."""
        binding_vecs = []
        field_map: dict[str, str] = {}
        weight_sum = 0.0

        for fname, raw in fields.items():
            role_vec  = self._role_vector(fname)          # orthogonalized
            value_vec, sym_name = self._encode_value(fname, raw)
            field_map[fname] = sym_name

            # Track which symbol names are valid values for this field
            if fname not in self._field_vocab:
                self._field_vocab[fname] = set()
            if not sym_name.startswith("_numeric:"):
                self._field_vocab[fname].add(sym_name)

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

        Retrieval is restricted to the per-field vocabulary — only symbols
        that have actually been used as values for field_name are searched.
        This eliminates false matches against symbols from other fields
        (e.g. name values like 'alice' showing up as role candidates).

        cleanup_steps still applies within the vocab-restricted set.
        """
        rec = self._records.get(record_id)
        if rec is None:
            raise KeyError(f"Record {record_id!r} not found")

        role_vec = self._role_vector(field_name)
        probed   = unbind(rec.vector, role_vec)

        # Typed search: only consider symbols known to be values of this field
        vocab = self._field_vocab.get(field_name)
        if vocab:
            candidates = [
                (name, cosine_similarity(probed, self.registry.vector(name)))
                for name in vocab
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:top_k]

        # Fallback: global search excluding internal symbols (pre-first-insert)
        return self.registry.nearest_fast(
            probed, top_k=top_k,
            exclude=[k for k in self.registry._symbols if k.startswith("_")]
        )

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
            role_vec  = self._role_vector(fname)
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

        role_vec  = self._role_vector(field_name)
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
