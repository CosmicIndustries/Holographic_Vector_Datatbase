"""
_registry.py — Symbol management layer.

SymbolRegistry   vocabulary of atomic phase vectors + vectorized cleanup memory
NumericEncoder   continuous scalar → phase rotation with ordinal similarity
SymbolSchema     structured symbol definitions via shared latent factors
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._algebra import (
    bind, normalize_phase, random_phase_vector,
    batch_similarity, similarity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Symbol
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Symbol:
    name: str
    vector: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"


# ─────────────────────────────────────────────────────────────────────────────
# SymbolRegistry
# ─────────────────────────────────────────────────────────────────────────────

class SymbolRegistry:
    """
    Vocabulary of atomic phase vectors.

    Maintains a cleanup-memory matrix (N×D complex) for vectorized
    nearest-neighbor lookup. The matrix is rebuilt lazily on first access
    after any insertion (_dirty flag).

    Role vectors (_role:*) and numeric internals (_num_*) are excluded from
    the cleanup matrix so they never pollute field-value searches.
    """

    # Prefixes excluded from cleanup matrix
    _INTERNAL_PREFIXES = ("_role:", "_num_", "_factor_base:", "_f:")

    def __init__(self, dim: int = 512, seed: Optional[int] = None):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._symbols: dict[str, Symbol] = {}
        self._matrix: Optional[np.ndarray] = None
        self._matrix_names: list[str] = []
        self._dirty = True

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def get_or_create(self, name: str, metadata: dict | None = None) -> Symbol:
        if name not in self._symbols:
            v = random_phase_vector(self.dim, self._rng)
            self._symbols[name] = Symbol(name=name, vector=v, metadata=metadata or {})
            self._dirty = True
        return self._symbols[name]

    def set(self, name: str, vector: np.ndarray, metadata: dict | None = None) -> Symbol:
        """Force-set a vector (used by SymbolSchema)."""
        sym = Symbol(name=name, vector=normalize_phase(vector), metadata=metadata or {})
        self._symbols[name] = sym
        self._dirty = True
        return sym

    def get(self, name: str) -> Optional[Symbol]:
        return self._symbols.get(name)

    def vector(self, name: str) -> np.ndarray:
        return self.get_or_create(name).vector

    def remove(self, name: str) -> None:
        if name in self._symbols:
            del self._symbols[name]
            self._dirty = True

    def all_names(self) -> list[str]:
        return list(self._symbols.keys())

    # ── Cleanup memory ────────────────────────────────────────────────────────

    def _rebuild_matrix(self) -> None:
        names, vecs = [], []
        for sym in self._symbols.values():
            if not any(sym.name.startswith(p) for p in self._INTERNAL_PREFIXES):
                names.append(sym.name)
                vecs.append(sym.vector)
        self._matrix_names = names
        self._matrix = np.stack(vecs) if vecs else np.zeros((0, self.dim), dtype=complex)
        self._dirty = False

    def nearest(self, query: np.ndarray, top_k: int = 5,
                include: list[str] | None = None,
                exclude: list[str] | None = None) -> list[tuple[str, float]]:
        """
        Vectorized nearest-neighbor in symbol space.

        include: if given, only search these symbols (typed field vocab).
        exclude: symbol names to skip regardless.
        """
        exclude_set = set(exclude or [])

        if include is not None:
            # Typed search: build mini matrix from vocab subset
            symbols = [self._symbols[n] for n in include if n in self._symbols]
            if not symbols:
                return []
            mat = np.stack([s.vector for s in symbols])
            sims = batch_similarity(mat, query)
            pairs = [(symbols[i].name, float(sims[i])) for i in range(len(symbols))
                     if symbols[i].name not in exclude_set]
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:top_k]

        # Global cleanup matrix search
        if self._dirty:
            self._rebuild_matrix()
        if self._matrix.shape[0] == 0:
            return []

        sims = batch_similarity(self._matrix, query)
        results = []
        for idx in np.argsort(-sims):
            name = self._matrix_names[idx]
            if name not in exclude_set:
                results.append((name, float(sims[idx])))
            if len(results) >= top_k:
                break
        return results

    def cleanup(self, v: np.ndarray, steps: int = 2,
                vocab: list[str] | None = None) -> np.ndarray:
        """Iteratively snap v to nearest valid symbol."""
        for _ in range(steps):
            hits = self.nearest(v, top_k=1, include=vocab)
            if not hits:
                break
            v = self.vector(hits[0][0])
        return v

    def __len__(self) -> int:
        return len(self._symbols)

    def __contains__(self, name: str) -> bool:
        return name in self._symbols

    def __repr__(self) -> str:
        return f"SymbolRegistry(dim={self.dim}, n={len(self)})"


# ─────────────────────────────────────────────────────────────────────────────
# NumericEncoder
# ─────────────────────────────────────────────────────────────────────────────

class NumericEncoder:
    """
    Continuous scalar encoding via multi-axis phase rotation.

    v = base ⊛ (dir_0 * exp(i·θ_0)) ⊛ (dir_1 * exp(i·θ_1)) ⊛ ...

    where θ_i = (value / (scale * 2^i)) * π/4

    Coarse axis (i=0) encodes full range; finer axes add sub-unit resolution.
    similarity(encode(a), encode(b)) ≈ cos(Δθ) — strictly ordinal.
    """

    def __init__(self, dim: int, registry: SymbolRegistry, n_axes: int = 3):
        self.dim = dim
        self._reg = registry
        self._scale: dict[str, float] = {}
        self._n_axes = n_axes

    def configure(self, field_name: str, scale: float) -> None:
        """Set expected value range. scale ≈ half the typical magnitude."""
        self._scale[field_name] = scale

    def encode(self, field_name: str, value: float) -> np.ndarray:
        scale = self._scale.get(field_name, 1.0)
        v = self._reg.vector(f"_num_base:{field_name}").copy()
        for i in range(self._n_axes):
            theta = (value / (scale * (2 ** i))) * (np.pi / 4.0)
            d = self._reg.vector(f"_num_dir:{field_name}:{i}")
            v = v * (d * np.exp(1j * theta))
        return normalize_phase(v)

    def similarity_to_range(self, field_name: str,
                            value: float, lo: float, hi: float) -> bool:
        """
        True if value lies within [lo, hi] using angular threshold.
        Compares the encoded value against encoded midpoint; checks that
        it is more similar to mid than to either boundary's outside edge.
        """
        mid = (lo + hi) / 2.0
        half_width = (hi - lo) / 2.0
        scale = self._scale.get(field_name, 1.0)

        v_value = self.encode(field_name, value)
        v_mid   = self.encode(field_name, mid)
        v_lo    = self.encode(field_name, lo - half_width)   # one half-width below lo
        v_hi    = self.encode(field_name, hi + half_width)   # one half-width above hi

        sim_mid  = similarity(v_value, v_mid)
        sim_edge = max(similarity(v_value, v_lo), similarity(v_value, v_hi))
        return sim_mid > sim_edge


# ─────────────────────────────────────────────────────────────────────────────
# SymbolSchema
# ─────────────────────────────────────────────────────────────────────────────

class SymbolSchema:
    """
    Structured symbol definitions via shared latent factors.

    Enables analogy purely from database structure — no training.

        schema.register_factor("gender", ["male", "female"])
        schema.register_factor("status", ["royal", "common"])
        schema.define_symbol("king",  gender="male",  status="royal")
        schema.define_symbol("queen", gender="female", status="royal")

        db.analogy("king", "queen", "man") → "woman" @ sim≈1.0
    """

    def __init__(self, registry: SymbolRegistry):
        self._reg = registry
        self._factors: dict[str, dict[str, np.ndarray]] = {}
        self._symbol_factors: dict[str, dict[str, str]] = {}

    def register_factor(self, axis: str, values: list[str]) -> "SymbolSchema":
        if axis not in self._factors:
            self._factors[axis] = {}
        base = self._reg.vector(f"_factor_base:{axis}")
        for val in values:
            key = f"{axis}:{val}"
            if val not in self._factors[axis]:
                seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
                angle = (seed / 0xFFFFFFFF) * 2 * np.pi
                v = normalize_phase(base * np.exp(1j * angle))
                self._factors[axis][val] = v
                self._reg.set(f"_f:{axis}:{val}", v)
        return self  # chainable

    def factor_vector(self, axis: str, value: str) -> np.ndarray:
        if axis not in self._factors or value not in self._factors[axis]:
            self.register_factor(axis, [value])
        return self._factors[axis][value]

    def define_symbol(self, name: str, **factors: str) -> np.ndarray:
        if not factors:
            raise ValueError("define_symbol requires at least one factor")
        axes = list(factors.items())
        v = self.factor_vector(*axes[0])
        for axis, val in axes[1:]:
            v = bind(v, self.factor_vector(axis, val))
        v = normalize_phase(v)
        self._reg.set(name, v)
        self._symbol_factors[name] = dict(factors)
        return v

    def factors_of(self, name: str) -> dict[str, str]:
        return dict(self._symbol_factors.get(name, {}))

    def symbols_sharing_factor(self, axis: str, value: str) -> list[str]:
        return [n for n, f in self._symbol_factors.items() if f.get(axis) == value]

    def axes(self) -> list[str]:
        return list(self._factors.keys())

    def values_for(self, axis: str) -> list[str]:
        return list(self._factors.get(axis, {}).keys())
