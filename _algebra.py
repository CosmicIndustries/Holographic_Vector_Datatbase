"""
_algebra.py — Phase HRR core operations.

All vectors are unit complex: v_i = exp(i*θ_i), θ_i ~ U[0, 2π].

  bind(a, b)      = a * b              elementwise complex multiply
  unbind(m, k)    = m * conj(k)        exact inverse (no flip approximation)
  superpose(*vs)  = normalize(Σ v)     holographic superposition
  similarity(a,b) = Re(v†·w)/(|v||w|) cosine in complex Hilbert space
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def random_phase_vector(dim: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, dim))


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def unbind(composite: np.ndarray, key: np.ndarray) -> np.ndarray:
    return composite * np.conj(key)


def superpose(*vectors: np.ndarray) -> np.ndarray:
    combined = np.sum(vectors, axis=0)
    mag = np.abs(combined)
    mag[mag < 1e-12] = 1.0
    return combined / mag


def normalize_phase(v: np.ndarray) -> np.ndarray:
    mag = np.abs(v)
    mag[mag < 1e-12] = 1.0
    return v / mag


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.real(np.vdot(a, b)) / (na * nb))


def batch_similarity(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Vectorized cosine similarity: matrix (N×D) vs query (D,).
    Returns shape (N,) float array.
    Single matmul — O(ND) not O(N) Python loop iterations.
    """
    sims = np.real(matrix @ np.conj(query))
    row_norms = np.linalg.norm(matrix, axis=1) + 1e-12
    q_norm = float(np.linalg.norm(query)) or 1.0
    return sims / (row_norms * q_norm)
