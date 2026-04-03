"""
_persist.py — Human-inspectable persistence for HoloDB.

Format:
  {path}.meta.json   — schema, field vocab, shard metadata, record raw_values
  {path}.vectors.npz — all complex numpy arrays (record + symbol vectors)

No pickle. JSON is inspectable/migratable; npz is compact and numpy-native.
"""

from __future__ import annotations

import json
import base64
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._db import HoloDB


def _vec_to_b64(v: np.ndarray) -> str:
    """Complex128 array → base64 string."""
    return base64.b64encode(v.astype(np.complex128).tobytes()).decode()


def _b64_to_vec(s: str, dim: int) -> np.ndarray:
    """Base64 string → complex128 array."""
    raw = base64.b64decode(s.encode())
    return np.frombuffer(raw, dtype=np.complex128).copy()


def save(db: "HoloDB", path: str | Path) -> None:
    """
    Persist the database to {path}.meta.json + {path}.vectors.npz.
    """
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    # ── Collect all vectors ───────────────────────────────────────────────────
    npz_arrays: dict[str, np.ndarray] = {}

    # Symbol vectors
    sym_names = list(db.symbols._symbols.keys())
    if sym_names:
        sym_mat = np.stack([db.symbols._symbols[n].vector for n in sym_names])
        npz_arrays["sym_vectors"] = sym_mat

    # Record vectors
    rec_ids = list(db._records.keys())
    if rec_ids:
        rec_mat = np.stack([db._records[rid].vector for rid in rec_ids])
        npz_arrays["rec_vectors"] = rec_mat

    # Shard vectors
    for i, shard in enumerate(db._shards):
        npz_arrays[f"shard_{i}"] = shard

    np.savez_compressed(str(base) + ".vectors.npz", **npz_arrays)

    # ── Metadata JSON ─────────────────────────────────────────────────────────
    # Schema
    schema_data: dict = {}
    if db.schema_def is not None:
        for fname, fdesc in db.schema_def._fields.items():
            schema_data[fname] = {
                "type": fdesc.type_name(),
                "weight": fdesc.weight,
            }
            if hasattr(fdesc, "values"):
                schema_data[fname]["values"] = fdesc.values
            if hasattr(fdesc, "scale"):
                schema_data[fname]["scale"] = fdesc.scale

    # Symbol metadata
    sym_meta = {}
    for n in sym_names:
        sym = db.symbols._symbols[n]
        sym_meta[n] = sym.metadata

    # Symbol schema (latent factor definitions)
    factor_axes: dict[str, list[str]] = {}
    for axis, vals in db.symbols_schema._factors.items():
        factor_axes[axis] = list(vals.keys())
    symbol_factor_map = dict(db.symbols_schema._symbol_factors)

    # Records
    records_data = {}
    for rid in rec_ids:
        rec = db._records[rid]
        records_data[rid] = {
            "fields": rec.fields,
            "raw_values": _serialize_raw(rec.raw_values),
            "metadata": rec.metadata,
        }

    # Field vocab
    field_vocab = {
        k: list(v) for k, v in db._field_vocab.items()
    }

    # Role cache keys (so we know field ordering on reload)
    role_cache_keys = list(db._role_cache.keys())
    role_cache_vecs: dict[str, np.ndarray] = {}
    for k, v in db._role_cache.items():
        npz_arrays[f"role_{k}"] = v

    # Re-save npz with role cache included
    np.savez_compressed(str(base) + ".vectors.npz", **npz_arrays)

    meta = {
        "dim": db.dim,
        "num_shards": db._num_shards,
        "sym_names": sym_names,
        "sym_meta": sym_meta,
        "rec_ids": rec_ids,
        "schema": schema_data,
        "factor_axes": factor_axes,
        "symbol_factor_map": symbol_factor_map,
        "records": records_data,
        "field_vocab": field_vocab,
        "role_cache_keys": role_cache_keys,
        "numeric_scales": dict(db.numeric._scale),
        "field_weights": db._field_weights,
    }

    with open(str(base) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load(path: str | Path) -> "HoloDB":
    """Reconstruct a HoloDB from {path}.meta.json + {path}.vectors.npz."""
    from ._db import HoloDB
    from ._registry import Symbol
    from ._schema import Schema, TextField, EnumField, NumericField, BoolField

    base = Path(path)
    with open(str(base) + ".meta.json") as f:
        meta = json.load(f)

    npz = np.load(str(base) + ".vectors.npz")

    dim = meta["dim"]
    num_shards = meta["num_shards"]

    # ── Rebuild schema ────────────────────────────────────────────────────────
    schema_def = None
    if meta.get("schema"):
        schema_def = Schema()
        for fname, fd in meta["schema"].items():
            t = fd["type"]
            w = fd.get("weight", 1.0)
            if t == "text":
                schema_def.field(fname, TextField(weight=w))
            elif t == "enum":
                schema_def.field(fname, EnumField(values=fd.get("values", []), weight=w))
            elif t == "numeric":
                schema_def.field(fname, NumericField(scale=fd.get("scale", 1.0), weight=w))
            elif t == "bool":
                schema_def.field(fname, BoolField(weight=w))

    # ── Create bare DB ────────────────────────────────────────────────────────
    db = HoloDB(dim=dim, num_shards=num_shards, schema=schema_def, _skip_init=True)
    db._field_weights = meta.get("field_weights", {})

    # ── Rebuild registry ──────────────────────────────────────────────────────
    sym_names = meta["sym_names"]
    sym_meta  = meta.get("sym_meta", {})
    if "sym_vectors" in npz:
        sym_mat = npz["sym_vectors"]
        for i, name in enumerate(sym_names):
            db.symbols._symbols[name] = Symbol(
                name=name,
                vector=sym_mat[i],
                metadata=sym_meta.get(name, {}),
            )
    db.symbols._dirty = True

    # ── Rebuild SymbolSchema ──────────────────────────────────────────────────
    for axis, vals in meta.get("factor_axes", {}).items():
        db.symbols_schema.register_factor(axis, vals)
    for sym_name, factors in meta.get("symbol_factor_map", {}).items():
        # Vector already in registry; just restore the mapping
        db.symbols_schema._symbol_factors[sym_name] = factors

    # ── Rebuild NumericEncoder ────────────────────────────────────────────────
    for fname, scale in meta.get("numeric_scales", {}).items():
        db.numeric.configure(fname, scale)

    # ── Restore shards ────────────────────────────────────────────────────────
    db._shards = []
    for i in range(num_shards):
        key = f"shard_{i}"
        db._shards.append(npz[key].copy() if key in npz else np.zeros(dim, dtype=complex))

    # ── Restore role cache ────────────────────────────────────────────────────
    for k in meta.get("role_cache_keys", []):
        key = f"role_{k}"
        if key in npz:
            db._role_cache[k] = npz[key].copy()

    # ── Restore records ───────────────────────────────────────────────────────
    from ._db import HoloRecord
    rec_ids = meta["rec_ids"]
    if "rec_vectors" in npz:
        rec_mat = npz["rec_vectors"]
        rdata   = meta["records"]
        for i, rid in enumerate(rec_ids):
            rd = rdata[rid]
            db._records[rid] = HoloRecord(
                id=rid,
                vector=rec_mat[i],
                fields=rd["fields"],
                raw_values=_deserialize_raw(rd["raw_values"]),
                metadata=rd.get("metadata", {}),
            )

    # ── Restore field vocab ───────────────────────────────────────────────────
    for k, v in meta.get("field_vocab", {}).items():
        db._field_vocab[k] = set(v)

    # ── Rebuild vectorized record matrix ────────────────────────────────────
    db._rebuild_record_matrix()

    return db


def _serialize_raw(d: dict) -> dict:
    """Make raw_values JSON-serializable (handle numpy scalars etc.)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _deserialize_raw(d: dict) -> dict:
    return dict(d)
