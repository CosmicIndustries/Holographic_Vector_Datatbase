"""
holographic — Phase HRR Associative Memory Database
====================================================

A vector database built on Holographic Reduced Representations.
All storage, retrieval, analogy, and arithmetic operate in a single
shared high-dimensional complex phase space.

Quick start:
    from holographic import HoloDB, Schema

    schema = (Schema()
        .text("name", weight=2.5)
        .enum("role", ["engineer", "designer", "manager"])
        .numeric("age", scale=40, lo=18, hi=80)
        .boolean("active"))

    db = HoloDB(dim=1024, schema=schema)

    db.insert("alice", name="alice", role="engineer", age=32, active=True)
    db.insert("bob",   name="bob",   role="designer", age=27, active=True)
    db.insert("carol", name="carol", role="engineer", age=41, active=False)

    # Structured query
    results = (db.query()
        .where(role="engineer")
        .range("age", lo=25, hi=45)
        .limit(5)
        .run())

    # Analogy (requires SymbolSchema structure)
    db.symbols_schema.register_factor("gender", ["male", "female"])
    db.symbols_schema.define_symbol("king",  gender="male",  status="royal")
    db.symbols_schema.define_symbol("queen", gender="female", status="royal")
    db.symbols_schema.define_symbol("man",   gender="male")
    db.symbols_schema.define_symbol("woman", gender="female")
    db.analogy("king", "queen", "man")   # → [("woman", ~1.0), ...]

    # Persistence
    db.save("mydb")
    db = HoloDB.load("mydb")
"""

from ._db import HoloDB, HoloRecord
from ._schema import Schema, TextField, EnumField, NumericField, BoolField
from ._query import QueryBuilder, QueryResult
from ._registry import SymbolRegistry, SymbolSchema, NumericEncoder
from ._algebra import bind, unbind, superpose, normalize_phase, similarity

__all__ = [
    # Core
    "HoloDB",
    "HoloRecord",
    # Schema
    "Schema",
    "TextField",
    "EnumField",
    "NumericField",
    "BoolField",
    # Query
    "QueryBuilder",
    "QueryResult",
    # Registry internals (for advanced use)
    "SymbolRegistry",
    "SymbolSchema",
    "NumericEncoder",
    # Algebra primitives
    "bind",
    "unbind",
    "superpose",
    "normalize_phase",
    "similarity",
]

__version__ = "3.0.0"
