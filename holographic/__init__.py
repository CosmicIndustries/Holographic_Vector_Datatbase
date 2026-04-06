"""
holographic — Phase HRR Associative Memory Database
====================================================

A vector database built on Holographic Reduced Representations.
All storage, retrieval, analogy, and arithmetic operate in a single
shared high-dimensional complex phase space.
"""

from ._db import HoloDB, HoloRecord
from ._schema import Schema, TextField, EnumField, NumericField, BoolField
from ._query import QueryBuilder, QueryResult
from ._registry import SymbolRegistry, SymbolSchema, NumericEncoder
from ._algebra import bind, unbind, superpose, normalize_phase, similarity

__all__ = [
    "HoloDB",
    "HoloRecord",
    "Schema",
    "TextField",
    "EnumField",
    "NumericField",
    "BoolField",
    "QueryBuilder",
    "QueryResult",
    "SymbolRegistry",
    "SymbolSchema",
    "NumericEncoder",
    "bind",
    "unbind",
    "superpose",
    "normalize_phase",
    "similarity",
]

__version__ = "3.0.0"
