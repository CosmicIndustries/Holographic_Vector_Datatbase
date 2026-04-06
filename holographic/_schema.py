"""
_schema.py — Typed field schema for HoloDB.

Field types:
  TextField()                    free-form string symbol
  EnumField(values)              constrained set of string values
  NumericField(scale)            continuous float with ordinal encoding
  BoolField()                    boolean (true/false symbols)
  VectorField()                  pre-computed external vector (pass-through)

Schema enforces types on insert and configures the numeric encoder
automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ─────────────────────────────────────────────────────────────────────────────
# Field type descriptors
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextField:
    """Free-form string. Any str value accepted, each becomes a symbol."""
    weight: float = 1.0

    def validate(self, value: Any) -> str:
        return str(value)

    def type_name(self) -> str:
        return "text"


@dataclass
class EnumField:
    """Constrained string enum. Rejects values not in the declared set."""
    values: list[str]
    weight: float = 1.5
    strict: bool = True  # if False, unknown values are accepted as new symbols

    def validate(self, value: Any) -> str:
        s = str(value)
        if self.strict and s not in self.values:
            raise ValueError(f"Value {s!r} not in enum {self.values}")
        return s

    def type_name(self) -> str:
        return "enum"


@dataclass
class NumericField:
    """
    Continuous float field with ordinal phase encoding.
    scale: expected half-range of values (e.g. 50 for age 0-100).
    """
    scale: float = 1.0
    weight: float = 1.0
    lo: float | None = None   # optional hard bounds for validation
    hi: float | None = None

    def validate(self, value: Any) -> float:
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"NumericField expects a number, got {value!r}")
        if self.lo is not None and f < self.lo:
            raise ValueError(f"Value {f} below minimum {self.lo}")
        if self.hi is not None and f > self.hi:
            raise ValueError(f"Value {f} above maximum {self.hi}")
        return f

    def type_name(self) -> str:
        return "numeric"


@dataclass
class BoolField:
    """Boolean field. Stored as symbols 'true' / 'false'."""
    weight: float = 1.0

    def validate(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no"):
                return False
        raise ValueError(f"BoolField expects bool, got {value!r}")

    def type_name(self) -> str:
        return "bool"


# Union type for all field descriptors
FieldType = TextField | EnumField | NumericField | BoolField


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

class Schema:
    """
    Declares the fields and their types for a HoloDB instance.

    Usage:
        schema = Schema()
        schema.field("name",  TextField(weight=2.5))
        schema.field("role",  EnumField(["engineer", "designer", "manager"]))
        schema.field("age",   NumericField(scale=40, lo=0, hi=120))
        schema.field("active", BoolField())

        db = HoloDB(schema=schema, dim=1024)

    Undeclared fields are accepted with TextField semantics (open schema mode).
    """

    def __init__(self, strict: bool = False):
        self._fields: dict[str, FieldType] = {}
        self.strict = strict  # if True, reject undeclared fields on insert

    def field(self, name: str, descriptor: FieldType) -> "Schema":
        """Declare a field. Chainable."""
        self._fields[name] = descriptor
        return self

    # Convenience methods
    def text(self, name: str, weight: float = 1.0) -> "Schema":
        return self.field(name, TextField(weight=weight))

    def enum(self, name: str, values: list[str],
             weight: float = 1.5, strict: bool = True) -> "Schema":
        return self.field(name, EnumField(values=values, weight=weight, strict=strict))

    def numeric(self, name: str, scale: float = 1.0,
                weight: float = 1.0,
                lo: float | None = None,
                hi: float | None = None) -> "Schema":
        return self.field(name, NumericField(scale=scale, weight=weight, lo=lo, hi=hi))

    def boolean(self, name: str, weight: float = 1.0) -> "Schema":
        return self.field(name, BoolField(weight=weight))

    # Inspection
    def get(self, name: str) -> FieldType | None:
        return self._fields.get(name)

    def declared_fields(self) -> list[str]:
        return list(self._fields.keys())

    def numeric_fields(self) -> list[str]:
        return [n for n, f in self._fields.items() if isinstance(f, NumericField)]

    def validate(self, fields: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and coerce a record dict against the schema.
        Returns cleaned values. Raises ValueError on constraint violation.
        """
        out = {}
        for name, value in fields.items():
            descriptor = self._fields.get(name)
            if descriptor is None:
                if self.strict:
                    raise ValueError(
                        f"Field {name!r} not declared in schema "
                        f"(declared: {self.declared_fields()})"
                    )
                out[name] = value  # pass through
            else:
                out[name] = descriptor.validate(value)
        return out

    def weight_for(self, field_name: str) -> float:
        d = self._fields.get(field_name)
        return d.weight if d else 1.0

    def __repr__(self) -> str:
        parts = [f"{n}: {d.type_name()}" for n, d in self._fields.items()]
        return f"Schema({', '.join(parts)})"
